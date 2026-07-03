"""Service layer for the live meta-forecast web app.

Framework-agnostic pipeline: load the frozen encoder + anchor codebook (and the
matchup model) once, scrape the most recent tournaments (with a cached
fallback), embed them into weekly anchor distributions, and forecast the next
week with glide-to-anchor. Produces JSON-able data for the frontend:
  - fine (pokemon-core) and coarse (team-archetype) current-vs-predicted shares,
  - a 3D PCA projection of the meta space (rotatable),
  - a matchup-advantage directed graph (who beats whom), so underrepresented
    archetypes that counter the field stand out.
"""

from __future__ import annotations

import html
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from poke_env.data.normalize import to_id_str

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.decomposition import PCA

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data import tournaments as tr
from vgc_team.meta import ucsv
from vgc_team.meta.matchup.predict import expected_winrate_vs, load_matchup
from vgc_team.meta.mdm.dataset import TeamFeatures
from vgc_team.meta.mdm.interpret import anchor_to_cluster_distribution
from vgc_team.meta.timeseries import iso_week_label
from vgc_team.models.frozen_encoder import (
    ReferenceSpace,
    embed_teams,
    load_frozen_encoder,
    select_device,
)
from vgc_team.teams.pokepaste import parse_pokepaste
from vgc_team.teams.schema import Team

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt"
DEFAULT_REFERENCE = DATA_DIR / "processed" / "multi_reg" / "reference.npz"
DEFAULT_MATCHUP = PROJECT_ROOT / "models" / "matchup" / "matchup.pt"
MULTI_REG_DIR = DATA_DIR / "processed" / "tournaments" / "multi_reg"

OFF_FORMAT = ("UU", "CUSTOM", "SVI", "BSS", "2v2", "Little Cup", "Metronome", "LGPE", "Gen 8", "Gen8")

SAMPLE_PER_UNIT = 25  # teams sampled per archetype/core to estimate matchups
EDGE_THRESHOLD = 0.53  # draw a directed "beats" edge when win prob exceeds this


def team_archetype(team: Team) -> str:
    """Coarse team-archetype family from a team's abilities / species / moves."""

    ab = "".join((p.ability or "") for p in team.pokemon).lower().replace(" ", "")
    sp = " ".join(p.species.lower() for p in team.pokemon)
    mv = "".join(" ".join(p.moves) for p in team.pokemon).lower().replace(" ", "")
    if "drought" in ab or "torkoal" in sp:
        return "Sun"
    if "drizzle" in ab or "pelipper" in sp or "politoed" in sp:
        return "Rain"
    if "sandstream" in ab:
        return "Sand"
    if "snowwarning" in ab or "ninetales-alola" in sp:
        return "Snow"
    if "trickroom" in mv:
        return "Trick Room"
    if "tailwind" in mv:
        return "Tailwind"
    if "followme" in mv or "ragepowder" in mv:
        return "Redirection"
    return "Balance"


@dataclass
class EmbeddedMeta:
    label: str
    source: str
    n_teams: int
    week_labels: list[str]
    P: np.ndarray  # (W, K) weekly anchor distributions
    current_reps: dict[int, str] = field(default_factory=dict)
    anchor_family: dict[int, str] = field(default_factory=dict)
    # cached matchup advantage matrices (win prob rows beat columns), key -> order
    fam_keys: list[str] = field(default_factory=list)
    fam_M: np.ndarray | None = None
    core_keys: list[int] = field(default_factory=list)
    core_M: np.ndarray | None = None
    # per-unit sampled opponent embeddings (matchup-standardized) for rating teams
    fam_samples: dict = field(default_factory=dict)
    core_samples: dict = field(default_factory=dict)
    # completion pool: real meta sets + per-species item/ability/move options
    sets_pool: list = field(default_factory=list)
    species_opts: dict = field(default_factory=dict)
    fetched_at: float = field(default_factory=lambda: 0.0)


class AppState:
    """Holds the loaded pipeline + last-embedded meta; forecasting is cheap."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, reference_npz: Path = DEFAULT_REFERENCE):
        self.device = select_device()
        self.model, self.kind_vocab, self.token_vocab, _ = load_frozen_encoder(checkpoint, self.device)
        self.reference = ReferenceSpace.load(reference_npz)
        self.reps = self._load_reps(reference_npz.parent)
        self._project_meta_space()
        self.matchup = None
        if DEFAULT_MATCHUP.exists():
            self.matchup, self.m_mean, self.m_std = load_matchup(DEFAULT_MATCHUP)
        self.embedded: EmbeddedMeta | None = None

    @staticmethod
    def _load_reps(ref_dir: Path) -> dict[int, str]:
        path = ref_dir / "cluster_representatives.csv"
        if not path.exists():
            return {}
        return {int(r.cluster_id): str(r.species) for r in pd.read_csv(path).itertuples()}

    def _project_meta_space(self) -> None:
        """3D PCA of the anchor codebook; project clusters into the same space."""

        pca = PCA(n_components=3, random_state=0).fit(self.reference.anchor_centroids)
        self.anchor_xyz = pca.transform(self.reference.anchor_centroids)  # (K, 3)
        self.cluster_xyz = pca.transform(self.reference.cluster_centroids)  # (C, 3)
        self.anchor_cluster = self.reference.anchor_to_cluster()  # (K,)

    # --- data acquisition ---

    def _scrape_recent(self, days: int, max_tournaments: int) -> tuple[list, np.ndarray, str]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        infos = tr.list_vgc_tournaments(
            start_ts=int(start.timestamp()), end_ts=int(end.timestamp()),
            exclude_keywords=OFF_FORMAT, max_pages=25,
        )
        infos = sorted(infos, key=lambda t: t.players, reverse=True)[:max_tournaments]
        teams, weights, _ = tr.scrape_tournament_teams(infos, format_id="live")
        label = f"Live: {start.date()} → {end.date()} ({len(infos)} events)"
        return teams, np.asarray(weights, dtype=np.float32), label

    def _load_cached(self) -> tuple[list, np.ndarray, str]:
        files = sorted(MULTI_REG_DIR.glob("reg*.json"))
        if not files:
            raise RuntimeError("no cached regulation files and live scrape unavailable")
        teams, weights = tr.load_tournament_teams(files[-1])
        return teams, weights, f"Cached: {files[-1].stem}"

    def refresh(self, source: str = "live", days: int = 45, max_tournaments: int = 40,
                min_teams: int = 300) -> EmbeddedMeta:
        """Fetch teams (live with cached fallback), embed once, bin weekly, cache matchups."""

        if source == "live":
            try:
                teams, weights, label = self._scrape_recent(days, max_tournaments)
                if len(teams) < min_teams:
                    raise RuntimeError(f"only {len(teams)} live teams (< {min_teams})")
            except Exception:
                teams, weights, label = self._load_cached()
                label += " (live scrape unavailable — fallback)"
        else:
            teams, weights, label = self._load_cached()

        # embed once (keep raw for the matchup model, standardized for anchors)
        kept = [(t, w) for t, w in zip(teams, weights, strict=True) if t.timestamp is not None]
        teams_k = [t for t, _ in kept]
        weights_k = np.array([w for _, w in kept], dtype=np.float32)
        raw = embed_teams(self.model, teams_k, self.kind_vocab, self.token_vocab,
                          device=self.device, show_progress=False)
        std = self.reference.standardize(raw)
        anchors = self.reference.assign_anchors(raw)
        labels = [iso_week_label(t.timestamp) for t in teams_k]
        ordered = sorted(set(labels))
        w2i = {w: i for i, w in enumerate(ordered)}
        week_index = np.array([w2i[label_] for label_ in labels], dtype=np.int64)

        features = TeamFeatures(embeddings=std.astype(np.float32), anchors=anchors.astype(np.int64),
                                week_index=week_index, weight=weights_k, weeks=ordered)
        P = ucsv.weekly_anchor_matrix(features, self.reference.n_anchors)
        species = [" | ".join(p.species for p in t.pokemon) for t in teams_k]
        current_reps = self._current_reps(features, species)
        anchor_family = self._anchor_family(anchors, teams_k)

        fam_keys, fam_M, core_keys, core_M = [], None, [], None
        fam_samples, core_samples = {}, {}
        if self.matchup is not None:
            (fam_samples, fam_keys, fam_M,
             core_samples, core_keys, core_M) = self._matchup_matrices(raw, teams_k, anchors)
        sets_pool, species_opts = self._completion_pool(teams_k)

        self.embedded = EmbeddedMeta(
            label=label, source=source, n_teams=len(teams_k), week_labels=ordered, P=P,
            current_reps=current_reps, anchor_family=anchor_family,
            fam_keys=fam_keys, fam_M=fam_M, core_keys=core_keys, core_M=core_M,
            fam_samples=fam_samples, core_samples=core_samples,
            sets_pool=sets_pool, species_opts=species_opts,
            fetched_at=time.time(),
        )
        return self.embedded

    @staticmethod
    def _completion_pool(teams: list[Team]) -> tuple[list, dict]:
        """Real meta sets (for a 6th mon) + per-species item/ability/move options."""

        set_counts: Counter = Counter()
        set_repr: dict = {}
        items: dict = defaultdict(Counter)
        abilities: dict = defaultdict(Counter)
        moves: dict = defaultdict(Counter)
        for team in teams:
            for p in team.pokemon:
                sid = to_id_str(p.species)
                key = (sid, to_id_str(p.item or ""), to_id_str(p.ability or ""),
                       tuple(sorted(to_id_str(m) for m in p.moves)))
                set_counts[key] += 1
                set_repr.setdefault(key, p)
                if p.item and p.item.lower() != "none":
                    items[sid][p.item] += 1
                if p.ability and p.ability.lower() != "none":
                    abilities[sid][p.ability] += 1
                for m in p.moves:
                    if m and m.lower() != "none":
                        moves[sid][m] += 1
        sets_pool = [set_repr[k] for k, _ in set_counts.most_common(150)]
        species_opts = {}
        for sid in set(items) | set(abilities) | set(moves):
            species_opts[sid] = {
                "items": [i for i, _ in items[sid].most_common(6)],
                "abilities": [a for a, _ in abilities[sid].most_common(3)],
                "moves": [m for m, _ in moves[sid].most_common(12)],
            }
        return sets_pool, species_opts

    @staticmethod
    def _anchor_family(anchors: np.ndarray, teams: list[Team]) -> dict[int, str]:
        by_anchor: dict[int, list[str]] = defaultdict(list)
        for a, team in zip(anchors, teams, strict=True):
            by_anchor[int(a)].append(team_archetype(team))
        return {a: Counter(fs).most_common(1)[0][0] for a, fs in by_anchor.items()}

    def _current_reps(self, features, species: list[str]) -> dict[int, str]:
        team_cluster = self.anchor_cluster[features.anchors]
        reps: dict[int, str] = {}
        for c in range(self.reference.n_clusters):
            members = np.flatnonzero(team_cluster == c)
            if len(members) == 0:
                continue
            d = np.linalg.norm(features.embeddings[members] - self.reference.cluster_centroids[c], axis=1)
            reps[c] = species[int(members[int(d.argmin())])]
        return reps

    # --- matchup advantage matrices (cached per refresh; independent of lambda) ---

    def _matchup_matrices(self, raw: np.ndarray, teams: list[Team], anchors: np.ndarray):
        mstd = ((raw - self.m_mean) / self.m_std).astype(np.float32)
        rng = np.random.default_rng(0)

        def sample(groups: dict) -> dict:
            out = {}
            for key, idx in groups.items():
                idx = np.asarray(idx)
                if len(idx) > SAMPLE_PER_UNIT:
                    idx = rng.choice(idx, SAMPLE_PER_UNIT, replace=False)
                out[key] = mstd[idx]
            return out

        fam_groups: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(teams):
            fam_groups[team_archetype(t)].append(i)
        core_groups: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(self.anchor_cluster[anchors]):
            core_groups[int(c)].append(i)

        fam_samples = sample(fam_groups)
        core_samples = sample(core_groups)
        fam_keys, fam_M = self._advantage(fam_samples)
        core_keys, core_M = self._advantage(core_samples)
        return fam_samples, fam_keys, fam_M, core_samples, core_keys, core_M

    def _advantage(self, samples: dict):
        """Pairwise mean win probability (row beats column) between unit samples."""

        keys = list(samples)
        n = len(keys)
        M = np.full((n, n), 0.5, dtype=np.float64)
        with torch.no_grad():
            for i in range(n):
                Ai = samples[keys[i]]
                for j in range(n):
                    if i == j:
                        continue
                    Bj = samples[keys[j]]
                    A = torch.from_numpy(np.repeat(Ai, len(Bj), axis=0))
                    B = torch.from_numpy(np.tile(Bj, (len(Ai), 1)))
                    M[i, j] = float(self.matchup.win_prob(A, B).mean())
        return keys, M

    @staticmethod
    def _graph(keys, M, shares: dict, labels: dict) -> dict:
        n = len(keys)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) if n else np.array([])
        nodes = []
        for i, k in enumerate(keys):
            num = den = 0.0
            for j, k2 in enumerate(keys):
                if i == j:
                    continue
                sj = shares.get(k2, 0.0)
                num += sj * M[i][j]
                den += sj
            nodes.append({
                "key": str(k), "label": labels.get(k, str(k)), "share": float(shares.get(k, 0.0)),
                "field_winrate": float(num / den) if den > 0 else 0.5,
                "x": float(np.cos(angles[i])), "y": float(np.sin(angles[i])),
            })
        edges = [
            {"source": i, "target": j, "advantage": float(M[i][j])}
            for i in range(n) for j in range(n)
            if i != j and M[i][j] > EDGE_THRESHOLD
        ]
        return {"nodes": nodes, "edges": edges}

    # --- forecasting ---

    def forecast(self, lam: float = 0.3) -> dict:
        if self.embedded is None:
            self.refresh(source="cached")
        e = self.embedded
        P = e.P
        current_a = P[-1]
        predicted_a = ucsv.Glide(lam).forecast(P)

        current_c = anchor_to_cluster_distribution(current_a, self.reference)
        predicted_c = anchor_to_cluster_distribution(predicted_a, self.reference)
        reps = e.current_reps
        cores = []
        for c in range(self.reference.n_clusters):
            rep = reps.get(int(c), "")
            cores.append({
                "id": int(c), "label": " / ".join(rep.split(" | ")[:3]) if rep else "",
                "representative": rep, "current": float(current_c[c]),
                "predicted": float(predicted_c[c]), "delta": float(predicted_c[c] - current_c[c]),
            })

        af = e.anchor_family
        cur: dict[str, float] = defaultdict(float)
        pred: dict[str, float] = defaultdict(float)
        for a in range(self.reference.n_anchors):
            fam = af.get(a)
            if fam is None:
                continue
            cur[fam] += float(current_a[a])
            pred[fam] += float(predicted_a[a])
        cs = sum(cur.values()) or 1.0
        ps = sum(pred.values()) or 1.0
        archetypes = [
            {"name": f, "current": cur[f] / cs, "predicted": pred[f] / ps,
             "delta": pred[f] / ps - cur[f] / cs}
            for f in sorted(cur, key=lambda k: -cur[k])
        ]

        # 3D meta space
        anchors3d = [
            {"id": int(i), "x": float(self.anchor_xyz[i, 0]), "y": float(self.anchor_xyz[i, 1]),
             "z": float(self.anchor_xyz[i, 2]), "current": float(current_a[i]),
             "delta": float(predicted_a[i] - current_a[i]),
             "label": reps.get(int(self.anchor_cluster[i]), "")}
            for i in range(self.reference.n_anchors)
        ]
        clusters3d = [
            {"id": int(c), "x": float(self.cluster_xyz[c, 0]), "y": float(self.cluster_xyz[c, 1]),
             "z": float(self.cluster_xyz[c, 2]), "label": (" / ".join(reps[c].split(" | ")[:2]))
             if reps.get(c) else "", "current": float(current_c[c])}
            for c in range(self.reference.n_clusters) if current_c[c] >= 0.01
        ]

        # matchup advantage graphs (cached matrices + current shares)
        matchup_families = matchup_cores = None
        if e.fam_M is not None:
            fam_share = {a["name"]: a["current"] for a in archetypes}
            matchup_families = self._graph(e.fam_keys, e.fam_M, fam_share,
                                           {k: k for k in e.fam_keys})
            core_share = {c: float(current_c[c]) for c in e.core_keys}
            core_label = {c: (" / ".join(reps[c].split(" | ")[:2]) if reps.get(c) else f"c{c}")
                          for c in e.core_keys}
            # only graph cores that are active in the current meta (readability)
            active = [c for c in e.core_keys if current_c[c] >= 0.01]
            idx = [e.core_keys.index(c) for c in active]
            sub_M = e.core_M[np.ix_(idx, idx)] if idx else np.zeros((0, 0))
            matchup_cores = self._graph(active, sub_M, core_share, core_label)

        return {
            "label": e.label, "source": e.source, "n_teams": e.n_teams,
            "n_weeks": len(e.week_labels), "week_labels": e.week_labels, "lam": lam,
            "fetched_at": e.fetched_at,
            "cores": cores, "archetypes": archetypes,
            "top_cores": sorted([c for c in cores if c["representative"]],
                                key=lambda d: -d["current"])[:5],
            "top_archetypes": sorted(archetypes, key=lambda d: -d["current"])[:5],
            "anchors3d": anchors3d, "clusters3d": clusters3d,
            "matchup_families": matchup_families, "matchup_cores": matchup_cores,
            "has_matchup": self.matchup is not None,
        }

    # --- team rating (paste -> win rate vs each archetype + aggregate score) ---

    @staticmethod
    def _fetch_paste(paste: str) -> str:
        s = paste.strip()
        if not (s.startswith("http://") or s.startswith("https://")):
            return s  # already the raw Showdown export
        url = s
        if "pokepast.es" in url and not url.rstrip("/").endswith("/raw"):
            url = url.rstrip("/") + "/raw"
        r = requests.get(url, timeout=15, headers={"User-Agent": "vgc-meta-app/0.1"})
        r.raise_for_status()
        text = r.text
        if text.lstrip().startswith("<"):  # got HTML — pull the <pre> export blocks
            pres = re.findall(r"<pre[^>]*>(.*?)</pre>", text, re.S)
            text = "\n\n".join(re.sub("<[^>]+>", "", p) for p in pres) or text
        return html.unescape(text)

    def rate_team(self, paste: str, lam: float = 0.3) -> dict:
        if self.matchup is None:
            return {"ok": False, "error": "matchup model not loaded"}
        if self.embedded is None:
            self.refresh(source="cached")
        try:
            pokemon = parse_pokepaste(self._fetch_paste(paste))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"failed to load paste: {exc}"}
        if not pokemon:
            return {"ok": False, "error": "could not parse any Pokémon from the paste"}
        team = Team(pokemon=pokemon, format_id="user")

        raw = embed_teams(self.model, [team], self.kind_vocab, self.token_vocab, device=self.device)
        team_std = ((raw[0] - self.m_mean) / self.m_std).astype(np.float32)

        fc = self.forecast(lam)
        arch_share = {a["name"]: (a["current"], a["predicted"]) for a in fc["archetypes"]}
        core_share = {c["id"]: (c["current"], c["predicted"], c["label"]) for c in fc["cores"]}

        archetypes = []
        for f in self.embedded.fam_keys:
            wr = expected_winrate_vs(self.matchup, team_std, self.embedded.fam_samples[f])
            cur, pred = arch_share.get(f, (0.0, 0.0))
            archetypes.append({"name": f, "winrate": wr, "current": cur, "predicted": pred})
        archetypes.sort(key=lambda d: -d["current"])

        cores = []
        for c in self.embedded.core_keys:
            cur, pred, label = core_share.get(c, (0.0, 0.0, f"c{c}"))
            if cur < 0.005:
                continue
            wr = expected_winrate_vs(self.matchup, team_std, self.embedded.core_samples[c])
            cores.append({"id": c, "label": label, "winrate": wr, "current": cur, "predicted": pred})
        cores.sort(key=lambda d: -d["current"])

        def weighted(rows: list[dict], key: str) -> float:
            w = sum(r[key] for r in rows) or 1.0
            return sum(r["winrate"] * r[key] for r in rows) / w

        return {
            "ok": True,
            "team_species": " / ".join(p.species for p in team.pokemon),
            "team_family": team_archetype(team),
            "lam": lam,
            "archetypes": archetypes,
            "cores": cores,
            "aggregate": {
                "current_by_cluster": weighted(cores, "current"),
                "predicted_by_cluster": weighted(cores, "predicted"),
                "current_by_archetype": weighted(archetypes, "current"),
                "predicted_by_archetype": weighted(archetypes, "predicted"),
            },
        }

    # --- team completion (fill empty slots to beat the team's worst matchups) ---

    def _matchup_std(self, teams: list[Team]) -> np.ndarray:
        raw = embed_teams(self.model, teams, self.kind_vocab, self.token_vocab, device=self.device)
        return ((raw - self.m_mean) / self.m_std).astype(np.float32)

    def recommend_completion(self, paste: str, lam: float = 0.3) -> dict:
        if self.matchup is None:
            return {"ok": False, "error": "matchup model not loaded"}
        if self.embedded is None:
            self.refresh(source="cached")
        e = self.embedded
        if not e.fam_keys:
            return {"ok": False, "error": "no matchup samples for the current meta"}
        try:
            pokemon = parse_pokepaste(self._fetch_paste(paste))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"failed to load paste: {exc}"}
        if not pokemon:
            return {"ok": False, "error": "could not parse any Pokémon from the paste"}
        team = Team(pokemon=pokemon, format_id="user")

        # the partial team's worst archetype matchups (what completions should fix)
        base_std = self._matchup_std([team])[0]
        base_wr = {f: expected_winrate_vs(self.matchup, base_std, e.fam_samples[f]) for f in e.fam_keys}
        worst = sorted(e.fam_keys, key=lambda f: base_wr[f])[:3]
        base_agg = float(np.mean([base_wr[f] for f in worst]))

        def rank(cand_teams: list[Team], labels: list[str]) -> list[dict]:
            if not cand_teams:
                return []
            stds = self._matchup_std(cand_teams)
            opts = []
            for i, lab in enumerate(labels):
                per = {f: expected_winrate_vs(self.matchup, stds[i], e.fam_samples[f]) for f in worst}
                avg = float(np.mean(list(per.values())))
                opts.append({"label": lab, "winrate_vs_worst": avg, "delta": avg - base_agg,
                             "per_archetype": per})
            opts.sort(key=lambda o: -o["winrate_vs_worst"])
            return opts[:5]

        def swap(i: int, **changes) -> Team:
            return Team(tuple(replace(pp, **changes) if j == i else pp
                              for j, pp in enumerate(team.pokemon)), format_id="user")

        recs = []
        # 1) missing Pokémon (fewer than 6) — add a real meta set that covers weaknesses
        if len(team.pokemon) < 6:
            have = {to_id_str(p.species) for p in team.pokemon}
            cands = [s for s in e.sets_pool if to_id_str(s.species) not in have][:120]
            teams_c = [Team(team.pokemon + (s,), "user") for s in cands]
            opts = rank(teams_c, [f"{s.species} @ {s.item or '—'}" for s in cands])
            if opts:
                recs.append({"slot": f"add a {len(team.pokemon) + 1}th Pokémon", "options": opts})

        # 2) empty item / ability / missing move on each mon
        for i, p in enumerate(team.pokemon):
            opts_sp = e.species_opts.get(to_id_str(p.species), {})
            if not p.item and opts_sp.get("items"):
                cands = opts_sp["items"][:6]
                o = rank([swap(i, item=c) for c in cands], cands)
                if o:
                    recs.append({"slot": f"item on {p.species}", "options": o})
            if not p.ability and opts_sp.get("abilities"):
                cands = opts_sp["abilities"][:3]
                o = rank([swap(i, ability=c) for c in cands], cands)
                if o:
                    recs.append({"slot": f"ability on {p.species}", "options": o})
            if len(p.moves) < 4 and opts_sp.get("moves"):
                existing = {to_id_str(m) for m in p.moves}
                cands = [m for m in opts_sp["moves"] if to_id_str(m) not in existing][:6]
                o = rank([swap(i, moves=p.moves + (c,)) for c in cands], cands)
                if o:
                    recs.append({"slot": f"a move on {p.species} ({len(p.moves)}/4)", "options": o})

        return {
            "ok": True,
            "team_species": " / ".join(p.species for p in team.pokemon),
            "n_pokemon": len(team.pokemon),
            "complete": len(recs) == 0,
            "worst_archetypes": [{"name": f, "base_winrate": base_wr[f]} for f in worst],
            "base_aggregate": base_agg,
            "recommendations": recs,
        }
