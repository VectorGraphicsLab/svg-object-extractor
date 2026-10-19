"""Genetic search over drawable leaf subsets; fitness = CLIP(image, text) on full-frame pruned render."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from svg_segmentator.clip_fitness import clip_image_text_score, load_clip
from svg_segmentator.extract import load_tagged_svg, run_leaf_mask_scoring
from svg_segmentator.grounding import query_to_candidate_labels, run_grounded_sam
from svg_segmentator.render import png_bytes_to_rgba, svg_file_to_png_bytes
from svg_segmentator.svg_subset import subset_png_bytes, write_subset_svg


@dataclass
class EvolutionConfig:
    population_size: int = 25
    generations: int = 40
    mutation_flips: int = 3
    crossover_or_prob: float = 0.5
    clip_model_id: str = "openai/clip-vit-base-patch32"
    seed: int = 0


def _initial_population(
    baseline: frozenset[str],
    all_lids: list[str],
    pop_size: int,
    rng: random.Random,
) -> list[frozenset[str]]:
    pop: list[frozenset[str]] = [baseline]
    n = len(all_lids)
    while len(pop) < pop_size:
        ind = set(baseline)
        k = rng.randint(1, max(1, min(len(baseline) + 5, n // 4 + 3)))
        for _ in range(k):
            lid = rng.choice(all_lids)
            if lid in ind:
                ind.discard(lid)
            else:
                ind.add(lid)
        pop.append(frozenset(ind))
    return pop


def _mutate(ind: frozenset[str], all_lids: list[str], flips: int, rng: random.Random) -> frozenset[str]:
    s = set(ind)
    for _ in range(flips):
        lid = rng.choice(all_lids)
        if lid in s:
            s.discard(lid)
        else:
            s.add(lid)
    return frozenset(s)


def _crossover(
    a: frozenset[str],
    b: frozenset[str],
    rng: random.Random,
    or_prob: float,
) -> frozenset[str]:
    if rng.random() < or_prob:
        return frozenset(set(a) | set(b))
    return frozenset(set(a) & set(b))


def run_clip_evolution(
    svg_path: Path,
    text_query: str,
    *,
    dpi: float,
    score_threshold: float,
    gdino_processor,
    gdino_model,
    sam_processor,
    sam_model,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
    evo_config: EvolutionConfig,
    out_dir: Path,
) -> tuple[frozenset[str], list[tuple[int, float, frozenset[str]]]]:
    """
    SAM mask baseline → leaf mask scores → baseline leaf set → evolve with CLIP.

    Returns best leaf-id frozenset and log ``(gen, best_score, best_ind)`` per generation.
    """
    rng = random.Random(evo_config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    root, pmap, leaves, leaf_by_lid = load_tagged_svg(svg_path)
    all_lids = [leaf.get("data-local-id", "") for leaf in leaves if leaf.get("data-local-id")]

    png_bytes = svg_file_to_png_bytes(svg_path, dpi=dpi)
    rgba = png_bytes_to_rgba(png_bytes)
    rgb = rgba.convert("RGB")
    text_labels = query_to_candidate_labels(text_query)
    t_mask, _object_masks, _object_meta, _dbg = run_grounded_sam(
        rgb,
        text_labels=text_labels,
        gdino_processor=gdino_processor,
        gdino_model=gdino_model,
        sam_processor=sam_processor,
        sam_model=sam_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    scores_mask = run_leaf_mask_scoring(
        root, pmap, leaves, np.asarray(t_mask), dpi, rgb, candidate_render_dir=None
    )
    baseline_list: list[str] = []
    for leaf, (lid, sc) in zip(leaves, scores_mask):
        if sc >= score_threshold and lid:
            baseline_list.append(lid)
    baseline = frozenset(baseline_list)

    clip_proc, clip_model, clip_dev = load_clip(evo_config.clip_model_id, device)

    def fitness(leaf_ids: frozenset[str]) -> float:
        png = subset_png_bytes(root, pmap, leaf_by_lid, leaf_ids, dpi)
        img = png_bytes_to_rgba(png).convert("RGB")
        return clip_image_text_score(img, text_query, clip_proc, clip_model, clip_dev)

    pop = _initial_population(baseline, all_lids, evo_config.population_size, rng)
    scored = [(fitness(ind), ind) for ind in pop]
    scored.sort(key=lambda x: -x[0])
    best_score, best_ind = scored[0]
    log: list[tuple[int, float, frozenset[str]]] = [(0, best_score, best_ind)]

    for gen in range(1, evo_config.generations + 1):
        next_pop: list[frozenset[str]] = [best_ind]
        while len(next_pop) < evo_config.population_size:
            if rng.random() < 0.3:
                parent = rng.choice(scored[: max(5, len(scored) // 2)])[1]
                child = _mutate(parent, all_lids, evo_config.mutation_flips, rng)
            else:
                p1 = rng.choice(scored[: max(5, len(scored) // 2)])[1]
                p2 = rng.choice(scored[: max(5, len(scored) // 2)])[1]
                child = _crossover(p1, p2, rng, evo_config.crossover_or_prob)
                if rng.random() < 0.5:
                    child = _mutate(child, all_lids, 1, rng)
            next_pop.append(child)
        scored = [(fitness(ind), ind) for ind in next_pop]
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] > best_score:
            best_score, best_ind = scored[0]
        log.append((gen, best_score, best_ind))

    write_subset_svg(out_dir / "best_evolved.svg", root, pmap, leaf_by_lid, best_ind)
    best_png = subset_png_bytes(root, pmap, leaf_by_lid, best_ind, dpi)
    (out_dir / "best_evolved.png").write_bytes(best_png)
    (out_dir / "baseline_leaf_ids.txt").write_text("\n".join(sorted(baseline)), encoding="utf-8")
    (out_dir / "best_leaf_ids.txt").write_text("\n".join(sorted(best_ind)), encoding="utf-8")

    np.save(out_dir / "evolution_log.npy", np.array([(g, s) for g, s, _ in log], dtype=np.float64))

    with (out_dir / "evolution_log.txt").open("w", encoding="utf-8") as f:
        for g, s, ind in log:
            f.write(f"gen={g} best_clip={s:.6f} n_leaves={len(ind)}\n")

    return best_ind, log
