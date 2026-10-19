#!/usr/bin/env python3
"""LLM-мост: ``predictions.ndjson`` + превью из v2 и эталонных GT-превью.

Копирует ``raster.png`` / ``source.svg`` / ``gt_extracted`` / ``oom_merged_*`` из
``output/gt_baseline_full/previews/<svg_id>/`` и пишет предикт в
``.../<detector>/<segmenter>/extracted_union.*`` + ``extracted_box0.*``.

Один SVG:

  python3 scripts/write_llm_bridge_examples.py \\
    --dataset 'dataset (1).jsonl' \\
    --v2-jsonl _unpacked_openrouter_samples_v2_20260508/predictions/openrouter_qwen3.6_35b_a3b_samples_v2_le32768.jsonl \\
    --reference-previews-root output/gt_baseline_full/previews \\
    --svg-id 68 --out output/llm_bridge_examples_svg68_tiny \\
    --detector llm --segmenter qwen-v2 --only-annotation-ids 177,232

Все eligible (~173 SVG, тот же список, что в v2 zip):

  python3 scripts/write_llm_bridge_examples.py \\
    --dataset 'dataset (1).jsonl' \\
    --v2-jsonl _unpacked_openrouter_samples_v2_20260508/predictions/openrouter_qwen3.6_35b_a3b_samples_v2_le32768.jsonl \\
    --reference-previews-root output/gt_baseline_full/previews \\
    --eligible-manifest _unpacked_openrouter_samples_v2_20260508/manifests/eligible_svg_samples_le32768.jsonl \\
    --all-eligible --out output/llm_bridge_full_qwen_eligible \\
    --detector llm --segmenter qwen-v2
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from svg_gt_baseline.dataset_io import filter_rows, load_dataset_rows
from svg_gt_baseline.masks import leaf_scores_from_binary_mask, node_count_from_root
from svg_gt_baseline.prediction_ndjson import build_llm_prediction_record
from svg_gt_baseline.types import DatasetAnnotationRow, InferenceSettings
from svg_gt_baseline.viz import write_png_once, write_svg_once
from svg_segmentator.extract import build_leaf_mask_cache, load_tagged_svg, prune_from_scores

LOG = logging.getLogger("write_llm_bridge_examples")
_PRUNE_TAU = 0.5


def _write_svg_raster_png_once(svg_path: Path, png_path: Path, raster_max_side: int) -> None:
    from svg_gt_baseline.svg_size import rasterize_svg_path_max_side

    png_b, _ = rasterize_svg_path_max_side(svg_path, raster_max_side)
    write_png_once(png_path, Image.open(io.BytesIO(png_b)).convert("RGB"))


def _write_pred_union_and_box0(
    *,
    svg_path: Path,
    raster_png_path: Path,
    pred_mask: list[int],
    seg_dir: Path,
    raster_max_side: int,
    settings: InferenceSettings,
) -> None:
    root, pmap, leaves, _ = load_tagged_svg(svg_path)
    if node_count_from_root(root) != len(pred_mask):
        raise ValueError(f"pred len {len(pred_mask)} != parsed nodes for {svg_path}")
    rgb = Image.open(raster_png_path).convert("RGB")
    rw, rh = rgb.size
    leaf_cache = build_leaf_mask_cache(
        root,
        pmap,
        leaves,
        settings.leaf_render_dpi,
        rgb,
        None,
        output_wh=(rw, rh),
    )
    scores = leaf_scores_from_binary_mask(leaf_cache, pred_mask)
    out_union, _ = prune_from_scores(leaf_cache, scores, _PRUNE_TAU)
    seg_dir.mkdir(parents=True, exist_ok=True)
    u_svg = seg_dir / "extracted_union.svg"
    write_svg_once(u_svg, out_union)
    _write_svg_raster_png_once(u_svg, seg_dir / "extracted_union.png", raster_max_side)
    b0_svg = seg_dir / "extracted_box0.svg"
    shutil.copy2(u_svg, b0_svg)
    shutil.copy2(seg_dir / "extracted_union.png", seg_dir / "extracted_box0.png")


def _row_key(r: DatasetAnnotationRow, pi: int) -> tuple[int, int, int]:
    return (r.svg_id, r.annotation_id, pi)


def _merged_gt_segmenter_dir(ref_svg: Path, anchor: int) -> Path | None:
    """Каталог сегментатора, где в эталоне лежит ``gt_merged_group_extracted.png`` для якоря."""
    base = ref_svg / f"ann_{anchor}" / "p0"
    if not base.is_dir():
        return None
    for png in base.rglob("gt_merged_group_extracted.png"):
        return png.parent
    return None


def build_records_and_previews_for_svg(
    *,
    svg_id: int,
    doc: dict[str, Any],
    ref: Path,
    out: Path,
    rows_this_svg: list[DatasetAnnotationRow],
    detector_id: str,
    segmenter_id: str,
    settings: InferenceSettings,
    ann_allow: frozenset[int] | None,
) -> list[dict[str, Any]]:
    if doc.get("ok") is False:
        LOG.warning("skip svg_id=%s (v2 ok=false)", svg_id)
        return []

    prev_out = out / "previews" / str(svg_id)
    prev_out.mkdir(parents=True, exist_ok=True)

    svg_src = ref / "source.svg"
    raster_src = ref / "raster.png"
    if not svg_src.is_file() or not raster_src.is_file():
        LOG.warning("skip svg_id=%s missing source or raster under %s", svg_id, ref)
        return []
    shutil.copy2(svg_src, prev_out / "source.svg")
    shutil.copy2(raster_src, prev_out / "raster.png")

    preds = doc.get("predictions") or []
    if not preds:
        LOG.warning("skip svg_id=%s (no predictions[])", svg_id)
        return []
    pred_by_sid = {str(p["sample_id"]): list(map(int, p["prediction"])) for p in preds}

    by_triple: dict[tuple[int, int, int], DatasetAnnotationRow] = {}
    for r in rows_this_svg:
        for pi, _pt in enumerate(r.prompt_texts):
            by_triple[_row_key(r, pi)] = r

    written_anns: set[int] = set()
    records: list[dict[str, Any]] = []
    oom_sample_ids_done: set[str] = set()
    wrote_any_oom_row = False
    oom_anchor_for_merged: int | None = None

    for s in doc.get("samples", []):
        sid = str(s["sample_id"])
        if sid not in pred_by_sid:
            LOG.warning("svg_id=%s skip sample %s (no prediction)", svg_id, sid)
            continue
        pred = pred_by_sid[sid]
        msc = int(s.get("merged_source_count", 1))
        kind = str(s.get("sample_kind", ""))

        if msc == 1 and kind != "one_visible_of_many":
            aids = [int(x) for x in s.get("source_annotation_ids", [])]
            pis = [int(x) for x in s.get("source_prompt_indices", [])]
            if len(aids) != 1 or len(pis) != 1:
                continue
            ann_id, pi = aids[0], pis[0]
            key = (svg_id, ann_id, pi)
            row = by_triple.get(key)
            if row is None:
                LOG.warning("svg_id=%s skip sample %s: no dataset row for %s", svg_id, sid, key)
                continue
            pt = row.prompt_texts[pi]
            if ann_allow is not None and ann_id not in ann_allow:
                continue
            rec = build_llm_prediction_record(
                row=row,
                prompt_index=pi,
                prompt_text=pt,
                detector_id=detector_id,
                segmenter_id=segmenter_id,
                pred_mask=pred,
            )
            records.append(rec)
            written_anns.add(ann_id)
            continue

        if kind == "one_visible_of_many" and msc > 1:
            if sid in oom_sample_ids_done:
                continue
            want_ids = frozenset(int(x) for x in s.get("source_annotation_ids", []))
            meta_path = None
            for p in ref.glob("oom_merged_*/meta.json"):
                meta = json.loads(p.read_text(encoding="utf-8"))
                got = frozenset(int(x) for x in meta.get("annotation_ids", []))
                if got == want_ids:
                    meta_path = p
                    break
            if meta_path is None:
                LOG.warning("svg_id=%s OOM sample %s: no meta under %s for ids %s", svg_id, sid, ref, sorted(want_ids))
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            members = sorted(int(x) for x in meta.get("annotation_ids", []))
            pts = list(meta.get("prompt_texts") or ())
            prompt_texts0 = str(pts[0]) if pts else str(s.get("prompt_text", ""))
            if str(s.get("prompt_text", "")) != prompt_texts0:
                continue
            oom_sample_ids_done.add(sid)
            anchor = min(members)
            oom_anchor_for_merged = anchor
            for aid in members:
                if ann_allow is not None and aid not in ann_allow:
                    continue
                row = next((r for r in rows_this_svg if r.annotation_id == aid), None)
                if row is None:
                    LOG.warning("svg_id=%s OOM member ann=%s missing in dataset slice", svg_id, aid)
                    continue
                rec = build_llm_prediction_record(
                    row=row,
                    prompt_index=0,
                    prompt_text=row.prompt_texts[0],
                    detector_id=detector_id,
                    segmenter_id=segmenter_id,
                    pred_mask=pred,
                )
                records.append(rec)
                written_anns.add(aid)
                wrote_any_oom_row = True

    keep_ann_ids = {int(r["annotation_id"]) for r in records}
    for p in list(prev_out.glob("ann_*")):
        if not p.is_dir():
            continue
        try:
            aid = int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if aid not in keep_ann_ids:
            shutil.rmtree(p, ignore_errors=True)

    if wrote_any_oom_row:
        for oom_dir in ref.glob("oom_merged_*"):
            if oom_dir.is_dir():
                dest = prev_out / oom_dir.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(oom_dir, dest)

    for aid in sorted(written_anns):
        src_gt = ref / f"ann_{aid}" / "gt_extracted.png"
        if src_gt.is_file():
            d = prev_out / f"ann_{aid}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_gt, d / "gt_extracted.png")
            s = ref / f"ann_{aid}" / "gt_extracted.svg"
            if s.is_file():
                shutil.copy2(s, d / "gt_extracted.svg")

    if wrote_any_oom_row and oom_anchor_for_merged is not None:
        src_seg = _merged_gt_segmenter_dir(ref, oom_anchor_for_merged)
        if src_seg is not None:
            dest_anchor = prev_out / f"ann_{oom_anchor_for_merged}" / "p0" / detector_id / segmenter_id
            dest_anchor.mkdir(parents=True, exist_ok=True)
            for name in ("gt_merged_group_extracted.png", "gt_merged_group_extracted.svg"):
                p = src_seg / name
                if p.is_file():
                    shutil.copy2(p, dest_anchor / name)

    for rec in records:
        ann_id = int(rec["annotation_id"])
        pi = int(rec["prompt_index"])
        seg_dir = prev_out / f"ann_{ann_id}" / f"p{pi}" / detector_id / segmenter_id
        try:
            _write_pred_union_and_box0(
                svg_path=prev_out / "source.svg",
                raster_png_path=prev_out / "raster.png",
                pred_mask=list(rec["pred_mask_combined"]),
                seg_dir=seg_dir,
                raster_max_side=settings.raster_max_side,
                settings=settings,
            )
        except Exception as e:
            LOG.error("svg_id=%s pred preview ann=%s p=%s: %s", svg_id, ann_id, pi, e)

    return records


def _load_v2_by_svg(path: Path) -> dict[int, dict[str, Any]]:
    by_svg: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_svg[int(d["svg_id"])] = d
    return by_svg


def _load_eligible_svg_ids(path: Path) -> list[int]:
    out: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(int(json.loads(line)["svg_id"]))
    return sorted(set(out))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--v2-jsonl", type=Path, required=True)
    ap.add_argument(
        "--reference-previews-root",
        type=Path,
        default=None,
        help="Родительский каталог ``previews`` (подкаталоги — числовые svg_id).",
    )
    ap.add_argument(
        "--reference-previews",
        type=Path,
        default=None,
        help="Устарело: каталог ``previews/<один_svg_id>/``. Используйте --reference-previews-root + --svg-id.",
    )
    ap.add_argument("--svg-id", type=int, default=None)
    ap.add_argument("--all-eligible", action="store_true", help="Обработать все svg_id из --eligible-manifest")
    ap.add_argument(
        "--eligible-manifest",
        type=Path,
        default=None,
        help="JSONL со строками {\"svg_id\": …} (eligible v2). Нужен с --all-eligible.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", type=str, default="llm")
    ap.add_argument("--segmenter", type=str, default="qwen-v2")
    ap.add_argument("--raster-max-side", type=int, default=640)
    ap.add_argument(
        "--only-annotation-ids",
        type=str,
        default=None,
        metavar="IDS",
        help="Через запятую id аннотаций (только с --svg-id)",
    )
    args = ap.parse_args()

    if bool(args.all_eligible) == (args.svg_id is not None):
        ap.error("укажите ровно одно: --svg-id … или --all-eligible")
    if args.all_eligible and not args.eligible_manifest:
        ap.error("--all-eligible требует --eligible-manifest")
    if args.reference_previews and args.reference_previews_root:
        ap.error("используйте либо --reference-previews-root, либо --reference-previews, не оба")
    if args.all_eligible:
        if args.reference_previews_root is None:
            ap.error("--all-eligible требует --reference-previews-root")
    elif args.reference_previews_root is None and args.reference_previews is None:
        ap.error("нужен --reference-previews-root или --reference-previews")

    settings = InferenceSettings(raster_max_side=int(args.raster_max_side))
    ann_allow: frozenset[int] | None = None
    if args.only_annotation_ids:
        if args.svg_id is None:
            ap.error("--only-annotation-ids только вместе с --svg-id")
        ann_allow = frozenset(int(x.strip()) for x in args.only_annotation_ids.split(",") if x.strip())

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ndjson_path = out / "predictions.ndjson"
    ndjson_path.unlink(missing_ok=True)

    rows_all = load_dataset_rows(args.dataset.resolve())
    rows_f = filter_rows(rows_all)
    by_svg_rows: dict[int, list[DatasetAnnotationRow]] = {}
    for r in rows_f:
        by_svg_rows.setdefault(r.svg_id, []).append(r)

    v2_path = args.v2_jsonl.resolve()
    v2_by_svg = _load_v2_by_svg(v2_path)

    if args.svg_id is not None:
        if args.reference_previews_root is not None:
            ref = args.reference_previews_root.resolve() / str(args.svg_id)
        else:
            ref = args.reference_previews.resolve()
        doc = v2_by_svg.get(args.svg_id)
        if doc is None:
            print(f"no v2 line for svg_id={args.svg_id}", file=sys.stderr)
            return 1
        recs = build_records_and_previews_for_svg(
            svg_id=args.svg_id,
            doc=doc,
            ref=ref,
            out=out,
            rows_this_svg=by_svg_rows.get(args.svg_id, []),
            detector_id=args.detector,
            segmenter_id=args.segmenter,
            settings=settings,
            ann_allow=ann_allow,
        )
        with ndjson_path.open("w", encoding="utf-8") as nf:
            for rec in recs:
                nf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(recs)} rows -> {ndjson_path}", file=sys.stderr)
        return 0

    ref_root = args.reference_previews_root.resolve()
    eligible = _load_eligible_svg_ids(args.eligible_manifest.resolve())
    t0 = time.monotonic()
    total_recs = 0
    skipped = 0
    with ndjson_path.open("w", encoding="utf-8") as nf:
        for i, sid in enumerate(eligible, start=1):
            ref = ref_root / str(sid)
            if not ref.is_dir():
                LOG.warning("[%d/%d] skip svg_id=%s (no reference previews dir)", i, len(eligible), sid)
                skipped += 1
                continue
            doc = v2_by_svg.get(sid)
            if doc is None:
                LOG.warning("[%d/%d] skip svg_id=%s (no v2 line)", i, len(eligible), sid)
                skipped += 1
                continue
            recs = build_records_and_previews_for_svg(
                svg_id=sid,
                doc=doc,
                ref=ref,
                out=out,
                rows_this_svg=by_svg_rows.get(sid, []),
                detector_id=args.detector,
                segmenter_id=args.segmenter,
                settings=settings,
                ann_allow=None,
            )
            for rec in recs:
                nf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_recs += len(recs)
            if i % 10 == 0 or i == len(eligible):
                LOG.info(
                    "[%d/%d] svg_id=%s rows_cum=%d elapsed=%.1fs",
                    i,
                    len(eligible),
                    sid,
                    total_recs,
                    time.monotonic() - t0,
                )

    print(
        f"Done eligible={len(eligible)} skipped_missing={skipped} ndjson_rows={total_recs} -> {ndjson_path}",
        file=sys.stderr,
    )
    print(
        "Eval: python3 scripts/eval_gt_baseline_metrics.py --out",
        str(out),
        "--summary-json",
        str(out / "metrics_summary.json"),
        "--csv",
        str(out / "metrics_per_row.csv"),
        "> ",
        str(out / "metrics_per_row.ndjson"),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
