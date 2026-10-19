#!/usr/bin/env python3
"""Evaluate docs/gt_baseline_metrics.md metrics on ``predictions.ndjson`` + previews."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from svg_gt_baseline.eval_metrics import (
    dedupe_canonical_combined,
    discover_oom_groups,
    extracted_pair_paths,
    inspect_png_rgb,
    leaf_metrics,
    load_ndjson,
    merge_binary_masks_or,
    ndjson_index_first,
    safe_visual_metrics,
)


def _build_oom_tasks(
    *,
    idx: dict[tuple[int, int, int, str, str], dict[str, Any]],
    oom_groups: list[dict[str, Any]],
    svg_allow: set[int] | None,
) -> list[tuple[dict[str, Any], int, int, int, str, str, list[list[int]]]]:
    oom_tasks: list[tuple[dict[str, Any], int, int, int, str, str, list[list[int]]]] = []
    for g in oom_groups:
        svg_id = int(g["svg_id"])
        if svg_allow is not None and svg_id not in svg_allow:
            continue
        members = sorted(int(x) for x in g["annotation_ids"])
        anchor = min(members)
        pts = list(g.get("prompt_texts") or [])
        prompt_index = 0 if len(pts) <= 1 else 0

        keys_anchor = [k for k in idx.keys() if k[0] == svg_id and k[1] == anchor and k[2] == prompt_index]
        det_segs = sorted({(k[3], k[4]) for k in keys_anchor})

        for det, seg in det_segs:
            masks: list[list[int]] = []
            skip = False
            for aid in members:
                rec = idx.get((svg_id, aid, prompt_index, det, seg))
                if rec is None:
                    skip = True
                    break
                masks.append(list(rec["binary_mask"]))
            if skip:
                continue
            oom_tasks.append((g, svg_id, anchor, prompt_index, det, seg, masks))
    return oom_tasks


def _json_float(x: object) -> object:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _first_n_svg_ids_in_ndjson_order(lines: list[dict[str, object]], n: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for d in lines:
        sid = int(d["svg_id"])
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
            if len(out) >= n:
                break
    return out


def _svg_ids_from_ndjson_file(path: Path) -> set[int]:
    """Уникальные svg_id из NDJSON (строки predictions или построчных метрик); строки _summary пропускаются."""
    out: set[int] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("row_kind") == "_summary":
                continue
            if "svg_id" not in d:
                continue
            out.add(int(d["svg_id"]))
    return out


METRICS_CSV_FIELDNAMES = [
    "row_kind",
    "svg_id",
    "annotation_id",
    "prompt_index",
    "detector_id",
    "segmenter_id",
    "annotation_kind",
    "iou_leaf",
    "f1",
    "exact",
    "mse_full",
    "mae_full",
    "psnr_full",
    "ssim_full",
    "mse_roi",
    "mae_roi",
    "psnr_roi",
    "ssim_roi",
    "roi_pixels",
    "roi_coverage",
    "gt_png",
    "pred_png",
    "visual_ok",
]


def _run_reaggregate_from_saved_metrics(
    *,
    reaggregate_from: Path,
    align_svg_ids_from: Path,
    summary_json: Path | None,
    csv_out: Path | None,
) -> int:
    src = reaggregate_from.resolve()
    align = align_svg_ids_from.resolve()
    if not src.is_file():
        print(f"missing {src}", file=sys.stderr)
        return 1
    if not align.is_file():
        print(f"missing {align}", file=sys.stderr)
        return 1
    svg_allow = _svg_ids_from_ndjson_file(align)
    rows_out: list[dict[str, object]] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("row_kind") == "_summary":
                continue
            if int(d["svg_id"]) not in svg_allow:
                continue
            rows_out.append(d)
    summary = build_summary(rows_out)
    summary["reaggregate"] = {
        "source_metrics_ndjson": str(src),
        "align_svg_ids_from": str(align),
        "n_svg_in_align_file": len(svg_allow),
        "n_rows_kept": len(rows_out),
    }
    for row in rows_out:
        print(json.dumps(row, ensure_ascii=False, default=str))
    print(json.dumps(summary, ensure_ascii=False, default=str))
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=METRICS_CSV_FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            for row in rows_out:
                w.writerow({k: row.get(k, "") for k in METRICS_CSV_FIELDNAMES})
    print(
        f"reaggregate: kept {len(rows_out)} rows for {len(svg_allow)} svg_id from {align.name} ← {src.name}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def _mean_numeric_rowvals(rows: list[dict[str, object]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in keys:
        vals: list[float] = []
        for r in rows:
            v = r.get(k)
            if v is None or v == "":
                continue
            if isinstance(v, bool):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            vals.append(fv)
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _fraction_exact(rows: list[dict[str, object]]) -> float:
    ok = [r.get("exact") for r in rows if isinstance(r.get("exact"), bool)]
    if not ok:
        return float("nan")
    return sum(1 for x in ok if x) / len(ok)


def build_summary(rows_out: list[dict[str, object]]) -> dict[str, object]:
    keys = [
        "iou_leaf",
        "f1",
        "mse_full",
        "mae_full",
        "psnr_full",
        "ssim_full",
        "mse_roi",
        "mae_roi",
        "psnr_roi",
        "ssim_roi",
        "roi_coverage",
    ]
    canon = [r for r in rows_out if r.get("row_kind") == "canonical_row"]
    oom = [r for r in rows_out if str(r.get("row_kind", "")).startswith("oom_merged")]

    def pack(sub: list[dict[str, object]]) -> dict[str, object]:
        means = _mean_numeric_rowvals(sub, keys)
        frac = _fraction_exact(sub)
        return {
            "n_rows": len(sub),
            "exact_match_rate": None if (isinstance(frac, float) and math.isnan(frac)) else frac,
            **{k: (None if math.isnan(means[k]) else means[k]) for k in keys},
        }

    by_seg: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows_out:
        det = str(r.get("detector_id", ""))
        seg = str(r.get("segmenter_id", ""))
        by_seg[f"{det}|{seg}"].append(r)

    by_seg_packed = {k: pack(v) for k, v in sorted(by_seg.items())}

    return {
        "row_kind": "_summary",
        "macro_all": pack(rows_out),
        "macro_canonical_row": pack(canon),
        "macro_oom_merged": pack(oom),
        "macro_by_detector_segmenter": by_seg_packed,
    }


def _flatten_visual(vis: dict[str, float] | None) -> dict[str, object]:
    if not vis:
        return {
            "mse_full": "",
            "mae_full": "",
            "psnr_full": "",
            "ssim_full": "",
            "mse_roi": "",
            "mae_roi": "",
            "psnr_roi": "",
            "ssim_roi": "",
            "roi_pixels": "",
            "roi_coverage": "",
        }
    return {k: _json_float(v) for k, v in vis.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="GT baseline metrics (vector + extracted PNG MSE/PSNR/SSIM)")
    ap.add_argument("--out", type=Path, default=None, help="Run directory with predictions.ndjson + previews/")
    ap.add_argument(
        "--inspect-png",
        type=Path,
        default=None,
        help="Print RGB stats / corners and exit (no NDJSON needed)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only first N canonical rows (stable sort) — для быстрого теста на 3–4 примерах",
    )
    ap.add_argument(
        "--svg-ids",
        type=str,
        default=None,
        metavar="IDS",
        help="Comma-separated svg_id filter (e.g. 5,68). Несовместимо с --smoke-svg-n.",
    )
    ap.add_argument(
        "--smoke-svg-n",
        type=int,
        default=None,
        metavar="N",
        help="Первые N уникальных svg_id в порядке появления в predictions.ndjson (как мини smoke с фулла). Несовместимо с --svg-ids.",
    )
    ap.add_argument(
        "--oom",
        action="store_true",
        help="Устарело: строки oom_merged считаются всегда, если есть previews/.../oom_merged_*/meta.json. Флаг оставлен для совместимости CLI.",
    )
    ap.add_argument("--csv", type=Path, default=None, help="Write per-row CSV (без строки _summary)")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Записать один JSON с macro-сводкой (и smoke_svg_ids при --smoke-svg-n)",
    )
    ap.add_argument(
        "--roi-gray-eps",
        type=float,
        default=2.0 / 255.0,
        help="ROI = pixels where max(gray_gt, gray_pr) > this (float01 scale after /255). Default 2/255.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        metavar="N",
        help="Прогресс в stderr каждые N шагов фазы (canonical и oom_merged); 1 = каждая строка + ETA (0 = выкл.).",
    )
    ap.add_argument(
        "--oom-metrics-only",
        action="store_true",
        help="Только строки oom_merged (по meta.json под previews); canonical не считается.",
    )
    ap.add_argument(
        "--reaggregate-from",
        type=Path,
        default=None,
        metavar="NDJSON",
        help="Не считать метрики заново: взять построчный NDJSON (как metrics_per_row_*.ndjson), "
        "отфильтровать по svg_id из --align-svg-ids-from, пересобрать macro (_summary). "
        "Несовместимо с --out.",
    )
    ap.add_argument(
        "--align-svg-ids-from",
        type=Path,
        default=None,
        metavar="NDJSON",
        help="Файл NDJSON (predictions.ndjson или metrics_per_row.ndjson LLM-прогона): "
        "множество svg_id для фильтра бейзлайна. Нужен с --reaggregate-from.",
    )
    args = ap.parse_args()

    if args.inspect_png is not None:
        if not args.inspect_png.is_file():
            print(f"missing {args.inspect_png}", file=sys.stderr)
            return 1
        print(json.dumps(inspect_png_rgb(args.inspect_png), ensure_ascii=False, indent=2))
        return 0

    if args.reaggregate_from is not None:
        if args.align_svg_ids_from is None:
            ap.error("--reaggregate-from requires --align-svg-ids-from")
        if args.out is not None:
            ap.error("--reaggregate-from cannot be combined with --out (omit --out)")
        if args.smoke_svg_n is not None or args.svg_ids:
            ap.error("--reaggregate-from is incompatible with --smoke-svg-n / --svg-ids")
        if args.limit is not None or args.oom_metrics_only:
            ap.error("--reaggregate-from is incompatible with --limit / --oom-metrics-only")
        return _run_reaggregate_from_saved_metrics(
            reaggregate_from=args.reaggregate_from,
            align_svg_ids_from=args.align_svg_ids_from,
            summary_json=args.summary_json,
            csv_out=args.csv,
        )

    if args.out is None:
        ap.error("--out is required unless --inspect-png or --reaggregate-from")

    nd_path = args.out / "predictions.ndjson"
    if not nd_path.is_file():
        print(f"missing {nd_path}", file=sys.stderr)
        return 1

    if args.svg_ids and args.smoke_svg_n is not None:
        ap.error("use either --svg-ids or --smoke-svg-n, not both")
    if args.oom_metrics_only and args.limit is not None:
        ap.error("--oom-metrics-only is incompatible with --limit")

    lines = load_ndjson(nd_path)
    svg_allow: set[int] | None = None
    smoke_ids_order: list[int] = []
    if args.smoke_svg_n is not None:
        smoke_ids_order = _first_n_svg_ids_in_ndjson_order(lines, args.smoke_svg_n)
        svg_allow = set(smoke_ids_order)
    elif args.svg_ids:
        svg_allow = {int(x.strip()) for x in args.svg_ids.split(",") if x.strip()}

    if args.oom_metrics_only:
        canon: list[dict[str, Any]] = []
    else:
        canon = dedupe_canonical_combined(lines)
        if svg_allow is not None:
            canon = [r for r in canon if int(r["svg_id"]) in svg_allow]
        if args.limit is not None:
            canon = canon[: max(0, args.limit)]

    idx = ndjson_index_first(lines)
    previews_root = args.out / "previews"
    oom_groups = discover_oom_groups(previews_root)

    def log_info(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    prog_n = max(0, int(args.progress_every))

    def fmt_eta(sec: float) -> str:
        if not math.isfinite(sec) or sec < 0:
            return "?"
        if sec < 1.0:
            return "<1s"
        if sec >= 3600:
            return f"{int(sec // 3600)}h{int((sec % 3600) // 60)}m"
        if sec >= 60:
            return f"{int(sec // 60)}m{int(sec % 60)}s"
        return f"{sec:.0f}s"

    def log_phase(
        phase: str,
        cur: int,
        total: int,
        t0: float,
        rows_out_n: int,
        *,
        detail: str = "",
    ) -> None:
        if prog_n == 0 or total <= 0:
            return
        if cur % prog_n != 0 and cur != total:
            return
        elapsed = time.monotonic() - t0
        if cur > 0 and elapsed > 0:
            eta_s = (total - cur) * (elapsed / cur)
            eta_txt = fmt_eta(eta_s)
            rate = cur / elapsed
        else:
            eta_txt = "?"
            rate = 0.0
        pct = 100.0 * cur / total
        tail = f" {detail}" if detail else ""
        log_info(
            f"eval_gt_baseline_metrics [{phase}] {cur}/{total} ({pct:.2f}%) "
            f"elapsed={elapsed:.1f}s ETA~{eta_txt} rate={rate:.2f}/s rows_out={rows_out_n}{tail}"
        )

    log_info(
        "eval_gt_baseline_metrics: "
        f"ndjson_lines={len(lines)} deduped_canonical={len(canon)} "
        f"oom_groups={len(oom_groups)} (one_visible_of_many только в oom_merged, не в canonical) "
        f"out={args.out} progress_every={prog_n or 'off'}"
        + (" oom_metrics_only=1" if args.oom_metrics_only else "")
    )

    fieldnames = METRICS_CSV_FIELDNAMES
    rows_out: list[dict[str, object]] = []

    def append_row(
        *,
        row_kind: str,
        svg_id: int,
        ann_id: int,
        pi: int,
        det: str,
        seg: str,
        kind: str,
        lm: dict[str, float | bool],
        gt_p: Path,
        pr_p: Path,
    ) -> None:
        vis = safe_visual_metrics(gt_p, pr_p, roi_gray_eps=args.roi_gray_eps)
        fx = _flatten_visual(vis)
        rows_out.append(
            {
                "row_kind": row_kind,
                "svg_id": svg_id,
                "annotation_id": ann_id,
                "prompt_index": pi,
                "detector_id": det,
                "segmenter_id": seg,
                "annotation_kind": kind,
                "iou_leaf": lm["iou_leaf"],
                "f1": lm["f1"],
                "exact": lm["exact"],
                **fx,
                "gt_png": str(gt_p),
                "pred_png": str(pr_p),
                "visual_ok": bool(vis),
            }
        )

    canon_len = len(canon)
    t_job = time.monotonic()
    t_canon = t_job
    if not args.oom_metrics_only:
        for ci, r in enumerate(canon, start=1):
            svg_id = int(r["svg_id"])
            ann_id = int(r["annotation_id"])
            pi = int(r["prompt_index"])
            det = str(r["detector_id"])
            seg = str(r["segmenter_id"])
            kind = str(r.get("annotation_kind", ""))
            if kind == "one_visible_of_many":
                continue

            lm = leaf_metrics(list(r["binary_mask"]), list(r["pred_mask_combined"]))
            gt_p, pr_p = extracted_pair_paths(
                args.out,
                svg_id=svg_id,
                annotation_id=ann_id,
                prompt_index=pi,
                detector_id=det,
                segmenter_id=seg,
                gt_is_merged_group=False,
            )
            append_row(
                row_kind="canonical_row",
                svg_id=svg_id,
                ann_id=ann_id,
                pi=pi,
                det=det,
                seg=seg,
                kind=kind,
                lm=lm,
                gt_p=gt_p,
                pr_p=pr_p,
            )
            log_phase(
                "canonical",
                ci,
                canon_len,
                t_canon,
                len(rows_out),
                detail=f"svg={svg_id} ann={ann_id} {det}|{seg} iou={lm['iou_leaf']:.4f}",
            )

    log_info(
        "eval_gt_baseline_metrics: canonical phase "
        + ("skipped (--oom-metrics-only)" if args.oom_metrics_only else f"done in {time.monotonic() - t_canon:.1f}s")
        + f" rows_out={len(rows_out)}"
    )

    oom_tasks = _build_oom_tasks(idx=idx, oom_groups=oom_groups, svg_allow=svg_allow)

    log_info(f"eval_gt_baseline_metrics: oom_merged tasks={len(oom_tasks)}")
    t_oom = time.monotonic()
    for oi, (g, svg_id, anchor, prompt_index, det, seg, masks) in enumerate(oom_tasks, start=1):
        merged = merge_binary_masks_or(masks)
        pred = list(idx[(svg_id, anchor, prompt_index, det, seg)]["pred_mask_combined"])
        lm = leaf_metrics(merged, pred)
        gt_p, pr_p = extracted_pair_paths(
            args.out,
            svg_id=svg_id,
            annotation_id=anchor,
            prompt_index=prompt_index,
            detector_id=det,
            segmenter_id=seg,
            gt_is_merged_group=True,
        )
        append_row(
            row_kind=f"oom_merged:{g['slug']}",
            svg_id=svg_id,
            ann_id=anchor,
            pi=prompt_index,
            det=det,
            seg=seg,
            kind="one_visible_of_many_merged",
            lm=lm,
            gt_p=gt_p,
            pr_p=pr_p,
        )
        log_phase(
            "oom_merged",
            oi,
            len(oom_tasks),
            t_oom,
            len(rows_out),
            detail=f"svg={svg_id} {det}|{seg} iou={lm['iou_leaf']:.4f}",
        )

    log_info(
        f"eval_gt_baseline_metrics: all phases done wall={time.monotonic() - t_job:.1f}s "
        f"rows_out={len(rows_out)} → stdout NDJSON + optional csv/summary"
    )

    for row in rows_out:
        print(json.dumps(row, ensure_ascii=False, default=str))

    summary = build_summary(rows_out)
    if smoke_ids_order:
        summary["smoke_svg_ids_in_order"] = smoke_ids_order
    print(json.dumps(summary, ensure_ascii=False, default=str))

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows_out:
                w.writerow({k: row.get(k, "") for k in fieldnames})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
