"""Batch-eval: HF causal LM extracts SVG objects per (file, tag); rasterize + CLIP (per object + combined)."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from svg_segmentator.repo_env import load_repo_dotenv

load_repo_dotenv()

from svg_segmentator.clip_fitness import clip_image_text_score, load_clip
from svg_segmentator.llm_hf.hf_causal import HfCausalSvgGenerator
from svg_segmentator.llm_hf.json_parse import extract_json, safe_filename, validate_svg_fragment
from svg_segmentator.llm_hf.prompts import SYSTEM_PROMPT, build_user_prompt
from svg_segmentator.render import png_bytes_to_rgba, svg_bytes_to_png_bytes


def _safe_segment(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _run_dir_name(stem: str, tag: str, max_tag_len: int = 80) -> str:
    seg = _safe_segment(tag)
    if len(seg) > max_tag_len:
        seg = seg[:max_tag_len]
    return f"{stem}__{seg}"


def _load_tags(cell: str) -> list[str]:
    cell = cell.strip()
    if not cell:
        return []
    data = json.loads(cell)
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for x in data:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def resolve_svg_for_row(rel: str, dataset_dir: Path, svg_root: Path | None) -> Path | None:
    """Match ``eval_dataset.py``: primary path is ``dataset_dir / rel``."""
    p = (dataset_dir / rel).resolve()
    if p.is_file():
        return p
    if svg_root is not None:
        q = (svg_root / rel).resolve()
        if q.is_file():
            return q
        q2 = (svg_root / Path(rel).name).resolve()
        if q2.is_file():
            return q2
    return None


RESULT_FIELDS = [
    "dataset_rel_path",
    "tag",
    "hf_model_id",
    "dpi",
    "clip_model_id",
    "n_objects",
    "clip_combined",
    "clip_combined_method",
    "clip_scores_json",
    "run_out_dir",
    "status",
    "error",
]


def process_tag_run(
    *,
    dataset_rel_path: str,
    svg_path: Path,
    svg_text: str,
    tag: str,
    run_dir: Path,
    gen: HfCausalSvgGenerator,
    clip_proc,
    clip_model,
    clip_dev: torch.device,
    dpi: float,
    resume: bool,
) -> dict:
    """Returns a dict suitable for one CSV row + writes artifacts under run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "result.json"
    if resume and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    types = [tag.strip().lower()]
    user = build_user_prompt(types, svg_text)
    raw = gen.chat(SYSTEM_PROMPT, user)

    raw_path = run_dir / "llm_raw.txt"
    raw_path.write_text(raw, encoding="utf-8")

    err = ""
    objects: list[dict] = []
    try:
        parsed = extract_json(raw)
        objs = parsed.get("objects")
        if objs is None:
            objs = parsed.get("buildings") or parsed.get("instances") or []
        if not isinstance(objs, list):
            raise ValueError(f"objects is not a list: {type(objs).__name__}")
        objects = [o for o in objs if isinstance(o, dict)]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        out = {
            "dataset_rel_path": dataset_rel_path,
            "tag": tag,
            "hf_model_id": gen.model_id,
            "dpi": dpi,
            "clip_model_id": getattr(clip_model.config, "_name_or_path", "clip"),
            "n_objects": 0,
            "clip_combined": None,
            "clip_combined_method": "",
            "clip_scores_json": "[]",
            "run_out_dir": str(run_dir.relative_to(_REPO)),
            "status": "error",
            "error": err,
        }
        meta_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    (run_dir / "source.svg").write_text(svg_text, encoding="utf-8")

    orig_png_bytes = svg_bytes_to_png_bytes(svg_text.encode("utf-8"), dpi=dpi)
    (run_dir / "source.png").write_bytes(orig_png_bytes)
    orig_rgba = png_bytes_to_rgba(orig_png_bytes)
    comp = Image.new("RGBA", orig_rgba.size, (0, 0, 0, 0))
    composite_ok = True

    clip_rows: list[dict] = []
    obj_scores: list[float] = []

    for i, entry in enumerate(objects):
        otype = str(entry.get("type") or "object").strip().lower() or "object"
        bid = str(entry.get("id") or f"{otype}_{i + 1}").strip()
        bid = safe_filename(bid) or f"obj_{i + 1}"
        base = f"object_{i:03d}_{bid}"
        svg_frag = entry.get("svg") or ""
        v_err = validate_svg_fragment(svg_frag)
        if v_err:
            clip_rows.append({"target": base, "clip_score": None, "error": v_err})
            continue

        frag_path = run_dir / f"{base}.svg"
        frag_path.write_text(svg_frag, encoding="utf-8")
        try:
            png_b = svg_bytes_to_png_bytes(svg_frag.encode("utf-8"), dpi=dpi)
        except Exception as e:
            clip_rows.append({"target": base, "clip_score": None, "error": f"render:{e}"})
            composite_ok = False
            continue

        png_path = run_dir / f"{base}.png"
        png_path.write_bytes(png_b)
        im = Image.open(io.BytesIO(png_b)).convert("RGBA")
        if im.size != comp.size:
            composite_ok = False
        else:
            comp = Image.alpha_composite(comp, im)

        rgb = im.convert("RGB")
        sc = clip_image_text_score(rgb, tag, clip_proc, clip_model, clip_dev)
        obj_scores.append(sc)
        clip_rows.append({"target": base, "clip_score": sc})

    combined_method = ""
    combined: float | None = None
    if obj_scores:
        if composite_ok:
            combined_method = "alpha_composite_same_size"
            combined = clip_image_text_score(
                comp.convert("RGB"), tag, clip_proc, clip_model, clip_dev
            )
        else:
            combined_method = "mean_per_object"
            combined = float(sum(obj_scores) / len(obj_scores))

    clip_rows.insert(0, {"target": "combined", "clip_score": combined, "method": combined_method})

    out = {
        "dataset_rel_path": dataset_rel_path,
        "tag": tag,
        "hf_model_id": gen.model_id,
        "dpi": dpi,
        "clip_model_id": getattr(clip_model.config, "_name_or_path", "openai/clip-vit-base-patch32"),
        "n_objects": len(obj_scores),
        "clip_combined": combined,
        "clip_combined_method": combined_method,
        "clip_scores_json": json.dumps(clip_rows, ensure_ascii=False),
        "run_out_dir": str(run_dir.relative_to(_REPO)),
        "status": "ok",
        "error": "",
    }
    meta_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=_REPO / "data" / "dataset_annotations")
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument("--svg-root", type=Path, default=None, help="Fallback if CSV path not found (uses basename under this dir).")
    p.add_argument("--out-root", type=Path, default=_REPO / "output" / "dataset_eval_llm_hf")
    p.add_argument("--hf-model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--device", type=str, default="", help="CLIP device; HF model uses device_map=auto on CUDA.")
    p.add_argument("--dpi", type=float, default=96.0)
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--max-tag-runs", type=int, default=0, help="Stop after N (svg,tag) runs (0 = all).")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    ann = args.annotations.resolve() if args.annotations else (dataset_dir / "annotations_filtered.csv")
    if not ann.is_file():
        print(f"Missing annotations: {ann}", file=sys.stderr)
        return 1

    out_root = args.out_root.resolve()
    model_seg = _safe_segment(args.hf_model.replace("/", "__"))
    combo_dir = out_root / model_seg
    combo_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / f"eval_llm_hf__{model_seg}.csv"

    clip_dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("[info] Hugging Face hub token present (HF_TOKEN / HUGGING_FACE_HUB_TOKEN).", file=sys.stderr)
    print(f"[info] Loading HF model {args.hf_model!r} (this may take a while)…", file=sys.stderr)
    gen = HfCausalSvgGenerator(args.hf_model, max_new_tokens=args.max_new_tokens)
    print("[info] Loading CLIP…", file=sys.stderr)
    clip_proc, clip_model, _ = load_clip(args.clip_model, device=clip_dev)

    runs = 0
    rows_written = 0
    skipped_empty_tags = 0
    tag_runs_ok = 0
    tag_runs_error = 0
    tag_runs_missing_svg = 0
    with results_csv.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        with ann.open(newline="", encoding="utf-8") as af:
            reader = csv.DictReader(af)
            if not reader.fieldnames or "path" not in reader.fieldnames or "tags" not in reader.fieldnames:
                print("CSV needs columns: path, tags", file=sys.stderr)
                return 2
            for row in reader:
                rel = (row.get("path") or "").strip()
                if not rel:
                    continue
                tags = _load_tags(row.get("tags") or "[]")
                if not tags:
                    skipped_empty_tags += 1
                    continue
                svg_path = resolve_svg_for_row(rel, dataset_dir, args.svg_root)
                if svg_path is None or not svg_path.is_file():
                    w.writerow(
                        {
                            "dataset_rel_path": rel,
                            "tag": "",
                            "hf_model_id": args.hf_model,
                            "dpi": args.dpi,
                            "clip_model_id": args.clip_model,
                            "n_objects": "",
                            "clip_combined": "",
                            "clip_combined_method": "",
                            "clip_scores_json": "",
                            "run_out_dir": "",
                            "status": "missing_svg",
                            "error": f"not a file: {dataset_dir / rel}",
                        }
                    )
                    out_f.flush()
                    rows_written += 1
                    tag_runs_missing_svg += 1
                    continue

                stem = svg_path.stem
                svg_text = svg_path.read_text(encoding="utf-8")
                for tag in tags:
                    if args.max_tag_runs and runs >= args.max_tag_runs:
                        break
                    run_dir = combo_dir / _run_dir_name(stem, tag)
                    rec = process_tag_run(
                        dataset_rel_path=rel,
                        svg_path=svg_path,
                        svg_text=svg_text,
                        tag=tag,
                        run_dir=run_dir,
                        gen=gen,
                        clip_proc=clip_proc,
                        clip_model=clip_model,
                        clip_dev=clip_dev,
                        dpi=args.dpi,
                        resume=args.resume,
                    )
                    w.writerow({k: rec.get(k, "") for k in RESULT_FIELDS})
                    out_f.flush()
                    runs += 1
                    rows_written += 1
                    if rec.get("status") == "ok":
                        tag_runs_ok += 1
                    else:
                        tag_runs_error += 1
                if args.max_tag_runs and runs >= args.max_tag_runs:
                    break

    summary_path = out_root / f"eval_summary_llm_hf__{model_seg}.json"
    summary_payload = {
        "hf_model": args.hf_model,
        "clip_model": args.clip_model,
        "results_csv": str(results_csv.relative_to(_REPO)),
        "combo_output_dir": str(combo_dir.relative_to(_REPO)),
        "csv_rows_written": rows_written,
        "tag_runs_completed": runs,
        "tag_runs_status_ok": tag_runs_ok,
        "tag_runs_status_error": tag_runs_error,
        "annotation_rows_missing_svg": tag_runs_missing_svg,
        "annotation_rows_skipped_empty_tags": skipped_empty_tags,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[done] wrote {results_csv} ({runs} tag runs, {rows_written} CSV rows)", file=sys.stderr)
    print(f"[done] summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
