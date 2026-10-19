# svg-object-extractor

Raster-grounded **text-guided SVG object extraction**: CairoSVG → Grounding DINO → box-fill / SAM → inclusion-aware leaf scoring → pruned editable sub-SVG.

Used for the CVEU@CVPR 2026 and ICML 2026 HAIcoCr workshop papers (VectorGraphicsLab).

**Core idea:** an input SVG is rasterized, **Grounding DINO** proposes boxes for a text query, a **segmenter** turns each box into a binary mask (SAM *or* axis-aligned box fill), then each drawable SVG primitive (“leaf”) is scored by **directed pixel overlap** \(|C\cap T|/|C|\) against the combined mask; leaves above a threshold are kept and written as a pruned SVG plus diagnostics (PNGs, CLIP scores).

## Layout (on the lab GPU host)

```
~/SVG_RELATED/svg-object-extractor/     # this git checkout (code)
  output -> /big_boy/.../output         # symlink; heavy eval runs (not in git)

/big_boy/izhanvarsky/SVG_RELATED/svg-object-extractor/
  output/                               # gt_baseline_full, dataset_eval_*, llm bridges, …
  data/                                 # datasets / jsonl / zips (not in git)
```

Related: workshop papers live in `VectorGraphicsLab/paper-2026-*-svg-extraction`.  
Dropped experiments (YOLO-World, SDXL raster preprocess): see `docs/ideas_backlog.md`.

---


## Lab host setup (symlink heavy dirs)

On `borya3090`, after clone:

```bash
mkdir -p /big_boy/izhanvarsky/SVG_RELATED/svg-object-extractor/{output,data}
ln -sfn /big_boy/izhanvarsky/SVG_RELATED/svg-object-extractor/output output
ln -sfn /big_boy/izhanvarsky/SVG_RELATED/svg-object-extractor/data data
```

Do **not** commit these symlinks (absolute paths are host-specific).

## Requirements

- **Python** 3.10+ recommended  
- **PyTorch** with a CUDA build if available (CPU works but is slow)  
- **Hugging Face Transformers** and model downloads (some checkpoints are gated; set `HF_TOKEN` if needed)  
- **CairoSVG** (system Cairo libraries)  
- **Pillow**, **NumPy**

```bash
pip install -r requirements.txt
```

Optional `.env` in the repo root:

```bash
HF_TOKEN=hf_...
# TRANSFORMERS_CACHE=...
```

---

## 1. Single SVG (one query)

```bash
python scripts/extract_svg_objects.py path/to/input.svg \
  --query "your noun phrase" \
  --detector gdino-tiny \
  --segmenter sam-vit-base \
  --out-dir output/my_run \
  --dpi 96 \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --score-threshold 0.8
```

**Detectors:** `gdino-tiny`, `gdino-base`, … (see `svg_segmentator/backends/registry.py`).  
**Segmenters:** `sam-vit-base` or `box-fill`.

**Detector×segmenter matrix:**

```bash
python scripts/extract_svg_objects.py input.svg --query cat \
  --detectors gdino-tiny,gdino-base \
  --segmenters box-fill,sam-vit-base \
  --matrix-mode full \
  --out-dir output/matrix_run
```

---

## 2. Batch evaluation

```bash
python scripts/eval_dataset.py \
  --dataset-dir data/dataset_annotations \
  --annotations data/dataset_annotations/annotations_filtered.csv \
  --out-root output/dataset_eval_run \
  --detector gdino-tiny \
  --segmenter box-fill \
  --dpi 96 \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --score-threshold 0.8
```

Annotations CSV: columns `path` (relative to `--dataset-dir`) and `tags` (JSON array of query strings).

---

## Default hyperparameters

| Parameter | CLI flag | Default |
|-----------|----------|---------|
| Raster DPI | `--dpi` | `96` |
| DINO box threshold | `--box-threshold` | `0.35` |
| DINO text threshold | `--text-threshold` | `0.25` |
| Leaf overlap threshold τ | `--score-threshold` | `0.8` |
| CLIP model | `--clip-model` | `openai/clip-vit-base-patch32` |

---

## Main code locations

| Component | Path |
|-----------|------|
| End-to-end run | `svg_segmentator/pipeline.py` |
| Cached models | `svg_segmentator/eval_pipeline.py` |
| Detectors / segmenters | `svg_segmentator/backends/` |
| Leaf masks + prune | `svg_segmentator/extract.py` |
| CLIs | `scripts/extract_svg_objects.py`, `scripts/eval_dataset.py` |

## Related papers (VectorGraphicsLab)

- [CVEU@CVPR 2026 extended abstract](https://github.com/VectorGraphicsLab/paper-2026-cvpr-cveu--svg-extraction) (private)
- [ICML 2026 HAIcoCr workshop paper](https://github.com/VectorGraphicsLab/paper-2026-icml-haicocr--svg-extraction) (private)
