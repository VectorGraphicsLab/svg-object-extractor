# Ideas backlog (not in code)

## YOLO-World detector backend (dropped)

**Idea:** Ultralytics YOLO-World (`yolo-world-*` ids, local `yolov8*-worldv2.pt`) as an alternative open-vocab box proposer instead of Grounding DINO.

**Why removed:** did not work well enough for the paper path; heavy extra dependency (`ultralytics` + weights). Paper baselines use Grounding DINO (+ box-fill / SAM).

## SDXL / diffusion as raster preprocess (dropped)

**Idea:** after CairoSVG rasterization, run a light **img2img** pass (e.g. SDXL, low strength ~0.1–0.2) so flat cartoon/UI SVGs look more “photoreal” before Grounding DINO / SAM — hoping open-vocab detectors fire more reliably on stylized art.

**Why it was tried:** domain gap between illustration rasters and natural-image pretraining of DINO/SAM.

**Why removed from the repo:** not used in CVEU/ICML results; heavy dependency (`diffusers` + SDXL weights); slow; easy to confuse with the actual method. Kept here only as a future experiment note.

**If revisited:** keep it behind an optional extra (`pip install …[diffusion]`), never on the default paper eval path; compare leaf IoU/F1 with/without on a fixed subset.
