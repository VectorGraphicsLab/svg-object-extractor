# SVG Object Extraction by Text Query — Working Design Document

## Purpose of this document

This document summarizes the current design discussion for a prototype / research pipeline that can:

1. take an existing **SVG**;
2. take a **text query** such as `remove the red flower`, `extract the background circle`, or `delete the cat`;
3. identify which SVG elements correspond to that query;
4. return either:
   - the matching SVG elements,
   - the SVG with those elements removed,
   - or an isolated SVG containing only those elements.

This is written as a handoff / context document for code generation in Cursor. It should help implement the prototype without losing the conceptual decisions we have already made.

---

# 1. Problem statement

We want to solve the following task:

> Given an SVG scene and a natural language query referring to an object or region in that scene, identify the SVG elements responsible for rendering that object and perform an edit operation such as extraction or deletion.

Examples:

- `remove the cat`
- `extract the blue icon in the top right`
- `delete the background`
- `remove the shadow behind the cup`
- `extract the left flower`

This is **not** only a segmentation problem and **not** only a vector parsing problem. It is a **cross-domain grounding** problem:

- the query is in language,
- the reference signal is easiest to obtain in raster space,
- but the final target is a subset of **SVG nodes**.

So the core challenge is:

> map a **text-conditioned raster mask** back to the **SVG elements** whose rendering best explains that mask.

---

# 2. Core idea we agreed on

The most practical initial idea is:

1. **Render SVG to raster**
2. **Use a text-conditioned grounding / segmentation model** to obtain a raster mask for the queried object
3. **Project the raster mask back onto SVG candidates** by rendering SVG elements separately and comparing overlaps
4. Select one or more SVG elements/groups whose rendered support best matches the query mask
5. Apply edit operation: delete / keep only / recolor / move

The key point is that we do **not** try to directly infer Bézier-level correspondence from language.
Instead, we use raster-space grounding as the semantic front-end and SVG-space subset selection as the structural back-end.

This should be treated as the baseline architecture.

---

# 3. Important conceptual clarification

We should avoid saying:

> "we get a mask and then simply overlay it onto the SVG"

That is too vague and not technically accurate.

The more correct formulation is:

> We obtain a text-conditioned raster mask and then solve a matching / subset selection problem over SVG elements by comparing the query mask to masks rendered from SVG candidates.

So the back-end is really:

- **candidate generation** in SVG,
- **candidate rendering** into raster masks,
- **mask-to-mask matching**,
- **subset selection**.

---

# 4. Why this is realistic

This idea is realistic because SVG rendering is deterministic.

For any SVG element or group, we can in principle determine what it contributes to the final rendered image by:

- rendering it alone,
- rendering its alpha support,
- rendering subtrees,
- using bounding boxes,
- using hit-testing,
- or using compositing-aware passes.

So the problem is not impossible. The main difficulty is that semantic objects do not always align neatly with SVG structure.

For example:

- one object may be split across many paths,
- one path may visually belong to multiple semantic components,
- clip paths / masks / opacity / z-order may complicate attribution,
- background elements may be structurally interleaved with foreground elements.

Still, this is very suitable for a prototype and potentially for a research task.

---

# 5. What task we are actually solving

The target is **not** “text-to-SVG generation”.
It is also **not** standard referring segmentation.

A more precise formulation is:

## Task name

**Text-guided object grounding and extraction in SVG scenes**

## Input

- SVG document `S`
- text query `q`

## Output

One of:

- subset of SVG nodes `N_q ⊆ N(S)`
- edited SVG `S \ N_q`
- isolated SVG containing only `N_q`

## Intermediate latent structure

- raster rendering `I = render(S)`
- query mask `M_q = segment(I, q)`
- candidate masks `M_i = render_mask(candidate_i)`

## Core inference problem

Find candidate or subset `C*` such that the rendered support of `C*` best matches the text-conditioned raster mask `M_q`.

---

# 6. Recommended first prototype architecture

## 6.1 High-level pipeline

```text
SVG + text query
    ↓
Render SVG to PNG
    ↓
Text-guided grounding / segmentation model
    ↓
Raster query mask
    ↓
Parse SVG into candidates (groups / paths / subtrees)
    ↓
Render candidate masks
    ↓
Match candidate masks to query mask
    ↓
Select best candidate or subset
    ↓
Apply edit operation in SVG space
```

## 6.2 Main modules

### Module A — SVG parsing

Responsibilities:

- parse SVG tree;
- preserve hierarchy;
- enumerate candidate nodes;
- keep metadata:
  - node id,
  - tag type (`g`, `path`, `rect`, `circle`, `ellipse`, `polygon`, `text`, `use`, etc.),
  - parent/children,
  - style attributes,
  - transform,
  - z-order / document order,
  - bbox if available.

### Module B — SVG rendering

Responsibilities:

- render the full SVG to raster;
- render individual candidates to masks;
- render alpha / binary masks per candidate;
- optionally render at multiple resolutions.

### Module C — Raster grounding / segmentation

Responsibilities:

- take rendered image + text query;
- return mask and optionally bounding boxes / confidence / phrases.

### Module D — Candidate scoring

Responsibilities:

- compare query mask to candidate masks;
- compute match score;
- rank candidates;
- optionally select subsets greedily.

### Module E — SVG editing

Responsibilities:

- remove selected nodes,
- isolate selected nodes,
- or perform alternate edit operations.

### Module F — Evaluation / debugging

Responsibilities:

- save overlay visualizations,
- save candidate rankings,
- save selected node ids,
- render before/after SVGs,
- compute metrics if ground truth is available.

---

# 7. Model choices we discussed

## 7.1 Main recommended baseline

### Grounding DINO + SAM / Grounded SAM / Grounded SAM 2

This is the current best first choice for the raster front-end.

Reason:

- natural language → object grounding works well,
- bounding box proposals can be text-conditioned,
- SAM refines boxes into masks,
- widely used and easy to reason about,
- clean modular design.

This should be the **primary baseline**.

Practical interpretation:

- Grounding DINO finds relevant boxes for the query,
- SAM or SAM 2 converts them into masks,
- the resulting raster mask is then used for SVG back-matching.

## 7.2 Secondary compact baseline

### Florence-2

Useful as a compact all-in-one baseline.

Why include it:

- it can handle multiple vision-language tasks,
- may provide detection/segmentation-like outputs from one model stack,
- useful for comparison against Grounded SAM pipeline.

This is good as a **comparative baseline**, not necessarily the main system.

## 7.3 Mask-first baseline for diffuse regions

### CLIPSeg

Useful for queries like:

- `background`
- `shadow`
- `blue region`
- `the red area`

Why include it:

- it can directly produce text-conditioned masks,
- may be more natural than box-based grounding for “stuff-like” queries.

This should be treated as an **additional baseline**, especially for non-object prompts.

## 7.4 Harder reasoning upgrade

### LISA / SEEM / similar reasoning segmentation models

These are for more complex queries such as:

- `the flower behind the vase`
- `the left icon under the title`
- `the small bird near the sun`

These are useful for a later version, but probably too heavy / unnecessary for the first prototype.

Recommended position:

- not required for MVP,
- useful for future hard-query experiments.

---

# 8. Strong recommendation for the first implementation

Do **not** start with a huge all-intelligent system.

The first implementation should be robust, interpretable, and debuggable.

## MVP recommendation

### Front-end
- Grounding DINO + SAM 2

### Candidate level
- `group` level first

### Back-end selection
- simple greedy or top-1 matching based on mask overlap

### Operation
- remove group or isolate group

This gives the fastest path to something that works and can be debugged.

---

# 9. How to map raster mask back to SVG

This is the most important engineering part.

## 9.1 Candidate generation

We should generate candidates at multiple granularity levels:

1. **group-level candidates** (`<g>`, logical subtrees)
2. **leaf shape candidates** (`path`, `rect`, `circle`, `polygon`, `text`, etc.)
3. **subtree candidates** for nodes that are not explicit groups but have meaningful descendants

For MVP, group-level only is enough.

Later, add path-level candidates.

## 9.2 Candidate rendering

For each candidate `i`, render a binary mask `M_i` indicating the pixels supported by that candidate.

Possible modes:

- binary support mask,
- alpha mask,
- premultiplied alpha contribution,
- visible-only rendering.

For MVP, binary or alpha mask is enough.

## 9.3 Candidate scoring

Given query mask `M_q` and candidate mask `M_i`, compute scores such as:

- IoU
- Dice / F1
- recall of query mask by candidate
- precision of candidate w.r.t. query mask
- extra area penalty

A reasonable score:

```text
score(i) = α * IoU(M_i, M_q)
         + β * Recall(M_i, M_q)
         - γ * ExtraArea(M_i, M_q)
```

Where:

- `Recall(M_i, M_q)` rewards candidates covering the query mask well
- `ExtraArea` penalizes candidates that include too much irrelevant area

## 9.4 Subset selection

Single candidate may not be enough.
One semantic object may require several SVG nodes.

So later we should support selecting a subset `S`.

A possible formulation:

```text
score(S) = IoU(M_q, union(M_i for i in S))
           - λ * |S|
           - μ * complexity(S)
```

Where:

- `|S|` penalizes selecting too many nodes,
- `complexity(S)` may penalize fragmentation, discontiguity, or mixed parentage.

## 9.5 Practical subset search

For MVP, use **greedy selection**:

1. pick best single candidate,
2. iteratively add the candidate that most improves score,
3. stop when improvement is below threshold.

No need for ILP or differentiable combinatorial optimization in v1.

---

# 10. Important failure modes we already anticipate

These should be assumed from the beginning.

## 10.1 Semantic object split across many paths

A “cat” may be:

- face path,
- ear paths,
- eye paths,
- whisker paths,
- shadow path.

Single-candidate matching may fail.

## 10.2 SVG structure may be poor

Autotraced or exported SVGs can have:

- many tiny paths,
- bad grouping,
- random ordering,
- non-semantic structure.

This makes back-matching harder.

## 10.3 Raster mask sees only visible result

If an object is partially occluded, clipped, or masked,
then the raster mask reflects visible pixels, not full logical ownership.

Need to decide whether we want:

- visible support,
- or full logical object membership.

For MVP, visible support is enough.

## 10.4 “Stuff” queries are hard

Queries like:

- `background`
- `shadow`
- `sky`

may not correspond to clean semantic objects.

Need separate handling or fallback.

## 10.5 Relational language is hard

Queries like:

- `the flower behind the vase`
- `the rightmost icon under the header`

likely require more than plain grounding.

This is for later stages.

---

# 11. What we should and should not optimize for at first

## Optimize for first

- clean architecture,
- interpretability,
- saved intermediate artifacts,
- easy debugging,
- reproducible scoring.

## Do not optimize for first

- full reasoning over complex relations,
- maximal performance on noisy SVGs,
- differentiable end-to-end training,
- clever combinatorial search.

The first goal is a **good baseline**, not a final SOTA system.

---

# 12. Evaluation plan

We discussed evaluation in three levels.

## 12.1 Level 1 — evaluate only raster front-end

Before involving SVG back-matching, measure how good the text-to-mask front-end is.

Possible public datasets / task families:

- phrase grounding segmentation,
- referring segmentation,
- reasoning segmentation.

Goal:

- understand if the model can even produce a useful raster mask for the query.

Useful metrics:

- mask IoU / mIoU,
- Dice / F1,
- no-target accuracy,
- metrics split by query type.

This isolates the quality of the grounding/segmentation model.

## 12.2 Level 2 — synthetic SVG benchmark

We likely need our own evaluation benchmark for the actual SVG task.

### Proposed structure

For each example:

- original SVG,
- rendered PNG,
- target object mask,
- ground truth set of SVG nodes corresponding to target object,
- text prompt(s).

### Prompt categories

1. **simple noun / attribute prompts**
   - `red star`
   - `blue circle`
   - `left flower`

2. **attribute-heavy prompts**
   - `small yellow icon`
   - `largest triangle`

3. **relational prompts**
   - `object behind the text`
   - `flower to the left of the vase`

4. **stuff / region prompts**
   - `background`
   - `shadow`
   - `blue region`

This benchmark is probably necessary for meaningful end-to-end evaluation.

## 12.3 Level 3 — SVG extraction / edit evaluation

Now evaluate the actual end-to-end result.

### Pixel-level metrics

Render the selected SVG subset and compare with GT mask:

- IoU,
- Dice,
- boundary F-score.

### Structure-level metrics

Compare predicted SVG nodes to GT nodes:

- node precision,
- node recall,
- node F1,
- exact-match rate,
- over-selection penalty,
- under-selection penalty.

### Edit-quality metrics

If operation is deletion:

- render edited SVG,
- verify target is removed,
- verify non-target content is preserved.

This may require additional metrics such as:

- post-edit LPIPS / SSIM on non-target region,
- target-removal success rate,
- collateral damage score.

---

# 13. Recommended experiment roadmap

## Version 0.1 — minimal working prototype

### Scope
- render SVG to PNG,
- run Grounded SAM pipeline,
- get one text mask,
- generate group-level candidates,
- score candidates,
- delete or isolate best group.

### Goal
Prove end-to-end feasibility.

## Version 0.2 — stronger baseline

### Add
- path-level candidates,
- greedy subset selection,
- Florence-2 baseline,
- CLIPSeg baseline.

### Goal
See whether mask quality or SVG matching is the bottleneck.

## Version 0.3 — harder queries

### Add
- relational prompts,
- soft reranking,
- reasoning segmentation model for hard cases,
- maybe candidate reranking with language model features.

### Goal
Handle realistic text queries beyond simple object names.

## Version 0.4 — research-grade improvements

### Add
- better grouping heuristics,
- support for clip-path/mask-aware attribution,
- differentiable or soft attribution,
- learned scoring or subset prediction.

### Goal
Turn robust baseline into research contribution.

---

# 14. Engineering recommendations for Cursor

The codebase should be organized around explicit stages.

## Recommended package structure

```text
project/
  svg_parser/
    parse_svg.py
    candidates.py
    tree_utils.py

  render/
    render_full.py
    render_candidate_mask.py
    render_utils.py

  grounding/
    grounded_sam.py
    florence2.py
    clipseg.py

  matching/
    metrics.py
    single_candidate.py
    greedy_subset.py

  editing/
    delete_nodes.py
    isolate_nodes.py
    save_svg.py

  eval/
    pixel_metrics.py
    structural_metrics.py
    benchmark_io.py

  debug/
    visualize_masks.py
    save_rankings.py
    save_overlays.py

  scripts/
    run_single_example.py
    run_benchmark.py
```

## Very important implementation principle

Every stage should save intermediate outputs.

At minimum save:

- rendered PNG,
- query mask,
- each top candidate mask,
- ranking table,
- selected node ids,
- edited SVG,
- rendered edited result.

This is crucial because most failures will be ambiguous unless visualized.

---

# 15. Data structures we should expose explicitly

## 15.1 SVG node representation

Each node should expose something like:

```python
class SvgNode:
    node_id: str
    tag: str
    parent_id: str | None
    child_ids: list[str]
    attributes: dict
    transform: Any
    bbox: tuple[float, float, float, float] | None
    z_index: int
```

## 15.2 Candidate representation

```python
class Candidate:
    candidate_id: str
    node_ids: list[str]
    level: str  # group / path / subtree
    bbox: tuple[int, int, int, int] | None
    metadata: dict
```

## 15.3 Query result representation

```python
class QueryMaskResult:
    text_query: str
    mask: np.ndarray
    boxes: list[Any]
    score: float | None
    metadata: dict
```

## 15.4 Match result representation

```python
class MatchResult:
    candidate_id: str
    node_ids: list[str]
    iou: float
    recall: float
    precision: float
    extra_area: float
    total_score: float
```

---

# 16. Candidate scoring details

This part is worth implementing cleanly.

## Suggested metrics

For candidate mask `M_i` and query mask `M_q`:

### IoU

```text
IoU = |M_i ∩ M_q| / |M_i ∪ M_q|
```

### Query recall

```text
Recall_q = |M_i ∩ M_q| / |M_q|
```

### Candidate precision

```text
Precision_i = |M_i ∩ M_q| / |M_i|
```

### Extra area

```text
ExtraArea = |M_i \ M_q| / |M_i|
```

## Suggested initial score

```text
score = 0.5 * IoU + 0.35 * Recall_q + 0.15 * Precision_i
```

Alternative with explicit penalty:

```text
score = 0.6 * IoU + 0.4 * Recall_q - 0.2 * ExtraArea
```

These values are placeholders and should be configurable.

---

# 17. Suggested heuristics for better matching

These are not mandatory for v1, but good to keep in mind.

## 17.1 BBox prefiltering

Before rendering every candidate at full resolution, use bbox overlap with query mask bbox.

This can reduce compute.

## 17.2 Area filtering

Discard candidates that are obviously too small or too large relative to query mask.

## 17.3 Parent-child consistency bonus

If multiple selected nodes share a parent or subtree, reward them slightly.

This may help prefer semantically coherent selections.

## 17.4 Color prior

If query contains colors like `red`, `blue`, `green`,
add a weak prior based on candidate style attributes or rendered average color.

## 17.5 Positional prior

If query contains words like `left`, `right`, `top`, `bottom`,
add a bbox-based prior.

These are useful later for harder prompts.

---

# 18. Handling special SVG constructs

The code should be designed with awareness of the following constructs, even if MVP handles them only partially:

- `clipPath`
- `mask`
- opacity / alpha compositing
- filters
- `use`
- nested transforms
- `text`
- stroke-only objects
- hidden elements / display:none

For the first version, it is acceptable to support a simpler subset of SVGs, but the architecture should not assume that SVG is only paths and groups.

---

# 19. What not to claim yet

We should not yet claim:

- exact semantic object decomposition,
- robust support for arbitrary messy SVG exports,
- reasoning-grade relational understanding,
- fully correct attribution under all compositing conditions.

The first prototype is a **feasibility baseline**, not a solved general problem.

---

# 20. Research framing ideas

If this turns into a paper/project, good framing options include:

## Framing A

**Text-guided object extraction in vector graphics**

Focus:
- natural language query,
- object selection in SVG,
- structural edit.

## Framing B

**Cross-domain grounding from text to vector scene structure via raster masks**

Focus:
- raster-language grounding,
- vector back-projection,
- subset selection.

## Framing C

**Open-vocabulary editing of existing SVG scenes**

Focus:
- editability,
- object removal / isolation,
- structure-aware matching.

For now, Framing B is probably the most technically precise.

---

# 21. Concrete implementation priorities for Cursor

If code generation starts now, the order should be:

## Priority 1

Implement full end-to-end pipeline on one SVG:

- parse SVG,
- render full image,
- obtain query mask,
- enumerate group candidates,
- render candidate masks,
- score candidates,
- select best candidate,
- delete candidate from SVG,
- save outputs.

## Priority 2

Add strong debugging tools:

- overlay query mask,
- show top-k candidate masks,
- save candidate scores to CSV/JSON,
- render before/after.

## Priority 3

Add path-level candidates and greedy subset selection.

## Priority 4

Add alternative raster front-ends (Florence-2, CLIPSeg).

## Priority 5

Build synthetic benchmark and evaluation tooling.

---

# 22. Suggested first CLI interface

A very useful first script:

```bash
python scripts/run_single_example.py \
  --svg input.svg \
  --query "remove the red flower" \
  --output_dir outputs/example_001 \
  --grounding_model grounded_sam2 \
  --candidate_level group \
  --operation delete
```

Expected outputs:

- `render_full.png`
- `query_mask.png`
- `top_candidates.json`
- `top_candidates_overlay.png`
- `selected_nodes.json`
- `edited.svg`
- `edited_render.png`

---

# 23. Pseudocode sketch

```python
def run_pipeline(svg_path: str, text_query: str, operation: str):
    svg_doc = parse_svg(svg_path)

    full_image = render_svg(svg_doc)
    query_result = get_text_conditioned_mask(full_image, text_query)
    query_mask = query_result.mask

    candidates = enumerate_candidates(svg_doc, level="group")

    match_results = []
    for cand in candidates:
        cand_mask = render_candidate_mask(svg_doc, cand)
        metrics = compute_mask_metrics(cand_mask, query_mask)
        score = score_candidate(metrics)
        match_results.append((cand, metrics, score))

    ranked = sort_by_score(match_results)
    selected = select_best_or_greedy_subset(ranked, query_mask)

    edited_svg = apply_operation(svg_doc, selected, operation)
    edited_render = render_svg(edited_svg)

    save_debug_outputs(
        full_image=full_image,
        query_mask=query_mask,
        ranked=ranked,
        selected=selected,
        edited_svg=edited_svg,
        edited_render=edited_render,
    )

    return edited_svg, ranked, selected
```

---

# 24. Final practical conclusion

The current best course of action is:

1. **Use Grounded SAM as the main text-to-mask front-end**
2. **Treat SVG back-matching as a candidate scoring / subset selection problem**
3. **Start with group-level candidates and a simple overlap-based score**
4. **Build strong visual debugging from the start**
5. **Only then add path-level matching, alternative models, and harder language cases**

This gives the best balance of:

- feasibility,
- engineering simplicity,
- interpretability,
- and research value.

---

# 25. Short summary for Cursor

If you need the shortest possible actionable summary:

> Build an end-to-end baseline that renders SVG to PNG, gets a text-conditioned raster mask using Grounded SAM, renders SVG groups as candidate masks, scores them against the query mask, selects the best group/subset, and edits the original SVG by deleting or isolating those nodes. Save every intermediate artifact for debugging. Start simple; do not overcomplicate reasoning or combinatorial optimization in the first version.

