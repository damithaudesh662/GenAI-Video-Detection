# Geometric Heuristics — Evaluator Guide

This document explains two rule-based metrics used alongside GradCAM3D in the GenAI video detection explainability pipeline. Both are computed on **colourised depth maps** (not RGB video), so they measure the **geometric plausibility** of a scene rather than its visual appearance.

**Source:** [`explainability_toolkit.py`](explainability_toolkit.py) — function `compute_geometric_features()`

---

## Spatial Smoothness — Laplacian Variance

### What the Laplacian does

The Laplacian is a second-order image derivative. Applied to a grayscale image, it computes — for every pixel — how different that pixel's intensity is from its immediate neighbours:

```
For a pixel at position (x, y):
Laplacian = pixel(x-1,y) + pixel(x+1,y) + pixel(x,y-1) + pixel(x,y+1) - 4×pixel(x,y)
```

- A pixel sitting on a **flat surface** → all neighbours are similar → Laplacian value ≈ 0
- A pixel sitting on a **sharp edge or texture** → neighbours differ strongly → large Laplacian value

### What variance adds

After computing the Laplacian image, the **variance** of all those values is taken. Variance measures the spread — how wildly the Laplacian values differ from their mean across the whole frame.

- **High variance** → many regions with rapid intensity change → rich surface detail, texture, edges
- **Low variance** → values all cluster near zero → the surface is uniformly flat and featureless

### Calculation steps

1. Convert each depth PNG → grayscale float image (0.0 – 1.0)
2. Apply Laplacian filter → response image (highlights edges and texture)
3. Compute variance of all pixel values in that response image
4. Repeat for all 16 sampled frames
5. Average the 16 variance values → one scalar: `spatial_smoothness`

### Intuitive analogy

> Imagine running your fingertips across a surface. A real wooden table has grain, bumps, and joins — your fingers feel constant variation. A CGI surface is often perfectly smooth plastic. The Laplacian variance is a mathematical "fingertip test" on the depth map — low variance means the geometry feels artificially smooth.

### Thresholds used in explanations

| Range | Label | Interpretation |
|-------|-------|----------------|
| `< 0.002` | Very smooth | Suspicious — AI generators often produce over-regularised depth fields |
| `0.002 – 0.015` | Intermediate | Inconclusive on its own |
| `> 0.015` | Textured | Consistent with real-world scenes captured on camera |

---

## Edge Density — Canny Edge Fraction

### What Canny edge detection does

Canny is a classical multi-step edge detector that finds the boundaries between objects. Given a grayscale image it:

1. **Blurs** the image slightly (Gaussian blur) to suppress noise
2. **Computes gradients** — finds pixels where intensity changes rapidly in any direction
3. **Non-maximum suppression** — thins edges to single-pixel-wide lines
4. **Double thresholding** — keeps strong edges (above 150) and weak edges connected to strong ones (above 50)
5. Outputs a binary mask: edge pixel = 255, non-edge = 0

### What "fraction" means

After Canny produces the binary edge mask:

```
edge_density = (number of edge pixels) / (total pixels in frame)
```

This gives a value between 0.0 and 1.0 representing what proportion of the depth map is occupied by object boundaries.

### Calculation steps

1. Convert each depth PNG → grayscale uint8 image (0 – 255)
2. Apply `Canny(threshold_low=50, threshold_high=150)` → binary edge mask
3. Count pixels where mask > 0, divide by total pixel count → fraction
4. Repeat for all 16 sampled frames
5. Average the 16 fractions → one scalar: `edge_density`

### Intuitive analogy

> Think of the depth map as an overhead map of a city. A real city map has crisp building outlines, road edges, and park boundaries. A poorly rendered CGI city has blurry or missing building edges — the boundaries between objects just fade out. Edge density counts how many of those clear boundary lines exist. An AI-generated video depth map often has very few crisp lines because the depth estimator "melts" object silhouettes together.

### Thresholds used in explanations

| Range | Label | Interpretation |
|-------|-------|----------------|
| `< 0.040` | Sparse | Suspicious — blurry or absent object boundaries |
| `0.040 – 0.120` | Average | Inconclusive on its own |
| `> 0.120` | Dense | Well-defined object boundaries; consistent with real geometry |

---

## Side-by-Side Summary

| Metric | Core question | Mathematical tool | Low value suggests | High value suggests |
|--------|---------------|-------------------|--------------------|---------------------|
| **Spatial Smoothness** | How much geometric texture exists? | Laplacian + variance | Artificially flat surface (AI-like) | Rich real-world texture |
| **Edge Density** | How many crisp object boundaries exist? | Canny detector + pixel fraction | Blurry/absent silhouettes (AI-like) | Well-defined geometry (real-like) |

### Pipeline overview

```
Depth frame (grayscale)
        │
        ├──► Laplacian filter ──► variance ──► spatial_smoothness
        │
        └──► Canny (50/150) ──► edge pixel count ÷ total ──► edge_density
```

Both scalars are averaged over the same 16 uniformly sampled depth frames used by the R3D-18 classifier. They appear in the explainability summary panel as plain-language bullets and in the technical footer with exact numeric values.

---

## Related metric: Temporal Stability

For completeness, the third geometric heuristic in the same pipeline is **temporal stability** — the mean absolute pixel difference between consecutive depth frames. It measures how much the depth field changes over time (frozen geometry vs natural motion). See `compute_geometric_features()` in [`explainability_toolkit.py`](explainability_toolkit.py) for implementation details.
