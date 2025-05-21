## Mathematical Formulation

### 1. Homography & Real-World Height

1. **Compute homography** from four reference points on the floor:

  $\mathbf{H} = \arg\min_H \sum_{i=1}^4 \|\,\mathbf{p}_i - H\,\mathbf{P}_i\|^2$,
   where $\mathbf{P}_i = [X_i, Y_i, 1]^T$ are known ground-plane coords (meters) and $\mathbf{p}_i = [x_i, y_i, 1]^T$ their pixel locations.

2. **Invert** to map image → ground:

  $\mathbf{P} = H^{-1}\mathbf{p},\quad (X,Y,W)^T = H^{-1}(x,y,1)^T,\quad (X_{\text{floor}}, Y_{\text{floor}}) = (X/W,\;Y/W)$.

3. **Estimate real height** $h_{\text{real}}$ of a person with pixel height $h_{\text{px}}$:

  $h_{\text{real}} = h_{\text{px}} \times \frac{D_{\mathrm{ref}}}{h_{\mathrm{ref,px}}}$,
   where $D_{\mathrm{ref}}$ is the known real height of your reference marker and $h_{\mathrm{ref,px}}$ its pixel height at the same location.

---

### 2. Gait Dynamics

* **Foot-point trajectory**: for each frame `t`, let `P_t` be the ground-plane coordinate of the bottom-center of the bbox.
* **Stride length** between two consecutive steps: `d = ||P_t2 - P_t1||` (meters).
* **Walking speed** over window `Δt`:  
  `v = (1 / (N - 1)) * Σ_{i=1}^{N-1} (||P_{t(i+1)} - P_{ti}||) / (Δt / N)` (m/s)



---

### 3. Silhouette & Shape

* **Binary mask** $M(x,y)$ via background subtraction or tiny segmentation.

* **Hu moments** $\{\phi_1,\dots,\phi_7\}$ computed from central moments $\mu_{pq}$:

 $\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^{1 + (p+q)/2}},\quad \phi_1 = \eta_{20} + \eta_{02},\;\dots$

* **Contour solidity**:

 $\text{solidity} = \frac{\text{Area of contour}}{\text{Area of its convex hull}}$.

---

### 4. Texture & Shape Descriptors

* **Local Binary Patterns (LBP)** histogram on the torso+leg region:

`LBP(x, y) = Σ_{k=0}^7 s(I_k - I_c) * 2^k`, where:

`s(z) = 1 if z ≥ 0, else 0`
.

* **Histogram of Oriented Gradients (HOG)**: split region into cells, compute gradient orientation histograms ($0–180°$) with block-wise normalization.


## Full Algorithm

```text
Inputs:
  - Monochrome CCTV video
  - Four floor reference markers with known real coords & pixel coords
  - Pretrained person detector (e.g. YOLOv5-nano)
  - Tracker (e.g. DeepSORT)
  - Labeled training data (~200 tracks per class)

Offline: Compute homography H from reference points.

For each incoming frame:
  1. Detect persons → get bboxes {b_i = (x1,y1,x2,y2)}.
  2. Track detections → assign track IDs and maintain trajectory of bbox bottoms.

Every window of N frames per track:
  A. For each frame t in window:
     a. Foot-pixel = ((x1+x2)/2, y2, 1)^T.
     b. Map → ground: P_t = H^{-1} · foot-pixel → normalize to (X_t,Y_t).
     c. Pixel height h_px = y1–y2; real height h_real = h_px·D_ref / h_ref_px.
     d. Aspect ratio ar = (y2–y1)/(x2–x1).
     e. Extract crop C_t; compute binary mask M_t via bg-sub.
     f. Compute Hu moments φ₁…φ₇ from M_t.
     g. Compute solidity s.
     h. Compute LBP histogram ℓ (e.g. 64-bin).
     i. Compute HOG descriptor g.

  B. Aggregate features over window:
     - Mean real height ĥ, std σ_h.
     - Mean aspect ratio ā.
     - Stride length d and speed v over trajectory {P_t}.
     - Mean Hu moments, solidity.
     - Average LBP ℓ̄, average HOG ḡ.

  C. Form feature vector:
     F = [ĥ, σ_h, ā, d, v, φ̄₁…φ̄₇, s, ℓ̄, ḡ].

  D. Classify with trained Random Forest → soft probabilities p = [p_kid, p_teen, p_adult].

  E. Temporal smoothing:
     - Store last K predictions per track.
     - If max vote fraction ≥ 0.6, emit label = argmax; else label = “Unknown”.

Output:
  - For each active track: label ∈ {Kid, Teen, Adult, Unknown}
```

---

### Notes & Next Steps

* **Training:** collect a balanced dataset in your specific scene, extract F per track, and fit a Random Forest (or XGBoost) on the three classes.
* **Zone-based tweaks:** if residual bias remains, split by “Near/Mid/Far” floor zones (via `$ Y_t $`) and either adjust thresholds or train separate classifiers.
* **Fallback:** tracks with too few frames or low confidence remain “Unknown,” avoiding false positives.

