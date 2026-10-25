// Builds docs/ClearSAR_Project_Guide.docx
const fs = require("fs");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, TableOfContents, PageBreak, Footer, Header, PageNumber, BorderStyle,
} = D;

const OUT = process.argv[2];
const CW = 9638; // A4 content width (DXA) with 2 cm margins
const ACCENT = "0F766E", DARK = "1F2937", MUTED = "6B7280", LIGHT = "E6F4F1", HEAD = "D1FAE5";

// ---------- inline markup: **bold**, *italic*, `code`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, color: "0F5132", ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const body = [];
const P = (t, o = {}) => body.push(new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 288 }, alignment: o.align, ...o.para }));
const H1 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true }));
const H2 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] }));
const H3 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] }));
const UL = (items, level = 0) => items.forEach((t) => body.push(new Paragraph({ numbering: { reference: "bul", level }, children: runs(t), spacing: { after: 60, line: 276 } })));
const OL = (items, ref = "num") => items.forEach((t) => body.push(new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 276 } })));
const F = (t) => body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 140 },
  children: [new TextRun({ text: t, font: "Cambria Math", size: 23, color: DARK })] }));
const NOTE = (title, t) => body.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: LIGHT, color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT, space: 8 } },
  spacing: { before: 120, after: 160, line: 288 }, indent: { left: 140, right: 140 },
  children: [new TextRun({ text: title + "  ", bold: true, color: ACCENT }), ...runs(t)],
}));
let numCount = 0;
function OLn(items) { numCount += 1; const ref = "num" + numCount; NUMREFS.push(ref); OL(items, ref); }
const NUMREFS = [];

function T(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CW / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CW - w.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "CBD5E1" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (txt, i, head) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: HEAD, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: String(txt).split("\n").map((line) => new Paragraph({ spacing: { after: 20 }, children: runs(line, head ? { bold: true, size: 19 } : { size: 19 }) })),
  });
  body.push(new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: w,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  }));
  body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
}

// =====================================================================================
// TITLE
body.push(new Paragraph({ spacing: { before: 2600, after: 200 }, children: [new TextRun({ text: "ClearSAR", bold: true, size: 72, color: ACCENT })] }));
body.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Seeing the ground through the clouds", size: 36, color: DARK })] }));
body.push(new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "SAR-to-optical cloud removal on SEN12MS-CR (Asia subset) and the ClearSAR Advisor expert system", size: 26, color: MUTED })] }));
body.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Complete project guide", bold: true, size: 26 })] }));
body.push(new Paragraph({ spacing: { after: 120 }, children: runs("Every decision, model, hyper-parameter, metric and formula in the project - what was done, how, and why.", { color: MUTED, size: 22 }) }));
body.push(new Paragraph({ spacing: { after: 120 }, children: runs("Code: github.com/karthikeya-bhamidipati/clearsar · October 2026", { color: MUTED, size: 22 }) }));
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 32, color: ACCENT })], spacing: { after: 200 } }));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(new Paragraph({ children: runs("If the table of contents is empty, right-click it in Word and choose *Update Field*.", { color: MUTED, size: 18 }) }));

// =====================================================================================
H1("1. The project in one page");
P("**The problem.** Optical satellites such as Sentinel-2 photograph the Earth in 13 spectral bands, but clouds hide the ground in a large share of all images - in the tropics and during monsoons, for weeks at a time. A cloudy image is useless for crop monitoring, mapping or disaster response.");
P("**The idea.** Radar satellites such as Sentinel-1 send their own microwave signal and measure the echo. Microwaves pass through clouds, so radar sees the ground in any weather - but it measures surface roughness and moisture, not colour. ClearSAR trains deep neural networks that take the radar image **and** the cloudy optical image and predict what the optical image would look like without clouds.");
P("**The deep-learning study.** Three architecture families are compared under one identical protocol - same data, same budget, same loss, same evaluation - so that differences in the results can be attributed to the architecture:");
UL(["**Model A - U-Net**: a convolutional encoder-decoder (strong locality bias).",
  "**Model B - Swin U-Net**: the same U-Net with shifted-window self-attention in its bottleneck (mixed bias).",
  "**Model C - pure Vision Transformer (ViT)**: no convolutions at all (weak bias, must learn locality from data)."]);
P("Around them, a set of controlled experiments asks *why* each model behaves as it does: how much data each needs (E1), how they cope with unfamiliar continents (E2), what the U-Net's skip connections and loss do (E3), what an adversarial (GAN) loss changes (E4), what matters inside Swin (E5) and the ViT (E6), and whether the models can estimate their own uncertainty (E7). Baselines, a leakage check and repeated seeds make the conclusions trustworthy.");
P("**The expert system.** A reconstruction is an estimate. The **ClearSAR Advisor** is a rule-based expert system that inspects each patch and the networks' outputs (cloud cover, agreement between architectures, predicted uncertainty, how much the network relied on radar) together with what the experiments showed, and decides how far a reconstruction can be trusted, which model to use, whether it fits the user's purpose and what to do next - explaining every step.");
P("**The main findings** (chapter 13). All models recover 5-7 dB PSNR over the cloudy image. Under an identical protocol no architecture is clearly best on overall PSNR - even a 3-layer CNN is competitive - but under thick cloud the Vision Transformer, which looks furthest across the image, is 2-4 dB better. A random patch split would have inflated scores by 5.1 dB. Of three uncertainty methods only the learned variance head is reasonably calibrated. The expert system's trust levels agree with true reconstruction quality in a pilot test.");
P("**The tools.** PyTorch training on Kaggle (2 × NVIDIA T4 GPUs, one 11-hour session for all 26 trainings), a Streamlit web frontend, and a knowledge base in YAML with a certainty-factor inference engine.");
NOTE("How to use this guide.", "Chapters 2-4 explain the data and why it is prepared as it is. Chapters 5-7 explain the models, training and metrics including every hyper-parameter. Chapter 8 lists every experiment and its hypothesis. Chapters 9-12 cover the infrastructure, debugging history, expert system and frontend. Chapter 14 is a bank of likely examiner questions with answers; chapter 15 is a glossary.");

// =====================================================================================
H1("2. Background: why radar can help remove clouds");
H2("2.1 Optical imagery: Sentinel-2");
P("Sentinel-2 (ESA, two satellites, 5-day revisit) carries a passive multispectral camera: it measures sunlight reflected by the surface in 13 bands from visible to short-wave infrared. Every band tells something different - red and near-infrared reveal vegetation, short-wave infrared reveals moisture and minerals. Because it depends on sunlight passing through the atmosphere twice, clouds and their shadows block it completely.");
T(["Band", "Name", "Centre wavelength", "Native resolution", "Typical use"], [
  ["B1", "Coastal aerosol", "443 nm", "60 m", "aerosols, coastal water"],
  ["B2", "Blue", "490 nm", "10 m", "true colour, water"],
  ["B3", "Green", "560 nm", "10 m", "true colour, vegetation vigour"],
  ["B4", "Red", "665 nm", "10 m", "true colour, chlorophyll absorption"],
  ["B5", "Red edge 1", "705 nm", "20 m", "vegetation stress"],
  ["B6", "Red edge 2", "740 nm", "20 m", "vegetation"],
  ["B7", "Red edge 3", "783 nm", "20 m", "vegetation"],
  ["B8", "Near infrared (NIR)", "842 nm", "10 m", "biomass, NDVI"],
  ["B8A", "Narrow NIR", "865 nm", "20 m", "vegetation, water vapour reference"],
  ["B9", "Water vapour", "945 nm", "60 m", "atmospheric correction"],
  ["B10", "Cirrus", "1375 nm", "60 m", "detecting thin, high clouds"],
  ["B11", "SWIR 1", "1610 nm", "20 m", "moisture, snow/cloud discrimination"],
  ["B12", "SWIR 2", "2190 nm", "20 m", "minerals, burn scars"],
], [8, 18, 15, 14, 30]);
P("In SEN12MS-CR all bands are resampled to 10 m and stored as integers (digital numbers, roughly reflectance × 10 000). The **true-colour (RGB) image** used for display and FID is made from **B4, B3, B2**.");
H2("2.2 Radar imagery: Sentinel-1");
P("Sentinel-1 carries a **synthetic aperture radar (SAR)** operating in C-band (about 5.4 GHz, 5.6 cm wavelength). It transmits microwave pulses and records the **backscatter** - the share of energy returned to the sensor. Microwaves pass through clouds, smoke and darkness. Backscatter depends on surface roughness, geometry and moisture: calm water acts like a mirror and returns almost nothing (very dark), cities with corner-like buildings return a lot (very bright), forests scatter in all directions (medium).");
UL(["**Polarisations.** Sentinel-1 sends vertically polarised waves and records the vertically (VV) and horizontally (VH) polarised echo. VV is sensitive to surface roughness; VH (cross-polarised) to volume scattering in vegetation.",
  "**Decibels.** Backscatter spans several orders of magnitude, so it is expressed in decibels: σ⁰[dB] = 10·log₁₀(σ⁰). Typical land values are −25 to 0 dB. SEN12MS-CR already stores S1 in dB.",
  "**Speckle.** SAR images contain grainy noise (speckle) caused by interference of echoes within one pixel. Filters such as the Lee filter reduce it at the cost of detail."]);
H2("2.3 Why the task is hard (ill-posed)");
P("Radar does not measure colour. A green field and a brown field of the same roughness look identical in radar. Under a thick cloud the network therefore has to **infer** plausible optical values from radar texture plus context learned from training data. This makes the problem *ill-posed*: several outputs are consistent with the input. This has three consequences that shape the project: (1) a loss that averages over possibilities (L2) produces blur, (2) adversarial losses produce sharp but partly invented detail, (3) estimates of **uncertainty** and an expert system that judges trust are genuinely useful.");

// =====================================================================================
H1("3. The dataset: SEN12MS-CR");
P("SEN12MS-CR (Ebel et al., 2021) is the standard benchmark for SAR-to-optical cloud removal. Every sample is a **triplet** of co-registered 256 × 256-pixel patches (2.56 km × 2.56 km at 10 m):");
T(["Part", "Bands", "Role in ClearSAR"], [
  ["Sentinel-1 SAR", "2 (VV, VH) in dB", "model input"],
  ["Sentinel-2 cloudy", "13", "model input"],
  ["Sentinel-2 cloud-free", "13 (acquired close in time)", "target (the answer the model must reproduce)"],
], [25, 30, 45]);
UL(["**Size:** 169 regions of interest (ROIs) worldwide × 4 seasons, **122 218 triplets** in total. The four community mirrors on Kaggle (spring, summer, fall, winter) together contain exactly 366 654 GeoTIFF files = 122 218 × 3.",
  "**Naming:** `ROIs1158_spring_s2_cloudy_12_p345.tif` = season ROIs1158_spring, modality s2_cloudy, scene 12, patch 345. The pipeline pairs the three modalities purely by this name.",
  "**Scene census:** each scene's centre coordinate was reverse-geocoded to a country and continent (`data/scans/scenes.csv`). 175 scenes were found: Africa 31, **Asia 37** (26 608 patches), Europe 43, North America 40, Oceania 9, South America 15."]);
NOTE("Why Asia?", "The project plan focuses on Asia (including India) to study a coherent but diverse region - monsoon tropics, deserts, temperate farmland - and to keep the training set manageable. Training on Asia and testing on other continents then gives a natural domain-shift experiment.");

// =====================================================================================
H1("4. Data pipeline");
H2("4.1 Indexing and pairing");
P("All `.tif` files under `/kaggle/input` are walked once and matched with the regular expression `(ROIs\\d+_[a-z]+)_(s1|s2_cloudy|s2)_(\\d+)_p(\\d+)\\.tif`. Files are grouped by (season, scene, patch); only complete triplets are kept. Duplicates across mirrors collapse onto the same key. **Why:** the original code paired three independently sorted file lists, which silently misaligns inputs and targets if any mirror differs.");
H2("4.2 Preprocessing (normalisation)");
T(["Modality", "Transformation", "Why"], [
  ["Sentinel-1 VV", "clip to [−25, 0] dB, rescale to [0, 1]: (x + 25) / 25", "standard SEN12MS-CR range; outliers clipped; networks train best on inputs of similar scale"],
  ["Sentinel-1 VH", "clip to [−32.5, 0] dB, rescale to [0, 1]", "VH is weaker than VV, so its range is wider"],
  ["S1 missing values", "NaN → lower bound (−25 / −32.5 dB)", "0 dB would mean maximum brightness - wrong"],
  ["S1 already linear?", "auto-detected (no negative values) and converted to dB", "robustness to other data sources"],
  ["Sentinel-2 (cloudy and clear)", "clip to [0, 10 000], divide by 10 000 → reflectance in [0, 1]", "standard SEN12MS-CR preprocessing; all 13 bands kept"],
  ["Lee speckle filter", "optional (`CLEARSAR_APPLY_LEE=1`), applied in the linear domain; **off** in all runs", "the published baselines do not use it; plan lists it as an ablation only"],
], [22, 40, 38]);
P("**Model input** = [VV, VH, 13 cloudy bands] = **15 channels**. **Model target** = 13 clear bands. The original pipeline converted the already-dB radar to dB a second time, which mapped almost every pixel to the same value and erased the radar signal; this was the most important bug found (chapter 10).");
H2("4.3 Splitting by scene");
P("Neighbouring patches overlap and look alike (**spatial autocorrelation**). If patches were split at random, near-duplicates of training patches would appear in the test set and inflate every score. ClearSAR therefore splits by **scene**:");
UL(["The 37 Asian scenes are shuffled with seed 42. About 20 % go to the test set, 20 % to validation and 60 % to training. One Indian scene is always forced into the test set so an India-only result can be reported.",
  "One random scene from each of Europe, Africa, North America and South America forms the **out-of-continent (OOD) test set**, used only for testing.",
  "Each split is then capped and sampled **evenly across its scenes** so no single scene dominates."]);
T(["Split", "Scenes", "Patches", "Countries"], [
  ["Train", "23", "3 000", "CN, IN, IQ, IR, JP, KR, TR, YE"],
  ["Validation", "7", "400", "CN, IR, JP, KZ, MM, PK, SA"],
  ["Test - Asia (in-domain)", "7", "600", "CN, IN, IR, LB, TM, TW"],
  ["Test - other continents (OOD)", "4", "600", "AR (Argentina), CF (Central African Rep.), FR (France), US"],
], [30, 12, 13, 45]);
P("**Validation** is used during training to pick the best checkpoint; the **test** sets are touched only once, at the end. This separation prevents test information from influencing any choice.");
H2("4.4 Caching");
P("Reading and decoding three GeoTIFFs per sample on every training step would starve the GPU (Kaggle has 4 CPU cores). Every selected patch is therefore read, preprocessed and written **once** to a float16 memory-mapped NumPy file (`/tmp/clearsar_cache`). One patch = 28 channels × 256 × 256 × 2 bytes ≈ 3.7 MB; all 4 600 patches ≈ 17 GB. Unreadable files are detected and excluded instead of crashing the run.");
H2("4.5 Augmentation and cropping");
UL(["**Training:** a random 128 × 128 crop from each 256 × 256 patch, then random horizontal flip, vertical flip and rotation by 0/90/180/270°. Each patch therefore yields many different training examples (data augmentation reduces overfitting). Satellite images have no canonical orientation, so flips and rotations are label-preserving.",
  "**Why 128 crops:** four times cheaper than full patches, more variety per epoch, and the ViT's learned positional embedding is defined for a fixed 16 × 16 token grid.",
  "**Evaluation:** every full 256 × 256 patch is cut into four 128 × 128 tiles, each tile is predicted and the four outputs are stitched back. All models therefore see exactly the input size they were trained on."]);
H2("4.6 Cloud-cover estimate (for binning results)");
P("To report results by cloud cover (0-20 %, …, 80-100 %), each test patch gets a cloud fraction: a pixel counts as cloudy if the cloudy image is brighter than the clear target by more than **0.08 reflectance** averaged over B2, B3, B4. This uses the target, which is acceptable because it is only used for grouping results, never as a model input. (The expert system uses a different, target-free cloud test - chapter 11.)");

// =====================================================================================
H1("5. The models");
P("All generators map a **15-channel** input to a **13-channel** output. The output passes through a **sigmoid**, so every predicted reflectance lies in [0, 1] - the same range as the target. Construction always uses keyword arguments (`in_channels=15, out_channels=13`); passing a boolean positionally - the bug that crashed the original Kaggle run - now raises an error.");
H2("5.1 Inductive bias - the idea behind the comparison");
P("An **inductive bias** is an assumption built into an architecture. Convolutions assume that nearby pixels are related and that the same pattern means the same thing anywhere in the image (translation equivariance). This is usually true for images, so CNNs learn quickly from little data. Self-attention assumes nothing about distance - every position can interact with every other - which is more flexible but must be learned from data. The three models span this range: A (strong bias) → B (mixed) → C (weak). Experiment E1 tests the classic prediction that strong bias wins with little data.");
H2("5.2 Model A - U-Net (7.71 M parameters)");
P("A U-Net (Ronneberger et al., 2015) is an encoder-decoder with **skip connections**. The encoder halves the resolution three times while doubling channels, capturing context; the decoder upsamples back; skip connections copy each encoder feature map directly into the decoder at the same resolution so fine detail does not have to be reconstructed from the compressed bottleneck.");
T(["Stage", "Operation", "Channels", "Resolution (128 input)"], [
  ["Block (all stages)", "2 × [Conv 3×3, no bias → BatchNorm → ReLU]", "-", "-"],
  ["enc1", "block", "15 → 64", "128 × 128"],
  ["enc2", "MaxPool 2×2 → block", "64 → 128", "64 × 64"],
  ["enc3", "MaxPool 2×2 → block", "128 → 256", "32 × 32"],
  ["bottleneck", "MaxPool 2×2 → block", "256 → 512", "16 × 16"],
  ["dec3", "ConvTranspose 2×2 stride 2 (512→256), concat enc3, block", "512 → 256", "32 × 32"],
  ["dec2", "ConvTranspose (256→128), concat enc2, block", "256 → 128", "64 × 64"],
  ["dec1", "ConvTranspose (128→64), concat enc1, block", "128 → 64", "128 × 128"],
  ["final", "Conv 1×1 → sigmoid", "64 → 13", "128 × 128"],
], [16, 46, 16, 22]);
UL(["**BatchNorm** normalises activations per batch, stabilising and speeding up training; convolution biases are dropped because BatchNorm adds its own.",
  "**Variants:** `use_skips=False` replaces each skip tensor by zeros (E3a, identical parameter count); `dropout=0.1` adds 2-D dropout after the bottleneck, dec3 and dec2 (E7b MC dropout); `uncertainty=True` doubles the output to 26 channels (E7a)."]);
H2("5.3 Model B - Swin U-Net (7.46 M parameters)");
P("Identical encoder and decoder to Model A; only the conv bottleneck is replaced by **4 Swin Transformer blocks** (Liu et al., 2021) operating on the 16 × 16 grid of 256-dimensional features, followed by a LayerNorm and a 1×1 convolution (256 → 512, BatchNorm, ReLU) so the decoder is unchanged.");
UL(["**Self-attention:** each position computes a query, key and value vector; its output is a weighted average of all values, weighted by query-key similarity: Attention(Q,K,V) = softmax(QKᵀ/√d)·V. Eight heads run in parallel on 32-dimensional slices.",
  "**Windows:** attention is computed inside non-overlapping 4 × 4 windows (16 tokens each) - cheap and local.",
  "**Shifted windows:** every second block cyclically shifts the grid by 2 positions before partitioning, so information crosses window borders. A mask (−100 before the softmax) stops tokens that were wrapped around the image edge from attending to each other.",
  "**Relative position bias:** a learned bias per head for every relative offset inside a window ((2·4 − 1)² = 49 entries) tells attention how far apart two tokens are.",
  "**Block structure (pre-norm):** x ← x + WindowAttention(LayerNorm(x)); x ← x + MLP(LayerNorm(x)), MLP = Linear(256→1024) → GELU → Linear(1024→256).",
  "**Variants:** no shift (E5a); window 8 and window 16 (E5b; window 16 covers the whole 16 × 16 grid = global attention)."]);
H2("5.4 Model C - pure Vision Transformer (4.84 M parameters)");
P("Follows Dosovitskiy et al. (2021): the image is cut into **8 × 8-pixel patches** (a 128 crop gives 16 × 16 = **256 tokens**). There are no convolutions anywhere.");
T(["Component", "Setting"], [
  ["Patch embedding", "flatten each 8×8×15 patch (960 values) → Linear → 192 dimensions"],
  ["Positional embedding", "learned, one 192-d vector per token position (256), initialised N(0, 0.02²); bicubically interpolated for other input sizes"],
  ["Encoder", "8 pre-norm Transformer layers: 3 heads, MLP 192 → 768 → 192 with GELU, no dropout"],
  ["Decoder", "2 more Transformer layers of the same kind, then LayerNorm"],
  ["Un-patchify", "Linear 192 → 13 × 8 × 8 per token, reshaped back into the image, sigmoid"],
  ["Variants", "no positional embedding (E6a, 4.79 M); patch size 16 (E6b, 5.84 M, 64 tokens)"],
], [25, 75]);
P("**Why positional embeddings matter:** self-attention is *permutation-invariant* - shuffle the tokens and the outputs are shuffled the same way. Without position information the model cannot know where a patch is. E6a tests this.");
H2("5.5 Baselines");
UL(["**Identity:** the output is simply the cloudy input. Surprisingly strong on lightly clouded patches; every useful model must beat it (21.18 dB on the Asian test set, 20.95 dB on other continents).",
  "**Tiny CNN (0.05 M):** Conv 3×3 (15→64) → ReLU → Conv 3×3 (64→64) → ReLU → Conv 3×3 (64→13) → sigmoid. If a large model cannot clearly beat it, either the task is too easy or the pipeline is broken."]);
H2("5.6 PatchGAN discriminator (for E4)");
P("A second network (Isola et al., 2017) that looks at (input, output) pairs and scores each ~70 × 70-pixel region as real or fake. Input: 15 + 13 = 28 channels. Layers: Conv 4×4 stride 2 (28→64, no BatchNorm) → Conv 4×4 s2 (→128) → Conv 4×4 s2 (→256) → Conv 4×4 stride 1 (→512) → Conv 4×4 (→1 logit per region); LeakyReLU(0.2) and BatchNorm in between. The generator is trained to fool it, which pushes outputs towards realistic texture.");
H2("5.7 Uncertainty heads (for E7)");
UL(["**Variance head (aleatoric, E7a):** the U-Net predicts 26 channels - 13 means (sigmoid) and 13 log-scales s. It is trained with a Laplace negative log-likelihood (§6.1), so it learns to output a large scale b = eˢ where errors are expected (e.g. under thick cloud).",
  "**MC dropout (epistemic, E7b):** a U-Net with dropout p = 0.1 is run 20 times at test time with dropout switched on; the mean is the prediction, the standard deviation is the uncertainty (Gal & Ghahramani, 2016).",
  "**Deep ensemble (epistemic, E7c):** three U-Nets trained with seeds 42, 43, 44; mean and standard deviation across them (Lakshminarayanan et al., 2017)."]);
P("*Aleatoric* uncertainty is noise in the data itself (information truly missing under a cloud); *epistemic* uncertainty is the model's ignorance (e.g. an unfamiliar continent) and should rise out of domain.");
H2("5.8 Output initialisation");
P("Before training, the bias of each model's output layer is set so that sigmoid(bias) equals the mean reflectance of that band over 10 training batches (range 0.002 for the cirrus band to ~0.23). Without it, every model starts by predicting 0.5 (far brighter than real surfaces at ~0.1) and convolutional models wasted much of a short run just learning that offset; in a 300-step check the fix raised the U-Net from 18.0 dB to 26.8 dB. It is applied identically to all runs (Karpathy's \"init well\" advice).");

// =====================================================================================
H1("6. Training");
H2("6.1 Loss functions");
H3("L1 loss (main runs)");
F("L1 = (1/N) Σ | ŷ − y |");
P("Mean absolute difference over all pixels and bands. **Why L1:** for an ill-posed problem L2 is minimised by the *mean* of all plausible answers (blur), while L1 is minimised by the *median*, which keeps edges sharper and is robust to outliers. L1 is the standard choice in Pix2Pix and SEN12MS-CR work.");
H3("L2 loss (E3b)");
F("L2 = (1/N) Σ ( ŷ − y )²");
H3("Adversarial loss (E4)");
F("L_D = ½ [ BCE(D(x, y), 1) + BCE(D(x, G(x)), 0) ]");
F("L_G = BCE(D(x, G(x)), 1) + λ · L1(G(x), y),   λ = 100 (and 10)");
P("BCE is binary cross-entropy on the discriminator's logits. The discriminator learns to tell real clear images from generated ones; the generator learns to make outputs the discriminator accepts, while the λ·L1 term keeps them close to the truth. Smaller λ = more weight on realism.");
H3("Laplace negative log-likelihood (E7a)");
F("L = (1/N) Σ [ |y − μ| · e^(−s) + s ],   b = e^s");
P("If the model predicts a large scale b it pays the penalty s, but errors are divided by b; the optimum is b ≈ the expected absolute error. s is clamped to [−10, 10] and the loss is computed in float32 for numerical safety.");
H2("6.2 Optimisation");
UL(["**Optimiser: AdamW** (Loshchilov & Hutter, 2019) - Adam with decoupled weight decay. Adam adapts the step size per parameter using running averages of gradients (β₁) and squared gradients (β₂). β = (0.9, 0.999) for L1 runs; (0.5, 0.999) for GAN runs, the Pix2Pix setting that stabilises adversarial training.",
  "**Weight decay 0.01**: gently shrinks weights towards zero (regularisation).",
  "**Learning-rate schedule:** linear warm-up over 500 steps (from ~0 to the peak) then cosine decay to 0 at the last step. Warm-up avoids unstable first steps, especially for transformers; cosine decay lets training settle into a good minimum.",
  "**Gradient clipping:** the generator's gradient norm is clipped to 1.0, protecting against occasional exploding gradients (important for attention models and GANs).",
  "**Mixed precision (AMP):** forward and backward passes run in float16 on the T4's tensor cores (≈2× faster, half the memory); a GradScaler multiplies the loss to avoid float16 underflow and skips steps with overflowing gradients. The generator and discriminator each have their own scaler.",
  "**Model selection (early stopping):** every 1 000 steps the model is evaluated on the 400 validation patches; the checkpoint with the lowest validation L1 is kept. The final model is the best one, not the last one."]);
H2("6.3 Complete hyper-parameter table");
T(["Hyper-parameter", "Value", "Notes"], [
  ["Training patches / val / test / OOD", "3 000 / 400 / 600 / 600", "sampled evenly across scenes"],
  ["Input / output channels", "15 / 13", "VV, VH + 13 cloudy bands → 13 clear bands"],
  ["Training crop", "128 × 128", "random position + flips + 90° rotations"],
  ["Batch size", "16", "evaluation batch 8 patches (= 32 tiles)"],
  ["Training steps per run", "15 000", "chosen by the GPU benchmark/planner; = 240 000 crops ≈ 80 passes over the 3 000 patches"],
  ["Peak learning rate", "3 × 10⁻⁴ (GAN runs 2 × 10⁻⁴)", "same for all architectures"],
  ["Warm-up / schedule", "500 steps linear / cosine to 0", ""],
  ["Optimiser", "AdamW, β = (0.9, 0.999), GAN (0.5, 0.999)", "weight decay 0.01"],
  ["Gradient clipping", "max norm 1.0", "generator"],
  ["Precision", "float16 AMP + GradScaler", ""],
  ["Validation interval", "1 000 steps", "best checkpoint by validation L1"],
  ["Seeds", "42 (main); 43, 44 (repeats)", "split seed fixed at 42"],
  ["GAN weight λ", "100 (E4), 10 (E4b)", ""],
  ["MC dropout", "p = 0.1, 20 passes", ""],
  ["Swin", "4 blocks, dim 256, 8 heads, window 4, shift 2, MLP ×4", ""],
  ["ViT", "patch 8, dim 192, 8 + 2 layers, 3 heads, MLP 768", ""],
  ["Output bias", "logit(mean band reflectance)", "from 10 training batches"],
  ["Data loader workers", "2 per job (4 CPUs shared by 2 jobs)", ""],
], [32, 34, 34]);

// =====================================================================================
H1("7. Evaluation and metrics");
P("All metrics are computed on the full 256 × 256 test patches (tiled prediction), on all 13 bands with values in [0, 1], one value per patch, then averaged. Results are reported for the Asian test set and the out-of-continent set separately, by cloud-cover bin and by country.");
H2("7.1 PSNR - peak signal-to-noise ratio");
F("PSNR = 10 · log₁₀( MAX² / MSE ),   MAX = 1");
P("Pixel fidelity on a logarithmic scale, in decibels. **Higher is better.** +3 dB ≈ halving the mean squared error. Typical values here: ~21 dB for the cloudy input, ~27-29 dB for trained models. Limitation: PSNR rewards blurry averages and says nothing about perceived realism.");
H2("7.2 SSIM - structural similarity");
F("SSIM(x, y) = [(2μₓμᵧ + C₁)(2σₓᵧ + C₂)] / [(μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂)]");
P("Compares local means (brightness), variances (contrast) and covariance (structure) inside a sliding Gaussian window (11 × 11, σ = 1.5), with C₁ = (0.01)², C₂ = (0.03)². Computed per band and averaged. **Range up to 1 (identical); higher is better.** More sensitive to blur and lost structure than PSNR. The GPU implementation is unit-tested against scikit-image (difference < 10⁻³).");
H2("7.3 SAM - spectral angle mapper");
F("SAM = arccos( ⟨ŷ, y⟩ / (‖ŷ‖ · ‖y‖) )   (per pixel, in degrees, then averaged)");
P("Treats each pixel's 13 band values as a vector and measures the angle between prediction and truth. It ignores overall brightness and measures whether the *spectral shape* (the 'colour' across all bands) is right - essential for vegetation and land-cover analysis. **Lower is better; 0° = perfect.**");
H2("7.4 MAE and RMSE");
F("MAE = (1/N) Σ |ŷ − y|          RMSE = √[ (1/N) Σ (ŷ − y)² ]");
P("Average error in reflectance units (0.03 = 3 % reflectance). RMSE penalises large errors more. **Lower is better.**");
H2("7.5 FID - Fréchet Inception Distance");
F("FID = ‖μᵣ − μ_g‖² + Tr( Σᵣ + Σ_g − 2 (Σᵣ Σ_g)^½ )");
P("Each RGB image (B4, B3, B2, brightness × 3, clipped) is passed through an Inception-v3 network pre-trained on ImageNet; its 2 048-dimensional features of real and generated images are modelled as Gaussians (mean μ, covariance Σ) and the distance between the two distributions is computed (Heusel et al., 2017). It measures **realism** of the image distribution rather than per-pixel accuracy. **Lower is better.** Computed with torchmetrics on the 600 test images per set - a small sample, so compare models with it, do not interpret absolute values (cloudy input: 128 / 164).");
H2("7.6 Uncertainty metrics (E7)");
T(["Metric", "Definition", "Ideal"], [
  ["coverage95", "share of true values inside the predicted 95 % interval: |error| ≤ b·ln 20 (Laplace) or ≤ 1.96σ (Gaussian)", "0.95; lower = over-confident, higher = under-confident"],
  ["median σ", "typical predicted uncertainty (robust to a few extreme pixels)", "should rise on other continents (epistemic)"],
  ["Spearman ρ(error, σ)", "rank correlation between predicted uncertainty and actual error", "high positive"],
  ["Sparsification curve", "MAE of the remaining pixels after removing the X % most uncertain pixels (X = 0 … 95 %)", "falls as fast as the oracle (removing the X % largest errors)"],
  ["AUSE", "area between the model's sparsification curve and the oracle's", "0"],
], [20, 55, 25]);
P("2 000 random pixels per test image are sampled (fixed seed) for these statistics.");
H2("7.7 Effective receptive field (E3c)");
P("For 16 validation crops, the gradient of the centre output pixel with respect to every input pixel is computed and its absolute value summed over channels and images (Luo et al., 2016). r50 / r90 are the radii containing 50 % / 90 % of that influence. It shows how far a model *actually* looks - usually much less than its theoretical receptive field.");
H2("7.8 Noise floor and verdicts");
P("Two trainings that differ only in their random seed give slightly different scores. A difference between two models smaller than this **seed variance** is not meaningful. The frontend therefore marks an experiment's change as *Within noise* if it is smaller than max(0.3 dB, 2 × the reference model's seed standard deviation) for PSNR (0.005 for SSIM, 5 for FID), and otherwise as *Supported* or *Contradicted* by comparing its sign with the prediction.");

// =====================================================================================
H1("8. Experiments");
P("Every experiment changes exactly one thing relative to a reference run and uses the identical protocol (data, steps, learning-rate schedule, evaluation). Predictions were written in the project plan before training. The 26 trainings run in this priority order, so if time ever runs short the least important are dropped first:");
T(["#", "Run", "Change vs reference", "Question / prediction"], [
  ["1", "A_unet", "- (core)", "Model A reference"],
  ["2", "B_swin", "- (core)", "Model B reference"],
  ["3", "C_vit", "- (core)", "Model C reference"],
  ["4", "tiny_cnn", "3-layer CNN", "Baseline: should be clearly worse than A"],
  ["5-10", "A/B/C_f25, A/B/C_f50", "25 % / 50 % of training patches (nested subsets, per scene)", "E1: all drop; ViT loses most; U-Net best at 25 %"],
  ["11", "A_random", "random patch split instead of scene split", "Leakage: score inflated"],
  ["12", "A_noskip", "skip connections zeroed", "E3a: blurrier, lower SSIM"],
  ["13", "A_gan", "+ PatchGAN, λ = 100, lr 2e-4, β₁ 0.5", "E4: sharper, lower FID, lower PSNR"],
  ["14", "A_unc", "+ variance head, Laplace NLL", "E7a: uncertainty highest under thick cloud"],
  ["15", "C_nopos", "no positional embedding", "E6a: spatial layout lost"],
  ["16", "B_noshift", "no window shift", "E5a: information can't cross windows"],
  ["17", "A_l2", "L2 instead of L1", "E3b: blurrier"],
  ["18", "A_gan10", "PatchGAN, λ = 10", "E4: even sharper, even lower PSNR"],
  ["19", "A_mcd", "dropout 0.1, 20 MC passes", "E7b: uncertainty rises out of domain"],
  ["20-21", "A_unet_s43, s44", "seeds 43, 44", "seed variance; members of the ensemble"],
  ["22-23", "B_swin_s43, C_vit_s43", "seed 43", "seed variance"],
  ["24", "C_p16", "patch size 16", "E6b: blocky artefacts"],
  ["25-26", "B_win8, B_win16", "window 8 / 16 (global)", "E5b: does more context help?"],
  ["post", "A_ens", "mean of A seeds 42/43/44", "E7c: ensemble beats single model; σ rises OOD"],
], [7, 20, 37, 36]);
H2("8.1 Why each experiment matters");
UL(["**E1 learning curves** - the central DL lesson: inductive bias versus data hunger. Nested subsets (the 25 % set is contained in the 50 % set) keep the comparison clean.",
  "**E2 domain shift** - every model is tested on four non-Asian scenes without retraining. Which architecture generalises best?",
  "**Leakage check** - demonstrates numerically why the scene split is necessary.",
  "**E3** - why U-Nets work (skips), what the loss optimises (L1 vs L2), how far the model looks (receptive field).",
  "**E4** - the perception-distortion trade-off (Blau & Michaeli, 2018): realism and pixel accuracy cannot both be maximised.",
  "**E5 / E6** - which design elements of attention models matter: window shifting, window size, positional information, patch size.",
  "**E7** - can a model know when it is wrong? Aleatoric (variance head) vs epistemic (MC dropout, ensemble) uncertainty; calibration and behaviour out of domain.",
  "**Seeds** - the noise floor that makes all other comparisons interpretable."]);

// =====================================================================================
H1("9. Infrastructure: how the 12-hour Kaggle run works");
OLn(["**Setup:** install rasterio, torchmetrics, torch-fidelity; clone the repository; detect both GPUs.",
  "**Index, split, cache** (≈ 11 minutes): find 122 218 triplets, split by scene, write the 17 GB cache.",
  "**Identity baseline** evaluated; Inception weights for FID downloaded once.",
  "**GPU benchmark:** time 20 training steps of each architecture (U-Net 79 ms, Swin 91 ms, GAN 103 ms, ViT 29 ms, tiny CNN 13 ms per step on a T4).",
  "**Planner:** predict each job's duration (steps × time per step × 1.25 + evaluation + start-up) and simulate scheduling on 2 GPUs; choose the largest common budget from {15 000, 12 000, 10 000, …} that fits 11 h minus a 25-minute safety margin. Result: 15 000 steps, predicted 5.6 h.",
  "**Scheduler:** run jobs in priority order, one per GPU in its own process. Before each start, skip a job if it cannot finish before the deadline; a running job stops early (keeping its best checkpoint) if it overruns. The real speed of finished jobs updates the predictions. Each job writes its results to `jobs/<run>/` when it finishes, so a crash never loses earlier work.",
  "**Post-hoc:** 3-seed ensemble, effective receptive fields.",
  "**Export:** tables, figures, gallery samples, config; zip everything."]);
H2("9.1 Output files");
T(["File", "Content"], [
  ["results/metrics_main.csv", "main comparison: mean ± std over seeds, per test set"],
  ["results/metrics_summary.csv", "every run × test set: PSNR, SSIM, SAM, MAE, RMSE, FID, parameters, time"],
  ["results/experiments.csv", "every experiment vs its reference, with deltas and the plan's prediction"],
  ["results/learning_curves.csv", "E1"],
  ["results/leakage_check.csv", "random vs scene split"],
  ["results/metrics_by_cloud_bin.csv / _by_country.csv", "breakdowns"],
  ["results/uncertainty_summary.csv, sparsification.csv", "E7"],
  ["results/erf_stats.csv, erf.png", "E3c"],
  ["results/per_patch_metrics.csv", "every test patch, every run"],
  ["results/*.png", "comparison grid, training curves, bar charts, learning curves, uncertainty"],
  ["results/samples.npz", "8 gallery patches with all predictions (frontend)"],
  ["checkpoints/best_<run>.pt", "best weights of every run (architecture settings included)"],
  ["logs/", "validation history of every run, job logs, full pipeline log"],
  ["config.json", "every setting, job status, measured speeds"],
], [42, 58]);
H2("9.2 How the code was verified before using GPU time");
UL(["**34 unit tests** (`tests/`): model shapes and gradients, the positional-argument bug, preprocessing of dB radar, SSIM and PSNR against scikit-image, tiling, splitting without scene overlap, nested subsets, ensembles, MC dropout, receptive field, and 8 expert-system scenarios.",
  "**Synthetic SEN12MS-CR** (`tests/make_synthetic_data.py`): fake GeoTIFFs with the real naming and real scene IDs (one deliberately corrupt) - the full pipeline with all 26 jobs, two parallel slots, ensemble and export ran end-to-end on a laptop.",
  "**Overfit test:** the U-Net drove its loss from 0.22 to 0.002 on 16 samples, proving inputs and targets are aligned and the model can learn.",
  "**Kaggle quick mode** (`CLEARSAR_QUICK=1`, 300 steps): the same code on the real data, which revealed the checkpoint-device bug and the output-offset problem before the long run."]);

// =====================================================================================
H1("10. What was wrong in the original pipeline, and why it matters");
P("Understanding these bugs is a good way to explain why the final design looks as it does.");
T(["Problem", "Effect", "Fix"], [
  ["`ModelA_UNet(False)`: the boolean was taken as in_channels", "network built with 0 input channels → crash", "keyword arguments + validation of channel counts"],
  ["S1 (already dB) converted to dB again", "almost every radar pixel became the same value → models trained without radar, silently", "clip and rescale the dB values directly"],
  ["Bands [:3] / [:4]", "used B1-B3 (coastal, blue, green), not RGB", "all 13 bands in and out; RGB = B4, B3, B2 for display"],
  ["Triplets paired by sorting three file lists", "any difference between mirrors misaligns input and target", "pair by (season, scene, patch) from the filename"],
  ["Random patch split; \"OOD\" = random 5 %", "spatial leakage; no real domain-shift test", "split by scene; real out-of-continent set"],
  ["\"FID\" on 8×8-pooled pixels per 16-image batch", "not FID; rank-deficient covariance", "Inception FID on the whole test set (or empty)"],
  ["Every model trained as a GAN; ViT without positions; Model B not Swin", "protocol did not match the plan; ViT permutation-invariant", "L1 core runs; GAN as ablation; real Swin; positional embedding"],
  ["Scene finder replaced by a stub guessing countries from a hash", "wrong country/continent labels", "original reverse-geocoding script restored"],
  ["Reloaded checkpoint stayed on the CPU (found in quick test)", "every evaluation crashed", "move model to the requested device"],
  ["Outputs started at 0.5 reflectance (found in quick test)", "U-Nets spent the short run learning the offset", "initialise output bias to mean reflectance"],
], [32, 34, 34]);

// =====================================================================================
H1("11. The ClearSAR Advisor - explainable expert system");
P("An expert system captures expert knowledge as explicit rules and applies it to the facts of a case, explaining its reasoning. Here the *case* is one image patch and its reconstructions. The full design is in `docs/expert_system.md`; this chapter summarises everything needed to explain it.");
H2("11.1 Architecture");
T(["Component", "File", "Role"], [
  ["Knowledge base", "src/expert/knowledge_base.yaml", "thresholds, 35 IF-THEN rules with certainty factors, labels, tie-breaking"],
  ["Fact extraction", "src/expert/facts.py", "measures the patch, the model outputs and the experiment results"],
  ["Explainable AI", "src/expert/xai.py", "radar-reliance by occlusion, band saliency by input × gradient"],
  ["Inference engine", "src/expert/engine.py", "fuzzification, stratified forward chaining, MYCIN certainty factors, trace"],
  ["Advisor", "src/expert/advisor.py", "runs the models, collects facts, calls the engine, returns decision + explanation"],
], [22, 33, 45]);
P("The knowledge (YAML) is separate from the reasoning code: rules can be changed without touching Python - the defining property of an expert system.");
H2("11.2 Facts");
T(["Fact", "How it is measured", "Levels (thresholds)"], [
  ["cloud_cover", "bright (mean B2-B4 > 0.20, calibrated on the test gallery) and white (colour spread / mean < 0.25) pixels in the cloudy input", "low < 0.2 ≤ moderate < 0.6 ≤ heavy"],
  ["thick_cloud", "cloudy and mean B2-B4 > 0.30", "no < 0.3 ≤ yes"],
  ["cirrus_fraction", "B10 > 0.03 (calibrated)", "no < 0.2 ≤ yes"],
  ["sar_gaps", "VV at the no-signal floor", "no < 0.1 ≤ yes"],
  ["water_fraction", "VV < −18 dB", "none < 0.05 ≤ some < 0.3 ≤ large"],
  ["model_disagreement", "mean |difference| between the three architectures' predictions", "low < 0.02 ≤ moderate < 0.04 ≤ high"],
  ["clear_pixel_error", "error on cloud-free pixels (where the input is the truth)", "good < 0.03 ≤ fair < 0.06 ≤ poor"],
  ["uncertainty", "median Laplace scale of the variance-head model under the clouds", "low < 0.02 ≤ moderate < 0.05 ≤ high"],
  ["sar_reliance", "occlusion: Δ(remove radar) / [Δ(remove radar) + Δ(remove optical)]", "low < 0.3 ≤ balanced < 0.6 ≤ high"],
  ["best_model_for_cloud_level", "highest PSNR in the experiments for this cloud bin", "a run name"],
  ["ood_drop", "PSNR lost on other continents in the experiments", "number used in explanations"],
  ["use_case, region", "chosen by the user; region auto-detected", "5 uses; in / out of region / unknown"],
], [22, 48, 30]);
P("**Fuzzification:** each number becomes a symbolic level. A value within 10 % of a threshold is *borderline* and gets certainty 0.7 instead of 1.0.");
H2("11.3 Rules (35, in five groups)");
T(["Group", "Concludes", "Example rule"], [
  ["observations (W1-W4)", "warnings: domain shift, radar gaps, cirrus, water", "W1: IF region = out_of_region THEN warning_domain_shift (CF 0.9)"],
  ["trust (T1-T12)", "trust = high / medium / low", "T4: IF cloud heavy AND radar reliance high AND disagreement ≤ moderate THEN trust = medium (0.5)"],
  ["model (M1-M3)", "recommended model; avoid-GAN warning", "M1: recommended = best model for this cloud level in the experiments (0.8)"],
  ["suitability (U1-U9)", "suitable / caution / unsuitable for the chosen use", "U4: IF use = vegetation monitoring AND cloud heavy THEN unsuitable (0.7)"],
  ["action (A1-A7)", "accept / accept with mask / radar-only / wait for a new image", "A4: IF use = flood mapping THEN radar-only products (0.8)"],
], [22, 30, 48]);
H2("11.4 Inference with certainty factors");
UL(["**Stratified forward chaining:** groups run in order observations → trust → model → suitability → action; each rule is tried once.",
  "**Firing:** a rule fires if every premise holds with certainty ≥ 0.2.",
  "**Propagation:** CF(conclusion) = CF(rule) × min(CF of premises).",
  "**Combination (MYCIN):** two rules supporting the same conclusion combine as CF₁ + CF₂ · (1 − CF₁).",
  "**Conflict resolution:** after each group, single-valued conclusions keep their highest-CF value; more specific rules carry higher CFs; exact ties go to the more cautious value (low trust before high)."]);
F("example:  T6 (0.7) and T7 (0.8) both say trust = low  →  0.7 + 0.8 · (1 − 0.7) = 0.94");
H2("11.5 Explanation facility");
UL(["**How?** - every fired rule with its premises, CF contribution and a sentence filled with the patch's real numbers.",
  "**What else?** - all competing values with their CF.",
  "**Why not?** - every rule that did not fire and the first premise that failed.",
  "**What did the network look at?** - radar-reliance map, band saliency chart, uncertainty and disagreement maps.",
  "**What if?** - a simulator re-runs the engine live as facts are changed with sliders.",
  "**Report** - downloadable Markdown and a JSON trace."]);
H2("11.6 Explainable-AI methods");
P("**Occlusion:** the radar channels (or the 13 optical channels) are replaced by their spatial mean - removing their spatial information - and the change in the prediction is measured pixel by pixel. Their ratio is the radar reliance. Under thick cloud a sound model should rely on radar; rule T12 lowers trust if it does not.");
P("**Input × gradient saliency:** the gradient of the predicted RGB (averaged over cloudy pixels) with respect to each input value, multiplied by the input, summed per channel and normalised - the share of influence of each of the 15 input bands.");
H2("11.7 How to evaluate the expert system");
UL(["Scenario tests (implemented): clear scene → accept; cloudy disagreeing scene → reject; flood mapping → radar-only; ties resolve cautiously; knowledge-base consistency.",
  "Agreement with ground truth: on test patches the true PSNR is known - patches rated *high trust* should have clearly higher PSNR than *low trust* ones. Pilot on the 8 gallery patches (§13.9): high 26.9 dB > medium 25.1 dB > low 24.9 dB.",
  "Sensitivity analysis with the what-if simulator; review of explanations by a remote-sensing user."]);

// =====================================================================================
H1("12. The frontend");
P("A Streamlit web app (`streamlit run app.py`) reads the folder extracted from the Kaggle output zip (`outputs/`).");
T(["Page", "Purpose"], [
  ["Home", "drag-to-compare slider (cloudy input vs reconstruction), headline numbers, the three architectures, map of scenes per split"],
  ["Model comparison", "one chart for PSNR / SSIM / SAM / MAE / FID with seed error bars, leaderboard, cloud-cover and country breakdowns"],
  ["Experiments", "one card per experiment: question, prediction, measured change, automatic verdict, write-up scaffold; E1 chart; leakage; receptive fields; uncertainty"],
  ["Explore patches", "any model on any gallery patch vs the cloudy input or the truth; error and uncertainty maps; per-patch scores"],
  ["Advisor", "the expert system: decision cards with certainty, warnings, visual explanation, how / why-not reasoning, facts, report download; works on gallery patches or uploaded GeoTIFFs"],
  ["Knowledge base", "architecture diagram, what-if simulator, all rules and thresholds, CF algebra"],
  ["How it works", "pipeline summary and limitations"],
], [22, 78]);

// =====================================================================================
H1("13. Results");
H2("13.1 The run");
P("The complete pipeline ran on Kaggle (2 × NVIDIA T4) on 21-22 October 2026. All **26 trainings finished with the full 15 000 steps** each, plus the 3-seed ensemble and the receptive-field analysis - no job failed, was skipped or stopped early. Total time **6.11 hours** (planner prediction 5.6 h of training plus ~12 minutes of indexing and caching). One training took 18-31 minutes; peak GPU memory was 5.5 GB. The best validation checkpoint was reached at step 14 000 for the U-Net, 6 000 for Swin and 2 000 for the ViT: the ViT stops improving on unseen scenes early (it overfits), the U-Net keeps improving.");
H2("13.2 Main comparison");
P("Mean over all 600 test patches per set, all 13 bands. ± = spread over training seeds (U-Net 3 seeds, Swin and ViT 2).");
T(["Model", "PSNR (dB)", "SSIM", "SAM (°)", "MAE", "FID"], [
  ["**Asia (in-domain test)**", "", "", "", "", ""],
  ["Identity (cloudy input)", "21.18", "0.740", "10.45", "0.0908", "128.4"],
  ["Tiny CNN (baseline)", "27.01", "0.844", "9.45", "0.0379", "224.6"],
  ["Model A - U-Net", "26.33 ± 0.06", "0.847", "9.43", "0.0409", "179.8"],
  ["Model B - Swin U-Net", "26.41 ± 0.20", "0.851", "9.55", "0.0409", "197.8"],
  ["Model C - Pure ViT", "26.90 ± 0.36", "0.844", "8.91", "0.0376", "256.0"],
  ["A + PatchGAN (λ = 100)", "25.00", "0.816", "10.04", "0.0455", "214.2"],
  ["A + variance head", "26.59", "0.847", "9.52", "0.0392", "190.7"],
  ["A + MC dropout", "26.41", "0.853", "9.30", "0.0402", "177.7"],
  ["A - 3-seed ensemble", "26.56", "0.857", "9.08", "0.0398", "176.5"],
  ["**Other continents (out-of-domain test)**", "", "", "", "", ""],
  ["Identity (cloudy input)", "20.95", "0.719", "12.93", "0.0916", "164.3"],
  ["Tiny CNN (baseline)", "28.01", "0.866", "8.68", "0.0303", "232.6"],
  ["Model A - U-Net", "27.82 ± 0.30", "0.874", "8.61", "0.0305", "176.1"],
  ["Model B - Swin U-Net", "27.68 ± 0.39", "0.871", "8.89", "0.0314", "189.5"],
  ["Model C - Pure ViT", "27.19 ± 0.41", "0.855", "9.97", "0.0341", "239.0"],
  ["A + PatchGAN (λ = 100)", "25.31", "0.810", "8.84", "0.0408", "208.6"],
  ["A + variance head", "28.20", "0.873", "8.11", "0.0296", "176.1"],
  ["A + MC dropout", "27.95", "0.876", "8.20", "0.0300", "169.7"],
  ["A - 3-seed ensemble", "28.26", "0.882", "8.31", "0.0290", "181.8"],
], [34, 18, 11, 11, 12, 14]);
UL(["**Every model removes most of the cloud damage:** the L1-trained models gain +5 to +7 dB PSNR over the cloudy input (the GAN +4 dB), SSIM rises from 0.72-0.74 to 0.84-0.88, and MAE is more than halved.",
  "**No architecture is clearly best overall.** In Asia the ViT leads by about 0.5 dB, on other continents the U-Net does - both differences are comparable to the seed spread (up to 0.4 dB). The U-Net-based ensemble has the best SSIM on both sets.",
  "**The tiny CNN is as good on PSNR.** This was flagged before interpretation and investigated (§13.4): it is a genuine result, not a bug.",
  "**FID favours the cloudy input (128).** All reconstructions are smoother than real images (L1 training), which an Inception network notices; pixel-accurate is not the same as realistic."]);
H2("13.3 Performance by cloud cover");
P("PSNR in dB by the cloud cover of the patch (seed-42 runs).");
T(["Cloud cover", "Patches", "Identity", "Tiny CNN", "U-Net", "Swin", "ViT", "GAN"], [
  ["**Asia**", "", "", "", "", "", "", ""],
  ["0-20 %", "304", "26.24", "28.65", "28.25", "27.70", "27.95", "26.07"],
  ["20-40 %", "77", "19.55", "27.09", "27.22", "26.71", "27.20", "24.75"],
  ["40-60 %", "56", "17.35", "24.53", "25.09", "24.78", "25.37", "23.76"],
  ["60-80 %", "64", "14.97", "25.02", "24.23", "24.74", "**27.29**", "24.17"],
  ["80-100 %", "99", "13.11", "24.60", "21.94", "23.01", "**26.24**", "23.18"],
  ["**Other continents**", "", "", "", "", "", "", ""],
  ["0-20 %", "253", "26.37", "28.85", "28.90", "29.11", "27.74", "26.29"],
  ["20-40 %", "92", "20.09", "26.90", "26.23", "27.07", "26.17", "24.84"],
  ["40-60 %", "92", "17.55", "26.82", "26.86", "27.10", "25.92", "24.64"],
  ["60-80 %", "65", "16.50", "27.45", "27.31", "27.68", "26.26", "24.16"],
  ["80-100 %", "98", "13.90", "28.35", "27.46", "27.50", "26.05", "24.59"],
], [18, 11, 11, 12, 11, 11, 11, 11]);
P("**Key finding:** under thick cloud in Asia (60-100 %) the **ViT is better by 2-4 dB** than the U-Net. With little optical signal left the model must use context from far away, and the ViT's global attention can (its effective receptive field is 56 px versus 21 px for the U-Net, §13.7). On the out-of-continent scenes the same advantage does not appear, which suggests the ViT learned scene-specific context that does not transfer.");
H2("13.4 Why does a 3-layer CNN match the large models?");
P("The project plan says a big model that cannot beat a tiny one signals a problem, so this was investigated before interpreting anything:");
UL(["**It is not a pipeline bug.** Results are stable across seeds (U-Net ± 0.06 dB), every model beats the identity baseline by 3.8-7.3 dB, and on the **validation scenes the large models do win** (best validation PSNR: U-Net 28.69, Swin 28.44, ViT 27.98, tiny CNN 27.88 dB).",
  "**The test ranking is decided by two scenes** (table below). With only 7 test scenes, the scene-to-scene differences (> 7 dB) are far larger than the differences between models (< 1 dB).",
  "**PSNR rewards cautious answers.** Under thick cloud the true colours are unknowable; a model that predicts a smooth, average-looking surface is penalised less than one that predicts confident but wrong texture. The tiny CNN (receptive field ~3 px) can only produce such local, conservative mappings.",
  "**Conclusion for the write-up:** with 3 000 patches from 23 training scenes, architectural sophistication does not translate into higher PSNR on new scenes; the limiting factor is the diversity of scenes, not the model."]);
T(["Test scene (Asia)", "Mean cloud", "Identity", "Tiny CNN", "U-Net", "Swin", "ViT"], [
  ["China, spring 119", "13 %", "24.88", "28.22", "29.08", "28.61", "28.37"],
  ["China, spring 124", "23 %", "21.67", "24.74", "25.66", "24.47", "25.90"],
  ["**India, summer 143**", "22 %", "22.58", "**29.84**", "25.45", "25.15", "28.18"],
  ["Iran, summer 127", "34 %", "21.62", "21.45", "23.51", "21.84", "21.14"],
  ["Lebanon, fall 112", "41 %", "16.39", "27.95", "28.49", "28.69", "28.33"],
  ["**Turkmenistan, fall 77**", "85 %", "14.79", "25.56", "21.01", "23.21", "**28.33**"],
  ["Taiwan, fall 109", "17 %", "26.28", "31.26", "31.21", "31.45", "30.56"],
], [28, 12, 12, 12, 12, 12, 12]);
P("The India-only slice (one scene): the tiny CNN scores 29.8 dB, the ViT 28.2 dB, the U-Net 25.5 dB.");
H2("13.5 Every experiment: prediction → result → verdict");
P("Δ = experiment − reference. Verdict uses the noise floor of §7.8 (max(0.3 dB, 2 × seed std) for PSNR; 0.005 or 2 × seed std for SSIM).");
T(["Exp.", "Run (vs reference)", "Prediction", "Δ PSNR Asia / other", "Δ SSIM", "Δ FID", "Verdict and interpretation"], [
  ["Base", "tiny CNN (vs U-Net)", "clearly worse", "+0.65 / +0.24", "−0.007", "+48", "**Contradicted** - see §13.4"],
  ["E1", "U-Net 25 % / 50 %", "worse with less data", "−0.17 / −0.17; −0.03 / +0.31", "−0.005", "+5", "Within noise - data quantity is not the bottleneck"],
  ["E1", "Swin 25 % / 50 %", "worse", "+0.22 / −0.32; +0.02 / +0.04", "+0.004", "−2", "Within noise"],
  ["E1", "ViT 25 % / 50 %", "worst drop", "−0.46 / +0.44; −0.37 / +1.10", "−0.012", "−28", "Largest in-domain drop, as predicted, but inside the ViT's large seed noise (floor 0.72 dB)"],
  ["Leak", "random patch split", "inflated", "**+5.12**", "+0.032", "−43", "**Supported, strongly** - the clearest result of the project"],
  ["E3a", "no skip connections", "blurrier", "−0.15 / −0.38", "**−0.010**", "**+106**", "**Supported** - structure and realism collapse"],
  ["E3b", "L2 loss", "blurrier", "+0.01 / +0.02", "+0.0005", "+14", "Within noise - L1 and L2 behave alike here"],
  ["E4", "GAN λ = 100", "lower PSNR, better FID", "**−1.35 / −2.46**", "−0.036", "+38", "PSNR part **supported**; FID did **not** improve"],
  ["E4", "GAN λ = 10", "even lower PSNR", "−0.79 / −1.68", "−0.067", "+55", "PSNR drop supported; FID worse (see note)"],
  ["E5a", "Swin without shift", "lower PSNR", "+0.38 / −0.14", "+0.001", "−7", "Within noise (floor 0.39 dB)"],
  ["E5b", "Swin window 8 / 16", "open question", "+0.00 / −0.14; +0.29 / −0.39", "+0.007", "−4 to −11", "No consistent benefit from more context in the bottleneck"],
  ["E6a", "ViT without positions", "lower PSNR", "−0.16 / +1.11", "−0.003", "−27", "Within noise - surprisingly, position information barely matters"],
  ["E6b", "ViT patch 16", "blocky, lower SSIM", "−0.28 / +0.95", "−0.007", "+13", "Within noise (floor 0.009)"],
  ["E7a", "U-Net + variance head", "no accuracy cost", "+0.23 / +0.44", "−0.005", "+14", "No cost"],
  ["E7b", "U-Net + MC dropout", "no accuracy cost", "+0.06 / +0.18", "+0.002", "+1", "No cost"],
  ["E7c", "3-seed ensemble", "better than one model", "+0.21 / +0.49", "+0.006", "0", "Small gain, within the PSNR noise floor; best SSIM of all models"],
], [6, 15, 13, 17, 8, 7, 34]);
NOTE("Why the GAN did not improve FID.", "The checkpoint is chosen by the lowest *validation L1*. For the GAN runs this happened very early (step 4 000 for λ = 100, step 1 000 for λ = 10), before adversarial training had shaped the texture; later, sharper checkpoints had higher L1 and were discarded. Selecting GAN checkpoints by L1 favours the least adversarial model - a genuine methodological lesson (selection criteria must match the goal).");
H2("13.6 E1 learning curves (PSNR, dB)");
T(["Model", "Asia 25 %", "Asia 50 %", "Asia 100 %", "Other 25 %", "Other 50 %", "Other 100 %"], [
  ["U-Net", "26.19", "26.33", "26.36", "27.60", "28.07", "27.77"],
  ["Swin U-Net", "26.43", "26.23", "26.21", "27.75", "28.12", "28.07"],
  ["Pure ViT", "26.80", "26.89", "27.26", "27.22", "27.88", "26.78"],
], [22, 13, 13, 13, 13, 13, 13]);
P("Curves are nearly flat: quadrupling the patches from 750 to 3 000 changes PSNR by less than 0.5 dB. Because all subsets come from the same 23 scenes, extra patches add little new information. The ViT gains most in-domain (+0.46 dB), in line with its weak inductive bias, but not reliably.");
H2("13.7 Effective receptive field (E3c)");
T(["Model", "50 % of influence within", "90 % within"], [
  ["Tiny CNN", "1.4 px", "3.2 px"], ["U-Net", "21.1 px", "37.7 px"], ["U-Net without skips", "19.1 px", "31.2 px"],
  ["Swin U-Net", "2.2 px", "33.8 px"], ["Swin without shift", "6.3 px", "29.7 px"], ["Pure ViT", "**56.0 px**", "**73.0 px**"],
  ["ViT without positions", "46.5 px", "69.6 px"],
], [40, 30, 30]);
P("The ViT really looks far: half of its output is influenced by pixels more than 56 px (560 m) away - which explains why it is best under thick cloud. Swin is mostly very local (r50 = 2.2 px, the skip connections dominate) with a long tail through its attention bottleneck.");
H2("13.8 Uncertainty (E7)");
T(["Method", "coverage95 Asia / other (ideal 0.95)", "median σ", "Spearman(err, σ)", "AUSE (lower better)"], [
  ["Variance head (aleatoric)", "**0.83 / 0.89**", "0.023 / 0.021", "**0.56** / 0.44", "**0.0080** / 0.0078"],
  ["MC dropout (epistemic)", "0.08 / 0.09", "0.002 / 0.002", "0.52 / 0.42", "0.0098 / 0.0082"],
  ["Deep ensemble (epistemic)", "0.32 / 0.37", "0.008 / 0.008", "0.46 / 0.40", "0.0111 / 0.0083"],
], [26, 26, 14, 16, 18]);
UL(["The **variance head** is by far the best calibrated (83-89 % of true values inside its 95 % interval) and ranks errors best (lowest AUSE).",
  "**MC dropout and the ensemble are severely over-confident** (8 % and 32 % coverage): three seeds or dropout at p = 0.1 produce far too little spread - a known weakness.",
  "**No method's uncertainty rises on the other continents** - but the out-of-continent scenes also had *lower* errors (MAE 0.030 vs 0.039), so they were easier, not harder: this test set could not reveal domain shift."]);
H2("13.9 Expert-system pilot evaluation");
P("The advisor was run on the 8 gallery patches (intended use: land cover) after calibrating two thresholds on them (brightness 0.17 → 0.20 halved the cloud-estimate error from 0.081 to 0.045; cirrus B10 0.01 → 0.03 because clear-sky B10 reaches 0.025 in some scenes).");
T(["Patch", "True / estimated cloud", "Trust (CF)", "Model chosen", "True PSNR", "Action"], [
  ["China (Asia)", "0.00 / 0.02", "high (0.80)", "U-Net", "29.7", "accept"],
  ["India (Asia)", "0.10 / 0.11", "high (0.28)", "U-Net", "22.9", "accept"],
  ["Lebanon (Asia)", "0.50 / 0.46", "medium (0.60)", "ViT", "25.1", "accept with mask"],
  ["Turkmenistan (Asia)", "0.97 / 1.00", "low (0.94)", "ViT", "28.4", "wait for new image"],
  ["Central African Rep.", "0.02 / 0.00", "high (0.80)", "U-Net", "28.0", "accept"],
  ["France", "0.20 / 0.06", "low (0.88)", "U-Net", "24.8", "wait for new image"],
  ["USA", "0.54 / 0.42", "low (0.87)", "ViT", "21.9", "wait for new image"],
  ["USA", "1.00 / 1.00", "low (0.88)", "ViT", "24.5", "wait for new image"],
], [22, 18, 15, 13, 12, 20]);
P("Mean true PSNR by trust level: **high 26.9 dB (n = 3) > medium 25.1 dB (n = 1) > low 24.9 dB (n = 4)** - the trust ranking agrees with the real reconstruction quality. It is conservative: the Turkmenistan patch gets low trust (97 % cloud, models disagree) although the ViT reconstructs it well. With 8 patches this is a pilot, not a proof; a full evaluation needs the advisor run on all 1 200 test patches.");
H2("13.10 Conclusions");
OLn(["SAR-optical fusion works: the L1-trained models recover 5-7 dB PSNR over the cloudy image (the GAN about 4 dB) and roughly halve the error.",
  "Under an identical protocol, **no architecture is clearly better on overall PSNR**; differences are of the size of seed noise and far smaller than scene-to-scene variation. A 3-layer CNN is competitive on pixel metrics.",
  "**The architectures differ where it matters most - under thick cloud:** the ViT's global receptive field gives it a 2-4 dB lead in Asia at 60-100 % cloud.",
  "**Splitting by scene is essential:** a random patch split inflates PSNR by 5.1 dB.",
  "Adding data from the same scenes barely helps; more *scenes* would.",
  "The GAN loses 0.8-1.4 dB PSNR as expected but did not gain realism under L1-based checkpoint selection.",
  "Inside Swin and the ViT, shift, window size, positional embedding and patch size made no measurable difference at this data scale; skip connections clearly matter for structure (SSIM, FID).",
  "Of three uncertainty methods only the learned variance head is reasonably calibrated.",
  "The expert system's trust levels agree with true quality in a small pilot, and its model choice (ViT under thick cloud) follows the experimental evidence."]);

H2("13.11 Limitations and deviations from the original plan");
T(["Topic", "Plan", "What was done and why"], [
  ["Training data", "15-20 Asian scenes, ~10-15k patches", "all 37 Asian scenes, but a capped 3 000-patch training set - keeps the cache (~17 GB) and the 26 trainings inside one 12-hour Kaggle session"],
  ["U-Net width", "4 down-blocks, base width 32 (~8 M)", "3 down-blocks, base width 64 (7.71 M) - similar size; keeps the bottleneck at 16 × 16 so the Swin bottleneck sees 16 × 16 tokens as planned"],
  ["Learning rate", "LR range test per model", "one fixed LR (3e-4, GAN 2e-4) for all models - identical protocol; a range test per model was cut for time"],
  ["Cloud masks", "s2cloudless", "brightness-based proxy (fewer dependencies); adequate for 20 %-wide bins but imperfect for bright deserts/snow"],
  ["Attention maps (E5c)", "visualise Swin attention", "not implemented (first in the plan's cut order); receptive fields (E3c) implemented instead for all models"],
  ["Seeds", "2 per main model", "2 for B and C, 3 for A (the third seed also forms the ensemble)"],
  ["Comparability", "-", "scores are not comparable to published SEN12MS-CR numbers, which use a different, global test split"],
], [18, 28, 54]);

// =====================================================================================
H1("14. Likely questions and answers");
const QA = [
  ["Why fuse SAR and optical instead of using SAR alone?", "SAR sees through clouds but not colour; the cloudy optical image still contains real colour wherever the cloud is thin or absent. Fusing both lets the model copy what is visible and infer only what is hidden."],
  ["Why not just use the previous cloud-free image?", "The ground changes (crops grow, floods, harvests). The target images in SEN12MS-CR are close in time precisely so that reconstruction, not time-series interpolation, is tested."],
  ["Why split by scene?", "Neighbouring patches overlap and look alike. A random split would let the model see almost the same image in training and testing, inflating scores. The leakage experiment measures this."],
  ["Why is the out-of-continent test important?", "It tests generalisation to landscapes never seen in training - the realistic situation when a model is deployed elsewhere."],
  ["Why L1 and not L2?", "With several plausible answers, L2 is minimised by their mean (blur); L1 by their median (sharper, robust). E3b tests it."],
  ["Why a sigmoid output?", "Reflectance lies in [0, 1]; the sigmoid guarantees valid values."],
  ["Why initialise the output bias?", "Otherwise every model starts predicting 0.5, five times brighter than typical surfaces, and convolutional models waste many steps learning that offset. Same rule for all models, so the comparison stays fair."],
  ["Why 128 × 128 crops and tiled evaluation?", "Cheaper and more varied training; the ViT's positional grid is fixed; tiling ensures every model sees the same input size at test time."],
  ["Why AdamW, warm-up and cosine decay?", "AdamW adapts per-parameter step sizes and applies weight decay correctly; warm-up prevents unstable early updates (important for attention); cosine decay settles into a good minimum."],
  ["What is mixed precision and why is it safe?", "Computation in float16 on tensor cores is about twice as fast; the GradScaler scales the loss so small gradients do not underflow and skips steps that overflow."],
  ["How was the best model chosen without touching the test set?", "Validation L1 on 400 patches from separate scenes every 1 000 steps; the test sets are evaluated once at the end."],
  ["What does PSNR measure and what does it miss?", "Pixel error on a log scale; it misses perceived realism and favours blur - hence SSIM, SAM and FID too."],
  ["What is SAM and why is it relevant here?", "The angle between predicted and true 13-band spectra per pixel; it checks the spectral shape needed for vegetation and land-cover analysis, independent of brightness."],
  ["Why is FID only compared relatively?", "With 600 images per set the 2 048-dimensional covariance is poorly estimated, so absolute values are biased; differences between models on the same set are still informative."],
  ["What is inductive bias?", "An assumption built into the architecture. Convolutions assume locality and translation equivariance; attention does not. Strong bias helps with little data."],
  ["How does Swin differ from a normal transformer?", "Attention only within small windows (linear cost), with shifted windows every other block so information still flows across the image; a relative position bias encodes distance."],
  ["Why does a ViT need positional embeddings?", "Self-attention is permutation-invariant; without positions the model cannot tell where a patch is. E6a tests it."],
  ["What does the GAN add?", "A learned realism loss: sharper, more natural texture (lower FID) but lower PSNR and possibly invented detail - the perception-distortion trade-off."],
  ["Aleatoric vs epistemic uncertainty?", "Aleatoric: noise or missing information in the data (thick cloud); cannot be reduced with more data. Epistemic: the model's ignorance (new continent); reducible with more data. Variance head estimates the first; MC dropout and ensembles the second."],
  ["How do you know uncertainty is meaningful?", "Calibration (coverage95 near 0.95), sparsification (removing uncertain pixels reduces error, AUSE small), correlation with error, and rising σ out of domain."],
  ["How large must a difference be to count?", "Larger than the seed noise: about 0.3 dB PSNR or twice the seed standard deviation."],
  ["What is an expert system and why build one here?", "A system that encodes expert knowledge as rules and reasons over facts with explanations. A reconstruction is an estimate; users need to know when to trust it and what to do - decisions that require domain knowledge and justification."],
  ["Why certainty factors and not probabilities?", "CFs (MYCIN) are simple to elicit from experts, combine incrementally and are easy to explain. A Bayesian network over the same facts would be a principled extension."],
  ["How does the expert system know which model is best?", "Experience facts: it reads the experiment results (best PSNR per cloud-cover bin), so retraining updates its knowledge automatically."],
  ["How does it explain decisions?", "Every fired rule with its numbers (how), alternatives with their CF, rules that did not fire with the failed premise (why not), visual XAI maps, and a what-if simulator."],
  ["What is occlusion-based explanation?", "Remove part of the input (here: radar or optical) and measure how much the output changes; large change = the model relied on it."],
  ["Why does a tiny CNN score as well as the large models?", "On validation scenes the large models do win; on the 7 test scenes two scenes decide the ranking and scene-to-scene differences (> 7 dB) dwarf model differences (< 1 dB). PSNR also rewards cautious, smooth predictions under thick cloud, which is all a 3-layer CNN can produce. With 23 training scenes, scene diversity - not architecture - limits generalisation."],
  ["Where do the architectures really differ?", "Under thick cloud (60-100 %) in Asia the ViT is 2-4 dB better, consistent with its effective receptive field of 56 px versus 21 px for the U-Net: with little optical signal, context from far away matters."],
  ["Why were the out-of-continent scores higher than the in-domain ones?", "The four foreign scenes happened to be easier (lower errors for every model, including the cloud-free part), so this small OOD set could not reveal a domain-shift penalty - an honest limitation of using 4 scenes."],
  ["Why did the GAN not improve FID?", "Its checkpoint was chosen by validation L1, which picked very early steps (1 000-4 000) before adversarial training shaped the texture. The selection criterion must match the goal."],
  ["What does the leakage experiment show?", "Training and testing on random patches of the same scenes raises PSNR by 5.1 dB - the size of the error a careless split would have introduced, larger than any architecture difference."],
  ["Why are MC dropout and the ensemble over-confident?", "Dropout p = 0.1 and three seeds produce too little spread between predictions; their 95 % intervals contain only 8 % and 32 % of true values. The learned variance head (83-89 %) is much better calibrated."],
  ["What are the main limitations?", "3 000-patch training subset; fixed GPU budget; heuristic cloud masks; FID on few images; results not comparable to published benchmarks; expert thresholds set by judgement."],
];
T(["Question", "Answer"], QA, [32, 68]);

// =====================================================================================
H1("15. Glossary");
const GL = [
  ["Ablation", "removing or changing one component to measure its contribution"],
  ["Activation function", "non-linearity between layers (ReLU, GELU, LeakyReLU, sigmoid)"],
  ["Attention / self-attention", "each position computes a weighted average of all others, weights from query-key similarity"],
  ["Backpropagation", "computing the gradient of the loss with respect to every weight via the chain rule"],
  ["Backscatter", "share of the radar signal returned to the sensor"],
  ["Batch normalisation", "normalises activations over the batch for faster, stabler training"],
  ["Certainty factor (CF)", "degree of belief in a conclusion between 0 and 1 (MYCIN)"],
  ["Checkpoint", "saved weights of a model at a point in training"],
  ["Convolution", "a small learned filter slid over the image; detects local patterns"],
  ["Data augmentation", "random label-preserving transformations (crops, flips, rotations) to increase variety"],
  ["Domain shift", "test data from a different distribution than training data"],
  ["Encoder-decoder", "network that compresses the input then expands it back to an image"],
  ["Epoch", "one pass over the training data"],
  ["Forward chaining", "data-driven inference: apply rules to facts to derive new facts"],
  ["Fuzzification", "mapping a number to symbolic levels (low / moderate / high)"],
  ["GAN", "generator and discriminator trained against each other"],
  ["Gradient clipping", "limiting the gradient's size to prevent unstable updates"],
  ["Inductive bias", "assumption built into a model's architecture"],
  ["Knowledge base", "the expert system's rules and thresholds, separate from the inference code"],
  ["Learning rate", "size of each optimisation step"],
  ["Overfitting", "fitting training data so closely that new data is predicted worse"],
  ["Patch (data)", "one 256 × 256 image tile; (ViT) one 8 × 8 piece turned into a token"],
  ["Positional embedding", "learned vector telling a transformer where each token is"],
  ["Reflectance", "fraction of incoming sunlight reflected by the surface, 0-1"],
  ["Seed", "number that fixes all random choices so a run is reproducible"],
  ["Skip connection", "direct link from encoder to decoder carrying fine detail"],
  ["Spatial autocorrelation", "nearby locations are similar"],
  ["Speckle", "grainy interference noise in radar images"],
  ["Token", "the unit a transformer processes (here an image patch)"],
  ["Validation set", "data used to choose checkpoints/settings, separate from the test set"],
];
T(["Term", "Meaning"], GL, [28, 72]);

// =====================================================================================
H1("16. References");
[
  "Ebel, P., Meraner, A., Schmitt, M., Zhu, X. X. (2021). Multisensor data fusion for cloud removal in global and all-season Sentinel-2 imagery. IEEE Transactions on Geoscience and Remote Sensing. (SEN12MS-CR)",
  "Schmitt, M., Hughes, L. H., Qiu, C., Zhu, X. X. (2019). SEN12MS - a curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery. ISPRS Annals.",
  "Meraner, A., Ebel, P., Zhu, X. X., Schmitt, M. (2020). Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion (DSen2-CR). ISPRS Journal of Photogrammetry and Remote Sensing.",
  "Ebel, P., Garnot, V. S. F., Schmitt, M., Wegner, J. D., Zhu, X. X. (2023). UnCRtainTS: uncertainty quantification for cloud removal in optical satellite time series. CVPR Workshops.",
  "Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. MICCAI.",
  "Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks (Pix2Pix). CVPR.",
  "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.",
  "Liu, Z. et al. (2021). Swin Transformer: hierarchical vision transformer using shifted windows. ICCV.",
  "Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: transformers for image recognition at scale. ICLR.",
  "Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing. (SSIM)",
  "Heusel, M. et al. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. NeurIPS. (FID)",
  "Kruse, F. A. et al. (1993). The spectral image processing system (SIPS). Remote Sensing of Environment. (SAM)",
  "Blau, Y., Michaeli, T. (2018). The perception-distortion tradeoff. CVPR.",
  "Kendall, A., Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? NeurIPS.",
  "Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian approximation. ICML.",
  "Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.",
  "Ilg, E. et al. (2018). Uncertainty estimates and multi-hypotheses networks for optical flow. ECCV. (sparsification, AUSE)",
  "Luo, W., Li, Y., Urtasun, R., Zemel, R. (2016). Understanding the effective receptive field in deep convolutional neural networks. NeurIPS.",
  "Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization (AdamW). ICLR; (2017) SGDR: stochastic gradient descent with warm restarts. ICLR.",
  "Micikevicius, P. et al. (2018). Mixed precision training. ICLR.",
  "Lee, J.-S. (1980). Digital image enhancement and noise filtering by use of local statistics. IEEE TPAMI. (Lee filter)",
  "Shortliffe, E. H., Buchanan, B. G. (1975). A model of inexact reasoning in medicine. Mathematical Biosciences. (MYCIN certainty factors)",
  "Karpathy, A. (2019). A recipe for training neural networks. Blog post.",
].forEach((r) => body.push(new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(r), spacing: { after: 80 } })));

// =====================================================================================
const numbering = {
  config: [
    { reference: "bul", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 560 } } } }] },
    ...["num", ...NUMREFS].map((ref) => ({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
  ],
};

const doc = new Document({
  creator: "ClearSAR", title: "ClearSAR - complete project guide",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22, color: DARK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, color: ACCENT, font: "Calibri" }, paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, color: DARK, font: "Calibri" }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: ACCENT, font: "Calibri" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering,
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "ClearSAR · complete project guide", size: 16, color: MUTED })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: MUTED })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
