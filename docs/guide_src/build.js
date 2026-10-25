// Builds docs/ClearSAR_Project_Guide.docx - version 2: the final model (Part A) + Phase 1 (Part B).
// Phase-1 chapters are taken from build.js (same folder) and renumbered.
const fs = require("fs");
const path = require("path");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, LevelFormat, TableOfContents, PageBreak, Footer, Header, PageNumber, BorderStyle, ImageRun,
} = D;

const OUT = process.argv[2];
const HERE = __dirname;
const CW = 9638;
const ACCENT = "0F766E", DARK = "1F2937", MUTED = "6B7280", LIGHT = "E6F4F1", HEAD = "D1FAE5";

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
const NUMREFS = [];
function OLn(items) { numCount += 1; const ref = "num" + numCount; NUMREFS.push(ref); OL(items, ref); }
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
function IMG(file, w, h, caption) {
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(HERE, file)), transformation: { width: w, height: h },
      altText: { title: caption, description: caption, name: file } })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(caption, { italics: true, size: 18, color: MUTED }) }));
}
const PART = (t, sub) => {
  body.push(new Paragraph({ pageBreakBefore: true, spacing: { before: 3000, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: t, bold: true, size: 48, color: ACCENT })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, children: runs(sub, { size: 24, color: MUTED }) }));
};

// ---------------------------------------------------------------- reuse Phase-1 chapters from build.js
const OLD = fs.readFileSync(path.join(HERE, "build_phase1.js"), "utf8");
function oldChapter(title, renumber, extra = []) {
  const start = OLD.indexOf(`H1("${title}`);
  if (start < 0) throw new Error("chapter not found: " + title);
  let end = OLD.indexOf("// =====", start);
  if (end < 0) end = OLD.length;
  let code = OLD.slice(start, end);
  const [from, to] = renumber;                       // e.g. ["5.", "12."]
  code = code.replace(new RegExp(`H([123])\\("${from.replace(".", "\\.")}`, "g"), `H$1("${to}`);
  const refs = [["§13.", "§15."], ["chapter 13", "chapter 15"], ["(§6.1)", "(§13.1)"], ["chapter 10", "chapter 17"],
    ["chapter 11", "chapter 18"], ["§7.8", "§9.8"], ...extra];
  for (const [a, b] of refs) code = code.split(a).join(b);
  eval(code);
}

// =====================================================================================
// TITLE
body.push(new Paragraph({ spacing: { before: 2600, after: 200 }, children: [new TextRun({ text: "ClearSAR", bold: true, size: 72, color: ACCENT })] }));
body.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Seeing the ground through the clouds", size: 36, color: DARK })] }));
body.push(new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "One SAR-optical deep-learning model for cloud removal, trained on the whole SEN12MS-CR dataset - and the ClearSAR Advisor expert system", size: 26, color: MUTED })] }));
body.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Complete project guide · version 2 (final model)", bold: true, size: 26 })] }));
body.push(new Paragraph({ spacing: { after: 120 }, children: runs("Every decision, model, hyper-parameter, metric and formula in the project - what was done, how, and why.", { color: MUTED, size: 22 }) }));
body.push(new Paragraph({ spacing: { after: 120 }, children: runs("Code: github.com/karthikeya-bhamidipati/clearsar · October 2026", { color: MUTED, size: 22 }) }));
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 32, color: ACCENT })], spacing: { after: 200 } }));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(new Paragraph({ children: runs("If the table of contents is empty, right-click it in Word and choose *Update Field*.", { color: MUTED, size: 18 }) }));

// =====================================================================================
H1("1. The project in one page");
P("**The problem.** Optical satellites such as Sentinel-2 photograph the Earth in 13 spectral bands, but clouds hide the ground in a large share of all images - in the tropics and during monsoons, for weeks at a time. A cloudy image is useless for crop monitoring, mapping or disaster response.");
P("**The idea.** Radar satellites such as Sentinel-1 send their own microwave signal and measure the echo. Microwaves pass through clouds, so radar sees the ground in any weather - but it measures surface roughness and moisture, not colour. ClearSAR trains a deep neural network that takes the radar image **and** the cloudy optical image (15 channels) and predicts the cloud-free optical image (13 channels).");
P("**The final model.** One network, **ClearSAR-Net**: a U-Net with two global self-attention blocks in its bottleneck and a learned **cloud gate** that decides, per pixel, whether to keep (and colour-correct) what the satellite saw or to rebuild it from the radar. It is trained in two parts on Kaggle's 2 × T4 GPUs in one 11-hour session on the **whole SEN12MS-CR dataset** - 102 871 training patches from 147 scenes on every continent and in all four seasons:");
UL(["**Part A - reconstruction** (55 % of the time): L1 + SSIM loss. Accurate but, like every pixel-loss model, slightly blurry. Saved as *before GAN*.",
  "**Part B - PatchGAN refinement** (45 %): a discriminator judges 70 × 70 windows - 'is this a real clear image for this input?' - and its gradient, held at 20 % of the pixel loss's, pushes the generator towards real texture."]);
P("**How we got here.** Phase 1 (Part B of this guide) compared 26 trainings - U-Net, Swin U-Net, Vision Transformer, ablations, a GAN and three uncertainty methods - on 3 000 Asian patches. It found no clearly best architecture, a 2-4 dB advantage for global context under thick cloud, almost flat learning curves (scene diversity, not architecture, was the limit) and a GAN ruined by its checkpoint selection. Each of these findings became a design decision of the final model (chapter 4).");
P("**The main results** (chapter 10). On 5 400 patches from 18 held-out scenes on every continent, the final model reaches **28.59 dB PSNR, SSIM 0.879** - +8.3 dB over the cloudy input (20.29 dB) and **+1.9 dB over the Phase 1 U-Net** (better in 17 of 18 scenes, 95 % CI +1.1 to +2.7 dB). The GAN part raised sharpness from 0.78 to 0.87 at a PSNR cost (−0.09 dB) that lies within noise. The cloud gate learned where the clouds are without ever being told (Spearman −0.84 with cloud cover). India: 29.45 dB (Phase 1: 25.50 dB).");
P("**The expert system.** A reconstruction is an estimate. The **ClearSAR Advisor** is a rule-based expert system that inspects each patch and the network's output (cloud cover, radar reliance, uncertainty, agreement with independent models) together with what the test showed, and decides how far a reconstruction can be trusted, which model to use (the GAN-refined one for visual use, the before-GAN one for measurements), whether it fits the user's purpose and what to do next - explaining every step.");
P("**The tools.** PyTorch (DistributedDataParallel, mixed precision) on Kaggle, a Streamlit web frontend, and a YAML knowledge base with a certainty-factor inference engine.");
NOTE("How to use this guide.", "Chapters 2-3 explain the satellites and the dataset. **Part A (chapters 4-11) is the final model**: why it is designed as it is, the data pipeline, the architecture, the GAN, training with every hyper-parameter, metrics, results and the engineering story. **Part B (chapters 12-17) is Phase 1**, the architecture study that led to it. Chapters 18-19 cover the expert system and the frontend; chapter 20 is a bank of likely examiner questions, 21 a glossary.");

// =====================================================================================
oldChapter("2. Background", ["2.", "2."]);
oldChapter("3. The dataset", ["3.", "3."], [
  ["NOTE(\"Why Asia?\", \"The project plan focuses on Asia (including India) to study a coherent but diverse region - monsoon tropics, deserts, temperate farmland - and to keep the training set manageable. Training on Asia and testing on other continents then gives a natural domain-shift experiment.\");",
   "NOTE(\"Which part of the data is used where?\", \"Phase 1 used a 3 000-patch subset of Asian scenes (Part B). The final model uses every complete triplet of every scene on every continent (chapter 5); India remains reported separately.\");"]]);

// =====================================================================================
PART("Part A", "The final model: ClearSAR-Net + PatchGAN on the whole dataset");

H1("4. From Phase 1 to one model");
P("After Phase 1 the reconstructions were measurably better than the cloudy input but did not *look* close to the real image. Instead of comparing more architectures, the project was re-planned around **one model, built as well as possible**, with every design choice justified by a Phase 1 finding or a measurement. (An intermediate staged plan, 'plan v4', is kept in `docs/project_plan_v4.md`; it was simplified to this single run because of time.)");
T(["Phase 1 finding / measurement", "Decision for the final model"], [
  ["Learning curves were flat (750 → 3 000 patches changed PSNR by < 0.5 dB); all subsets came from the same 23 scenes. 3 000 patches are 2.5 % of SEN12MS-CR.", "**Train on everything**: all 147 training scenes, all 102 871 patches, every continent and season. Diversity of scenes, not more patches of the same scenes."],
  ["No architecture clearly best; U-Net stable across seeds and best SSIM; ViT 2-4 dB better under 60-100 % cloud thanks to its 56 px receptive field.", "**U-Net backbone + global self-attention in the bottleneck**: convolutions for local detail, attention so every region can use context from the whole patch."],
  ["Even a cloud-free input scores only 25.9-27.1 dB against the 'clear' target; after a per-band brightness fit 31-40 dB. Most visible-pixel error is the date difference between the two images.", "**Hybrid output**: a learned gate keeps visible pixels *and corrects their colour*; report **PSNR with the date shift removed** next to plain PSNR."],
  ["The GAN lost 1.35 dB and did not improve FID because its checkpoint was chosen by L1 after 1-4 k steps.", "**GAN as a second part** on top of a converged reconstruction model; **adaptive** adversarial weight; checkpoint chosen by **accuracy × sharpness**, only once the GAN is fully on."],
  ["Training on 128 × 128 crops with tiled evaluation limits context (a ViT position grid was fixed).", "**Full 256 × 256 patches** in training and evaluation; attention sees the whole patch."],
  ["Ranking was decided by 2 of 7 test scenes.", "**18 test scenes** (300 patches each) on every continent; differences tested with a **scene bootstrap**."],
  ["The user wants one model, not a comparison.", "Comparisons are only **references**: the cloudy input, the Phase 1 U-Net (same test patches) and the same model before its GAN part."],
], [50, 50]);

// =====================================================================================
H1("5. Data pipeline of the final model");
H2("5.1 Splitting the whole dataset by scene");
P("All 366 654 GeoTIFF files (122 218 complete triplets, 175 scenes) are indexed by filename as in Phase 1 (§3). Scenes - never individual patches - are assigned to splits, so neighbouring, near-identical patches can never appear in both training and test (Phase 1 measured the inflation a random split causes: +5.1 dB). The 41 scenes used by Phase 1 keep their roles (its 11 test scenes stay in the test set), so the Phase 1 U-Net can be evaluated on exactly the same test patches. The remaining scenes are dealt out round-robin per continent until validation and test reach their target sizes.");
T(["Split", "Scenes", "Patches", "Continents (scenes)"], [
  ["Train", "147", "**102 871 (all)**", "Europe 39, North America 37, Africa 27, Asia 23, South America 13, Oceania 8"],
  ["Validation", "10", "600 (60 per scene)", "Asia 7, Africa 1, Europe 1, North America 1"],
  ["Test", "18", "5 400 (300 per scene, of 12 539)", "Asia 7 (CN ×2, IN, IR, LB, TM, TW), Africa 3 (CF, ET, MG), Europe 3 (FR, PT, RU), North America 2 (US, CA), South America 2 (AR ×2), Oceania 1 (NZ)"],
], [14, 10, 22, 54]);
P("Capping the test set at 300 patches per scene keeps every scene's weight equal and the test pass under 10 minutes; validation (600 patches) is read once into RAM and scored every 20 minutes.");
H2("5.2 Preprocessing");
P("Identical to Phase 1 - the standard SEN12MS-CR preprocessing:");
T(["Modality", "Transformation", "Why"], [
  ["Sentinel-1 VV", "clip to [−25, 0] dB, rescale to [0, 1]: (x + 25) / 25", "standard SEN12MS-CR range; outliers clipped; networks train best on inputs of similar scale"],
  ["Sentinel-1 VH", "clip to [−32.5, 0] dB, rescale to [0, 1]", "VH is weaker than VV, so its range is wider"],
  ["S1 missing values", "NaN → lower bound (−25 / −32.5 dB)", "0 dB would mean maximum brightness - wrong"],
  ["S1 already linear?", "auto-detected (no negative values) and converted to dB", "robustness to other data sources"],
  ["Sentinel-2 (cloudy and clear)", "clip to [0, 10 000], divide by 10 000 → reflectance in [0, 1]", "all 13 bands kept"],
  ["Lee speckle filter", "not used", "the published baselines do not use it"],
], [22, 40, 38]);
P("**Model input** = [VV, VH, 13 cloudy bands] = **15 channels**; **target** = 13 clear bands. (The original pipeline converted the already-dB radar to dB a second time, erasing the radar signal - the most important bug found, chapter 17.)");
H2("5.3 Reading the GeoTIFFs on the fly");
P("A cache like Phase 1's would need ~450 GB (3.7 MB per patch × 122 218) and hours to build. Instead, DataLoader worker processes read the three GeoTIFFs of each patch directly while the GPUs train:");
UL(["**4 worker processes per GPU**, each reading the S1, cloudy-S2 and clear-S2 files of a patch **in three parallel threads**. Kaggle's dataset storage is limited by latency, not CPU: the quick test read only ~15 patches/s cold; overlapping the waits raised it to 42 patches/s on one machine and 15-21 (mean 18.5) on the slower disk of the final run.",
  "Unreadable or mis-shaped files are dropped by the collate function instead of crashing the run."]);
H2("5.4 Data echoing");
P("Two T4 GPUs consume ~36 patches/s, more than the final run's disk delivered. Rather than letting the GPUs wait, a background thread fills a queue with fresh batches; when none is ready, the training step re-uses patches from a pool of the **256 most recent patches** (kept on the GPU) with a **new random flip/rotation** - *data echoing* (Choi et al., 2019). At most 4 echoed batches may follow each other, so at least 1 in 5 is always fresh. In the final run **52 % of batches were fresh**; every training patch was read about 6 times in 10.8 hours.");
H2("5.5 Augmentation and patch size");
P("Every batch - fresh or echoed - gets one of the **8 dihedral transforms** (identity, flips, 90° rotations and combinations) per patch, applied on the GPU. Satellite images have no canonical orientation, so these are label-preserving. Training uses **full 256 × 256 patches** (no cropping), so the attention bottleneck sees the whole 2.56 km patch.");
H2("5.6 Cloud cover for reporting");
P("For grouping results only, a pixel counts as cloudy if the cloudy image is brighter than the clear target by more than 0.08 reflectance (mean of B2-B4). Bins: **0-10, 10-30, 30-60, 60-90, 90-100 %** - narrower at the low end, where most patches lie (1 878 of the 5 400 test patches are below 10 %).");

// =====================================================================================
H1("6. The generator: ClearSAR-Net");
P("Code: `src/models/clearsar_net.py`. Settings of the final model: `output_mode=\"hybrid\"`, `attn_layers=2`, `attn_heads=8`, `attn_mlp_ratio=2`, `width=64`, `upsample=\"bilinear\"` - **12.78 M parameters**.");
H2("6.1 Layer by layer (256 × 256 input)");
T(["Stage", "Operation", "Channels", "Resolution"], [
  ["Block (all conv stages)", "2 × [Conv 3×3, no bias → BatchNorm → ReLU]", "-", "-"],
  ["enc1", "block", "15 → 64", "256 × 256"],
  ["enc2", "MaxPool 2×2 → block", "64 → 128", "128 × 128"],
  ["enc3", "MaxPool 2×2 → block", "128 → 256", "64 × 64"],
  ["bottleneck", "MaxPool 2×2 → block", "256 → 512", "32 × 32"],
  ["attention", "1 024 tokens + 2-D sinusoidal position code → 2 Transformer blocks → LayerNorm", "512", "32 × 32"],
  ["dec3", "bilinear ×2 → Conv 3×3 (512→256) → concat enc3 → block", "512 → 256", "64 × 64"],
  ["dec2", "bilinear ×2 → Conv 3×3 (256→128) → concat enc2 → block", "256 → 128", "128 × 128"],
  ["dec1", "bilinear ×2 → Conv 3×3 (128→64) → concat enc1 → block", "128 → 64", "256 × 256"],
  ["heads", "three 1×1 convolutions: gate (64→1), correction (64→13), generator (64→13)", "→ 13", "256 × 256"],
], [16, 50, 14, 20]);
P("**Skip connections** copy each encoder feature map into the decoder at the same resolution, so fine detail need not be squeezed through the bottleneck (Phase 1's no-skip ablation lost structure: SSIM −0.010, FID +106).");
H2("6.2 Global self-attention in the bottleneck");
P("At 32 × 32 the bottleneck has 1 024 positions ('tokens') of 512 features. Each Transformer block (pre-norm, as in ViT) computes:");
F("x ← x + MHSA(LN(x))        x ← x + MLP(LN(x)),   MLP = Linear(512→1024) → GELU → Linear(1024→512)");
F("Attention(Q, K, V) = softmax( Q Kᵀ / √d ) · V,   8 heads, d = 64");
UL(["**Why here:** every token can attend to every other, so a pixel under a large cloud can use evidence from clear ground or radar structure anywhere in the 2.56 km patch - the property that made the ViT best under thick cloud in Phase 1. At 32 × 32 the 1 024² attention matrix is affordable; at full resolution it would not be.",
  "**Position code:** attention is permutation-invariant, so a fixed 2-D sine/cosine encoding (Vaswani et al., 2017, extended to rows and columns) is added to the tokens: half of the 512 channels encode the row, half the column, at geometrically spaced frequencies. Unlike a learned embedding it works for any grid size.",
  "**Implementation:** the attention is written out by hand (readable) and, in training, run through PyTorch's fused `scaled_dot_product_attention` kernel (same result, verified by a unit test to 10⁻⁷, without storing the 1 024 × 1 024 matrix).",
  "**Numerical safety:** the residual stream x is kept in **float32** and a final **LayerNorm** normalises it before the decoder. Without these, the first full Kaggle run overflowed fp16 (> 65 504) at step ~4 900 and went NaN (§11.3)."]);
H2("6.3 The hybrid output and the cloud gate");
F("out = m · σ( logit(c) + r ) + (1 − m) · σ( g ),    m = σ(gate)");
UL(["**c** = the 13 cloudy bands of the input; **r** = a learned per-pixel correction (initialised to 0); **g** = the generator head; **m ∈ [0, 1]** = the gate; σ = sigmoid.",
  "**Keep branch** σ(logit(c) + r): reproduces what the satellite saw, plus a learned brightness/colour shift - needed because the clear target was photographed on another day (§4).",
  "**Generate branch** σ(g): rebuilds the surface from radar and context where the input is hidden.",
  "**Initialisation:** r = 0 and gate bias +2 (m ≈ 0.88), so the untrained model starts close to the cloudy input; the generator's bias is set to logit(mean clear reflectance per band) from 48 training patches.",
  "**Interpretability:** m is a map of where the network trusts the input. It is never told where clouds are; on the test set the share of pixels it keeps falls with cloud cover (Spearman −0.84, mean 36 %)."]);
H2("6.4 Bilinear upsampling");
P("Phase 1's U-Net upsampled with 2 × 2 transposed convolutions. Transposed convolutions can create periodic 'checkerboard' artefacts (Odena et al., 2016) which a GAN discriminator tends to amplify. ClearSAR-Net upsamples bilinearly and then applies a 3 × 3 convolution - smooth by construction.");
H2("6.5 Size");
P("12.78 M parameters (Phase 1 U-Net: 7.71 M): 7.7 M for the U-Net, 4.2 M for the two attention blocks, 0.9 M for the bilinear-upsampling convolutions and heads. With batch 8 per GPU at 256 × 256 and the discriminator, peak memory was 10.4 GB of a T4's 15 GB.");

// =====================================================================================
H1("7. The GAN part: a PatchGAN critic");
H2("7.1 Why a GAN at all");
P("Under thick cloud several outputs are consistent with the input. A pixel loss (L1, L2) is minimised by a *compromise* between them, which looks blurry - measured as a **sharpness ratio** below 1 (§9.9): 0.78 for the model after part A. A GAN adds a *learned* loss: a second network, the discriminator, is trained to tell real clear images from generated ones, and the generator is trained to fool it. That rewards realistic texture. The price is the **perception-distortion trade-off** (Blau & Michaeli, 2018): realism and pixel accuracy cannot both be maximised, so the GAN may cost some PSNR and may invent plausible detail.");
H2("7.2 The discriminator");
T(["Layer", "Operation", "Channels", "Output (256 input)"], [
  ["input", "concat(x, y): radar + cloudy input and a candidate clear image", "15 + 13 = 28", "256 × 256"],
  ["1", "SN-Conv 4×4 stride 2 → LeakyReLU 0.2", "28 → 64", "128 × 128"],
  ["2", "SN-Conv 4×4 stride 2 → LeakyReLU 0.2", "64 → 128", "64 × 64"],
  ["3", "SN-Conv 4×4 stride 2 → LeakyReLU 0.2", "128 → 256", "32 × 32"],
  ["4", "SN-Conv 4×4 stride 1 → LeakyReLU 0.2", "256 → 512", "31 × 31"],
  ["5", "SN-Conv 4×4 stride 1", "512 → 1", "**30 × 30 scores**"],
], [10, 52, 16, 22]);
UL(["**PatchGAN** (Isola et al., 2017): each of the 30 × 30 outputs judges one overlapping **70 × 70-pixel window**. A window-level critic is a texture/sharpness critic, exactly what L1 lacks; the global layout is already handled by L1.",
  "**Conditional:** it sees the input x next to the candidate y, so it judges 'is this a plausible clear image *for this radar and cloudy input*' - not just 'is this a realistic image'.",
  "2.79 M parameters; no batch normalisation (spectral normalisation instead)."]);
H2("7.3 Spectral normalisation");
F("W_SN = W / σ(W),   σ(W) = largest singular value of W");
P("Every convolution weight is divided by its largest singular value (Miyato et al., 2018), estimated with one power-iteration step per training pass and computed in float32 even under mixed precision. Each layer can then stretch its input by at most a factor 1 (the discriminator is 1-Lipschitz per layer), so it changes smoothly and gives the generator well-behaved gradients - the standard cure for unstable GAN training. Verified against `torch.linalg.matrix_norm` (σ estimate 0.7804 vs true 0.7810 after a few passes).");
H2("7.4 Losses (least-squares GAN)");
F("L_D = ½ (D(x, y) − 1)² + ½ D(x, G(x))²");
F("L_G = L1(G(x), y) + 0.05 · (1 − SSIM) + λ · ½ (D(x, G(x)) − 1)²");
P("The least-squares form (LSGAN, Mao et al., 2017) pushes real scores to 1 and generated ones to 0; its gradients do not vanish when the discriminator is confident, which makes it more stable than the original cross-entropy GAN. Phase 1 used cross-entropy with λ = 1/100 relative to L1.");
H2("7.5 The adaptive adversarial weight λ");
F("λ = 0.2 · ‖∇_w L_rec‖ / ‖∇_w L_adv‖     (w = last shared generator layer)");
P("A fixed λ is fragile: the right value depends on the raw scales of both losses, which change during training. Following VQGAN (Esser et al., 2021), λ is set so that the adversarial gradient at the generator's last shared convolution is **20 % of the reconstruction gradient** - 'the GAN may push one fifth as hard as the pixel loss', whatever the scales. It is recomputed every 10 steps (gradient norms of both losses with respect to that layer), smoothed with factor 0.9, and phased in: 300 discriminator-only steps, then a linear ramp over 1 000 steps. As the discriminator grew stronger its gradients grew, so λ fell automatically from a peak of 0.06 to ~0.0045 (chapter 10).");
H2("7.6 Choosing the final checkpoint");
F("score = L1_val · (1 + |1 − sharpness_val|)     (lower is better)");
P("Phase 1 chose GAN checkpoints by L1 alone and so always picked the least adversarial one. The final model is chosen by a **balanced score** that rewards both accuracy and sharpness close to 1, and only among validations **after the GAN is fully on** (after the 1 300-step warm-up and ramp). The best score (0.0352) was reached at step 85 121 of 86 380.");

// =====================================================================================
H1("8. Training the final model");
H2("8.1 Two parts on a time-based schedule");
P("The schedule is defined in fractions of a **training budget** measured in time, not steps: the script measures the disk speed at start-up, reserves time for the test and export (40 minutes in the final run) and gives the rest to training (**10.84 h**). Part A runs until 55 % of the budget, part B to the end. Whatever the speed of the machine Kaggle assigns, the run always finishes inside the 12-hour limit.");
T(["", "Part A - reconstruction", "Part B - GAN refinement"], [
  ["Time / steps", "5.96 h, steps 0 - 47 940", "4.88 h, steps 47 940 - 86 380"],
  ["Generator loss", "L1 + 0.05 (1 − SSIM)", "L1 + 0.05 (1 − SSIM) + λ · LSGAN"],
  ["Generator LR", "warm-up 1 000 steps to 3 × 10⁻⁴, cosine to 1.5 × 10⁻⁵", "5 × 10⁻⁵, cosine to 5 × 10⁻⁷"],
  ["Discriminator", "not used", "Adam, LR 2 × 10⁻⁴ cosine to 2 × 10⁻⁶"],
  ["Checkpoint", "best validation L1 → `clearsar_net_before_gan.pt`", "best balanced score → `clearsar_net_final.pt`"],
], [18, 41, 41]);
P("The generator's learning rate restarts at 5 × 10⁻⁵ in part B (a *warm restart*): the new adversarial signal needs room to change the weights, but a lower peak than part A keeps the reconstruction from being destroyed.");
H2("8.2 Reconstruction loss and its weights");
F("L_rec = L1 + 0.05 · (1 − SSIM)");
P("**L1** (median-seeking, robust) is the main term; **SSIM** punishes lost local structure that L1 tolerates. The weight 0.05 was not guessed: on real patches the gradient of (1 − SSIM) with respect to the prediction was measured to be **14 × larger** than that of L1 (Phase 1 U-Net outputs: ‖∇L1‖ = 3.8 × 10⁻⁴, ‖∇SSIM‖ = 5.4 × 10⁻³). The initially planned 0.2 would have let SSIM dominate (2.8 × L1); 0.05 makes it pull about 0.7 × as hard as L1. A VGG perceptual loss was considered and dropped: it is built for 3-band photographs, costs ~30 % more time and its weight could not be calibrated on real data in time - the GAN covers texture.");
H2("8.3 Optimisation");
UL(["**AdamW** for the generator, β = (0.9, 0.999), weight decay 10⁻⁴ on weights only (not on biases or normalisation parameters); **Adam** for the discriminator, β = (0.5, 0.999) - the standard GAN setting that keeps the discriminator responsive.",
  "**Gradient clipping** of the generator at norm 1.0.",
  "**EMA (exponential moving average) of the weights**, decay 0.999: θ_EMA ← 0.999 θ_EMA + 0.001 θ after every step (batch-norm statistics averaged too). The EMA weights are what is validated and saved: they average out step-to-step noise and generalise better (Polyak averaging).",
  "**Batch size 8 per GPU (16 in total)**, chosen automatically: before training, a full part-B step (generator + discriminator, forward and backward) is tried at 8, 6 and 4 and the largest that fits in 85 % of the GPU memory is used; the probe's side effects on batch-norm and spectral-norm statistics are undone."]);
H2("8.4 Mixed precision");
P("Forward and backward passes run in **float16** on the T4's tensor cores (≈ 2 × faster, half the memory). Small gradients would underflow in float16, so a **GradScaler** multiplies the loss by a scale S before backpropagation and divides the gradients by S afterwards; if any gradient overflows, that step is skipped and S is halved; after 2 000 good steps S is doubled. In the final run S moved between 65 536 and 1 048 576 - healthy. Generator and discriminator have separate scalers. Spectral-norm power iterations, the attention residual stream, SSIM and all losses run in float32.");
H2("8.5 Two GPUs: DistributedDataParallel");
UL(["One process per T4; both hold an identical copy of both networks. Each processes its own half of every global batch, then the gradients are **averaged across GPUs (all-reduce)** before the optimiser step, so the copies stay identical. Gradients are sent in float16 (half the traffic).",
  "Time-based decisions (validate now? switch to part B? stop?) are made by GPU 0 from its clock and broadcast every 10 steps, so both processes always do the same thing.",
  "During the generator's adversarial step the discriminator is frozen and called without its DDP wrapper, so no gradient synchronisation is triggered for it."]);
H2("8.6 The NaN guard");
P("After the failed first full run (§11.3) the trainer gained a safety net: every 1 000 good steps it keeps a snapshot (generator, discriminator, optimisers, EMA) in RAM. If more than half of the steps are non-finite for 200 steps in a row, or a validation returns NaN, **both GPUs roll back to the snapshot, halve the learning rate and continue**; after 3 rollbacks training stops early and the best checkpoints are kept. Tested by deliberately setting a weight to infinity during a local run (detected at the next validation, rolled back, recovered). In the final run: **no events**.");
H2("8.7 Complete hyper-parameter table");
T(["Hyper-parameter", "Value", "Why / note"], [
  ["Data", "102 871 train / 600 val / 5 400 test patches; 147 / 10 / 18 scenes", "whole dataset, split by scene"],
  ["Input / output", "15 / 13 channels, 256 × 256", "full patches"],
  ["Augmentation", "8 dihedral transforms per patch, on the GPU", "label-preserving"],
  ["Generator", "ClearSAR-Net, width 64, 3 levels, 2 attention blocks (8 heads, MLP ×2), hybrid output, bilinear upsampling", "12.78 M parameters"],
  ["Discriminator", "conditional SN-PatchGAN, base 64, 70 × 70 windows", "2.79 M parameters"],
  ["Batch", "8 per GPU × 2 = 16", "probed automatically"],
  ["Training budget", "10.84 h (time-based); part A 55 %, part B 45 %", "86 380 steps in total"],
  ["Loss part A", "L1 + 0.05 (1 − SSIM)", "SSIM gradient measured 14 × L1"],
  ["Loss part B", "+ λ · LSGAN, λ adaptive with factor 0.2", "recomputed every 10 steps, smoothed 0.9"],
  ["GAN warm-up", "300 D-only steps, then 1 000-step linear ramp", ""],
  ["Generator optimiser", "AdamW β (0.9, 0.999), wd 10⁻⁴, clip 1.0", ""],
  ["Generator LR", "A: 1 000-step warm-up to 3 × 10⁻⁴, cosine to 5 % · B: 5 × 10⁻⁵ cosine to 1 %", ""],
  ["Discriminator optimiser", "Adam β (0.5, 0.999), LR 2 × 10⁻⁴ cosine to 1 %", ""],
  ["EMA decay", "0.999", "validated and saved weights"],
  ["Precision", "fp16 autocast + GradScaler (G and D separate); fp32 attention stream, SN, losses", ""],
  ["Parallelism", "DDP over 2 T4, fp16 gradient all-reduce, sync every 10 steps", ""],
  ["Data loading", "4 workers × 3 threads per GPU; echo pool 256, ≤ 4 echoes in a row", "52 % fresh batches"],
  ["Validation", "every 20 min, 600 patches, EMA weights", "33 validations"],
  ["Checkpoint selection", "A: lowest val L1; B: lowest L1 · (1 + |1 − sharpness|), GAN fully on", ""],
  ["NaN guard", "snapshot every 1 000 steps; rollback after 20 × 10 bad steps or NaN validation; ≤ 3 rollbacks", "no events"],
  ["Seed", "42", "same split as Phase 1 for shared scenes"],
], [24, 46, 30]);

// =====================================================================================
oldChapter("7. Evaluation and metrics", ["7.", "9."], [
  ["P(\"All metrics are computed on the full 256 × 256 test patches (tiled prediction), on all 13 bands with values in [0, 1], one value per patch, then averaged. Results are reported for the Asian test set and the out-of-continent set separately, by cloud-cover bin and by country.\");",
   "P(\"All metrics are computed on full 256 × 256 test patches, on all 13 bands with values in [0, 1], one value per patch, then averaged. §9.1-9.4 and §9.9-9.13 are used for the final model; §9.5-9.8 (FID, uncertainty calibration, receptive field, seed noise floor) belong to Phase 1.\");"],
  ["H2(\"9.5 FID - Fréchet Inception Distance\");", "H2(\"9.5 FID - Fréchet Inception Distance (Phase 1)\");"],
  ["H2(\"9.6 Uncertainty metrics (E7)\");", "H2(\"9.6 Uncertainty metrics (Phase 1, E7)\");"],
  ["H2(\"9.7 Effective receptive field (E3c)\");", "H2(\"9.7 Effective receptive field (Phase 1, E3c)\");"],
  ["H2(\"9.8 Noise floor and verdicts\");", "H2(\"9.8 Noise floor and verdicts (Phase 1)\");"],
]);
H2("9.9 Sharpness ratio");
F("sharpness = E(ŷ) / E(y),   E(z) = mean |z(i, j+1) − z(i, j)| + mean |z(i+1, j) − z(i, j)|");
P("Edge energy of the prediction relative to the truth. **1 = as sharp as reality**, < 1 blurry, > 1 more edges than reality (noise or invented detail; the cloudy input scores 1.88 because cloud edges are sharp). This is the measure the GAN part is meant to improve.");
H2("9.10 Coarse colour error and fine-detail error");
P("The error is split into two scales: the **colour error** is the MAE between 16 × 16-averaged prediction and truth (large-scale brightness and colour); the **detail error** is the MAE between the high-frequency residuals (image minus its 16 × 16 average). A model can get colours right but lose detail, or the opposite - two problems with different fixes.");
H2("9.11 PSNR with the date shift removed");
F("for each image and band:  ŷ' = a · (ŷ − mean ŷ) + mean y,   a = cov(ŷ, y) / var(ŷ);   PSNR(ŷ', y)");
P("The 'clear' target was taken on a different day: brightness, moisture and vegetation have changed. A per-band least-squares brightness/contrast fit removes that global difference before PSNR is computed, so the number measures how well structure and relative values are reconstructed. The cloudy input already scores 33.7 dB on it (its visible parts are perfect structure), the final model 34.5 dB.");
H2("9.12 Uncertainty from test-time augmentation");
P("The model is run on the patch and on three flipped versions (horizontal, vertical, both); the outputs are flipped back and their per-pixel standard deviation is the uncertainty map. Where the four answers disagree the model is unsure. Its usefulness is measured by the Spearman rank correlation between a patch's mean spread and its real MAE (0.55 on the test set).");
H2("9.13 Is a difference real? Scene bootstrap");
P("Patches from the same scene are correlated, so a difference between two models is judged per **scene**: the mean paired difference of each of the 18 test scenes is resampled 5 000 times with replacement (bootstrap, Efron & Tibshirani, 1993) to give a 95 % confidence interval, and the number of scenes in which one model wins is reported. A difference whose interval contains 0 is not established.");

// =====================================================================================
H1("10. Results of the final model");
H2("10.1 The run");
P("Kaggle, 2 × NVIDIA T4, 24-25 October 2026, one committed session of **11.1 hours**: 6.5 min indexing, 10.84 h training (**86 380 steps** of 16 patches, about 6 passes over the 102 871 training patches, 52 % fresh batches, 2.2-2.3 steps/s), 9 min test, export. **No NaN-guard events**, no skipped steps beyond normal GradScaler adjustments.");
UL(["**Part A:** validation PSNR rose from 25.6 dB (step 2 551) to 29.1 dB within 2 hours and peaked at **29.48 dB** (step 31 791, validation L1 0.0264); sharpness rose slowly from 0.41 to 0.64.",
  "**Part B** (from step 47 940): D(real) rose from 0.45 to 0.86 and D(generated) fell from 0.44 to 0.14 - the discriminator became strong but never saturated at 1/0. Validation sharpness jumped from 0.64 to 0.70-0.71 while validation PSNR eased from 29.4 to 29.1-29.2 dB."]);
IMG("fig_curves.png", 640, 171, "Figure 1 - Training of the final model. Left: L1 loss (train and validation). Middle: validation PSNR and sharpness. Right: discriminator scores in part B. Dotted line: the GAN joins.");
H2("10.2 Main result");
P("5 400 test patches from 18 held-out scenes, all 13 bands. Identical patches for all four rows.");
T(["", "PSNR (dB)", "SSIM", "SAM (°)", "MAE", "PSNR, date shift removed", "Sharpness", "Colour err.", "Detail err."], [
  ["Cloudy input (no model)", "20.29", "0.703", "13.37", "0.1001", "33.71", "1.88", "0.0980", "0.0194"],
  ["Phase 1 U-Net (3 000 Asian patches)", "26.70", "0.853", "9.78", "0.0377", "33.86", "0.90", "0.0353", "0.0115"],
  ["ClearSAR-Net before GAN", "**28.68**", "**0.881**", "8.38", "**0.0304**", "**34.58**", "0.78", "**0.0280**", "**0.0104**"],
  ["**ClearSAR-Net final (GAN)**", "28.59", "0.879", "**8.33**", "0.0305", "34.53", "**0.87**", "0.0280", "0.0106"],
], [26, 9, 8, 8, 9, 12, 9, 9, 10]);
H2("10.3 Which differences are real?");
T(["Comparison", "Mean Δ", "95 % CI over scenes", "Scenes where the first is better"], [
  ["Final − cloudy input (PSNR)", "+8.30 dB", "+6.69 to +9.94", "18 / 18"],
  ["Final − Phase 1 U-Net (PSNR)", "**+1.89 dB**", "+1.14 to +2.72", "17 / 18"],
  ["Before GAN − Phase 1 U-Net (PSNR)", "+1.98 dB", "+1.14 to +2.90", "16 / 18"],
  ["Final − before GAN (PSNR)", "−0.09 dB", "−0.34 to +0.16 (contains 0)", "7 / 18"],
  ["Final − before GAN (sharpness)", "**+0.086**", "+0.054 to +0.114", "16 / 18"],
], [34, 14, 28, 24]);
P("**Interpretation:** the new model is clearly and consistently better than Phase 1 - the gain is about 4 × the Phase 1 seed spread and holds in 17 of 18 scenes. The GAN part did what it was designed to do: a significant sharpness gain with no measurable PSNR loss.");
H2("10.4 By cloud cover");
T(["Cloud cover", "Patches", "Cloudy input", "Phase 1 U-Net", "Before GAN", "Final (GAN)", "Sharpness before → after GAN"], [
  ["0-10 %", "1 878", "27.07", "28.03", "29.92", "29.78", "0.91 → 1.02"],
  ["10-30 %", "966", "20.82", "27.04", "28.86", "28.75", "0.81 → 0.88"],
  ["30-60 %", "930", "16.99", "26.30", "27.82", "27.74", "0.70 → 0.78"],
  ["60-90 %", "817", "14.57", "25.27", "26.89", "26.94", "0.73 → 0.77"],
  ["90-100 %", "809", "13.51", "25.12", "28.40", "28.29", "0.61 → 0.69"],
], [14, 10, 14, 14, 13, 13, 22]);
IMG("fig_cloud.png", 520, 260, "Figure 2 - PSNR by cloud cover of the test patch.");
UL(["The **gap to the cloudy input grows with cloud** (+2.7 dB at 0-10 %, +14.8 dB at 90-100 %): the more is hidden, the more the network contributes.",
  "The gain over Phase 1 is **largest under thick cloud** (+3.2 dB at 90-100 %), where global context and the larger training set matter most.",
  "90-100 % scores higher than 60-90 %: fully clouded patches are often bright, homogeneous landscapes (deserts) whose average appearance is easy to predict from radar, while partly clouded ones mix visible detail with hidden detail.",
  "The GAN sharpens at every cloud level; under thin cloud it reaches the sharpness of reality (1.02)."]);
H2("10.5 By continent and for India");
T(["Continent", "Patches", "Cloudy input", "Phase 1 U-Net", "Before GAN", "Final (GAN)"], [
  ["Africa", "900", "20.66", "27.18", "30.39", "30.10"],
  ["Asia", "2 100", "20.98", "26.23", "27.90", "27.84"],
  ["Europe", "900", "16.67", "25.82", "27.46", "27.71"],
  ["North America", "600", "22.51", "29.25", "32.20", "32.22"],
  ["Oceania", "300", "18.91", "28.57", "30.37", "30.17"],
  ["South America", "600", "21.22", "25.47", "26.36", "25.87"],
  ["**India** (1 scene)", "300", "22.06", "25.50", "**29.64**", "29.45"],
], [24, 12, 16, 16, 16, 16]);
P("The model improves on Phase 1 on every continent - including Asia, Phase 1's own training region (+1.6 dB), because it saw far more scenes. South America is hardest (two Argentinian scenes with farmland whose field patterns cannot be inferred under cloud).");
H2("10.6 What the GAN changed");
P("Same generator before and after part B: PSNR −0.09 dB (not significant), SSIM −0.003, SAM 8.38° → 8.33° (slightly better spectra), sharpness **0.78 → 0.87**, fine-detail error +0.0002. The adaptive weight kept the adversarial gradient at one fifth of the reconstruction gradient, so the GAN added texture without the large accuracy loss of Phase 1's GAN (−1.35 dB). Sharpness is still below 1 under cloud: a stronger GAN (higher factor than 0.2) would sharpen further, at a growing risk of invented detail - the perception-distortion trade-off in practice.");
H2("10.7 The learned gate and the uncertainty");
UL(["**Gate:** the share of pixels whose input is kept falls from ~70 % on clear patches to ~5 % on fully clouded ones (Spearman −0.84 with cloud cover, mean 36 %). The gate was never given a cloud mask - it learned to detect clouds from the loss alone.",
  "**Uncertainty (flip test):** Spearman 0.55 between a patch's predicted uncertainty and its real error - the model 'knows' fairly well which patches it gets wrong, comparable to Phase 1's best method (variance head, 0.44-0.56). The expert system uses it (§18.8)."]);
H2("10.8 What the images show");
IMG("fig_gallery.png", 560, 742, "Figure 3 - Representative test patches (per cloud level the patch closest to the median score, not the best), from 18 scenes never seen in training.");
UL(["**Clear or lightly clouded** (China): very close to the truth; the colours are even shifted towards the date of the clear image (the hybrid output's correction branch).",
  "**Partly cloudy** (India, France, Argentina): clouds and shadows are removed and colours are plausible; detail under the clouds is soft and fine patterns (field boundaries, villages) are missing.",
  "**Heavily or fully clouded** (Portugal, Turkmenistan): the large-scale layout and colours are recovered from radar, fine detail is not - 10 m radar does not contain it.",
  "**Date differences:** some 'errors' are changes on the ground between the two dates (Argentina: different crop stage), which no model can know."]);
H2("10.9 Conclusions");
OLn(["One model trained on the whole dataset beats the best Phase 1 model by **+1.9 dB PSNR** (17 of 18 scenes) and the cloudy input by +8.3 dB; SSIM 0.879, spectral angle 8.3°.",
  "The largest gains are under thick cloud (+3.2 dB over Phase 1 at 90-100 %), where global attention and scene diversity matter most.",
  "Two-part training with an adaptive GAN weight works: sharpness +0.09 (16 of 18 scenes) with no significant PSNR cost - unlike Phase 1's GAN.",
  "The hybrid gate learned to detect clouds without supervision; the flip-test uncertainty tracks the real error (ρ = 0.55).",
  "Remaining error is dominated by what cannot be known: detail under thick cloud and changes between the two acquisition dates (PSNR 28.6 dB, but 34.5 dB once the date shift is removed)."]);
H2("10.10 Limitations");
UL(["One training run (seed 42): no seed spread for the final model; the scene bootstrap partly compensates.",
  "18 test scenes cover every continent but not every landscape; one Indian scene.",
  "Scores are not directly comparable to published SEN12MS-CR results, which use the official split and sometimes different normalisation (published single-image models report roughly 27-29 dB).",
  "The cloud fraction used for binning uses the target image; the gate and the expert system never see the target.",
  "No FID for the final model (dropped to save time; sharpness and detail error measure texture instead)."]);

// =====================================================================================
H1("11. Engineering: from quick test to the final run");
P("Five Kaggle sessions were needed. Each problem was diagnosed from the log, fixed, tested locally and pushed to GitHub before the next session (Kaggle clones the repository at start).");
T(["Session", "What happened", "Diagnosis and fix"], [
  ["Quick test (15 min, 400 patches)", "worked: DDP, batch 8 per GPU, 10.4 GB, 3.0 it/s; GAN healthy; 27.5 dB after 10 minutes", "cold reads only ~15 patches/s; fast later only because 400 patches fit in RAM → **parallel threaded reads**, 4 workers, larger echo pool"],
  ["Full run 1", "healthy until step ~4 800 (val 26.9 dB), then every loss and validation NaN for 3.5 h; speed rose", "fp16 overflow in the unnormalised attention residual stream: inf → NaN in every forward pass, GradScaler skipped every step → **fp32 residual stream + final LayerNorm + NaN guard** (§8.6) + regression test"],
  ["Full run 2", "Kaggle showed no log at all after 45 minutes", "the code printed its first line within a second from a fresh clone; Kaggle's log stream was not attached → cancelled and restarted"],
  ["Full run 3 (final)", "slower disk (start-up benchmark 4.2 patches/s, 15-21 in training)", "handled by data echoing (52 % fresh) and the time budget (40 min test reserve); finished in 11.1 h with no guard events"],
], [18, 40, 42]);
H2("11.1 Verification before using GPU time");
UL(["**56 unit tests** (`tests/`), including the hybrid output, spectral norm, adaptive λ (gradient ratio exactly 0.2), learning-rate schedule, data echoing, checkpoint round trip, the attention overflow regression and the expert system.",
  "**Synthetic end-to-end runs** on fake SEN12MS-CR GeoTIFFs (with real scene IDs): indexing → split → both training parts → test → gallery → zip; a run with the GAN fully on (tiny model) and a run with an injected infinite weight to test the NaN guard.",
  "**Fresh-clone test** of the exact pushed commit; headless tests of every frontend page with and without results."]);
H2("11.2 Output files");
T(["File (outputs/final/)", "Content"], [
  ["checkpoints/clearsar_net_final.pt, clearsar_net_before_gan.pt", "EMA weights + settings (51 MB each); load with `load_model_from_checkpoint`"],
  ["results/test_summary.csv", "the main table (§10.2)"],
  ["results/test_by_cloud.csv, _by_continent.csv, _by_country.csv", "breakdowns"],
  ["results/test_per_patch.csv", "every test patch × every model, incl. gate share and flip-test uncertainty"],
  ["results/samples.npz", "10 representative gallery patches with predictions, gate and uncertainty maps"],
  ["results/scene_splits.csv, test_patches.csv, uncertainty.json, training_curves.png", "split, test list, uncertainty summary, curves"],
  ["logs/train_history.csv, val_history.csv, train_summary.json, train_log.txt, pipeline_log.txt", "every 200 steps / every validation / run summary / full logs"],
  ["config.json", "every setting of the run"],
], [42, 58]);
H2("11.3 The NaN incident in detail");
P("The first full run's log showed real losses up to step 4 800 and 'nan' for every 200-step window from step 5 000 on; every validation was NaN, so no checkpoint was saved after step ~3 200. Because a GradScaler skips any step with non-finite gradients, weights cannot become NaN through the optimiser; persistent NaN *losses* with frozen weights therefore point to an overflow in the **forward** pass. The prime suspect was the attention bottleneck: its residual stream (CNN features + attention + MLP outputs) was added up in fp16 and fed, un-normalised, into the decoder's first convolution, whose fp16 output overflows above 65 504. The fix keeps that stream in float32 and adds a final LayerNorm (standard in Vision Transformers); a unit test multiplies the attention weights by 10⁴ and checks that the decoder input stays bounded. The rerun passed step 5 000 and finished cleanly. Lesson: mixed precision needs normalisation wherever activations can grow, and a long run needs an automatic guard rather than a human watching the log.");

// =====================================================================================
PART("Part B", "Phase 1: the architecture study that led to the final model");
H1("12. Phase 1 - data split and models");
P("Phase 1 trained on Asian scenes only: 37 Asian scenes were split by scene (seed 42, one Indian scene forced into the test set) into 23 training scenes (3 000 patches), 7 validation scenes (400) and 7 test scenes (600), plus one scene from each of four other continents as an out-of-continent test set (600). Patches were cached once as float16 memory maps (~17 GB) and trained as random 128 × 128 crops with flips and rotations; evaluation tiled each 256 patch into four 128 tiles.");
oldChapter("5. The models", ["5.", "12."], [["H1(\"12. The models\");", ""]]);
oldChapter("6. Training", ["6.", "13."], [["H1(\"13. Training\");", "H1(\"13. Phase 1 - training\");"]]);
oldChapter("8. Experiments", ["8.", "14."], [["H1(\"14. Experiments\");", "H1(\"14. Phase 1 - experiments\");"]]);
oldChapter("13. Results", ["13.", "15."], [["H1(\"15. Results\");", "H1(\"15. Phase 1 - results\");"]]);
oldChapter("9. Infrastructure", ["9.", "16."], [["H1(\"16. Infrastructure: how the 12-hour Kaggle run works\");", "H1(\"16. Phase 1 - infrastructure of the 26-run Kaggle session\");"]]);
oldChapter("10. What was wrong", ["10.", "17."]);

// =====================================================================================
oldChapter("11. The ClearSAR Advisor", ["11.", "18."], [["(§15.9)", "(§15.9 and §18.8)"], ["35 IF-THEN", "36 IF-THEN"], ["Rules (35, in five groups)", "Rules (36, in five groups)"], ["\"model (M1-M3)\", \"recommended model; avoid-GAN warning\"", "\"model (M1-M4)\", \"recommended model; avoid-GAN warning (M4: §18.8)\""]]);
H2("18.8 The advisor with the final model");
P("When `outputs/final/` contains the ClearSAR-Net checkpoints, the advisor switches to *final mode* (`Advisor(..., final_dir=\"outputs/final\")`); without them it works exactly as in Phase 1.");
T(["Part", "Phase 1 mode", "Final mode"], [
  ["Reconstruction", "best Phase 1 model per cloud level", "**ClearSAR-Net** - final (GAN) or before-GAN, whole 256 × 256 patch"],
  ["Experience: best model per cloud level", "Phase 1 test, Asian scenes, 20 %-wide bins", "the final model's own test (18 scenes): the higher-PSNR of before-GAN / final in that bin (0-10, 10-30, 30-60, 60-90, 90-100 %)"],
  ["Model disagreement", "U-Net vs Swin vs ViT", "unchanged: the three Phase 1 models act as independent second opinions"],
  ["Uncertainty fact", "median Laplace scale of the variance-head U-Net", "flip-test spread × k, k = median(MAE) / median(spread) on the test patches (clamped to 1-20), so it reads as an expected error comparable with the knowledge-base thresholds"],
  ["GAN rule M2 (visual inspection)", "recommends A_gan", "recommends the GAN-trained final model (`value_from: gan_model`)"],
  ["New rule M4 (CF 0.85)", "-", "for vegetation monitoring, land cover and change detection recommend the most faithful non-adversarial model - the before-GAN ClearSAR-Net; outranks M1 (0.8)"],
  ["Region", "inside / outside Asia (domain-shift warning)", "always inside the training data - the final model was trained on every continent"],
  ["XAI", "occlusion and saliency on 128 tiles", "on the whole patch (`tile=None`)"],
], [20, 34, 46]);
P("The knowledge base now has **36 rules**. Check on a test patch: visual inspection → *final (GAN)*, CF 0.80; vegetation monitoring → *before GAN*, CF 0.85, together with the *avoid GAN* warning - consistent advice.");

// =====================================================================================
H1("19. The frontend");
P("A Streamlit web app: `.venv\\Scripts\\python -m streamlit run app.py` (or `streamlit run app.py` inside the activated environment). It reads `outputs/final/` (final model) and `outputs/` (Phase 1); every page works with whatever is present. The repository contains both, including all checkpoints, so a fresh clone runs the full app without training.");
T(["Section", "Page", "What it shows"], [
  ["ClearSAR-Net", "Home", "drag-to-compare slider (cloudy input or truth vs the final model) on held-out patches - opens on the Indian patch; headline numbers; the two-part model"],
  ["", "Results", "all metrics for the cloudy input, Phase 1 U-Net, before-GAN and final; by cloud cover, continent and country; what the GAN changed (table + per-patch sharpness scatter); uncertainty vs error; the learned gate vs cloud cover"],
  ["", "How it learned", "loss, validation PSNR and sharpness over the 11 hours with the GAN start marked; D(real) / D(generated) and the adaptive λ; disk and GPU throughput; NaN-guard events; every hyper-parameter"],
  ["", "Explore patches", "every gallery patch and model against the cloudy input or the truth; radar, error, gate and uncertainty maps; per-patch scores"],
  ["Expert system", "Advisor", "decision cards (trust · model · fit for purpose · action) with certainty factors, warnings, visual explanation, how / why-not reasoning, facts, downloadable report; gallery patches or uploaded GeoTIFFs"],
  ["", "Knowledge base", "architecture diagram, what-if simulator, all rules and thresholds, CF algebra"],
  ["Earlier experiments", "Phase 1 model comparison / experiments", "the 26-run study of Part B"],
  ["About", "How it works", "data, model, training, evaluation, expert system, limitations"],
], [16, 18, 66]);

// =====================================================================================
H1("20. Likely questions and answers");
H2("20.1 The final model");
T(["Question", "Answer"], [
  ["Why one model instead of comparing architectures?", "Phase 1 already compared them and found no clear winner at that data scale. The remaining limit was data diversity and training design, so the effort went into one model built from Phase 1's lessons and trained on everything."],
  ["Why a U-Net with attention and not a pure Transformer?", "The U-Net was the most stable and best at structure (SSIM); the ViT's advantage came from global context under thick cloud. Attention in the 32 × 32 bottleneck adds exactly that context at low cost, while convolutions and skip connections keep local detail."],
  ["What does the cloud gate do and how was it trained?", "It outputs m ∈ [0, 1] per pixel: keep and colour-correct the input (m → 1) or generate from radar (m → 0). It received no cloud labels; minimising the loss alone taught it to open where the input is reliable (Spearman −0.84 with cloud cover)."],
  ["Why the extra correction on kept pixels?", "The target was photographed on another day; even cloud-free pixels differ (a cloud-free input scores only ~26-27 dB). The correction branch learns that brightness/colour shift instead of copying the pixels."],
  ["Why train in two parts instead of a GAN from the start?", "A discriminator gives noisy, unhelpful gradients while the generator still produces blur; starting from a converged reconstruction model lets the GAN refine texture. It also gives a clean before/after comparison of the same network."],
  ["What is a PatchGAN and why conditional?", "A discriminator that scores overlapping 70 × 70 windows instead of the whole image - a texture critic. Conditional: it sees the radar and cloudy input next to the candidate, so it judges plausibility for this specific input."],
  ["Why spectral normalisation and LSGAN?", "Both stabilise GAN training: SN limits how sharply the discriminator can change (1-Lipschitz layers), LSGAN keeps gradients informative when the discriminator is confident."],
  ["How was the adversarial weight chosen?", "Adaptively: λ is recomputed so that the GAN's gradient at the last shared layer is 20 % of the pixel loss's gradient (VQGAN). This is independent of the raw loss scales, which a single 12-hour run could not afford to tune."],
  ["Why did λ fall from 0.06 to 0.0045?", "As the discriminator got stronger its gradients grew; to keep the 20 % ratio, λ shrank. That is the intended behaviour."],
  ["Is D(real) = 0.86, D(fake) = 0.14 a problem?", "The discriminator became strong but did not saturate at 1/0, and the adaptive weight capped its influence; validation sharpness rose while PSNR stayed within noise. A collapse would show D at exactly 1/0 and degrading images."],
  ["How was the final checkpoint chosen?", "By the lowest L1 · (1 + |1 − sharpness|) on validation, only after the GAN was fully on - rewarding accuracy and realism together. Phase 1's GAN failed because L1 alone picked the least adversarial checkpoint."],
  ["Why L1 + 0.05 SSIM and not 0.2?", "The SSIM gradient was measured to be 14 × larger than L1's on real patches; 0.2 would have let SSIM dominate. 0.05 balances the two."],
  ["What is EMA and why validate the EMA weights?", "An exponential moving average of the weights over the last ~1 000 steps; it smooths out the noise of individual steps and usually generalises better than the raw weights."],
  ["How do two GPUs train one model?", "DistributedDataParallel: each GPU computes gradients on half of the batch, the gradients are averaged across GPUs, and both apply the same update - mathematically one batch of 16."],
  ["What went wrong in the first full run?", "An fp16 overflow in the attention residual stream turned every forward pass into NaN after ~5 000 steps; the GradScaler skipped every update. Fixed with an fp32 residual stream and a final LayerNorm, plus an automatic rollback guard."],
  ["What is data echoing?", "When the disk is slower than the GPUs, recent patches are reused with a new random flip/rotation instead of idling the GPU. At least one batch in five is always fresh; 52 % were in the final run."],
  ["Is +1.9 dB over Phase 1 significant?", "Yes: the 95 % bootstrap interval over the 18 test scenes is +1.1 to +2.7 dB and the new model is better in 17 of 18 scenes."],
  ["Did the GAN make things worse?", "PSNR changed by −0.09 dB with an interval that contains 0 (−0.34 to +0.16): no measurable loss. Sharpness improved significantly (+0.086, 16 of 18 scenes)."],
  ["Why is PSNR only ~28.6 dB if the images look good?", "Part of the 'error' is real change on the ground between the two acquisition dates and detail that radar cannot see. With the date shift removed the model reaches 34.5 dB."],
  ["Why is the 90-100 % cloud bin better than 60-90 %?", "Fully clouded test patches are often homogeneous landscapes (e.g. desert in Turkmenistan) whose appearance radar predicts well; partly clouded ones hide detailed mixed scenes."],
  ["How is the uncertainty of the final model obtained?", "Test-time augmentation: four flipped predictions, their spread per pixel. It correlates with the real error (Spearman 0.55); the advisor scales it to an expected error."],
  ["Would it work on a new region?", "It was trained on all continents and tested on 18 unseen scenes, so it generalises to new places within Sentinel-1/2 conditions; very unusual landscapes or a different sensor would need testing."],
], [32, 68]);
H2("20.2 Data, metrics and Phase 1");
const OLDQA = OLD.slice(OLD.indexOf("const QA = ["), OLD.indexOf("T([\"Question\", \"Answer\"], QA"));
eval(OLDQA.replace("const QA", "var QA"));
T(["Question", "Answer"], QA.filter((q) => !/^(What are the main limitations\?)$/.test(q[0])), [32, 68]);

// =====================================================================================
H1("21. Glossary");
const GL_OLD = OLD.slice(OLD.indexOf("const GL = ["), OLD.indexOf("T([\"Term\", \"Meaning\"], GL"));
eval(GL_OLD.replace("const GL", "var GL"));
const GL_NEW = [
  ["Adaptive adversarial weight", "λ recomputed so the GAN's gradient is a fixed share (here 20 %) of the pixel loss's gradient"],
  ["Bilinear upsampling", "enlarging a feature map by interpolation; used with a 3 × 3 convolution instead of transposed convolutions"],
  ["Bootstrap confidence interval", "uncertainty of a mean estimated by resampling the data (here: the 18 test scenes) with replacement"],
  ["Checkerboard artefact", "periodic grid pattern produced by transposed convolutions"],
  ["Conditional GAN", "GAN whose discriminator also sees the input the image was generated from"],
  ["Data echoing", "re-using recent training examples with new augmentations when data loading is slower than the GPU"],
  ["Date shift", "difference between the cloudy and the clear image caused by their different acquisition dates"],
  ["DistributedDataParallel (DDP)", "one process per GPU, gradients averaged across GPUs every step"],
  ["Discriminator", "the GAN network that judges real vs generated images"],
  ["EMA (weights)", "exponential moving average of the weights over recent steps; the version that is validated and saved"],
  ["Gate (hybrid output)", "learned per-pixel weight between keeping the input and generating"],
  ["GradScaler / loss scaling", "multiplying the loss before backpropagation so fp16 gradients do not underflow"],
  ["LayerNorm", "normalises each token's features to zero mean and unit variance"],
  ["LSGAN", "least-squares GAN loss: real → 1, fake → 0 with squared error"],
  ["Mixed precision", "computing in float16 where safe and float32 where needed"],
  ["NaN", "'not a number' - the result of an invalid computation, e.g. ∞ − ∞; spreads through every later computation"],
  ["PatchGAN", "discriminator that outputs one score per image window instead of one per image"],
  ["Perception-distortion trade-off", "realism and pixel accuracy cannot both be maximised"],
  ["Residual stream", "the running sum x + attention + MLP that flows through Transformer blocks"],
  ["Sharpness ratio", "edge energy of the prediction divided by that of the truth (1 = as sharp as reality)"],
  ["Spectral normalisation", "dividing a weight matrix by its largest singular value to bound the layer's gain"],
  ["Test-time augmentation (TTA)", "predicting transformed copies of an input and combining them; their spread estimates uncertainty"],
  ["Time-based schedule", "learning rate and training phases defined by elapsed time instead of step count"],
];
T(["Term", "Meaning"], [...GL, ...GL_NEW].sort((a, b) => a[0].localeCompare(b[0])), [28, 72]);

// =====================================================================================
H1("22. References");
const REF_OLD = OLD.slice(OLD.indexOf("H1(\"16. References\");") + "H1(\"16. References\");".length, OLD.indexOf("].forEach((r) => body.push"));
const refs = eval(REF_OLD.trim() + "]");
[...refs,
  "Mao, X. et al. (2017). Least squares generative adversarial networks. ICCV. (LSGAN)",
  "Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y. (2018). Spectral normalization for generative adversarial networks. ICLR.",
  "Esser, P., Rombach, R., Ommer, B. (2021). Taming transformers for high-resolution image synthesis. CVPR. (VQGAN, adaptive adversarial weight)",
  "Odena, A., Dumoulin, V., Olah, C. (2016). Deconvolution and checkerboard artifacts. Distill.",
  "Choi, D., Passos, A., Shallue, C. J., Dahl, G. E. (2019). Faster neural network training with data echoing. arXiv:1907.05550.",
  "Zhao, H., Gallo, O., Frosio, I., Kautz, J. (2017). Loss functions for image restoration with neural networks. IEEE Transactions on Computational Imaging.",
  "Li, S. et al. (2020). PyTorch distributed: experiences on accelerating data parallel training. VLDB.",
  "Polyak, B. T., Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization.",
  "Efron, B., Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.",
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
  creator: "ClearSAR", title: "ClearSAR - complete project guide (final model)",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22, color: DARK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, color: ACCENT, font: "Calibri" }, paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 0, keepNext: true, keepLines: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, color: DARK, font: "Calibri" }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1, keepNext: true, keepLines: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: ACCENT, font: "Calibri" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2, keepNext: true, keepLines: true } },
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
