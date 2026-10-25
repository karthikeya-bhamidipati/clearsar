// ClearSAR final presentation - structured deck (theme, layouts, sections, native charts, speaker notes).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const A = (f) => path.join(__dirname, "assets", f);
const DATA = JSON.parse(fs.readFileSync(A("data.json"), "utf8"));
const OUT = process.argv[2] || path.join(__dirname, "ClearSAR_Final_Presentation.pptx");

const THEME = {
  name: "ClearSAR Orbit",
  headFontFace: "Segoe UI Semibold",
  bodyFontFace: "Segoe UI",
  colors: {
    dk1: "0A1022", lt1: "FFFFFF", dk2: "131D3A", lt2: "AAB5C8",
    accent1: "2DD4BF", accent2: "38BDF8", accent3: "A78BFA", accent4: "FBBF24", accent5: "FB7185", accent6: "A3E635",
    hlink: "7DD3FC", folHlink: "A78BFA",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                     // 13.333 x 7.5 in
pres.title = "ClearSAR - final presentation";
pres.author = "Karthikeya Bhamidipati";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, M = 0.6;

// ------------------------------------------------------------------ icons
async function icon(name, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "CS_TITLE",
  background: { color: C.text1 },
  objects: [{ image: { path: A("bg_title.jpg"), x: 0, y: 0, w: W, h: H } }],
  slideNumber: { x: W - 0.9, y: H - 0.45, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});
pres.defineSlideMaster({
  title: "CS_CONTENT",
  background: { color: C.text1 },
  objects: [
    { image: { path: A("bg_content.jpg"), x: 0, y: 0, w: W, h: H } },
    { text: { text: "ClearSAR · SAR-to-optical cloud removal", options: { x: M, y: H - 0.45, w: 6, h: 0.3, fontSize: 10, color: C.background2, margin: 0 } } },
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.38, w: 8, h: 0.32, fontSize: 11, bold: true, color: C.accent1, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.68, w: W - 2 * M, h: 0.8, fontSize: 32, bold: true, color: C.background1, margin: 0, valign: "top", align: "left" }, text: "" } },
  ],
  slideNumber: { x: W - 0.9, y: H - 0.45, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

// ------------------------------------------------------------------ helpers
function panel(s, x, y, w, h, name, fill = C.text2, line = C.accent1) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill, transparency: 8 },
    line: { color: line, width: 0.75, transparency: 55 }, objectName: name });
}
function T(s, text, o) { s.addText(text, Object.assign({ isTextBox: true, margin: 0, color: C.background1, fontSize: 14, valign: "top" }, o)); }
async function iconDot(s, name, x, y, d, color, objectName) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 }, objectName: objectName + "_dot" });
  s.addImage({ data: await icon(name), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52, objectName: objectName + "_icon" });
}
function img(s, file, x, y, w, h, alt, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x - 0.04, y: y - 0.04, w: w + 0.08, h: h + 0.08, rectRadius: 0.08,
    fill: { color: C.text2 }, line: { color: C.accent2, width: 0.75, transparency: 50 }, objectName: name + "_frame" });
  s.addImage({ path: A(file), x, y, w, h, altText: alt, objectName: name });
}
function caption(s, text, x, y, w, align = "left") { T(s, text, { x, y, w, h: 0.32, fontSize: 11, color: C.background2, align }); }
function stat(s, value, label, x, y, w, color, name) {
  panel(s, x, y, w, 1.45, name + "_panel");
  T(s, value, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.7, fontSize: 34, bold: true, color, fontFace: THEME.headFontFace });
  T(s, label, { x: x + 0.25, y: y + 0.9, w: w - 0.5, h: 0.45, fontSize: 12, color: C.background2 });
}
function content(section, kicker, title) {
  const s = pres.addSlide({ masterName: "CS_CONTENT", sectionTitle: section });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}
const chartText = { catAxisLabelColor: HEX.lt2, valAxisLabelColor: HEX.lt2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.lt2,
  valGridLine: { color: "26325A", size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: false, valAxisLineShow: false,
  titleColor: HEX.lt1, titleFontFace: "+mn-lt", titleFontSize: 13 };

(async () => {
  // ================================================================ 1. title
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "CS_TITLE", sectionTitle: "Opening" });
  T(s, "MSC DEEP LEARNING PROJECT · OCTOBER 2026", { x: M, y: 1.35, w: 7, h: 0.35, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3 });
  T(s, "ClearSAR", { x: M, y: 1.8, w: 7, h: 1.1, fontSize: 60, bold: true, fontFace: THEME.headFontFace });
  T(s, "Seeing the ground through the clouds", { x: M, y: 2.95, w: 7, h: 0.6, fontSize: 26, color: C.accent2, fontFace: THEME.headFontFace });
  T(s, "One deep-learning model that rebuilds cloud-free Sentinel-2 images from radar and the cloudy image - and an expert system that decides when to trust it.",
    { x: M, y: 3.75, w: 6.3, h: 1.0, fontSize: 16, color: C.background2 });
  T(s, "Karthikeya Bhamidipati", { x: M, y: 5.6, w: 6, h: 0.4, fontSize: 16, bold: true });
  T(s, "github.com/karthikeya-bhamidipati/clearsar", { x: M, y: 6.0, w: 6, h: 0.35, fontSize: 12, color: C.background2 });
  img(s, "CN2_cloudy.png", 7.75, 1.45, 2.35, 2.35, "Hazy Sentinel-2 test patch from China", "t_cloudy");
  img(s, "CN2_final.png", 10.35, 1.45, 2.35, 2.35, "ClearSAR-Net reconstruction of the same patch", "t_final");
  img(s, "CN2_radar.png", 7.75, 4.2, 2.35, 2.35, "Sentinel-1 radar (VV) of the same patch", "t_radar");
  img(s, "CN2_truth.png", 10.35, 4.2, 2.35, 2.35, "True clear image of the same patch", "t_truth");
  caption(s, "hazy input (China, test scene)", 7.6, 3.9, 2.65, "center"); caption(s, "ClearSAR-Net", 10.35, 3.9, 2.35, "center");
  caption(s, "radar (sees through cloud)", 7.6, 6.65, 2.65, "center"); caption(s, "true clear image", 10.35, 6.65, 2.35, "center");
  s.addNotes("Open with the picture, not the title. Top left is a real Sentinel-2 image over China, washed out by haze. Bottom left is the radar of the same place - it sees through the clouds but has no colour. Top right is what my model rebuilds; bottom right is the real clear image taken on another day. This is a test scene the model never saw. In the next 15 minutes: the problem, the model, the results, and an expert system that tells you when to trust it.");

  // ================================================================ 2. problem
  pres.addSection({ title: "Problem" });
  s = content("Problem", "01 · THE PROBLEM", "Clouds hide the ground - radar does not");
  const rows = [["FaCloud", C.accent2, "Optical satellites are blind under cloud", "Sentinel-2 sees 13 spectral bands - colour, vegetation, water - but needs sunlight through a clear sky."],
    ["FaTowerBroadcast", C.accent3, "Radar sees through cloud, day and night", "Sentinel-1 sends its own microwaves (C-band, 5.6 cm). The echo shows roughness and moisture - but no colour."],
    ["FaWandMagicSparkles", C.accent1, "Fuse the two", "Use the radar plus whatever is still visible to predict the cloud-free optical image. Under thick cloud this is an educated guess - so we also need to know when to trust it."]];
  for (let i = 0; i < rows.length; i++) {
    const y = 1.85 + i * 1.6;
    await iconDot(s, rows[i][0], M, y, 0.75, rows[i][1], "p_icon" + i);
    T(s, rows[i][2], { x: M + 1.05, y: y - 0.02, w: 6.2, h: 0.4, fontSize: 18, bold: true });
    T(s, rows[i][3], { x: M + 1.05, y: y + 0.42, w: 6.2, h: 0.95, fontSize: 14, color: C.background2 });
  }
  img(s, "PT81_cloudy.png", 8.45, 1.85, 2.0, 2.0, "Heavily clouded Sentinel-2 patch in Portugal", "p_cloudy");
  img(s, "PT81_radar.png", 10.75, 1.85, 2.0, 2.0, "Sentinel-1 radar of the same patch", "p_radar");
  img(s, "PT81_truth.png", 9.6, 4.35, 2.0, 2.0, "True clear image of the same patch", "p_truth");
  caption(s, "optical, 81 % cloud", 8.45, 3.9, 2.0, "center"); caption(s, "radar", 10.75, 3.9, 2.0, "center");
  caption(s, "what is really there", 9.6, 6.4, 2.0, "center");
  s.addNotes("Clouds are the single biggest obstacle for optical Earth observation - in the monsoon an area can stay hidden for weeks. Radar works through cloud but measures something different: surface roughness and moisture, not colour. The idea is to fuse the two. Note the honest limitation from the start: under thick cloud the colours are not observed at all, the model has to infer them - which is why the second half of the project is an expert system that judges trust.");

  // ================================================================ 3. task
  s = content("Problem", "01 · THE TASK", "15 channels in, 13 channels out");
  panel(s, M, 2.2, 3.6, 3.3, "task_in");
  T(s, "INPUT", { x: M + 0.3, y: 2.45, w: 3, h: 0.3, fontSize: 11, bold: true, color: C.accent1, charSpacing: 2 });
  T(s, [{ text: "2 radar bands", options: { bold: true, breakLine: true } }, { text: "Sentinel-1 VV, VH (dB, scaled to 0-1)", options: { color: C.background2, fontSize: 13, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "13 cloudy optical bands", options: { bold: true, breakLine: true } }, { text: "Sentinel-2 B1-B12 incl. red-edge, NIR, SWIR (reflectance 0-1)", options: { color: C.background2, fontSize: 13 } }],
    { x: M + 0.3, y: 2.9, w: 3.0, h: 2.4, fontSize: 16 });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.45, y: 3.55, w: 0.9, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "task_arrow1" });
  panel(s, 5.5, 2.4, 2.4, 2.9, "task_model", C.text2, C.accent3);
  await iconDot(s, "FaBrain", 6.3, 2.7, 0.8, C.accent3, "task_brain");
  T(s, "ClearSAR-Net", { x: 5.6, y: 3.7, w: 2.2, h: 0.45, fontSize: 18, bold: true, align: "center" });
  T(s, "12.8 M parameters", { x: 5.6, y: 4.15, w: 2.2, h: 0.35, fontSize: 12, color: C.background2, align: "center" });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 8.05, y: 3.55, w: 0.9, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "task_arrow2" });
  panel(s, 9.1, 2.2, 3.6, 3.3, "task_out");
  T(s, "OUTPUT", { x: 9.4, y: 2.45, w: 3, h: 0.3, fontSize: 11, bold: true, color: C.accent1, charSpacing: 2 });
  T(s, [{ text: "13 clear optical bands", options: { bold: true, breakLine: true } }, { text: "the cloud-free Sentinel-2 image - all bands, not just colour, so vegetation and water indices still work", options: { color: C.background2, fontSize: 13 } }],
    { x: 9.4, y: 2.9, w: 3.0, h: 2.4, fontSize: 16 });
  T(s, "The answer is ill-posed: under thick cloud several outputs fit the radar equally well. A pixel loss then averages them (blur); a GAN can add realistic texture - at the risk of inventing it.",
    { x: M, y: 6.1, w: W - 2 * M, h: 0.7, fontSize: 14, color: C.background2 });
  s.addNotes("Precisely: every 2.56 km patch is 256 by 256 pixels. Input is 15 channels - 2 radar plus the 13 cloudy optical bands. Output is all 13 clear bands, so the result is usable for science, not just a pretty RGB picture. The key word is ill-posed: more than one answer is consistent with the input, and that drives two choices later - the loss, and the GAN.");

  // ================================================================ 4. data
  pres.addSection({ title: "Approach" });
  s = content("Approach", "02 · DATA", "The whole SEN12MS-CR dataset, split by scene");
  stat(s, "122,218", "patch triplets: radar + cloudy + clear", M, 1.9, 2.85, C.accent1, "d1");
  stat(s, "175", "scenes on every continent", M + 3.05, 1.9, 2.85, C.accent2, "d2");
  stat(s, "4", "seasons", M + 6.1, 1.9, 2.85, C.accent3, "d3");
  stat(s, "+5.1 dB", "fake gain if patches were split at random (Phase 1 leakage test)", M + 9.15, 1.9, 2.95, C.accent5, "d4");
  T(s, "Scene split (no scene is in two splits)", { x: M, y: 3.75, w: 8, h: 0.4, fontSize: 16, bold: true });
  const split = [["Train", 147, "102,871 patches - all of them", C.accent1], ["Val", 10, "600 patches", C.accent4], ["Test", 18, "5,400 patches, every continent", C.accent3]];
  let x0 = M;
  for (const [name, n, lab, col] of split) {
    const w = (W - 2 * M) * n / 175;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0, y: 4.3, w: Math.max(w - 0.06, 0.5), h: 0.7, rectRadius: 0.08, fill: { color: col }, line: { color: col, width: 0 }, objectName: "split_" + name });
    T(s, name === "Val" ? "Val" : `${name} · ${n}`, { x: x0 + 0.1, y: 4.45, w: Math.max(w - 0.16, 0.5), h: 0.4, fontSize: 13, bold: true, color: C.text1 });
    x0 += w;
  }
  T(s, [{ text: "Train: ", options: { bold: true, color: C.accent1 } }, { text: "147 scenes, 102,871 patches - all of them.   ", options: { color: C.background2 } },
    { text: "Validation: ", options: { bold: true, color: C.accent4 } }, { text: "10 scenes, 600 patches.   ", options: { color: C.background2 } },
    { text: "Test: ", options: { bold: true, color: C.accent3 } }, { text: "18 unseen scenes, 5,400 patches - incl. India.", options: { color: C.background2 } }],
    { x: M, y: 5.2, w: W - 2 * M, h: 0.5, fontSize: 14 });
  T(s, "Neighbouring patches look almost identical. Splitting by scene means the test measures generalisation to new places, not memory.",
    { x: M, y: 5.95, w: W - 2 * M, h: 0.6, fontSize: 14, color: C.background2 });
  s.addNotes("The data: SEN12MS-CR, 122 thousand co-registered triplets from 175 scenes, four seasons, every continent. Phase 1 used only 3,000 Asian patches - 2.5 % of this. The final model uses every training patch. Most important design rule: split by scene. Phase 1 measured that a random split inflates PSNR by 5.1 dB - more than any architecture difference - because neighbouring patches are near-duplicates.");

  // ================================================================ 5. phase 1
  s = content("Approach", "02 · HOW WE GOT HERE", "Phase 1: 26 trainings taught four lessons");
  T(s, "U-Net vs Swin U-Net vs Vision Transformer + ablations, a GAN, three uncertainty methods - on 3,000 Asian patches.",
    { x: M, y: 1.55, w: W - 2 * M, h: 0.4, fontSize: 15, color: C.background2 });
  const lessons = [["No architecture clearly best (26.3-27.0 dB)", "U-Net backbone - the most stable"],
    ["Global context wins under thick cloud: ViT +2-4 dB at 60-100 % cloud", "Self-attention in the U-Net's bottleneck"],
    ["Learning curves flat: 750 → 3,000 patches barely helped - scene diversity was the limit", "Train on all 147 scenes"],
    ["GAN gained no realism: checkpoint chosen by L1 after 1-4 k steps", "Two-part training, select by accuracy × sharpness"]];
  for (let i = 0; i < lessons.length; i++) {
    const y = 2.4 + i * 1.1;
    panel(s, M, y, 6.0, 0.92, "l_find" + i, C.text2, C.accent5);
    T(s, lessons[i][0], { x: M + 0.25, y: y + 0.12, w: 5.6, h: 0.7, fontSize: 14, valign: "middle" });
    s.addShape(pres.shapes.RIGHT_ARROW, { x: 6.85, y: y + 0.26, w: 0.6, h: 0.4, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "l_arrow" + i });
    panel(s, 7.65, y, 5.08, 0.92, "l_dec" + i);
    T(s, lessons[i][1], { x: 7.9, y: y + 0.12, w: 4.7, h: 0.7, fontSize: 14, bold: true, color: C.accent1, valign: "middle" });
  }
  T(s, "FINDING", { x: M, y: 2.07, w: 3, h: 0.28, fontSize: 11, bold: true, color: C.accent5, charSpacing: 2 }); T(s, "DECISION FOR THE FINAL MODEL", { x: 7.65, y: 2.07, w: 5, h: 0.28, fontSize: 11, bold: true, color: C.accent1, charSpacing: 2 });
  s.addNotes("Phase 1 was a controlled comparison - 26 trainings. It did not produce a winner, but it taught four lessons, and each became a design decision. Left the finding, right what I did about it. The last one is the most instructive: the GAN looked like a failure, but the real fault was the checkpoint selection - choosing by L1 picks the least adversarial model.");

  // ================================================================ 6. date shift
  s = content("Approach", "02 · A HIDDEN CEILING", "The 'clear' image was taken on another day");
  img(s, "CN2_cloudy.png", M, 1.75, 3.0, 3.0, "Nearly cloud-free Sentinel-2 input, China", "ds_in");
  img(s, "CN2_truth.png", M + 3.35, 1.75, 3.0, 3.0, "The clear target image of the same place", "ds_truth");
  caption(s, "input - only 2 % cloud", M, 4.82, 3.0, "center"); caption(s, "'clear' target, different date", M + 3.35, 4.82, 3.0, "center");
  stat(s, "24-27 dB", "a nearly cloud-free input scores only this against its target", 7.4, 1.75, 5.33, C.accent5, "ds1");
  stat(s, "31-40 dB", "after a simple per-band brightness fit - the rest was the date difference", 7.4, 3.35, 5.33, C.accent1, "ds2");
  T(s, [{ text: "So: ", options: { bold: true, color: C.accent1 } },
    { text: "the model may correct the colours of visible pixels (hybrid output, next slides), and every result is also reported as ", options: { color: C.background2 } },
    { text: "PSNR with the date shift removed.", options: { bold: true } }], { x: 7.4, y: 5.05, w: 5.33, h: 1.3, fontSize: 15 });
  T(s, "Measured on the nearly cloud-free Phase 1 gallery patches.", { x: M, y: 5.3, w: 6.3, h: 0.35, fontSize: 11, color: C.background2 });
  s.addNotes("This was the insight that changed my evaluation. Even a cloud-free input only scores 24 to 27 dB against its 'clear' target - because the target was photographed on another day: different light, moisture, crop stage. After a simple per-band brightness fit the same pairs score 31 to 40 dB. So plain PSNR has a ceiling well below perfect, and I report a second number with the date shift removed. It also motivated the hybrid output: the model is allowed to correct colours, not just copy visible pixels.");

  // ================================================================ 7. architecture
  pres.addSection({ title: "Model" });
  s = content("Model", "03 · THE MODEL", "ClearSAR-Net: a U-Net that can look across the whole patch");
  const blocks = [["enc1", "64 ch\n256²", 1.6], ["enc2", "128 ch\n128²", 1.35], ["enc3", "256 ch\n64²", 1.1], ["bottleneck", "512 ch\n32²", 0.9]];
  let bx = M;
  for (let i = 0; i < blocks.length; i++) {
    const [n, d, hh] = blocks[i];
    const y = 3.55 - hh;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: 1.05, h: hh * 2, rectRadius: 0.08, fill: { color: C.accent2, transparency: 25 }, line: { color: C.accent2, width: 0.75 }, objectName: "enc_" + n });
    T(s, n, { x: bx, y: 3.25, w: 1.05, h: 0.3, fontSize: 12, bold: true, color: C.text1, align: "center" });
    T(s, d, { x: bx, y: 3.55, w: 1.05, h: 0.55, fontSize: 10, color: C.text1, align: "center" });
    bx += 1.25;
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: 2.25, w: 2.3, h: 2.6, rectRadius: 0.12, fill: { color: C.accent3, transparency: 10 }, line: { color: C.accent3, width: 0.75 }, objectName: "attn_block" });
  T(s, "2 × global self-attention", { x: bx + 0.1, y: 2.45, w: 2.1, h: 0.6, fontSize: 14, bold: true, color: C.text1, align: "center" });
  T(s, "1,024 tokens, 8 heads\n2-D sine position code\nfp32 stream + LayerNorm", { x: bx + 0.1, y: 3.1, w: 2.1, h: 1.2, fontSize: 11, color: C.text1, align: "center" });
  bx += 2.5;
  for (let i = 2; i >= 0; i--) {
    const [n, d, hh] = blocks[i];
    const y = 3.55 - hh;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: 1.05, h: hh * 2, rectRadius: 0.08, fill: { color: C.accent1, transparency: 25 }, line: { color: C.accent1, width: 0.75 }, objectName: "dec_" + i });
    T(s, "dec" + (i + 1), { x: bx, y: 3.25, w: 1.05, h: 0.3, fontSize: 12, bold: true, color: C.text1, align: "center" });
    T(s, d, { x: bx, y: 3.55, w: 1.05, h: 0.55, fontSize: 10, color: C.text1, align: "center" });
    bx += 1.25;
  }
  T(s, "skip connections carry fine detail from each encoder level to the decoder", { x: M, y: 5.35, w: 9.5, h: 0.35, fontSize: 12, color: C.background2 });
  const feats = [["Attention where it is cheap", "at 32 × 32 every region of the 2.56 km patch can use evidence from every other - what made the ViT best under thick cloud."],
    ["Bilinear upsampling", "instead of transposed convolutions: no checkerboard artefacts for the GAN to amplify."],
    ["Full 256 × 256 patches", "in training and test - no tiling, the whole context is visible."]];
  for (let i = 0; i < 3; i++) {
    T(s, [{ text: feats[i][0] + " - ", options: { bold: true } }, { text: feats[i][1], options: { color: C.background2 } }],
      { x: M + i * 4.1, y: 5.85, w: 3.85, h: 1.1, fontSize: 13 });
  }
  s.addNotes("The generator is a U-Net - three levels down, three up, with skip connections. The change is in the bottleneck: at 32 by 32 there are 1,024 positions, and two Transformer blocks let every position attend to every other. That gives the global context that made the Vision Transformer best under thick cloud in Phase 1, at a cost that fits a T4. Two engineering details: bilinear upsampling avoids checkerboard artefacts, and the attention stream runs in float32 with a final LayerNorm - that fixed a crash, I come back to it.");

  // ================================================================ 8. gate
  s = content("Model", "03 · THE KEY IDEA", "The cloud gate: keep what is visible, rebuild the rest");
  panel(s, M, 1.75, W - 2 * M, 1.0, "gate_formula");
  T(s, [{ text: "output = ", options: { color: C.background2 } }, { text: "m", options: { bold: true, color: C.accent4 } },
    { text: " · (input + learned colour correction)  +  (1 − ", options: { color: C.background1 } }, { text: "m", options: { bold: true, color: C.accent4 } },
    { text: ") · (generated from radar + context)", options: { color: C.background1 } }],
    { x: M + 0.3, y: 1.95, w: W - 2 * M - 0.6, h: 0.6, fontSize: 18, fontFace: THEME.headFontFace, align: "center", valign: "middle" });
  img(s, "PT81_cloudy.png", M, 3.15, 2.6, 2.6, "Cloudy input, Portugal, 81 % cloud", "g_in");
  img(s, "PT81_gate.png", M + 2.85, 3.15, 2.6, 2.6, "Gate map m: bright in the cloud gap, where the input is kept; dark under cloud", "g_gate");
  img(s, "PT81_final.png", M + 5.7, 3.15, 2.6, 2.6, "Reconstruction of the same patch", "g_out");
  caption(s, "cloudy input (Portugal, 81 %)", M, 5.82, 2.6, "center"); caption(s, "gate m (bright = keep input)", M + 2.85, 5.82, 2.6, "center"); caption(s, "result", M + 5.7, 5.82, 2.6, "center");
  stat(s, "51 % → 15 %", "share of pixels kept, from < 10 % cloud to 90-100 % cloud", 9.25, 3.15, 3.48, C.accent4, "g1");
  stat(s, "ρ = −0.84", "gate vs cloud cover - learned without any cloud labels", 9.25, 4.75, 3.48, C.accent1, "g2");
  s.addNotes("This is the idea I am most proud of. The network outputs a gate m between 0 and 1 for every pixel. Where m is high it keeps the cloudy input - plus a learned colour correction for the date shift. Where m is low it generates the pixel from radar and context. Nobody told it where the clouds are: on the test set, the share of pixels it keeps falls from 51 % on clear patches to 15 % under full cloud, Spearman minus 0.84. It learned cloud detection from the reconstruction loss alone. In the picture: the gate is bright exactly in the gap between the clouds, and dark under them.");

  // ================================================================ 9. two-part training
  s = content("Model", "03 · TRAINING", "Two parts: first accurate, then realistic");
  const tl = [[0, 0.55, "Part A · reconstruction", "5.96 h · steps 0 - 47,940", "loss = L1 + 0.05 · (1 − SSIM)", "saves the model 'before GAN'", C.accent2],
    [0.55, 0.45, "Part B · PatchGAN refinement", "4.88 h · steps 47,940 - 86,380", "+ adversarial loss, adaptive weight", "saves the final model", C.accent3]];
  const TW = W - 2 * M;
  for (const [st, fr, ttl, when, loss, save, col] of tl) {
    const x = M + st * TW, w = fr * TW - 0.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.85, w, h: 0.62, rectRadius: 0.1, fill: { color: col }, line: { color: col, width: 0 }, objectName: "tl_" + st });
    T(s, ttl, { x: x + 0.2, y: 1.95, w: w - 0.4, h: 0.42, fontSize: 16, bold: true, color: C.text1, valign: "middle" });
    panel(s, x, 2.65, w, 1.75, "tl_panel" + st);
    T(s, [{ text: when, options: { bold: true, breakLine: true } }, { text: loss, options: { color: C.background2, breakLine: true } }, { text: save, options: { color: C.background2 } }],
      { x: x + 0.25, y: 2.85, w: w - 0.5, h: 1.4, fontSize: 14, paraSpaceAfter: 6 });
  }
  const tips = [["FaScaleBalanced", "SSIM weight measured", "Its gradient is 14 × L1's - 0.2 would have dominated; 0.05 balances them."],
    ["FaSliders", "Adaptive GAN weight λ", "Set so the GAN's gradient is 20 % of the pixel loss's, whatever the loss scales (VQGAN)."],
    ["FaBullseye", "Selection that matches the goal", "Final checkpoint = lowest L1 × (1 + |1 − sharpness|), only once the GAN is fully on."]];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1;
    await iconDot(s, tips[i][0], x, 4.95, 0.6, C.accent1, "tip" + i);
    T(s, tips[i][1], { x: x + 0.8, y: 4.98, w: 3.15, h: 0.4, fontSize: 15, bold: true });
    T(s, tips[i][2], { x: x + 0.8, y: 5.42, w: 3.15, h: 1.2, fontSize: 13, color: C.background2 });
  }
  s.addNotes("Training runs on a time budget: 55 % reconstruction, 45 % with the GAN. Starting the GAN from a converged model means the discriminator gives useful advice instead of noise, and it gives a clean before-and-after of the same network. Three choices are measured rather than guessed: the SSIM weight, from a gradient measurement; the GAN weight, adapted so the GAN always pushes one fifth as hard as the pixel loss; and the checkpoint selection, which rewards accuracy and sharpness together - the fix for Phase 1's GAN.");

  // ================================================================ 10. GAN
  s = content("Model", "03 · THE GAN PART", "A PatchGAN critic judges texture, 70 × 70 pixels at a time");
  const gan = [["FaEye", "Conditional PatchGAN", "Sees radar + cloudy input next to a candidate clear image; scores 30 × 30 overlapping windows: is this window a plausible clear image for this input?"],
    ["FaShieldHalved", "Spectral normalisation", "Every layer is limited to a gain of 1 - the discriminator changes smoothly and gives the generator usable gradients."],
    ["FaSquareRootVariable", "Least-squares loss (LSGAN)", "Real → 1, generated → 0; gradients do not vanish when the critic is confident."]];
  for (let i = 0; i < 3; i++) {
    const y = 1.8 + i * 1.6;
    await iconDot(s, gan[i][0], M, y, 0.65, C.accent3, "gan" + i);
    T(s, gan[i][1], { x: M + 0.9, y: y - 0.02, w: 5.0, h: 0.4, fontSize: 16, bold: true });
    T(s, gan[i][2], { x: M + 0.9, y: y + 0.4, w: 5.0, h: 1.1, fontSize: 13, color: C.background2 });
  }
  panel(s, 6.75, 1.75, 5.98, 4.85, "gan_chart_panel");
  s.addChart(pres.charts.LINE, [{ name: "D(real image)", labels: DATA.gan_h.map((v) => v.toFixed(1)), values: DATA.d_real },
    { name: "D(generated image)", labels: DATA.gan_h.map((v) => v.toFixed(1)), values: DATA.d_fake }],
    Object.assign({ x: 6.9, y: 1.9, w: 5.7, h: 4.55, chartColors: [HEX.accent1, HEX.accent5], lineSize: 2.25, lineDataSymbol: "none",
      showLegend: true, legendPos: "b", showTitle: true, title: "Discriminator scores during part B (hours)", valAxisMinVal: 0, valAxisMaxVal: 1,
      valAxisMajorUnit: 0.2, catAxisLabelFrequency: 20, objectName: "gan_chart" }, chartText));
  s.addNotes("The discriminator is a conditional PatchGAN with spectral normalisation and a least-squares loss - the standard recipe for stable GAN training. The chart is the game itself: the score for real images rose to about 0.86, for generated ones it fell to about 0.14. The critic became strong but never saturated at exactly 1 and 0 - had it, the generator would get no useful signal. The adaptive weight shrank automatically as the critic grew stronger, keeping its push at 20 %.");

  // ================================================================ 11. engineering
  s = content("Model", "03 · ENGINEERING", "11 hours on two Kaggle T4 GPUs - built not to fail");
  const eng = [["FaMicrochip", "Two GPUs, one model", "DistributedDataParallel, mixed precision (fp16), batch size probed to fit the 15 GB."],
    ["FaHardDrive", "No 450 GB cache", "GeoTIFFs streamed with parallel reads; when the disk lags, recent patches are re-used with new flips (52 % fresh)."],
    ["FaClock", "Time-based schedule", "Learning rate and parts defined in hours, so the run always ends inside Kaggle's 12-hour limit."],
    ["FaTriangleExclamation", "The NaN incident", "The first full run went NaN at step ~4,900: an fp16 overflow in the attention stream. Fixed with fp32 + LayerNorm."],
    ["FaLifeRing", "Automatic NaN guard", "Snapshots every 1,000 steps; rolls back and halves the LR if training breaks. Final run: no events."],
    ["FaVial", "Tested before using GPU time", "60 unit tests, synthetic end-to-end runs, a deliberately broken run to test the guard."]];
  for (let i = 0; i < 6; i++) {
    const x = M + (i % 3) * 4.1, y = 1.8 + Math.floor(i / 3) * 2.45;
    panel(s, x, y, 3.85, 2.2, "eng" + i, C.text2, i === 3 ? C.accent5 : C.accent1);
    await iconDot(s, eng[i][0], x + 0.25, y + 0.25, 0.6, i === 3 ? C.accent5 : C.accent1, "engi" + i);
    T(s, eng[i][1], { x: x + 1.0, y: y + 0.3, w: 2.7, h: 0.5, fontSize: 15, bold: true, valign: "middle" });
    T(s, eng[i][2], { x: x + 0.25, y: y + 1.0, w: 3.4, h: 1.1, fontSize: 12.5, color: C.background2 });
  }
  s.addNotes("A one-shot 12-hour run has to be engineered not to fail. The most instructive story is the red card: the first full run silently went NaN after about 5,000 steps and wasted 3.5 hours - half-precision overflow in the attention bottleneck. I found it from the log, fixed the cause with a float32 stream and a LayerNorm, and added an automatic guard that rolls back if it ever happens again. The final run had no events. Data echoing handled a slow disk: only about half the batches came fresh from disk, and the run still made 6 complete passes over the 102,871 training patches.");

  // ================================================================ 12. training curves
  s = content("Model", "03 · HOW IT LEARNED", "Accuracy first, then sharpness");
  panel(s, M, 1.75, 8.2, 4.95, "curve_panel");
  const hrs = DATA.val_h.map((v) => v.toFixed(1));
  s.addChart(pres.charts.LINE, [{ name: "validation PSNR (dB)", labels: hrs, values: DATA.val_psnr }],
    Object.assign({ x: M + 0.15, y: 1.9, w: 7.9, h: 2.35, chartColors: [HEX.accent1], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5,
      showLegend: false, showTitle: true, title: "Validation PSNR (dB)", valAxisMinVal: 25, valAxisMaxVal: 30, valAxisMajorUnit: 1, catAxisLabelFrequency: 4, objectName: "val_psnr" }, chartText));
  s.addChart(pres.charts.LINE, [{ name: "validation sharpness", labels: hrs, values: DATA.val_sharp }],
    Object.assign({ x: M + 0.15, y: 4.3, w: 7.9, h: 2.3, chartColors: [HEX.accent4], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 5,
      showLegend: false, showTitle: true, title: "Validation sharpness (1 = as sharp as reality) · x-axis: hours", valAxisMinVal: 0.3, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.1, catAxisLabelFrequency: 4, objectName: "val_sharp" }, chartText));
  stat(s, "29.48 dB", "best validation PSNR in part A (hour 4.1)", 9.1, 1.75, 3.63, C.accent1, "c1");
  stat(s, "0.64 → 0.71", "validation sharpness jumps when the GAN joins (hour 6.0)", 9.1, 3.35, 3.63, C.accent4, "c2");
  stat(s, "86,380", "training steps · 6 passes over the data", 9.1, 4.95, 3.63, C.accent2, "c3");
  s.addNotes("Validation every 20 minutes on 10 unseen scenes. PSNR rises to 29 dB within the first hour and peaks at 29.5. Sharpness - edge energy compared with the real image - creeps up slowly under the pixel loss and then jumps when the GAN joins at hour six. PSNR eases by only a few tenths: the trade-off, measured live.");

  // ================================================================ 13. results headline
  pres.addSection({ title: "Results" });
  s = content("Results", "04 · RESULTS", "One honest test: 5,400 patches from 18 unseen scenes");
  stat(s, "28.59 dB", "PSNR of the final model, all 13 bands", M, 1.75, 2.9, C.accent1, "r1");
  stat(s, "+1.89 dB", "over the Phase 1 U-Net · better in 17 of 18 scenes", M + 3.07, 1.75, 2.9, C.accent2, "r2");
  stat(s, "+8.3 dB", "over the cloudy input itself · 18 of 18 scenes", M + 6.14, 1.75, 2.9, C.accent3, "r3");
  stat(s, "34.53 dB", "PSNR with the date shift removed", M + 9.21, 1.75, 2.92, C.accent4, "r4");
  const S = DATA.summary;
  const hdr = { bold: true, color: C.background1, fill: { color: C.text2 } };
  const row = (name, r, b) => [{ text: name, options: { bold: b } }, ...["psnr", "ssim", "sam", "psnr_aligned", "sharpness"].map((k) =>
    ({ text: k === "ssim" ? S[r][k].toFixed(3) : S[r][k].toFixed(2), options: { bold: b, align: "right" } }))];
  s.addTable([[{ text: "Model", options: hdr }, ...["PSNR (dB)", "SSIM", "SAM (°)", "PSNR, date shift removed", "Sharpness"].map((t) => ({ text: t, options: Object.assign({ align: "right" }, hdr) }))],
    row("Cloudy input (no model)", "identity"), row("Phase 1 U-Net (3,000 Asian patches)", "p1_unet"),
    row("ClearSAR-Net before GAN", "before_gan"), row("ClearSAR-Net final (GAN)", "final", true)],
    { x: M, y: 3.5, w: W - 2 * M, colW: [4.0, 1.5, 1.3, 1.3, 2.5, 1.53], fontSize: 14, color: C.background1, fill: { color: C.text1 },
      border: { type: "solid", pt: 0.75, color: "26325A" }, rowH: 0.48, margin: 0.08, objectName: "results_table" });
  T(s, [{ text: "Scene bootstrap (18 scenes, 5,000 resamples): +1.89 dB over Phase 1 has a 95 % interval of +1.1 to +2.7 dB.", options: { breakLine: true } },
    { text: "Sharpness = edge energy relative to the true image (1 = as sharp as reality); above 1 means extra edges - the cloudy input scores 1.88 because cloud edges are sharp." }],
    { x: M, y: 6.05, w: W - 2 * M, h: 0.75, fontSize: 13, color: C.background2, paraSpaceAfter: 4 });
  s.addNotes("The test is run once, at the end, on 18 scenes from every continent that were never seen. Same patches for every row. The final model reaches 28.6 dB - 8.3 above the cloudy input, and 1.9 above Phase 1's best model, better in 17 of the 18 scenes; a scene bootstrap gives a 95 % interval of 1.1 to 2.7 dB, so this is not noise. SSIM 0.88, spectral angle down from 13 to 8 degrees. With the date shift removed: 34.5 dB.");

  // ================================================================ 14. by cloud
  s = content("Results", "04 · WHERE IT IS HARD", "The more cloud, the more the network adds");
  panel(s, M, 1.75, 8.4, 5.0, "cloud_panel");
  const lbl = { identity: "Cloudy input", p1_unet: "Phase 1 U-Net", before_gan: "Before GAN", final: "Final (GAN)" };
  s.addChart(pres.charts.LINE, Object.keys(lbl).map((r) => ({ name: lbl[r], labels: DATA.cloud_bins, values: DATA.cloud[r] })),
    Object.assign({ x: M + 0.15, y: 1.9, w: 8.1, h: 4.7, chartColors: [HEX.lt2, HEX.accent3, HEX.accent4, HEX.accent1], lineSize: 2.75,
      lineDataSymbol: "circle", lineDataSymbolSize: 7, showLegend: true, legendPos: "b", showTitle: true, title: "Test PSNR (dB) by cloud cover of the patch",
      valAxisMinVal: 10, valAxisMaxVal: 32, valAxisMajorUnit: 5, objectName: "cloud_chart" }, chartText));
  stat(s, "+14.8 dB", "over the cloudy input at 90-100 % cloud (+2.7 dB at 0-10 %)", 9.3, 1.75, 3.43, C.accent1, "b1");
  stat(s, "+3.2 dB", "over Phase 1 under full cloud - global attention and scene diversity pay off most here", 9.3, 3.35, 3.43, C.accent3, "b2");
  T(s, "Why is 90-100 % higher than 60-90 %? Scene mix, not physics: that bin holds easy scenes (Canada, Central African Rep., New Zealand).",
    { x: 9.3, y: 5.0, w: 3.43, h: 1.7, fontSize: 12.5, color: C.background2 });
  s.addNotes("By cloud cover: the grey line is the cloudy input - it collapses as cloud grows, by definition. The model stays between 27 and 30 dB. The gain over Phase 1 is largest under full cloud, plus 3.2 dB - exactly where global attention and the bigger, more diverse training set matter. If someone asks why the last bin goes up again: it is the scene mix - that bin happens to contain easy scenes; the desert scene scores the same in both bins.");

  // ================================================================ 15. GAN effect
  s = content("Results", "04 · WHAT THE GAN CHANGED", "Sharper, with no measurable loss of accuracy");
  stat(s, "0.78 → 0.87", "sharpness before → after the GAN · better in 16 of 18 scenes", M, 1.75, 4.0, C.accent4, "ge1");
  stat(s, "−0.09 dB", "PSNR change · 95 % interval −0.34 to +0.16 dB (contains 0)", M, 3.35, 4.0, C.accent1, "ge2");
  T(s, "Phase 1's GAN lost 1.35 dB. Starting from a converged model and capping the GAN's push at 20 % made the difference. By eye the change is subtle: crisper edges, slightly more texture.",
    { x: M, y: 5.05, w: 4.0, h: 1.6, fontSize: 13, color: C.background2 });
  img(s, "CN2_zoom_before_gan.png", 5.05, 1.8, 2.4, 2.4, "Zoomed crop before the GAN part", "z_before");
  img(s, "CN2_zoom_final.png", 7.7, 1.8, 2.4, 2.4, "Same crop after the GAN part", "z_final");
  img(s, "CN2_zoom_truth.png", 10.35, 1.8, 2.4, 2.4, "Same crop of the true clear image", "z_truth");
  caption(s, "before GAN", 5.05, 4.27, 2.4, "center"); caption(s, "final (GAN)", 7.7, 4.27, 2.4, "center"); caption(s, "true image", 10.35, 4.27, 2.4, "center");
  panel(s, 5.05, 4.9, 7.7, 1.75, "ge_note");
  T(s, [{ text: "The perception-distortion trade-off, measured: ", options: { bold: true } },
    { text: "realism and pixel accuracy pull in different directions. A stronger GAN would sharpen further - and risk inventing detail. That is why the expert system recommends the before-GAN model for measurements.", options: { color: C.background2 } }],
    { x: 5.3, y: 5.05, w: 7.2, h: 1.5, fontSize: 13.5 });
  s.addNotes("The GAN did exactly what it was designed for. Sharpness rose from 0.78 to 0.87 - significant, 16 of 18 scenes. PSNR moved by minus 0.09 dB, and the interval contains zero, so there is no measurable accuracy cost - unlike Phase 1's GAN, which lost 1.35 dB. Be honest about the picture: the difference is subtle by eye. And it is a trade-off: that is why, for scientific use, the advisor picks the before-GAN model.");

  // ================================================================ 16. gallery
  s = content("Results", "04 · WHAT THE IMAGES SHOW", "Representative test patches - not the best ones");
  const gal = [["CN2", "China · 2 % cloud", "near-perfect; colours shifted to the target date"], ["FR31", "France · 31 %", "clouds and shadows gone; detail soft"],
    ["PT81", "Portugal · 81 %", "layout and colours from radar; fine detail lost"], ["TM99", "Turkmenistan · 99 %", "only the large-scale pattern survives"]];
  const gx = M + 1.95, gw = 1.17, gs = 1.3;
  ["cloudy input", "ClearSAR-Net", "true image"].forEach((t, k) => T(s, t, { x: gx + k * gs - 0.1, y: 1.55, w: gw + 0.2, h: 0.3, fontSize: 12, bold: true, color: C.background2, align: "center" }));
  for (let i = 0; i < 4; i++) {
    const [tag, lab, note] = gal[i];
    const y = 1.9 + i * 1.28;
    T(s, lab, { x: M, y: y + 0.25, w: 1.85, h: 0.6, fontSize: 13, bold: true });
    ["cloudy", "final", "truth"].forEach((k, j) => s.addImage({ path: A(`${tag}_${k}.png`), x: gx + j * gs, y, w: gw, h: gw,
      altText: `${lab}: ${k}`, objectName: `gal_${tag}_${k}` }));
    T(s, note, { x: gx + 3 * gs + 0.1, y: y + 0.2, w: 2.45, h: 0.8, fontSize: 13, color: C.background2 });
  }
  panel(s, 9.15, 1.9, 3.58, 4.9, "gal_panel");
  T(s, [{ text: "Honest reading", options: { bold: true, breakLine: true } },
    { text: "Light cloud: close to the truth.", options: { color: C.background2, breakLine: true } },
    { text: "Heavy cloud: plausible layout and colour, no fine detail - 10 m radar does not contain it.", options: { color: C.background2, breakLine: true } },
    { text: "Some 'errors' are real changes between the two dates that no model can know.", options: { color: C.background2 } }],
    { x: 9.4, y: 2.1, w: 3.1, h: 4.5, fontSize: 14, paraSpaceAfter: 10 });
  s.addNotes("Per cloud level I show the patch closest to the median score - representative, not cherry-picked. Under light cloud it is close to the truth, even correcting colours towards the target date. Under heavy cloud it recovers layout and colour from radar but not fine detail - that information is not in a 10-metre radar image. Say this before anyone asks; it shows you understand the limits.");

  // ================================================================ 17. expert system
  pres.addSection({ title: "Expert system" });
  s = content("Expert system", "05 · EXPLAINABLE DECISION SUPPORT", "The ClearSAR Advisor: should you trust this reconstruction?");
  const flow = [["FaMagnifyingGlass", "Facts", "measured on the patch and the model: cloud cover, cirrus, open water, radar reliance (occlusion), model disagreement, flip-test uncertainty, test-set experience per cloud level", C.accent2],
    ["FaListCheck", "38 IF-THEN rules", "knowledge base in YAML, five groups: observations → trust → model → suitability → action; MYCIN certainty factors", C.accent3],
    ["FaCircleCheck", "Decision + explanation", "trust · recommended model · fit for purpose · action - with how, why-not and visual evidence", C.accent1]];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.15;
    panel(s, x, 1.8, 3.8, 2.5, "flow" + i, C.text2, flow[i][3]);
    await iconDot(s, flow[i][0], x + 0.25, 2.0, 0.65, flow[i][3], "flowi" + i);
    T(s, flow[i][1], { x: x + 1.05, y: 2.1, w: 2.6, h: 0.45, fontSize: 17, bold: true });
    T(s, flow[i][2], { x: x + 0.25, y: 2.85, w: 3.35, h: 1.7, fontSize: 12.5, color: C.background2 });
    if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 3.83, y: 2.85, w: 0.3, h: 0.4, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "flowarrow" + i });
  }
  T(s, "Example decisions on real test patches", { x: M, y: 4.7, w: 8, h: 0.4, fontSize: 15, bold: true });
  const ex = [["Visual inspection", "→ final (GAN) model", "rule M5: the test shows the GAN costs < 0.3 dB at this cloud level"],
    ["Vegetation monitoring", "→ before-GAN model", "rules M3, M4: measurements need faithful pixels, not invented texture"],
    ["Flood mapping", "→ radar-only products", "radar detects open water directly"]];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.15;
    T(s, [{ text: ex[i][0] + " ", options: { bold: true } }, { text: ex[i][1], options: { bold: true, color: C.accent1, breakLine: true } }, { text: ex[i][2], options: { color: C.background2, fontSize: 12 } }],
      { x, y: 5.2, w: 3.85, h: 1.4, fontSize: 14 });
  }
  s.addNotes("The second project: a classic, explainable expert system on top of the deep-learning output. Facts are measured from the patch and the model - including how much it relied on radar, found by occlusion, and how uncertain it is, from a flip test. 38 rules with MYCIN certainty factors chain from observations to a decision. Every decision explains itself: which rules fired, which did not and why. Verification on real patches found two flaws - the GAN model was never recommended, and one cloud situation had no trust rule - both fixed with evidence-based rules M5 and T13.");

  // ================================================================ 18. demo
  s = content("Expert system", "05 · LIVE DEMO", "Everything is in an interactive web app");
  img(s, "app_home_crop.png", M, 1.8, 7.3, 5.0, "Screenshot of the ClearSAR web app home page", "app_shot");
  const demo = [["Home", "drag the slider: cloudy → reconstruction on the Indian test patch"], ["Results", "metrics by cloud cover, continent, country; the GAN's effect"],
    ["How it learned", "training curves and the GAN game, from the real log"], ["Explore patches", "every model, gate and uncertainty map"],
    ["Advisor", "a decision with its reasoning, live"], ["Knowledge base", "what-if simulator - move a slider, watch a rule fire"]];
  for (let i = 0; i < demo.length; i++) {
    const y = 1.8 + i * 0.82;
    s.addShape(pres.shapes.OVAL, { x: 8.3, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "demo_n" + i });
    T(s, String(i + 1), { x: 8.3, y: y + 0.07, w: 0.42, h: 0.38, fontSize: 13, bold: true, color: C.text1, align: "center", valign: "middle" });
    T(s, [{ text: demo[i][0] + "  ", options: { bold: true } }, { text: demo[i][1], options: { color: C.background2, fontSize: 12.5 } }], { x: 8.9, y, w: 3.85, h: 0.75, fontSize: 14 });
  }
  T(s, "Runs offline: all results and both trained models are in the repository.", { x: 8.3, y: 6.62, w: 4.45, h: 0.4, fontSize: 11, color: C.background2 });
  s.addNotes("Switch to the browser here - start the app before the talk and open the Advisor page once, so its analyses are cached. Suggested 3-minute path: Home slider on the Indian patch; Results by cloud cover; Explore one heavily clouded patch with its gate map; Advisor on the same patch, switching the intended use from visual inspection to vegetation monitoring to show the recommendation change; finish in the Knowledge base what-if simulator. If the demo fails, the screenshot on this slide is the fallback.");

  // ================================================================ 19. limitations
  pres.addSection({ title: "Closing" });
  s = content("Closing", "06 · LIMITS AND NEXT STEPS", "What this model cannot do - yet");
  const lim = [["Detail under thick cloud", "10 m radar does not contain fine optical detail; the model infers layout and colour only."],
    ["The date shift caps PSNR", "the 'clear' target is another day; 28.6 dB plain vs 34.5 dB date-shift-free."],
    ["One training run", "no seed spread for the final model; the scene bootstrap partly compensates."],
    ["Uncertainty calibrated on the test set", "fine for a demo; a deployment needs separate calibration data."]];
  const nxt = [["Time series", "use several earlier acquisitions, not one cloudy image (e.g. SEN12MS-CR-TS)."],
    ["Stronger, measured GAN", "tune the 20 % push with a perceptual metric such as LPIPS."],
    ["Learned uncertainty head", "Phase 1's best-calibrated method, trained jointly with the GAN."],
    ["Official benchmark split", "for a direct comparison with published models."]];
  T(s, "Limitations", { x: M, y: 1.7, w: 5.9, h: 0.4, fontSize: 18, bold: true, color: C.accent5 });
  T(s, "Next steps", { x: 6.95, y: 1.7, w: 5.8, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  for (let i = 0; i < 4; i++) {
    const y = 2.25 + i * 1.15;
    panel(s, M, y, 5.9, 1.0, "lim" + i, C.text2, C.accent5);
    T(s, [{ text: lim[i][0], options: { bold: true, breakLine: true } }, { text: lim[i][1], options: { color: C.background2, fontSize: 12.5 } }], { x: M + 0.25, y: y + 0.1, w: 5.45, h: 0.85, fontSize: 14 });
    panel(s, 6.95, y, 5.78, 1.0, "nxt" + i);
    T(s, [{ text: nxt[i][0], options: { bold: true, breakLine: true } }, { text: nxt[i][1], options: { color: C.background2, fontSize: 12.5 } }], { x: 7.2, y: y + 0.1, w: 5.33, h: 0.85, fontSize: 14 });
  }
  s.addNotes("Close the technical part with honest limits - examiners respect this. The biggest physical limit: radar cannot supply fine optical detail. The biggest measurement limit: the date shift. Each limitation has a concrete next step on the right; time series is the most promising, because earlier clear images contain exactly the detail radar lacks.");

  // ================================================================ 20. takeaways
  s = pres.addSlide({ masterName: "CS_TITLE", sectionTitle: "Closing" });
  T(s, "TAKEAWAYS", { x: M, y: 0.9, w: 6, h: 0.35, fontSize: 12, bold: true, color: C.accent1, charSpacing: 3 });
  T(s, "One model, built from evidence", { x: M, y: 1.3, w: 11, h: 0.8, fontSize: 36, bold: true, fontFace: THEME.headFontFace });
  const tk = [["FaDatabase", "Data diversity beat architecture", "all 147 scenes: +1.9 dB over the best Phase 1 model, 17 of 18 test scenes."],
    ["FaWandMagicSparkles", "A gate that learned cloud detection by itself", "keeps 51 % → 15 % of the input as cloud grows (ρ = −0.84)."],
    ["FaScaleBalanced", "A GAN that sharpens without costing accuracy", "sharpness 0.78 → 0.87, PSNR −0.09 dB (within noise)."],
    ["FaCircleCheck", "An expert system that knows when to trust it", "38 explainable rules; different advice for viewing and for measuring."]];
  for (let i = 0; i < 4; i++) {
    const y = 2.4 + i * 0.98;
    await iconDot(s, tk[i][0], M, y, 0.62, [C.accent2, C.accent4, C.accent3, C.accent1][i], "tk" + i);
    T(s, [{ text: tk[i][1] + "  ", options: { bold: true } }, { text: tk[i][2], options: { color: C.background2 } }], { x: M + 0.9, y: y + 0.05, w: 10.5, h: 0.6, fontSize: 16, valign: "middle" });
  }
  T(s, "Thank you - questions?", { x: M, y: 6.45, w: 8, h: 0.6, fontSize: 24, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  s.addNotes("Four takeaways, one sentence each - then stop talking and invite questions. If there is time, offer to show any page of the app again. The backup slides after this one hold the hyper-parameters, the Phase 1 numbers and short answers to likely questions.");

  // ================================================================ backup
  pres.addSection({ title: "Backup" });
  s = content("Backup", "BACKUP", "Hyper-parameters of the final run");
  const hp = [["Generator", "ClearSAR-Net, width 64, 3 levels, 2 attention blocks (8 heads), hybrid output · 12.78 M"], ["Discriminator", "conditional SN-PatchGAN, 70 × 70 windows · 2.79 M"],
    ["Batch", "8 per GPU × 2 T4 = 16 (probed)"], ["Budget", "10.84 h, time-based: part A 55 %, part B 45 % · 86,380 steps"],
    ["Loss A / B", "L1 + 0.05 (1 − SSIM) / + λ · LSGAN, λ adaptive (factor 0.2)"], ["Optimiser G", "AdamW β (0.9, 0.999), wd 1e-4, clip 1.0; LR 3e-4 warm-up 1,000 + cosine; part B 5e-5"],
    ["Optimiser D", "Adam β (0.5, 0.999), LR 2e-4 cosine"], ["EMA / precision", "0.999 / fp16 + GradScaler, fp32 attention stream"],
    ["Data", "102,871 / 600 / 5,400 patches; 8 dihedral augmentations; echo pool 256"], ["Selection", "A: lowest val L1 · B: lowest L1 × (1 + |1 − sharpness|)"]];
  s.addTable(hp.map(([k, v]) => [{ text: k, options: { bold: true, color: C.accent1 } }, { text: v }]),
    { x: M, y: 1.75, w: W - 2 * M, colW: [2.6, W - 2 * M - 2.6], fontSize: 13, color: C.background1, fill: { color: C.text1 },
      border: { type: "solid", pt: 0.75, color: "26325A" }, rowH: 0.47, margin: 0.08, objectName: "hp_table" });
  s.addNotes("Backup - only if asked.");

  s = content("Backup", "BACKUP", "Phase 1 numbers (Asian test scenes)");
  const p1 = [["Cloudy input", "21.18", "0.740"], ["Tiny CNN (baseline)", "27.01", "0.844"], ["U-Net", "26.33 ± 0.06", "0.847"], ["Swin U-Net", "26.41 ± 0.20", "0.851"],
    ["Pure ViT", "26.90 ± 0.36", "0.844"], ["U-Net + PatchGAN", "25.00", "0.816"], ["U-Net 3-seed ensemble", "26.56", "0.857"]];
  s.addTable([[{ text: "Model", options: hdr }, { text: "PSNR (dB)", options: Object.assign({ align: "right" }, hdr) }, { text: "SSIM", options: Object.assign({ align: "right" }, hdr) }],
    ...p1.map(([a, b, c]) => [{ text: a }, { text: b, options: { align: "right" } }, { text: c, options: { align: "right" } }])],
    { x: M, y: 1.75, w: 6.6, colW: [3.4, 1.8, 1.4], fontSize: 14, color: C.background1, fill: { color: C.text1 }, border: { type: "solid", pt: 0.75, color: "26325A" }, rowH: 0.5, margin: 0.08, objectName: "p1_table" });
  panel(s, 7.6, 1.75, 5.13, 4.0, "p1_notes");
  T(s, [{ text: "Other Phase 1 findings", options: { bold: true, breakLine: true } }, { text: "Random split inflates PSNR by +5.1 dB.", options: { color: C.background2, breakLine: true } },
    { text: "ViT +2-4 dB at 60-100 % cloud (receptive field 56 px vs 21 px).", options: { color: C.background2, breakLine: true } },
    { text: "Variance head: best-calibrated uncertainty (83-89 % coverage).", options: { color: C.background2, breakLine: true } },
    { text: "No skip connections: SSIM −0.010, FID +106.", options: { color: C.background2 } }], { x: 7.85, y: 1.95, w: 4.65, h: 3.6, fontSize: 14, paraSpaceAfter: 10 });
  s.addNotes("Backup - only if asked.");

  s = content("Backup", "BACKUP", "Short answers to likely questions");
  const qa = [["Why not a pure GAN?", "GANs add texture but not accuracy; as a second part they sharpen without measurable PSNR loss."],
    ["Why is PSNR only 28.6 dB?", "The target is another day and radar lacks fine detail; 34.5 dB with the date shift removed."],
    ["Is +1.9 dB real?", "Scene bootstrap 95 % CI +1.1 to +2.7 dB; better in 17 of 18 scenes."],
    ["How does the gate know about clouds?", "It does not - it learned from the loss alone (ρ = −0.84 with cloud cover)."],
    ["Why certainty factors?", "Easy to elicit, combine and explain; a Bayesian network would be the principled extension."],
    ["What went wrong?", "fp16 overflow after ~5,000 steps in run 1 - fixed with fp32 + LayerNorm and a NaN guard."]];
  for (let i = 0; i < 6; i++) {
    const x = M + (i % 2) * 6.17, y = 1.75 + Math.floor(i / 2) * 1.62;
    panel(s, x, y, 5.95, 1.45, "qa" + i);
    T(s, [{ text: qa[i][0], options: { bold: true, color: C.accent1, breakLine: true } }, { text: qa[i][1], options: { color: C.background2 } }], { x: x + 0.25, y: y + 0.15, w: 5.45, h: 1.2, fontSize: 14 });
  }
  s.addNotes("Backup - only if asked.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
