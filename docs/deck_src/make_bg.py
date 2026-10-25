import numpy as np
from PIL import Image
W, H = 2667, 1500
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.array([10, 16, 34], np.float32)
def glow(cx, cy, rx, ry, col, a):
    d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    return np.exp(-d)[..., None] * (np.array(col, np.float32) - base) * a
for name, glows in {
    "bg_title": [(0.05, -0.05, 0.55, 0.6, (45, 212, 191), .30), (0.98, 0.0, 0.5, 0.55, (167, 139, 250), .30), (0.6, 1.1, 0.6, 0.45, (56, 189, 248), .14)],
    "bg_content": [(0.0, -0.1, 0.45, 0.35, (45, 212, 191), .16), (1.0, -0.1, 0.4, 0.35, (167, 139, 250), .14)],
}.items():
    img = np.broadcast_to(base, (H, W, 3)).copy()
    for cx, cy, rx, ry, col, a in glows:
        img += glow(cx * W, cy * H, rx * W, ry * H, col, a)
    rng = np.random.default_rng(0)
    img += rng.normal(0, 1.2, img.shape)  # dither against banding
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(f"assets/{name}.jpg", quality=90)
print("ok")
