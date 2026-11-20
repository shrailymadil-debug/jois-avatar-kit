"""Offline orthographic preview of the last build: python3 preview.py out.png "jawOpen=1,eyeBlinkLeft=1" [zoom]"""
import sys, numpy as np
from PIL import Image
d = np.load(__file__.replace("preview.py", "last_build.npz"))
V, N, UV, F, D, names, tex = d["V"].astype(np.float64), d["N"], d["UV"], d["F"], d["D"], list(d["names"]), d["base"]
out = sys.argv[1]; spec = sys.argv[2] if len(sys.argv) > 2 else ""
for kv in filter(None, spec.split(",")):
    k, w = kv.split("="); V = V + float(w) * D[names.index(k)]
yaw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
c, s = np.cos(yaw), np.sin(yaw)
R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
cen = np.array([0, float(d["eye"][1]) - 0.035, 0.0]) if "eye" in d.files else np.array([0, 1.395, 0.0]); half = float(sys.argv[3]) if len(sys.argv) > 3 else 0.14
P = (V - cen) @ R.T; Nn = N @ R.T
W = H = int(sys.argv[5]) if len(sys.argv) > 5 else 700
X = (P[:, 0] / half * 0.5 + 0.5) * W; Y = (0.5 - P[:, 1] / half * 0.5) * H; Z = P[:, 2]
img = np.full((H, W, 3), 40, np.uint8); zb = np.full((H, W), -1e9)
L = np.array([0.35, 0.45, 0.82]); L /= np.linalg.norm(L)
th, tw = tex.shape[:2]
for f in F:
    xs, ys = X[f], Y[f]
    x0, x1 = int(max(0, xs.min())), int(min(W - 1, xs.max() + 1)); y0, y1 = int(max(0, ys.min())), int(min(H - 1, ys.max() + 1))
    if x1 < x0 or y1 < y0: continue
    den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
    if abs(den) < 1e-9: continue
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
    l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
    l2 = 1 - l0 - l1
    m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    if not m.any(): continue
    z = l0 * Z[f[0]] + l1 * Z[f[1]] + l2 * Z[f[2]]
    iy, ix = gy[m].astype(int), gx[m].astype(int); z = z[m]
    vis = z > zb[iy, ix]
    if not vis.any(): continue
    iy, ix, z = iy[vis], ix[vis], z[vis]
    lw = np.stack([l0[m][vis], l1[m][vis], l2[m][vis]], 1)
    uv = lw @ UV[f]; n = lw @ Nn[f]; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    col = tex[np.clip((uv[:, 1] * th).astype(int), 0, th - 1), np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)].astype(np.float64)
    sh = 0.35 + 0.75 * np.clip(n @ L, 0, 1)
    zb[iy, ix] = z; img[iy, ix] = np.clip(col * sh[:, None], 0, 255).astype(np.uint8)
Image.fromarray(img).save(out)
