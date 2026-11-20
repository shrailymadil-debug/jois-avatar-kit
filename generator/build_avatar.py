"""
Jois Avatar Kit — procedural avatar generator.

Builds a stylized real-time humanoid avatar entirely from code and writes a
single-material, single-draw-call skinned .glb that follows the Jois spec:
  * 15k–35k triangles
  * humanoid skeleton (Mixamo-style names, LeftEye/RightEye bones)
  * 52 ARKit blendshapes + 15 visemes (sil PP FF TH DD kk CH SS nn RR aa E ih oh ou)
  * one PBR metallic-roughness material with a 2K atlas (BaseColor, Normal, ORM)

Usage:  python3 build_avatar.py <out_dir>
"""
import io, json, math, struct, sys, os, time
import numpy as np
from PIL import Image
from scipy import sparse, ndimage
from skimage import measure
import fast_simplification

rng = np.random.default_rng(7)
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

# ----------------------------------------------------------------- helpers
def v3(*a): return np.array(a, dtype=np.float64)
def norm(a, axis=1): return np.linalg.norm(a, axis=axis)
def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def sd_sphere(P, c, r): return norm(P - c) - r
def sd_ell(P, c, r):
    q = (P - c) / r
    k0 = norm(q); k1 = norm((P - c) / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
def sd_oell(P, c, axes, r):
    q = (P - c) @ axes.T          # axes rows = local axes
    k0 = norm(q / r); k1 = norm(q / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
def sd_cap(P, a, b, ra, rb):
    pa = P - a; ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return norm(pa - np.outer(h, ba)) - (ra + (rb - ra) * h)
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)
def smax(a, b, k): return -smin(-a, -b, k)

def rot_about(P, center, axis, ang):
    axis = axis / np.linalg.norm(axis)
    q = P - center
    c, s = math.cos(ang), math.sin(ang)
    return center + q * c + np.cross(axis, q) * s + np.outer(q @ axis, axis) * (1 - c)

# ----------------------------------------------------------------- meshing
def mesh_sdf(fn, bmin, bmax, vox):
    bmin, bmax = np.array(bmin, float), np.array(bmax, float)
    n = np.ceil((bmax - bmin) / vox).astype(int) + 1
    xs = bmin[0] + np.arange(n[0]) * vox
    ys = bmin[1] + np.arange(n[1]) * vox
    zs = bmin[2] + np.arange(n[2]) * vox
    vol = np.empty(n, np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    for i, x in enumerate(xs):
        P = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1)
        vol[i] = fn(P).reshape(Y.shape)
    V, F, _, _ = measure.marching_cubes(vol, 0.0, spacing=(vox, vox, vox))
    V = V + bmin
    F = F.astype(np.int64)
    if signed_volume(V, F) < 0: F = F[:, [0, 2, 1]]
    return V, F

def signed_volume(V, F):
    return np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6

def taubin(V, F, it=8, lam=0.5, mu=-0.53):
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()[:, None]; deg[deg == 0] = 1
    for _ in range(it):
        V = V + lam * (A @ V / deg - V)
        V = V + mu * (A @ V / deg - V)
    return V

def decimate(V, F, target):
    if len(F) <= target: return V, F
    V2, F2 = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), 1 - target / len(F))
    V2 = V2.astype(np.float64); F2 = F2.astype(np.int64)
    if signed_volume(V2, F2) < 0: F2 = F2[:, [0, 2, 1]]
    return V2, F2

def drop_faces(V, F, keep):
    F = F[keep]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[F]

def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3): np.add.at(N, F[:, k], fn)
    l = norm(N)[:, None]; l[l == 0] = 1
    return N / l

# ----------------------------------------------------------------- layout
HC = v3(0, 1.40, 0)                     # head centre (world)
EYE_X, EYE_Y = 0.037, 0.006
R_EYE = 0.0275
MY, MZ, MW = -0.0665, 0.100, 0.022      # mouth (head-local)

# ---- head (head-local coordinates): soft, young, stylised
def head_base(P):
    d = sd_ell(P, v3(0, 0.025, -0.01), v3(0.104, 0.117, 0.111))
    d = smin(d, sd_ell(P, v3(0, -0.033, 0.020), v3(0.084, 0.082, 0.083)), 0.035)
    d = smin(d, sd_ell(P, v3(0, -0.082, 0.036), v3(0.047, 0.036, 0.050)), 0.03)
    d = smin(d, sd_sphere(P, v3(0, -0.101, 0.050), 0.021), 0.022)
    for sx in (1, -1):
        d = smin(d, sd_ell(P, v3(0.050 * sx, -0.026, 0.056), v3(0.045, 0.037, 0.043)), 0.025)
        d = smin(d, sd_ell(P, v3(0.106 * sx, -0.004, -0.008), v3(0.013, 0.026, 0.019)), 0.012)  # ears
    # small soft nose
    d = smin(d, sd_ell(P, v3(0, -0.020, 0.096), v3(0.009, 0.019, 0.013)), 0.012)
    d = smin(d, sd_sphere(P, v3(0, -0.033, 0.104), 0.0105), 0.009)
    for sx in (1, -1):
        d = smin(d, sd_sphere(P, v3(0.0085 * sx, -0.037, 0.099), 0.0065), 0.006)
    # soft muzzle under the lips
    d = smin(d, sd_ell(P, v3(0, -0.067, 0.090), v3(0.026, 0.016, 0.012)), 0.01)
    # neck
    d = smin(d, sd_cap(P, v3(0, -0.07, -0.015), v3(0, -0.26, -0.02), 0.037, 0.040), 0.03)
    return d

def _surface_z(x, y):
    zs = np.linspace(0.14, 0.0, 1401); P = np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], 1)
    return zs[np.argmax(head_base(P) < 0)]
_EZ = _surface_z(EYE_X, EYE_Y) - R_EYE + 0.0025       # eyeball sits flush, 2.5 mm proud of the skin
E_L = HC + v3(EYE_X, EYE_Y, _EZ)        # left eye centre (character's left = +x)
E_R = HC + v3(-EYE_X, EYE_Y, _EZ)
R_LID, R_SOCK = R_EYE + 0.0016, R_EYE + 0.0026
MOUTH_C, MOUTH_R = v3(0, MY, 0.091), v3(0.0170, 0.0075, 0.020)
LIP_FRONT = _surface_z(0.0, MY) + 0.0045

def head_sdf(P):
    d = head_base(P)
    for e in (E_L, E_R):
        d = smax(d, -sd_sphere(P, e - HC, R_SOCK), 0.006)
    d = smax(d, -sd_ell(P, MOUTH_C, MOUTH_R), 0.003)       # mouth cavity; the lips are separate geometry
    return d

def hair_sdf(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    d = sd_ell(P, v3(0, 0.030, -0.010), v3(0.114, 0.125, 0.121))
    d = smin(d, sd_sphere(P, v3(0.02, 0.095, -0.02), 0.075), 0.03)                   # crown volume
    d = smin(d, sd_ell(P, v3(0, -0.045, -0.035), v3(0.112, 0.125, 0.098)), 0.035)     # long back
    for sx in (1, -1):
        d = smin(d, sd_ell(P, v3(0.093 * sx, -0.055, 0.010), v3(0.030, 0.090, 0.050)), 0.03)  # side curtains
    # side-swept fringe: high on the parting side (+x), sweeping lower across the forehead
    fringe = 0.059 + 0.011 * ss(-0.09, 0.08, x) + 0.0035 * np.sin(x * 95 + 1.0) + 0.002 * np.sin(x * 41)
    window = np.maximum(np.maximum(0.002 - z, np.abs(x) - 0.083), y - fringe)
    d = smax(d, -window, 0.010)
    d = smax(d, -(y + 0.150), 0.03)                 # soft ends at shoulder height
    return d

# ---- body (world coordinates)
ARM_D = v3(0.707, -0.707, 0.0)
def body_parts():
    parts = []
    def add(name, label, f): parts.append((name, label, f))
    add("chest", "top", lambda P: sd_ell(P, v3(0, 1.10, 0), v3(0.135, 0.13, 0.095)))
    add("waist", "top", lambda P: sd_ell(P, v3(0, 0.965, 0), v3(0.115, 0.11, 0.085)))
    add("hips", "pants", lambda P: sd_ell(P, v3(0, 0.845, 0), v3(0.135, 0.09, 0.095)))
    for s, side in ((1, "L"), (-1, "R")):
        sh, el, wr = v3(0.15 * s, 1.20, -0.01), v3(0.32 * s, 1.03, -0.01), v3(0.476 * s, 0.874, 0.0)
        d = np.array([0.707 * s, -0.707, 0.0])
        add(f"shoulder{side}", "top", lambda P, sh=sh, s=s: sd_cap(P, v3(0.02 * s, 1.19, -0.01), sh, 0.065, 0.058))
        add(f"uarm{side}", "top", lambda P, sh=sh, el=el: sd_cap(P, sh, el, 0.052, 0.044))
        add(f"farm{side}", "top", lambda P, el=el, wr=wr: sd_cap(P, el, wr, 0.044, 0.037))
        ax = np.stack([d, np.cross(d, v3(0, 0, 1)), v3(0, 0, 1)])
        add(f"hand{side}", "hand", lambda P, wr=wr, d=d, ax=ax: sd_oell(P, wr + d * 0.05, ax, v3(0.052, 0.020, 0.040)))
        add(f"thumb{side}", "hand", lambda P, wr=wr, d=d: sd_cap(P, wr + d * 0.02 + v3(0, 0, 0.022), wr + d * 0.055 + v3(0, 0, 0.042), 0.012, 0.010))
        hip, kn, an = v3(0.085 * s, 0.82, 0), v3(0.09 * s, 0.45, 0), v3(0.09 * s, 0.09, -0.01)
        add(f"thigh{side}", "pants", lambda P, hip=hip, kn=kn: sd_cap(P, hip, kn, 0.078, 0.058))
        add(f"shin{side}", "pants", lambda P, kn=kn, an=an: sd_cap(P, kn, an, 0.056, 0.050))
        add(f"shoe{side}", "shoe", lambda P, s=s: np.maximum(sd_ell(P, v3(0.09 * s, 0.045, 0.03), v3(0.052, 0.052, 0.115)), -P[:, 1]))
    return parts
PARTS = body_parts()
def body_sdf(P):
    ds = [f(P) for _, _, f in PARTS]
    d = ds[0]
    for x in ds[1:]: d = smin(d, x, 0.025)
    return d
def body_part_id(P):
    return np.argmin(np.stack([f(P) for _, _, f in PARTS], 1), 1)

# ----------------------------------------------------------------- build geometry
log("meshing head")
Vh, Fh = mesh_sdf(head_sdf, (-0.125, -0.265, -0.13), (0.125, 0.16, 0.14), 0.0016)
Vh = taubin(Vh, Fh, 6)
# remove skin fully hidden under the hair cap
cen = Vh[Fh].mean(1)
Vh, Fh = drop_faces(Vh, Fh, hair_sdf(cen) > -0.006)
log("head raw tris", len(Fh))
Vh, Fh = decimate(Vh, Fh, 13000)
# light post-decimation smoothing, keeping the mouth slit and eye sockets sharp
_keep = ((np.abs(Vh[:, 1] - MY) < 0.012) & (Vh[:, 2] > 0.07) & (np.abs(Vh[:, 0]) < 0.04))
for _e in (E_L, E_R): _keep |= norm(Vh - (_e - HC)) < R_SOCK + 0.004
_Vs = taubin(Vh, Fh, 4)
Vh = np.where(_keep[:, None], Vh, _Vs)

log("head tris", len(Fh))

log("meshing hair")
Vr, Fr = mesh_sdf(hair_sdf, (-0.14, -0.10, -0.16), (0.14, 0.18, 0.15), 0.0025)
Vr = taubin(Vr, Fr, 8)
cen = Vr[Fr].mean(1)
Vr, Fr = drop_faces(Vr, Fr, head_sdf(cen) > -0.004)    # drop hair surface buried in the skull
Vr, Fr = decimate(Vr, Fr, 4200)
log("hair tris", len(Fr))

log("meshing body")
Vb, Fb = mesh_sdf(body_sdf, (-0.60, -0.01, -0.16), (0.60, 1.25, 0.20), 0.005)
Vb = taubin(Vb, Fb, 8)
Vb, Fb = decimate(Vb, Fb, 9500)
log("body tris", len(Fb))

def uv_sphere(c, r, nlat, nlon):
    V, F = [], []
    for i in range(nlat + 1):
        th = math.pi * i / nlat
        for j in range(nlon):
            ph = 2 * math.pi * j / nlon
            V.append(c + r * v3(math.sin(th) * math.sin(ph), math.cos(th), math.sin(th) * math.cos(ph)))
    for i in range(nlat):
        for j in range(nlon):
            a, b = i * nlon + j, i * nlon + (j + 1) % nlon
            c2, d = a + nlon, b + nlon
            F += [[a, c2, b], [b, c2, d]]
    V, F = np.array(V), np.array(F)
    return drop_faces(V, F, np.ones(len(F), bool)) if False else (V, F)

def lid_mesh(E, e0, e1, a0, a1, ne=10, na=18):
    V, P = [], []
    for i in range(ne + 1):
        e = e0 + (e1 - e0) * i / ne
        for j in range(na + 1):
            a = a0 + (a1 - a0) * j / na
            V.append(E + R_LID * v3(math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)))
            P.append(((j / na), (i / ne)))
    F = []
    for i in range(ne):
        for j in range(na):
            a = i * (na + 1) + j; b = a + 1; c = a + na + 1; d = c + 1
            F += [[a, b, c], [b, d, c]]
    V, F = np.array(V), np.array(F)
    # make outward-facing
    if (np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]]) @ (V[F[0]].mean(0) - E)) < 0:
        F = F[:, [0, 2, 1]]
    return V, F, np.array(P)

LID_OPEN = 0.98      # upper lid rotated up from its closed configuration (rad)
X = v3(1, 0, 0)

# ----------------------------------------------------------------- parts list
# each part: dict(name, kind, V, F, N, uv, tile)
parts = []
def add_part(name, kind, V, F, N=None, extra=None):
    if N is None: N = vertex_normals(V, F)
    p = dict(name=name, kind=kind, V=V, F=F, N=N)
    if extra is not None: p["extra"] = extra
    parts.append(p); return p

# head: split mouth interior faces into their own group (duplicated seam verts)
Vhw = Vh + HC
Nh = vertex_normals(Vhw, Fh)
c = Vh[Fh].mean(1)
_fn = Nh[Fh].mean(1)
mouth_face = (norm((c - MOUTH_C) / (MOUTH_R * 1.12)) < 1) & (c[:, 2] < LIP_FRONT - 0.006)
def sub(V, N, F, mask):
    Fm = F[mask]; used = np.unique(Fm)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], N[used], remap[Fm]
Vs, Ns, Fs = sub(Vhw, Nh, Fh, ~mouth_face); add_part("head", "head", Vs, Fs, Ns)
Vs, Ns, Fs = sub(Vhw, Nh, Fh, mouth_face); add_part("mouth", "head", Vs, Fs, Ns)

Vrw = Vr + HC
add_part("hair", "hair", Vrw, Fr)

for side, E in (("L", E_L), ("R", E_R)):
    V, F = uv_sphere(E, R_EYE, 16, 24)
    add_part(f"eye{side}", f"eye{side}", V, F)
    V, F, P = lid_mesh(E, -0.50, 1.60, -1.25, 1.25)
    V = rot_about(V, E, X, -LID_OPEN)
    add_part(f"lidU{side}", f"lidU{side}", V, F, extra=P)
    V, F, P = lid_mesh(E, -1.55, -0.52, -1.25, 1.25, 8)
    add_part(f"lidD{side}", f"lidD{side}", V, F, extra=P)

V, F = uv_sphere(HC + v3(0, MY - 0.0065, 0.072), 1.0, 8, 12)
V = HC + v3(0, MY - 0.0065, 0.072) + (V - HC - v3(0, MY - 0.0065, 0.072)) * v3(0.017, 0.005, 0.022)
add_part("tongue", "tongue", V, F)

def tube(center_fn, ry_fn, rz_fn, nu=34, nv=18):
    V, UVp, F = [], [], []
    for i in range(nu + 1):
        u = -1 + 2 * i / nu
        c, ry, rz = center_fn(u), ry_fn(u), rz_fn(u)
        for j in range(nv):
            a = 2 * math.pi * j / nv
            V.append(c + v3(0, ry * math.sin(a), rz * math.cos(a))); UVp.append((i / nu, j / (nv - 1) if j < nv else 1))
    for i in range(nu):
        for j in range(nv):
            a0 = i * nv + j; a1 = i * nv + (j + 1) % nv; b0 = a0 + nv; b1 = a1 + nv
            F += [[a0, b0, a1], [a1, b0, b1]]
    V, F = np.array(V), np.array(F)
    if signed_volume(V - V.mean(0), F) < 0: F = F[:, [0, 2, 1]]
    return V, F, np.array(UVp)

def taper(u, p=0.55, lo=0.12): return lo + (1 - lo) * (1 - u * u) ** p
LW = 0.0215
def lip_center(yc):
    return lambda u: HC + v3(u * LW, MY + (yc - MY) * taper(u), LIP_FRONT - 0.0058 - 0.017 * u * u)
for nm, yc, ry, rz in (("lipU", MY + 0.0040, 0.0040, 0.0058), ("lipD", MY - 0.0047, 0.0047, 0.0062)):
    V, F, UVp = tube(lip_center(yc), lambda u, ry=ry: ry * taper(u), lambda u, rz=rz: rz * taper(u, 0.45, 0.25))
    add_part(nm, "head", V, F, extra=UVp)
for nm, y0, y1 in (("teethU", MY - 0.0036, MY + 0.0045), ("teethD", MY - 0.0070, MY - 0.0040)):
    V, P_, F = [], [], []
    nu = 16
    for i in range(nu + 1):
        u = -1 + 2 * i / nu; x = u * 0.0150; z = LIP_FRONT - 0.0136 - 0.012 * u * u
        for y in (y0, y1):
            V.append(HC + v3(x, y, z)); P_.append((i / nu, 0 if y == y0 else 1))
    for i in range(nu):
        a, b, c2, d2 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        F += [[a, c2, b], [b, c2, d2]]
    V, F = np.array(V), np.array(F)
    N_ = vertex_normals(V, F)
    if N_[:, 2].mean() < 0: F = F[:, [0, 2, 1]]
    add_part(nm, nm, V, F, extra=np.array(P_))

# body: split by primitive
Nb = vertex_normals(Vb, Fb)
fpid = body_part_id(Vb[Fb].mean(1))
def body_group(pname):
    if pname in ("chest", "waist", "hips") or pname.startswith("shoulder"): return "torso"
    side = pname[-1]
    if pname.startswith(("uarm", "farm", "hand", "thumb")): return "arm" + side
    return "leg" + side
fgrp = np.array([body_group(PARTS[i][0]) for i in fpid])
for gname_ in ("torso", "armL", "armR", "legL", "legR"):
    Vs, Ns, Fs = sub(Vb, Nb, Fb, fgrp == gname_)
    add_part(gname_, "body", Vs, Fs, Ns)

# horizontal faces (shoulder tops, soles, under the chin) get their own top-down projection
def cap_split(name, thr=0.72):
    i = [k for k, p in enumerate(parts) if p["name"] == name][0]; p = parts[i]
    fny = p["N"][p["F"]].mean(1)[:, 1]
    up, dn = fny > thr, fny < -thr
    out = [dict(p)]
    out[0]["V"], out[0]["N"], out[0]["F"] = sub(p["V"], p["N"], p["F"], ~(up | dn))
    for m, suffix in ((up, "_cap"), (dn, "_bot")):
        if m.sum() < 1: continue
        b = dict(p); b["V"], b["N"], b["F"] = sub(p["V"], p["N"], p["F"], m); b["name"] = name + suffix; out.append(b)
    parts[i:i + 1] = out
for _n in ("head", "torso", "legL", "legR"): cap_split(_n, 0.62)
cap_split("hair", 0.8)

# ----------------------------------------------------------------- UV atlas
ATLAS = 2048
TILES = {
    "head":   (0, 0, 1024, 768),
    "hair":   (1024, 0, 512, 512),
    "eye":    (1024, 512, 256, 256),
    "lidU":   (1280, 512, 128, 64),
    "lidD":   (1280, 576, 128, 64),
    "mouth":  (1408, 512, 64, 64),
    "tongue": (1472, 512, 64, 64),
    "lipU": (1280, 640, 256, 64), "lipD": (1280, 704, 256, 64), "teeth": (1408, 576, 64, 64),
    "torso":  (0, 768, 1024, 768),       # top + waistband, one cylinder
    "arm":    (1024, 768, 1024, 512),    # sleeve + hand; L top / R bottom
    "leg":    (0, 1536, 1024, 512),      # trousers + shoe; L top / R bottom
    "head_cap": (1536, 0, 512, 512), "torso_cap": (1024, 1536, 512, 512),
    "legL_cap": (1536, 1536, 256, 256), "hair_cap": (1536, 512, 256, 256), "legR_cap": (1792, 1536, 256, 256),
    "head_bot": (1792, 512, 256, 256), "torso_bot": (1024, 1280, 256, 256), "legL_bot": (1280, 1280, 256, 256),
    "legR_bot": (1536, 1280, 256, 256), "hair_bot": (1792, 1280, 256, 256),
}
PAD = 6

def cyl(V, O, a, r0):
    a = a / np.linalg.norm(a); r0 = r0 - a * (r0 @ a); r0 /= np.linalg.norm(r0)
    b = np.cross(a, r0)
    q = V - O; t = q @ a; q = q - np.outer(t, a)
    ang = np.arctan2(q @ b, q @ r0)
    return ang / (2 * math.pi) + 0.5, t

def fix_seam(p, u):
    """duplicate vertices on faces that wrap around the cylinder seam"""
    F = p["F"]; uf = u[F]
    wrap = (uf.max(1) - uf.min(1)) > 0.5
    if not wrap.any(): return u
    newV, newN, newU, extra_idx = [], [], [], {}
    n = len(p["V"])
    F = F.copy()
    for fi in np.where(wrap)[0]:
        for k in range(3):
            vi = F[fi, k]
            if u[vi] < 0.5:
                if vi not in extra_idx:
                    extra_idx[vi] = n + len(newV)
                    newV.append(p["V"][vi]); newN.append(p["N"][vi]); newU.append(u[vi] + 1.0)
                F[fi, k] = extra_idx[vi]
    p["V"] = np.vstack([p["V"], newV]); p["N"] = np.vstack([p["N"], newN]); p["F"] = F
    return np.concatenate([u, newU])

def to_tile(tile, s, t, half=None):
    x0, y0, w, h = tile
    if half == 0: h = h // 2
    if half == 1: y0 += h // 2; h = h // 2
    px = x0 + PAD + np.clip(s, 0, 1) * (w - 2 * PAD)
    py = y0 + PAD + np.clip(t, 0, 1) * (h - 2 * PAD)
    return np.stack([px / ATLAS, py / ATLAS], 1)

def lohi(x): return (x - x.min()) / max(x.max() - x.min(), 1e-9)

for p in parts:
    n, k, V = p["name"], p["kind"], p["V"]
    if n.endswith(("_cap", "_bot")):
        x_, z_ = V[:, 0], V[:, 2]; sc = max(np.ptp(x_), np.ptp(z_), 1e-6)
        p["uv"] = to_tile(TILES[n], (x_ - x_.min()) / sc, (z_ - z_.min()) / sc)
    elif n in ("head",):
        u, t = cyl(V, HC, v3(0, 1, 0), v3(0, 0, 1)); u = fix_seam(p, u); V = p["V"]
        p["uv"] = to_tile(TILES["head"], u / 1.03, 1 - (V[:, 1] - (HC[1] - 0.265)) / 0.43)
    elif n == "hair":   # azimuthal-equidistant from the crown: no pole, no seam
        q = V - (HC + v3(0, -0.01, -0.012)); rq = norm(q)
        th = np.arccos(np.clip(q[:, 1] / rq, -1, 1)) / 3.0; lam = np.arctan2(q[:, 0], q[:, 2])
        p["uv"] = to_tile(TILES["hair"], 0.5 + 0.5 * th * np.sin(lam), 0.5 + 0.5 * th * np.cos(lam))
    elif n == "mouth":
        p["uv"] = to_tile(TILES["mouth"], lohi(V[:, 0]), lohi(V[:, 1]))
    elif n in ("lipU", "lipD"):
        P_ = p["extra"]; p["uv"] = to_tile(TILES[n], P_[:, 0], P_[:, 1])
    elif n.startswith("teeth"):
        P_ = p["extra"]; p["uv"] = to_tile(TILES["teeth"], P_[:, 0], P_[:, 1])
    elif n == "tongue":
        p["uv"] = to_tile(TILES["tongue"], lohi(V[:, 0]), lohi(V[:, 2]))
    elif k.startswith("eye"):
        E = E_L if k.endswith("L") else E_R
        sgn = 1 if k.endswith("L") else -1
        q = rot_about((V - E) / R_EYE, v3(0, 0, 0), v3(0, 1, 0), -0.17 * sgn)
        p["uv"] = to_tile(TILES["eye"], 0.5 + 0.5 * q[:, 0], 0.5 - 0.5 * q[:, 1])
    elif k.startswith("lidU"):
        P = p["extra"]; p["uv"] = to_tile(TILES["lidU"], P[:, 0], P[:, 1])
    elif k.startswith("lidD"):
        P = p["extra"]; p["uv"] = to_tile(TILES["lidD"], P[:, 0], 1 - P[:, 1])
    else:  # body
        side = 0 if n.endswith("L") else 1
        s_ = 1 if side == 0 else -1
        if n == "torso":
            u, t = cyl(V, v3(0, 0, 0), v3(0, 1, 0), v3(0, 0, 1)); u = fix_seam(p, u); V = p["V"]
            p["uv"] = to_tile(TILES["torso"], u / 1.03, 1 - (V[:, 1] - 0.72) / 0.54)
        elif n.startswith("arm"):
            sh = v3(0.15 * s_, 1.20, -0.01); ax = v3(0.707 * s_, -0.707, 0)
            u, t = cyl(V, sh, ax, v3(0.707 * s_, 0.707, 0)); u = fix_seam(p, u); V = p["V"]
            t = (V - sh) @ ax
            p["uv"] = to_tile(TILES["arm"], (t + 0.08) / 0.74, u / 1.03, side)
        elif n.startswith("leg"):
            hip = v3(0.085 * s_, 0.82, 0)
            u, t = cyl(V, hip, v3(0, -1, 0), v3(s_, 0, 0)); u = fix_seam(p, u); V = p["V"]
            p["uv"] = to_tile(TILES["leg"], (0.88 - V[:, 1]) / 0.89, u / 1.03, side)

# ----------------------------------------------------------------- skeleton
BONES = [  # name, parent, head(world)
    ("Hips", None, v3(0, 0.86, 0)),
    ("Spine", "Hips", v3(0, 0.95, 0)),
    ("Spine1", "Spine", v3(0, 1.06, 0)),
    ("Spine2", "Spine1", v3(0, 1.16, 0)),
    ("Neck", "Spine2", v3(0, 1.25, -0.005)),
    ("Head", "Neck", v3(0, 1.33, -0.01)),
    ("HeadTop_End", "Head", v3(0, 1.56, -0.01)),
    ("LeftEye", "Head", E_L), ("RightEye", "Head", E_R),
]
for s, S in ((1, "Left"), (-1, "Right")):
    BONES += [
        (f"{S}Shoulder", "Spine2", v3(0.03 * s, 1.21, -0.01)),
        (f"{S}Arm", f"{S}Shoulder", v3(0.15 * s, 1.20, -0.01)),
        (f"{S}ForeArm", f"{S}Arm", v3(0.32 * s, 1.03, -0.01)),
        (f"{S}Hand", f"{S}ForeArm", v3(0.476 * s, 0.874, 0.0)),
        (f"{S}HandTip_End", f"{S}Hand", v3(0.55 * s, 0.80, 0.01)),
    ]
for s, S in ((1, "Left"), (-1, "Right")):
    BONES += [
        (f"{S}UpLeg", "Hips", v3(0.085 * s, 0.82, 0)),
        (f"{S}Leg", f"{S}UpLeg", v3(0.09 * s, 0.45, 0)),
        (f"{S}Foot", f"{S}Leg", v3(0.09 * s, 0.09, -0.01)),
        (f"{S}ToeBase", f"{S}Foot", v3(0.09 * s, 0.02, 0.09)),
        (f"{S}Toe_End", f"{S}ToeBase", v3(0.09 * s, 0.02, 0.15)),
    ]
BI = {b[0]: i for i, b in enumerate(BONES)}
BPOS = {b[0]: b[2] for b in BONES}
SEG = {  # bone -> segment used for skin-weight distance
    "Hips": (v3(0, 0.78, 0), v3(0, 0.95, 0)), "Spine": (BPOS["Spine"], BPOS["Spine1"]),
    "Spine1": (BPOS["Spine1"], BPOS["Spine2"]), "Spine2": (BPOS["Spine2"], BPOS["Neck"]),
    "Neck": (BPOS["Neck"], BPOS["Head"]), "Head": (BPOS["Head"] + v3(0, 0.03, 0), v3(0, 1.55, -0.01)),
}
for S in ("Left", "Right"):
    SEG[f"{S}Shoulder"] = (BPOS[f"{S}Shoulder"], BPOS[f"{S}Arm"])
    SEG[f"{S}Arm"] = (BPOS[f"{S}Arm"], BPOS[f"{S}ForeArm"])
    SEG[f"{S}ForeArm"] = (BPOS[f"{S}ForeArm"], BPOS[f"{S}Hand"])
    SEG[f"{S}Hand"] = (BPOS[f"{S}Hand"], BPOS[f"{S}HandTip_End"])
    SEG[f"{S}UpLeg"] = (BPOS[f"{S}UpLeg"], BPOS[f"{S}Leg"])
    SEG[f"{S}Leg"] = (BPOS[f"{S}Leg"], BPOS[f"{S}Foot"])
    SEG[f"{S}Foot"] = (BPOS[f"{S}Foot"], BPOS[f"{S}ToeBase"])
    SEG[f"{S}ToeBase"] = (BPOS[f"{S}ToeBase"], BPOS[f"{S}Toe_End"])

def seg_dist(P, a, b):
    ab = b - a; h = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return norm(P - a - np.outer(h, ab))

def skin(P, allowed, tau=0.018):
    D = np.stack([seg_dist(P, *SEG[b]) for b in allowed], 1)
    W = np.exp(-(D - D.min(1, keepdims=True)) / tau)
    idx = np.argsort(-W, 1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w[w < 0.02] = 0
    w /= w.sum(1, keepdims=True)
    J = np.array([BI[allowed[i]] for i in range(len(allowed))])[idx]
    if J.shape[1] < 4:
        padn = 4 - J.shape[1]
        J = np.hstack([J, np.zeros((len(J), padn), int)]); w = np.hstack([w, np.zeros((len(w), padn))])
    return J, w

BODY_BONES = [b for b in SEG if b != "Head"]
for p in parts:
    k, V = p["kind"], p["V"]
    if k in ("hair", "tongue") or k.startswith("lid") or p["name"] in ("mouth", "lipU", "lipD", "teethU", "teethD"):
        p["J"] = np.tile([BI["Head"], 0, 0, 0], (len(V), 1)); p["W"] = np.tile([1., 0, 0, 0], (len(V), 1))
    elif k.startswith("eye"):
        b = BI["LeftEye" if k.endswith("L") else "RightEye"]
        p["J"] = np.tile([b, 0, 0, 0], (len(V), 1)); p["W"] = np.tile([1., 0, 0, 0], (len(V), 1))
    elif k == "head":
        p["J"], p["W"] = skin(V, ["Head", "Neck", "Spine2"], 0.012)
    else:
        p["J"], p["W"] = skin(V, BODY_BONES)

# ----------------------------------------------------------------- blendshapes
ARKIT = ["eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
         "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
         "jawForward", "jawLeft", "jawRight", "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
         "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight",
         "mouthStretchLeft", "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
         "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
         "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
         "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "tongueOut"]
assert len(ARKIT) == 52
VIS = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou"]
VISEME_MIX = {  # composition of base shapes; 'jaw' = fraction of full jawOpen
    "sil": {},
    "PP": {"mouthClose": 1.0, "mouthPressLeft": 0.6, "mouthPressRight": 0.6},
    "FF": {"jaw": 0.12, "mouthRollLower": 0.8, "mouthUpperUpLeft": 0.3, "mouthUpperUpRight": 0.3},
    "TH": {"jaw": 0.25, "tongueOut": 0.35, "mouthUpperUpLeft": 0.2, "mouthUpperUpRight": 0.2},
    "DD": {"jaw": 0.28, "mouthStretchLeft": 0.25, "mouthStretchRight": 0.25},
    "kk": {"jaw": 0.32, "mouthStretchLeft": 0.35, "mouthStretchRight": 0.35},
    "CH": {"jaw": 0.22, "mouthFunnel": 0.6, "mouthPucker": 0.25},
    "SS": {"jaw": 0.10, "mouthStretchLeft": 0.55, "mouthStretchRight": 0.55, "mouthSmileLeft": 0.1, "mouthSmileRight": 0.1},
    "nn": {"jaw": 0.18, "mouthStretchLeft": 0.15, "mouthStretchRight": 0.15},
    "RR": {"jaw": 0.20, "mouthPucker": 0.45, "mouthFunnel": 0.3},
    "aa": {"jaw": 0.75, "mouthLowerDownLeft": 0.3, "mouthLowerDownRight": 0.3},
    "E":  {"jaw": 0.40, "mouthStretchLeft": 0.55, "mouthStretchRight": 0.55, "mouthSmileLeft": 0.2, "mouthSmileRight": 0.2},
    "ih": {"jaw": 0.30, "mouthStretchLeft": 0.40, "mouthStretchRight": 0.40},
    "oh": {"jaw": 0.50, "mouthFunnel": 0.75},
    "ou": {"jaw": 0.25, "mouthPucker": 0.9, "mouthFunnel": 0.2},
}
JAW_MAX = 0.22
JAW_PIVOT = v3(0, -0.03, -0.015)

def g3(P, c, s):
    return np.exp(-(((P - c) / s) ** 2).sum(1))

def jaw_w(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax = np.abs(x)
    yl = MY + np.maximum(0, ax - 0.016) * 0.55
    wd = 0.00045 + np.maximum(0, ax - 0.011) * 0.4
    below = ss(-wd, wd, yl - y)
    return below * ss(-0.04, 0.02, z) * ss(-0.18, -0.12, y) * (1 - ss(0.075, 0.105, ax))

def jaw_delta(P, ang):
    return jaw_w(P)[:, None] * (rot_about(P, JAW_PIVOT, X, ang) - P)

def lips_masks(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    front = ss(0.065, 0.085, z)
    up = g3(P, v3(0, MY + 0.008, 0.10), v3(0.045, 0.016, 0.04)) * ss(MY + 0.0001, MY + 0.0012, y) * front
    lo = g3(P, v3(0, MY - 0.009, 0.10), v3(0.042, 0.018, 0.04)) * ss(MY - 0.0001, MY - 0.0012, y) * front
    left = ss(-0.01, 0.01, x)
    return up, lo, left, front

def base_delta(name, P, kind):
    """delta (N,3) of one ARKit shape for vertices P (head-local) of a given part kind"""
    Z = np.zeros_like(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    side = 1 if name.endswith("Left") else -1
    # ---- eyes
    if name.startswith("eye"):
        eye_side = "L" if name.endswith("Left") else "R"
        E = (E_L if eye_side == "L" else E_R) - HC
        if name.startswith("eyeLook") and kind == f"eye{eye_side}":
            a = 0.35
            if "Up" in name: return rot_about(P, E, X, -a) - P
            if "Down" in name: return rot_about(P, E, X, a) - P
            outward = 1 if eye_side == "L" else -1
            sgn = outward if "Out" in name else -outward
            return rot_about(P, E, v3(0, 1, 0), a * sgn) - P
        if name.startswith("eyeBlink") and kind == f"lidU{eye_side}":
            return rot_about(P, E, X, LID_OPEN) - P
        if name.startswith("eyeWide") and kind == f"lidU{eye_side}":
            return rot_about(P, E, X, -0.32) - P
        if name.startswith("eyeSquint"):
            if kind == f"lidD{eye_side}": return rot_about(P, E, X, -0.42) - P
            if kind == f"lidU{eye_side}": return rot_about(P, E, X, 0.18) - P
            if kind == "head":
                return g3(P, E + v3(0, -0.028, 0.01), v3(0.022, 0.009, 0.03))[:, None] * v3(0, 0.003, 0.001)
        return Z
    if kind == "teethU": return Z
    if kind in ("tongue", "teethD"):
        if name == "tongueOut": return np.tile(v3(0, -0.004, 0.034), (len(P), 1)) if kind == "tongue" else Z
        if name == "jawOpen": return jaw_delta(P, JAW_MAX)
        if name == "jawForward": return jaw_w(P)[:, None] * v3(0, 0, 0.008)
        if name in ("jawLeft", "jawRight"): return jaw_w(P)[:, None] * v3(0.008 * side, 0, 0)
        return Z
    if kind != "head": return Z
    up, lo, left, front = lips_masks(P)
    sideM = left if side == 1 else (1 - left)
    corner = g3(P, v3((MW + 0.002) * side, MY, 0.092), v3(0.020, 0.018, 0.03)) * front
    M = g3(P, v3(0, MY, 0.10), v3(0.042, 0.022, 0.04)) * front
    if name == "jawOpen": return jaw_delta(P, JAW_MAX)
    if name == "jawForward": return jaw_w(P)[:, None] * v3(0, 0, 0.008)
    if name in ("jawLeft", "jawRight"): return jaw_w(P)[:, None] * v3(0.008 * side, 0, 0)
    if name == "mouthClose": return (up[:, None] * v3(0, -0.0018, 0) + lo[:, None] * v3(0, 0.0018, 0)) * ss(0.0, 0.006, 0.012 - np.abs(y - MY))[:, None]
    if name == "mouthFunnel":
        d = np.stack([-x * 0.22, (y - MY) * 0.5, np.full_like(x, 0.006)], 1); return d * M[:, None]
    if name == "mouthPucker":
        d = np.stack([-x * 0.38, (MY - y) * 0.15, np.full_like(x, 0.008)], 1); return d * M[:, None]
    if name in ("mouthLeft", "mouthRight"): return M[:, None] * v3(0.008 * side, 0, 0)
    if name.startswith("mouthSmile"):
        cheek = g3(P, v3(0.045 * side, -0.035, 0.075), v3(0.022, 0.02, 0.03))
        return corner[:, None] * v3(0.0045 * side, 0.0125, -0.005) + cheek[:, None] * v3(0.001 * side, 0.0055, 0.002)
    if name.startswith("mouthFrown"): return corner[:, None] * v3(0.001 * side, -0.010, -0.001)
    if name.startswith("mouthDimple"): return corner[:, None] * v3(0.002 * side, 0.001, -0.005)
    if name.startswith("mouthStretch"): return corner[:, None] * v3(0.008 * side, -0.002, -0.002)
    if name == "mouthRollLower": return lo[:, None] * v3(0, 0.002, -0.004)
    if name == "mouthRollUpper": return up[:, None] * v3(0, -0.002, -0.004)
    if name == "mouthShrugLower":
        chin = g3(P, v3(0, -0.10, 0.06), v3(0.03, 0.02, 0.03))
        return lo[:, None] * v3(0, 0.004, 0.002) + chin[:, None] * v3(0, 0.003, 0.002)
    if name == "mouthShrugUpper": return up[:, None] * v3(0, 0.003, 0.001)
    if name.startswith("mouthPress"):
        return (sideM * up)[:, None] * v3(0, -0.0012, -0.0015) + (sideM * lo)[:, None] * v3(0, 0.0012, -0.0015)
    if name.startswith("mouthLowerDown"): return (sideM * lo)[:, None] * v3(0, -0.006, 0.001)
    if name.startswith("mouthUpperUp"): return (sideM * up)[:, None] * v3(0, 0.005, 0.001)
    fr = ss(0.05, 0.075, z)
    if name.startswith("browDown"):
        return (g3(P, v3(0.035 * side, 0.048, 0.09), v3(0.026, 0.016, 0.04)) * fr)[:, None] * v3(-0.003 * side, -0.008, 0.002)
    if name == "browInnerUp":
        m = (g3(P, v3(0.018, 0.048, 0.096), v3(0.019, 0.017, 0.04)) + g3(P, v3(-0.018, 0.048, 0.096), v3(0.019, 0.017, 0.04))) * fr
        return m[:, None] * v3(0, 0.011, 0)
    if name.startswith("browOuterUp"):
        return (g3(P, v3(0.058 * side, 0.045, 0.08), v3(0.021, 0.017, 0.04)) * fr)[:, None] * v3(0, 0.010, 0)
    if name == "cheekPuff":
        m = g3(P, v3(0.05, -0.05, 0.07), v3(0.03, 0.025, 0.035)); m2 = g3(P, v3(-0.05, -0.05, 0.07), v3(0.03, 0.025, 0.035))
        return m[:, None] * v3(0.005, 0, 0.006) + m2[:, None] * v3(-0.005, 0, 0.006)
    if name.startswith("cheekSquint"):
        return g3(P, v3(0.045 * side, -0.016, 0.075), v3(0.022, 0.013, 0.03))[:, None] * v3(0, 0.006, 0.0015)
    if name.startswith("noseSneer"):
        return g3(P, v3(0.012 * side, -0.030, 0.10), v3(0.012, 0.013, 0.02))[:, None] * v3(0.001 * side, 0.003, 0)
    return Z

def viseme_delta(v, P, kind):
    d = np.zeros_like(P)
    for k, w in VISEME_MIX[v].items():
        if k == "jaw":
            if kind in ("head", "tongue", "teethD"): d += jaw_delta(P, JAW_MAX * w)
        else:
            d += w * base_delta(k, P, kind)
    return d

TARGETS = ARKIT + [f"viseme_{v}" for v in VIS]
log("computing", len(TARGETS), "morph targets")
for p in parts:
    P = p["V"] - HC
    # lips belong to one side of the mouth line: evaluate the fields just inside their own side
    if p["name"] == "lipU": P = P + v3(0, 0.0025, 0)
    if p["name"] == "lipD": P = P - v3(0, 0.0025, 0)
    D = []
    for t in TARGETS:
        if t.startswith("viseme_"): d = viseme_delta(t[7:], P, p["kind"])
        else: d = base_delta(t, P, p["kind"])
        d = np.asarray(d, float)
        d[np.abs(d).max(1) < 2e-6] = 0
        D.append(d)
    p["D"] = D

# ----------------------------------------------------------------- texture painting
log("rasterising position maps")
POS = np.zeros((ATLAS, ATLAS, 3), np.float32)
NRM = np.zeros((ATLAS, ATLAS, 3), np.float32)
GRP = -np.ones((ATLAS, ATLAS), np.int16)
GROUP_NAMES = []
def raster(p, gi):
    UV = p["uv"] * ATLAS - 0.5
    for f in p["F"]:
        a, b, c = UV[f]
        x0, x1 = int(max(0, math.floor(min(a[0], b[0], c[0])))), int(min(ATLAS - 1, math.ceil(max(a[0], b[0], c[0]))))
        y0, y1 = int(max(0, math.floor(min(a[1], b[1], c[1])))), int(min(ATLAS - 1, math.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0: continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12: continue
        l0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / den
        l1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / den
        l2 = 1 - l0 - l1
        m = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
        if not m.any(): continue
        L = np.stack([l0[m], l1[m], l2[m]], 1)
        POS[ys[m], xs[m]] = L @ p["V"][f]
        NRM[ys[m], xs[m]] = L @ p["N"][f]
        GRP[ys[m], xs[m]] = gi
for gi, p in enumerate(parts):
    GROUP_NAMES.append(p["name"]); raster(p, gi)

# dilate coverage to hide seams
cov = GRP >= 0
for _ in range(10):
    idx = ndimage.distance_transform_edt(~cov, return_distances=False, return_indices=True)
    break
POS = POS[idx[0], idx[1]]; NRM = NRM[idx[0], idx[1]]; GRP = GRP[idx[0], idx[1]]
gname = np.array(GROUP_NAMES + ["none"])[GRP]

log("painting")
def hexc(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
def noise(scale, octaves=3, seed=0):
    r = np.random.default_rng(seed); out = np.zeros((ATLAS, ATLAS), np.float32); amp = 1
    for o in range(octaves):
        n = r.random((ATLAS // scale + 2, ATLAS // scale + 2)).astype(np.float32)
        out += amp * ndimage.zoom(n, scale, order=3)[:ATLAS, :ATLAS]; amp *= 0.5; scale = max(2, scale // 2)
    return (out - out.min()) / (out.max() - out.min())

SKIN, SKIN2 = hexc("#F3CDB0"), hexc("#E8A08E")
LIP = hexc("#DB8580"); BROW = hexc("#3B2A2E")
HAIR, HAIR2 = hexc("#3A2620"), hexc("#8A5A40")
TOP, TOP2 = hexc("#2F7C7E"), hexc("#1F5859")
ACC = hexc("#F2E3C6")
PANTS = hexc("#2B2D36"); SHOE = hexc("#F1EFEA"); SOLE = hexc("#9A8F86")

base = np.zeros((ATLAS, ATLAS, 3), np.float32)
rough = np.full((ATLAS, ATLAS), 0.6, np.float32)
height = np.zeros((ATLAS, ATLAS), np.float32)
n1, n2, n3 = noise(64, 4, 1), noise(16, 3, 2), noise(4, 2, 3)
P = POS - HC
px, py, pz = P[..., 0], P[..., 1], P[..., 2]

def g2(cx, cy, sx, sy): return np.exp(-(((px - cx) / sx) ** 2 + ((py - cy) / sy) ** 2))

skinmask = np.isin(gname, ["head", "head_cap", "head_bot", "lidUL", "lidUR", "lidDL", "lidDR"])
col = SKIN[None, None] * (0.94 + 0.08 * n1[..., None]) + (SKIN2 - SKIN)[None, None] * (0.25 * n2[..., None] * 0.4)
# blush / warmth
face = (gname == "head") & (pz > 0.03)
blush = (g2(0.053, -0.030, 0.021, 0.015) + g2(-0.053, -0.030, 0.021, 0.015)) * face
col = col + (SKIN2 - SKIN)[None, None] * (0.8 * blush)[..., None]
nose_tip = g2(0, -0.036, 0.012, 0.01) * face
col = col + (SKIN2 - SKIN)[None, None] * (0.35 * nose_tip)[..., None]
# lips
lipm = 0 * np.clip(((px / 0.0245) ** 2 + ((py - MY + 0.0005) / 0.0125) ** 2 < 1) * 1.0, 0, 1) * face * (pz > 0.08)
lipm = ndimage.gaussian_filter(lipm.astype(np.float32), 1.6)
col = col * (1 - lipm[..., None]) + LIP[None, None] * lipm[..., None]
# inner walls of the mouth slit (vertical-facing surfaces between the lips) are mouth interior, not lip
nrm_y = NRM[..., 1] / (np.linalg.norm(NRM, axis=-1) + 1e-9)
inner = face & (np.abs(py - MY) < 0.0040) & (np.abs(px) < MW + 0.004) & (pz > 0.07) & ((np.abs(nrm_y) > 0.45) | (pz < 0.097))
inner = 0 * ndimage.gaussian_filter(inner.astype(np.float32), 0.8)
col = col * (1 - inner[..., None]) + hexc("#4A2226")[None, None] * inner[..., None]
# eyebrows
bx = np.abs(px)
by = 0.0470 + 0.006 * np.sin(np.clip((bx - 0.014) / 0.05, 0, 1) * math.pi) - 0.004 * np.clip((bx - 0.014) / 0.05, 0, 1)
thick = 0.0033 * (1 - 0.6 * np.clip((bx - 0.014) / 0.05, 0, 1))
brow = (np.abs(py - by) < thick) & (bx > 0.013) & (bx < 0.066) & face & (pz > 0.06)
brow = ndimage.gaussian_filter(brow.astype(np.float32), 1.2)
col = col * (1 - brow[..., None]) + BROW[None, None] * brow[..., None]
# nostrils
nos = (g2(0.0075, -0.0425, 0.003, 0.0022) + g2(-0.0075, -0.0425, 0.003, 0.0022)) * face
col = col * (1 - 0.6 * nos[..., None])
# upper-lid lash line
for ln in ("lidUL", "lidUR", "lidDL", "lidDR"):
    m = gname == ln
    x0, y0, w, h = TILES[ln[:4]]
    yy = (np.arange(ATLAS)[:, None] - y0 - PAD) / (h - 2 * PAD)
    yy = np.broadcast_to(yy, (ATLAS, ATLAS))
    lash = (ss(0.15, 0.04, yy) if ln.startswith("lidU") else ss(0.10, 0.0, 1 - yy) * 0.6)
    lash = lash * m
    col = col * (1 - lash[..., None] * 0.92) + BROW[None, None] * 0.4 * lash[..., None] * 0.92
base[skinmask] = col[skinmask]
rough[skinmask] = 0.52 - 0.12 * lipm[skinmask]
height[skinmask] = 0.05 * n2[skinmask]

# lips: soft colour, darker where they meet, a touch of gloss on the lower lip
lipmask = np.isin(gname, ["lipU", "lipD"])
contact = np.exp(-((py - MY) / 0.0016) ** 2)
lc = LIP[None, None] * (0.94 + 0.08 * n2[..., None])
lc = lc * (1 - 0.55 * contact[..., None])
edge = ss(0.016, 0.022, np.abs(px))                      # blend into skin at the corners
lc = lc * (1 - edge[..., None]) + SKIN2[None, None] * edge[..., None]
base[lipmask] = lc[lipmask]; rough[lipmask] = (0.42 - 0.12 * (gname == "lipD"))[lipmask]; height[lipmask] = 0.05 * n2[lipmask]
m = np.char.startswith(gname, "teeth"); base[m] = hexc("#F2EEE6"); rough[m] = 0.3
# mouth interior + tongue
m = gname == "mouth"; base[m] = hexc("#3E1C20"); rough[m] = 0.5
m = gname == "tongue"; base[m] = hexc("#C46670") * (0.9 + 0.1 * n2[m][:, None]); rough[m] = 0.35

# eyes (painted analytically in the tile's own disc coordinates)
x0, y0, w, h = TILES["eye"]
yy_, xx_ = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
qx = ((xx_ + 0.5 - x0 - PAD) / (w - 2 * PAD)) * 2 - 1; qy = -(((yy_ + 0.5 - y0 - PAD) / (h - 2 * PAD)) * 2 - 1)
r = np.sqrt(qx ** 2 + qy ** 2); ang = np.arctan2(qy, qx)
sclera = hexc("#F4F1EC")[None, None] * (1 - 0.14 * ss(0.55, 1.0, r))[..., None]
IRIS_R, PUP_R = 0.60, 0.25
irisc = hexc("#9A6A3C")[None, None] * (1 - ss(0.28, IRIS_R, r))[..., None] + hexc("#4A2E1E")[None, None] * ss(0.28, IRIS_R, r)[..., None]
streak = 0.85 + 0.15 * np.sin(ang * 23 + 3 * np.sin(ang * 7))
irisc = irisc * streak[..., None] + hexc("#C9A66B")[None, None] * (ss(PUP_R + 0.12, PUP_R, r) * 0.5)[..., None]
limbal = ss(IRIS_R - 0.07, IRIS_R, r) * ss(IRIS_R + 0.02, IRIS_R, r)
c = np.where((r < IRIS_R)[..., None], irisc, sclera) * (1 - 0.75 * limbal)[..., None]
c = np.where((r < PUP_R)[..., None], hexc("#0D0F12")[None, None], c)
hl = np.exp(-(((qx + 0.22) / 0.11) ** 2 + ((qy - 0.24) / 0.11) ** 2)) + 0.75 * np.exp(-(((qx - 0.2) / 0.05) ** 2 + ((qy + 0.2) / 0.05) ** 2))
c = c + (1 - c) * np.clip(hl, 0, 1)[..., None]
base[y0:y0 + h, x0:x0 + w] = c; rough[y0:y0 + h, x0:x0 + w] = 0.08

# hair
m = np.char.startswith(gname, "hair")
x0, y0, w, h = TILES["hair"]
xx = np.broadcast_to(np.arange(ATLAS)[None, :], (ATLAS, ATLAS)).astype(np.float32)
yy = np.broadcast_to(np.arange(ATLAS)[:, None], (ATLAS, ATLAS)).astype(np.float32)
lam = np.arctan2(px, pz + 0.012)
strand = (0.5 + 0.22 * np.sin(lam * 61 + 2.5 * np.sin(lam * 9)) + 0.16 * np.sin(lam * 143 + 1.3) + 0.12 * np.sin(lam * 29 + 4 * np.sin(py * 40)))
strand = np.clip(strand + 0.15 * (n2 - 0.5), 0, 1).astype(np.float32)
sheen = np.exp(-((py - 0.085) / 0.018) ** 2) * (pz > -0.02)
hc = HAIR[None, None] * (0.75 + 0.5 * strand[..., None]) + (HAIR2 - HAIR)[None, None] * (0.8 * sheen * strand)[..., None]
base[m] = hc[m]; rough[m] = 0.42 + 0.15 * strand[m]; height[m] = strand[m]

# clothing — colours follow continuous position lines, not mesh partitions
wy = POS[..., 1]
weave = 0.5 + 0.25 * np.sin(xx * 1.4) * np.sin(yy * 1.4)
twill = 0.5 + 0.5 * np.sin((xx + yy) * 1.1)
topc = TOP[None, None] * (0.92 + 0.12 * n1[..., None]) * (0.96 + 0.08 * weave[..., None])
pantc = PANTS[None, None] * (0.9 + 0.15 * n1[..., None]) * (0.95 + 0.08 * twill[..., None])
skinc = SKIN[None, None] * (0.94 + 0.08 * n1[..., None])
def mix(a, b, w): return a * (1 - w[..., None]) + b * w[..., None]
# torso: hoodie above the waistband, trousers below
m = np.char.startswith(gname, "torso")
w_p = ss(0.902, 0.896, wy)                       # 1 = trousers
hem = ss(0.92, 0.912, wy) * (1 - w_p)
c = mix(mix(topc, TOP2[None, None] * (0.95 + 0.08 * weave[..., None]), hem), pantc, w_p)
ex, ey = POS[..., 0], POS[..., 1] - 1.115
front = (POS[..., 2] > 0.04)
rr = np.sqrt(ex ** 2 + ey ** 2)
ring = (np.abs(rr - 0.030) < 0.0032)
bars = np.zeros_like(rr, bool)
for i, hgt in zip(range(-2, 3), (0.007, 0.013, 0.019, 0.013, 0.007)):
    bars |= (np.abs(ex - i * 0.0085) < 0.0024) & (np.abs(ey) < hgt)
emb = ndimage.gaussian_filter(((ring | bars) & front & m).astype(np.float32), 1.0)
c = mix(c, ACC[None, None] * np.ones_like(c), emb)
base[m] = c[m]; rough[m] = (0.88 - 0.06 * w_p - 0.4 * emb)[m]; height[m] = (0.6 * weave * (1 - w_p) + 0.5 * twill * w_p - 0.4 * emb)[m]
# arms: sleeve up to the wrist, then hand
for side, s_ in (("L", 1), ("R", -1)):
    m = gname == "arm" + side
    sh = v3(0.15 * s_, 1.20, -0.01); ax = v3(0.707 * s_, -0.707, 0)
    t = (POS - sh) @ ax
    w_s = ss(0.452, 0.458, t)                    # 1 = skin
    cuff = ss(0.415, 0.422, t) * (1 - w_s)
    c = mix(mix(topc, TOP2[None, None] * (0.95 + 0.08 * weave[..., None]), cuff), skinc, w_s)
    base[m] = c[m]; rough[m] = (0.88 - 0.36 * w_s)[m]; height[m] = (0.6 * weave * (1 - w_s))[m]
# legs: trousers, then shoe with a sole
for side in ("L", "R"):
    m = np.char.startswith(gname, "leg" + side)
    w_sh = ss(0.112, 0.104, wy)                  # 1 = shoe
    sole = ss(0.028, 0.022, wy)
    shoe = mix(SHOE[None, None] * (0.95 + 0.06 * n2[..., None]), SOLE[None, None] * np.ones((ATLAS, ATLAS, 1), np.float32), sole)
    c = mix(pantc, shoe, w_sh)
    base[m] = c[m]; rough[m] = (0.82 - 0.3 * w_sh + 0.25 * sole)[m]; height[m] = (0.5 * twill * (1 - w_sh))[m]

# baked cavity: darken mouth crease and eye sockets on the skin
crease = 0 * np.exp(-((py - MY) / 0.0022) ** 2) * (np.abs(px) < MW + 0.004) * face
base = base * (1 - 0.45 * crease[..., None])

# normal map from height
hgt = ndimage.gaussian_filter(height, 0.7)
gx = ndimage.sobel(hgt, axis=1); gy = ndimage.sobel(hgt, axis=0)
strength = 1.6
nm = np.stack([-gx * strength, gy * strength, np.ones_like(gx)], -1)
nm /= np.linalg.norm(nm, axis=-1, keepdims=True)
nimg = ((nm * 0.5 + 0.5) * 255).astype(np.uint8)

bimg = (np.clip(base, 0, 1) ** (1 / 1.0) * 255).astype(np.uint8)
mr = np.zeros((ATLAS, ATLAS, 3), np.uint8)
mr[..., 0] = 255                                   # occlusion (unused)
mr[..., 1] = (np.clip(rough, 0.04, 1) * 255).astype(np.uint8)
mr[..., 2] = 0                                     # metallic

def img_bytes(arr, fmt="JPEG", q=90):
    b = io.BytesIO(); Image.fromarray(arr).save(b, fmt, quality=q, optimize=True); return b.getvalue()
os.makedirs(OUT, exist_ok=True)
Image.fromarray(bimg).resize((1024, 1024)).save(os.path.join(OUT, "atlas_preview.jpg"), quality=85)
IMG = [("T_Jois_BaseColor", img_bytes(bimg, q=90)), ("T_Jois_Normal", img_bytes(nimg, q=92)), ("T_Jois_ORM", img_bytes(mr, q=92))]

# ----------------------------------------------------------------- assemble mesh
log("assembling")
Vs, Ns, UVs, Js, Ws, Fs = [], [], [], [], [], []
D_all = [[] for _ in TARGETS]
off = 0
for p in parts:
    Vs.append(p["V"]); Ns.append(p["N"]); UVs.append(p["uv"]); Js.append(p["J"]); Ws.append(p["W"])
    Fs.append(p["F"] + off); off += len(p["V"])
    for i, d in enumerate(p["D"]): D_all[i].append(d)
V = np.vstack(Vs).astype(np.float32); N = np.vstack(Ns).astype(np.float32)
UV = np.vstack(UVs).astype(np.float32); J = np.vstack(Js).astype(np.uint8); W = np.vstack(Ws).astype(np.float32)
F = np.vstack(Fs).astype(np.uint32)
D_all = [np.vstack(d).astype(np.float32) for d in D_all]
N /= np.linalg.norm(N, axis=1, keepdims=True)
log(f"verts {len(V)}  tris {len(F)}  parts {len(parts)}")
assert len(V) < 65536

# ----------------------------------------------------------------- glTF writer
class GLB:
    def __init__(s): s.bin = bytearray(); s.views = []; s.acc = []
    def view(s, data, target=None):
        while len(s.bin) % 4: s.bin.append(0)
        o = len(s.bin); s.bin += data
        v = {"buffer": 0, "byteOffset": o, "byteLength": len(data)}
        if target: v["target"] = target
        s.views.append(v); return len(s.views) - 1
    def accessor(s, arr, ctype, typ, target=None, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": s.view(arr.tobytes(), target), "componentType": ctype, "count": len(arr), "type": typ}
        if normalized: a["normalized"] = True
        if minmax:
            a["min"] = arr.reshape(len(arr), -1).min(0).tolist(); a["max"] = arr.reshape(len(arr), -1).max(0).tolist()
        s.acc.append(a); return len(s.acc) - 1
    def sparse_vec3(s, d):
        nz = np.where(np.abs(d).max(1) > 0)[0]
        if len(nz) == 0: nz = np.array([0])
        ib = s.view(nz.astype(np.uint16 if len(d) < 65536 else np.uint32).tobytes())
        vb = s.view(np.ascontiguousarray(d[nz]).astype(np.float32).tobytes())
        a = {"componentType": 5126, "count": len(d), "type": "VEC3",
             "min": d.min(0).tolist(), "max": d.max(0).tolist(),
             "sparse": {"count": int(len(nz)), "indices": {"bufferView": ib, "componentType": 5123 if len(d) < 65536 else 5125},
                        "values": {"bufferView": vb}}}
        s.acc.append(a); return len(s.acc) - 1

g = GLB()
ARRAY, ELEM = 34962, 34963
a_pos = g.accessor(V, 5126, "VEC3", ARRAY, minmax=True)
a_nrm = g.accessor(N, 5126, "VEC3", ARRAY)
a_uv = g.accessor(UV, 5126, "VEC2", ARRAY)
a_j = g.accessor(J, 5121, "VEC4", ARRAY)
a_w = g.accessor(W, 5126, "VEC4", ARRAY)
a_idx = g.accessor(F.reshape(-1).astype(np.uint16), 5123, "SCALAR", ELEM)
morph_acc = [g.sparse_vec3(d) for d in D_all]

images, textures = [], []
for name, data in IMG:
    images.append({"name": name, "mimeType": "image/jpeg", "bufferView": g.view(data)})
    textures.append({"sampler": 0, "source": len(images) - 1, "name": name})

nodes = []
bone_node = {}
for i, (bn, parent, pos) in enumerate(BONES):
    ppos = BPOS[parent] if parent else v3(0, 0, 0)
    nodes.append({"name": bn, "translation": (pos - ppos).tolist()})
    bone_node[bn] = i
for i, (bn, parent, pos) in enumerate(BONES):
    if parent: nodes[bone_node[parent]].setdefault("children", []).append(i)
ibm = np.stack([np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-p[0], -p[1], -p[2], 1]], np.float32) for _, _, p in BONES])
a_ibm = g.accessor(ibm.reshape(len(BONES), 16), 5126, "MAT4")
mesh_node = len(nodes)
nodes.append({"name": "Jois_Avatar", "mesh": 0, "skin": 0})
root = len(nodes)
nodes.append({"name": "Armature", "children": [bone_node["Hips"], mesh_node]})

EMOTIONS = {
    "joy": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0, "cheekSquintLeft": 0.8, "cheekSquintRight": 0.8,
            "eyeSquintLeft": 0.65, "eyeSquintRight": 0.65, "browInnerUp": 0.2, "browOuterUpLeft": 0.25, "browOuterUpRight": 0.25,
            "mouthDimpleLeft": 0.2, "mouthDimpleRight": 0.2},
    "surprise": {"browInnerUp": 1.0, "browOuterUpLeft": 1.0, "browOuterUpRight": 1.0, "eyeWideLeft": 1.0, "eyeWideRight": 1.0,
                 "jawOpen": 0.45, "mouthFunnel": 0.35},
    "focus": {"browDownLeft": 0.85, "browDownRight": 0.85, "eyeSquintLeft": 0.55, "eyeSquintRight": 0.55,
              "mouthPressLeft": 0.6, "mouthPressRight": 0.6, "noseSneerLeft": 0.15, "noseSneerRight": 0.15, "mouthLeft": 0.2},
    "empathy": {"browInnerUp": 1.0, "browDownLeft": 0.15, "browDownRight": 0.15, "mouthFrownLeft": 0.55, "mouthFrownRight": 0.55,
                "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3, "mouthPressLeft": 0.25, "mouthPressRight": 0.25,
                "eyeBlinkLeft": 0.15, "eyeBlinkRight": 0.15, "mouthShrugLower": 0.3},
}
gltf = {
    "asset": {"version": "2.0", "generator": "Jois Avatar Kit generator 1.0",
              "extras": {"jois": {"arkit": ARKIT, "visemes": [f"viseme_{v}" for v in VIS], "emotions": EMOTIONS,
                                   "visemeRecipes": VISEME_MIX}}},
    "scene": 0, "scenes": [{"name": "Jois", "nodes": [root]}],
    "nodes": nodes,
    "meshes": [{"name": "SK_Jois_Avatar", "extras": {"targetNames": TARGETS},
                "weights": [0.0] * len(TARGETS),
                "primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv,
                                               "JOINTS_0": a_j, "WEIGHTS_0": a_w},
                                "indices": a_idx, "material": 0, "mode": 4,
                                "targets": [{"POSITION": a} for a in morph_acc]}]}],
    "skins": [{"name": "SKN_Jois", "joints": list(range(len(BONES))), "inverseBindMatrices": a_ibm, "skeleton": bone_node["Hips"]}],
    "materials": [{"name": "M_Jois_Atlas",
                   "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 2},
                                            "metallicFactor": 1.0, "roughnessFactor": 1.0},
                   "normalTexture": {"index": 1, "scale": 0.6}}],
    "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
    "textures": textures, "images": images,
    "accessors": g.acc, "bufferViews": g.views,
}
while len(g.bin) % 4: g.bin.append(0)
gltf["buffers"] = [{"byteLength": len(g.bin)}]
js = json.dumps(gltf, separators=(",", ":")).encode()
while len(js) % 4: js += b" "
glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(g.bin))
glb += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(g.bin), 0x004E4942) + bytes(g.bin)
path = os.path.join(OUT, "jois_avatar.glb")
open(path, "wb").write(glb)
stats = {"triangles": int(len(F)), "vertices": int(len(V)), "morphTargets": len(TARGETS), "bones": len(BONES),
         "drawCalls": 1, "materials": 1, "textures": 3, "textureSize": ATLAS, "fileMB": round(len(glb) / 1e6, 2),
         "parts": {p["name"]: int(len(p["F"])) for p in parts}}
json.dump(stats, open(os.path.join(OUT, "avatar_stats.json"), "w"), indent=1)
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_build.npz"), V=V, N=N, UV=UV, F=F,
                    D=np.stack(D_all), names=np.array(TARGETS), base=bimg)
log("wrote", path, stats)
