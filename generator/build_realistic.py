"""
Jois Avatar — realistic build on the MakeHuman (CC0) base mesh.

young woman, chin-length bob with straight bangs, school-style outfit
(camel blazer, white shirt, red tie, grey pleated skirt, black tights, black shoes),
52 ARKit + 15 visemes baked from the MakeHuman face pose units, reduced humanoid skeleton.

python3 build_realistic.py <out_dir>
"""
import io, json, math, os, struct, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from skimage import measure
import fast_simplification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mh_base import MH, load_obj, apply_targets, skeleton, load_weights, face_units

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
rng = np.random.default_rng(11)
def v3(*a): return np.array(a, float)
def norm(a, axis=1): return np.linalg.norm(a, axis=axis)
def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0); return t * t * (3 - 2 * t)

# ------------------------------------------------------------------ 1. shape the body
log("loading MakeHuman base")
V0, VT, FACES, FUV, FG = load_obj()
TARGETS_W = {
    "macrodetails/asian-female-young.target": 0.55,
    "macrodetails/caucasian-female-young.target": 0.45,
    "macrodetails/universal-female-young-averagemuscle-minweight.target": 0.35,
    "macrodetails/proportions/female-young-averagemuscle-averageweight-idealproportions.target": 1.0,
    "macrodetails/height/female-young-averagemuscle-averageweight-maxheight.target": 0.15,
    "breast/female-young-averagemuscle-averageweight-mincup-averagefirmness.target": 0.55,
    "head/head-oval.target": 0.5, "head/head-fat-decr.target": 0.25,
    "eyes/l-eye-scale-incr.target": 0.5, "eyes/r-eye-scale-incr.target": 0.5,
    "eyes/l-eye-corner2-up.target": 0.35, "eyes/r-eye-corner2-up.target": 0.35,
    "eyes/l-eye-bag-decr.target": 0.6, "eyes/r-eye-bag-decr.target": 0.6,
    "breast/nipple-size-decr.target": 1.0, "breast/nipple-point-decr.target": 1.0,
    "nose/nose-hump-decr.target": 0.4, "nose/nose-point-up.target": 0.25, "nose/nose-volume-decr.target": 0.2,
    "mouth/mouth-cupidsbow-incr.target": 0.35, "mouth/mouth-scale-horiz-decr.target": 0.12,
    "cheek/l-cheek-volume-decr.target": 0.25, "cheek/r-cheek-volume-decr.target": 0.25,
    "eyes/l-eye-height2-incr.target": 0.25, "eyes/r-eye-height2-incr.target": 0.25,
    "nose/nose-scale-horiz-decr.target": 0.3, "nose/nose-scale-depth-decr.target": 0.15,
    "nose/nose-point-width-decr.target": 0.35, "nose/nose-nostrils-width-decr.target": 0.3,
    "mouth/mouth-lowerlip-volume-decr.target": 0.3, "mouth/mouth-upperlip-volume-decr.target": 0.2, "mouth/mouth-scale-horiz-decr.target": 0.05,
    "mouth/mouth-angles-up.target": 0.3,
    "chin/chin-width-decr.target": 0.5, "chin/chin-height-decr.target": 0.15,
    "cheek/l-cheek-bones-incr.target": 0.15, "cheek/r-cheek-bones-incr.target": 0.15,
    "head/head-invertedtriangular.target": 0.35, "head/head-age-decr.target": 0.5, "head/head-scale-horiz-decr.target": 0.12,
    "chin/chin-triangle.target": 0.3, "chin/chin-bones-decr.target": 0.4, "chin/chin-jaw-drop-decr.target": 0.3,
    "neck/neck-scale-horiz-decr.target": 0.35, "cheek/l-cheek-trans-up.target": 0.15, "cheek/r-cheek-trans-up.target": 0.15,
}
V = apply_targets(V0, TARGETS_W)
BONES = skeleton(V); WTS = load_weights()
log("baking face pose units")
UNITS, _ = face_units(V, BONES, WTS)

V = V + 0.45 * UNITS["lowerLipUp"] + 0.15 * UNITS["UpperLipForward"]
GROUP = {}
for k, g in enumerate(FG): GROUP.setdefault(g, []).append(k)
def verts_of(g): return np.unique(np.concatenate([FACES[k] for k in GROUP[g]]))

# ------------------------------------------------------------------ eye proxy (mhclo fitting)
def load_mhclo(path):
    lines = open(path).read().splitlines(); refs = []; scales = {}; inv = False
    for ln in lines:
        p = ln.split()
        if not p or p[0].startswith("#"): continue
        if p[0] in ("x_scale", "y_scale", "z_scale"): scales[p[0][0]] = (int(p[1]), int(p[2]), float(p[3])); continue
        if p[0] == "verts": inv = True; continue
        if inv:
            if len(p) == 9 and all(c.lstrip("-").replace(".", "").isdigit() for c in p):
                refs.append((int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]), float(p[7]), float(p[8])))
            elif len(p) == 1 and p[0].isdigit():
                refs.append((int(p[0]), int(p[0]), int(p[0]), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            else:
                inv = False
    R = np.array(refs)
    return R[:, :3].astype(int), R[:, 3:6], R[:, 6:9], scales
EYE_IDX, EYE_W, EYE_OFF, EYE_SC = load_mhclo(f"{MH}/eyes/high-poly/high-poly.mhclo")
def fit_eyes(Vb):
    s = np.array([abs(Vb[EYE_SC[a][0], i] - Vb[EYE_SC[a][1], i]) / EYE_SC[a][2] for i, a in enumerate("xyz")])
    return (Vb[EYE_IDX] * EYE_W[:, :, None]).sum(1) + EYE_OFF * s
def load_proxy_obj(path):
    Vp, VTp, Fp, FUp = [], [], [], []
    for line in open(path):
        if line.startswith("v "): Vp.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "): VTp.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            pp = line.split()[1:]; Fp.append([int(q.split("/")[0]) - 1 for q in pp]); FUp.append([int(q.split("/")[1]) - 1 for q in pp])
    return np.array(VTp), Fp, FUp
EYE_VT, EYE_F, EYE_FU = load_proxy_obj(f"{MH}/eyes/high-poly/high-poly.obj")
EYE_V = fit_eyes(V)
assert len(EYE_V) == len(EYE_IDX)

# ------------------------------------------------------------------ 2. ARKit + visemes from pose units
X_DM = V[:, 0]; Y_DM = V[:, 1]
LEFT = ss(-0.12, 0.12, X_DM); RIGHT = 1 - LEFT
eyeL_dm = BONES["eye.L"]["head"]; eyeR_dm = BONES["eye.R"]["head"]
teeth_up = V[verts_of("helper-upper-teeth")]; teeth_lo = V[verts_of("helper-lower-teeth")]
MOUTH_Y_DM = (teeth_up[:, 1].min() + teeth_lo[:, 1].max()) / 2
LID_BOTTOM_DM = eyeL_dm[1] - 0.16
MOUTH_ZONE = ss(LID_BOTTOM_DM - 0.05, LID_BOTTOM_DM - 0.35, Y_DM)       # 1 below the eyes
EYE_ZONE = ss(MOUTH_Y_DM + 0.18, MOUTH_Y_DM + 0.32, Y_DM)               # 1 above the nose tip
U = UNITS
def u(n, w=1.0, mask=None):
    d = U[n] * w
    return d if mask is None else d * mask[:, None]
ARKIT = {
    "eyeBlinkLeft": lambda: u("LeftUpperLidClosed"), "eyeBlinkRight": lambda: u("RightUpperLidClosed"),
    "eyeLookDownLeft": lambda: u("LeftEyeDown"), "eyeLookDownRight": lambda: u("RightEyeDown"),
    "eyeLookUpLeft": lambda: u("LeftEyeUp"), "eyeLookUpRight": lambda: u("RightEyeUp"),
    "eyeLookInLeft": lambda: u("LeftEyeturnRight"), "eyeLookOutLeft": lambda: u("LeftEyeturnLeft"),
    "eyeLookInRight": lambda: u("RightEyeturnLeft"), "eyeLookOutRight": lambda: u("RightEyeturnRight"),
    "eyeSquintLeft": lambda: u("LeftLowerLidUp") + u("LeftCheekUp", 0.25, EYE_ZONE * LEFT),
    "eyeSquintRight": lambda: u("RightLowerLidUp") + u("RightCheekUp", 0.25, EYE_ZONE * RIGHT),
    "eyeWideLeft": lambda: u("LeftUpperLidOpen"), "eyeWideRight": lambda: u("RightUpperLidOpen"),
    "jawForward": lambda: u("ChinForward"), "jawLeft": lambda: u("ChinLeft"), "jawRight": lambda: u("ChinRight"),
    "jawOpen": lambda: u("JawDrop"),
    "mouthClose": lambda: u("lowerLipUp", 0.6) + u("UpperLipForward", 0.2),
    "mouthFunnel": lambda: u("LipsKiss", 0.55) + u("lowerLipDown", 0.35) + u("UpperLipUp", 0.35),
    "mouthPucker": lambda: u("LipsKiss"),
    "mouthLeft": lambda: u("MouthMoveLeft"), "mouthRight": lambda: u("MouthMoveRight"),
    "mouthSmileLeft": lambda: u("MouthLeftPullUp"), "mouthSmileRight": lambda: u("MouthRightPullUp"),
    "mouthFrownLeft": lambda: u("MouthLeftPullDown"), "mouthFrownRight": lambda: u("MouthRightPullDown"),
    "mouthDimpleLeft": lambda: u("MouthLeftPullSide", 0.45), "mouthDimpleRight": lambda: u("MouthRightPullSide", 0.45),
    "mouthStretchLeft": lambda: u("MouthLeftPullSide") + u("MouthLeftPlatysma", 0.3),
    "mouthStretchRight": lambda: u("MouthRightPullSide") + u("MouthRightPlatysma", 0.3),
    "mouthRollLower": lambda: u("lowerLipBackward"), "mouthRollUpper": lambda: u("UpperLipBackward"),
    "mouthShrugLower": lambda: u("lowerLipUp", 0.5) + u("lowerLipForward", 0.4),
    "mouthShrugUpper": lambda: u("UpperLipUp", 0.4) + u("UpperLipForward", 0.3),
    "mouthPressLeft": lambda: u("lowerLipUp", 0.35, LEFT) + u("UpperLipBackward", 0.3, LEFT),
    "mouthPressRight": lambda: u("lowerLipUp", 0.35, RIGHT) + u("UpperLipBackward", 0.3, RIGHT),
    "mouthLowerDownLeft": lambda: u("lowerLipDown", 1, LEFT), "mouthLowerDownRight": lambda: u("lowerLipDown", 1, RIGHT),
    "mouthUpperUpLeft": lambda: u("UpperLipUp", 1, LEFT), "mouthUpperUpRight": lambda: u("UpperLipUp", 1, RIGHT),
    "browDownLeft": lambda: u("LeftBrowDown"), "browDownRight": lambda: u("RightBrowDown"),
    "browInnerUp": lambda: u("LeftInnerBrowUp") + u("RightInnerBrowUp"),
    "browOuterUpLeft": lambda: u("LeftOuterBrowUp"), "browOuterUpRight": lambda: u("RightOuterBrowUp"),
    "cheekPuff": lambda: u("CheeksPump"),
    "cheekSquintLeft": lambda: u("LeftCheekUp"), "cheekSquintRight": lambda: u("RightCheekUp"),
    "noseSneerLeft": lambda: u("NoseWrinkler", 1, LEFT), "noseSneerRight": lambda: u("NoseWrinkler", 1, RIGHT),
    "tongueOut": lambda: u("TongueOut"),
}
ARKIT_ORDER = ["eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "tongueOut"]
assert len(ARKIT_ORDER) == 52
VIS = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou"]
VISEME_MIX = {
    "sil": {}, "PP": {"mouthClose": 0.6, "mouthPressLeft": 0.6, "mouthPressRight": 0.6},
    "FF": {"jawOpen": 0.10, "mouthRollLower": 0.7, "mouthUpperUpLeft": 0.25, "mouthUpperUpRight": 0.25},
    "TH": {"jawOpen": 0.22, "tongueOut": 0.3}, "DD": {"jawOpen": 0.28, "mouthStretchLeft": 0.2, "mouthStretchRight": 0.2},
    "kk": {"jawOpen": 0.32, "mouthStretchLeft": 0.3, "mouthStretchRight": 0.3},
    "CH": {"jawOpen": 0.2, "mouthFunnel": 0.6, "mouthPucker": 0.2},
    "SS": {"jawOpen": 0.08, "mouthStretchLeft": 0.45, "mouthStretchRight": 0.45, "mouthSmileLeft": 0.15, "mouthSmileRight": 0.15},
    "nn": {"jawOpen": 0.16, "mouthStretchLeft": 0.15, "mouthStretchRight": 0.15},
    "RR": {"jawOpen": 0.18, "mouthPucker": 0.45, "mouthFunnel": 0.3},
    "aa": {"jawOpen": 0.62, "mouthLowerDownLeft": 0.3, "mouthLowerDownRight": 0.3, "mouthUpperUpLeft": 0.15, "mouthUpperUpRight": 0.15},
    "E": {"jawOpen": 0.35, "mouthStretchLeft": 0.5, "mouthStretchRight": 0.5, "mouthSmileLeft": 0.25, "mouthSmileRight": 0.25},
    "ih": {"jawOpen": 0.26, "mouthStretchLeft": 0.35, "mouthStretchRight": 0.35},
    "oh": {"jawOpen": 0.45, "mouthFunnel": 0.75}, "ou": {"jawOpen": 0.2, "mouthPucker": 0.85, "mouthFunnel": 0.2},
}
log("composing blendshapes")
SHAPES = {}
for n in ARKIT_ORDER:
    d = ARKIT[n]()
    if n.startswith(("jaw", "mouth", "tongue", "cheekPuff")): d = d * MOUTH_ZONE[:, None]
    if n.startswith(("eye", "brow")): d = d * EYE_ZONE[:, None]
    SHAPES[n] = d
for v in VIS:
    d = np.zeros_like(V)
    for k, w in VISEME_MIX[v].items(): d += w * SHAPES[k]
    SHAPES["viseme_" + v] = d
TARGET_NAMES = ARKIT_ORDER + [f"viseme_{v}" for v in VIS]
EYE_SHAPES = {n: fit_eyes(V + SHAPES[n]) - EYE_V for n in TARGET_NAMES}

# ------------------------------------------------------------------ 3. units: decimetres -> metres, feet on the floor
body_v = verts_of("body")
FLOOR = V[body_v, 1].min()
def to_m(P): return (P - v3(0, FLOOR, 0)) * 0.1
VM = to_m(V); EYE_VM = to_m(EYE_V)
SHAPES = {n: d * 0.1 for n, d in SHAPES.items()}; EYE_SHAPES = {n: d * 0.1 for n, d in EYE_SHAPES.items()}
B = {n: dict(parent=b["parent"], head=to_m(b["head"]), tail=to_m(b["tail"])) for n, b in BONES.items()}
EYE_L, EYE_R = B["eye.L"]["head"], B["eye.R"]["head"]
MOUTH_Y = (MOUTH_Y_DM - FLOOR) * 0.1
HEAD_C = (EYE_L + EYE_R) / 2
log("eyes at", EYE_L.round(3), "mouth y", round(MOUTH_Y, 3))

# dominant MakeHuman bone per base vertex (used for regions and the reduced rig)
NB = len(VM)
DOM_W = np.zeros(NB); DOM = np.array([""] * NB, dtype=object)
for bn, (idx, w) in WTS.items():
    upd = w > DOM_W[idx]; DOM_W[idx[upd]] = w[upd]; DOM[idx[upd]] = bn

# ------------------------------------------------------------------ 4. reduced humanoid skeleton
ANCHOR = {"root": "Hips", "spine05": "Spine", "spine03": "Spine1", "spine02": "Spine2", "neck01": "Neck", "head": "Head",
          "eye.L": "LeftEye", "eye.R": "RightEye",
          "clavicle.L": "LeftShoulder", "upperarm01.L": "LeftArm", "lowerarm01.L": "LeftForeArm", "wrist.L": "LeftHand",
          "clavicle.R": "RightShoulder", "upperarm01.R": "RightArm", "lowerarm01.R": "RightForeArm", "wrist.R": "RightHand",
          "upperleg01.L": "LeftUpLeg", "lowerleg01.L": "LeftLeg", "foot.L": "LeftFoot",
          "upperleg01.R": "RightUpLeg", "lowerleg01.R": "RightLeg", "foot.R": "RightFoot"}
def reduced(bn):
    while bn is not None:
        if bn in ANCHOR: return ANCHOR[bn]
        bn = BONES[bn]["parent"]
    return "Hips"
RB = [("Hips", None, "root"), ("Spine", "Hips", "spine05"), ("Spine1", "Spine", "spine03"), ("Spine2", "Spine1", "spine02"),
      ("Neck", "Spine2", "neck01"), ("Head", "Neck", "head"), ("LeftEye", "Head", "eye.L"), ("RightEye", "Head", "eye.R")]
for S in ("Left", "Right"):
    s = S[0]
    RB += [(f"{S}Shoulder", "Spine2", f"clavicle.{s}"), (f"{S}Arm", f"{S}Shoulder", f"upperarm01.{s}"),
           (f"{S}ForeArm", f"{S}Arm", f"lowerarm01.{s}"), (f"{S}Hand", f"{S}ForeArm", f"wrist.{s}"),
           (f"{S}UpLeg", "Hips", f"upperleg01.{s}"), (f"{S}Leg", f"{S}UpLeg", f"lowerleg01.{s}"), (f"{S}Foot", f"{S}Leg", f"foot.{s}")]
RBI = {r[0]: i for i, r in enumerate(RB)}
RBPOS = {r[0]: B[r[2]]["head"] for r in RB}
RBPOS["HeadTop_End"] = v3(0, VM[body_v, 1].max(), B["head"]["head"][2]); RB.append(("HeadTop_End", "Head", None)); RBI["HeadTop_End"] = len(RB) - 1
# per-base-vertex reduced weights (top 4)
RW = np.zeros((NB, len(RB)))
for bn, (idx, w) in WTS.items(): RW[idx, RBI[reduced(bn)]] += w
s_ = RW.sum(1); RW[s_ == 0, RBI["Hips"]] = 1; RW /= RW.sum(1, keepdims=True)
def top4(Wm):
    idx = np.argsort(-Wm, 1)[:, :4]; w = np.take_along_axis(Wm, idx, 1); w[w < 0.01] = 0
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9); return idx, w

# ------------------------------------------------------------------ 5. mesh parts
def tris_from(faces_list, fuv_list):
    T, TU = [], []
    for f, fu in zip(faces_list, fuv_list):
        for a, b, c in ((0, 1, 2), (0, 2, 3)) if len(f) == 4 else ((0, 1, 2),):
            T.append([f[a], f[b], f[c]]); TU.append([fu[a], fu[b], fu[c]])
    return np.array(T), np.array(TU)

def vertex_normals(P, T):
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]); N = np.zeros_like(P)
    for k in range(3): np.add.at(N, T[:, k], fn)
    l = norm(N)[:, None]; l[l == 0] = 1; return N / l

parts = []
def part_from_group(groups, name, uv_xform):
    ks = [k for g in groups for k in GROUP[g]]
    T, TU = tris_from([FACES[k] for k in ks], [FUV[k] for k in ks])
    # unique (vertex, uv) pairs -> render vertices
    key = T.astype(np.int64) * 100000 + TU
    uniq, inv = np.unique(key.ravel(), return_inverse=True)
    vi = uniq // 100000; ui = uniq % 100000
    p = dict(name=name, base=vi, F=inv.reshape(-1, 3), uvsrc=VT[ui])
    p["uv"] = uv_xform(VT[ui])
    parts.append(p); return p

BODY_U = 0.75   # body occupies the left 3/4 of the 4096 atlas
def body_uv(t): return np.stack([t[:, 0] * BODY_U, 1 - t[:, 1]], 1)
body = part_from_group(["body"], "body", body_uv)

def tile_uv(tile):
    x0, y0, w, h = tile
    def f(t):
        tt = (t - t.min(0)) / np.maximum(np.ptp(t, 0), 1e-9)
        return np.stack([(x0 + 4 + tt[:, 0] * (w - 8)) / 4096, (y0 + 4 + (1 - tt[:, 1]) * (h - 8)) / 4096], 1)
    return f
TILES = {"hair": (3072, 0, 1024, 1024), "skirt": (3072, 1024, 1024, 1024), "eye": (3072, 2048, 512, 512),
         "teethU": (3584, 2048, 256, 128), "teethD": (3584, 2176, 256, 128), "tongue": (3840, 2048, 256, 256),
         "hair_cap": (3072, 2560, 512, 512), "hair_bot": (3584, 2560, 512, 512)}
skirt = part_from_group(["helper-skirt"], "skirt", tile_uv(TILES["skirt"]))
part_from_group(["helper-upper-teeth"], "teethU", tile_uv(TILES["teethU"]))
part_from_group(["helper-lower-teeth"], "teethD", tile_uv(TILES["teethD"]))
part_from_group(["helper-tongue"], "tongue", tile_uv(TILES["tongue"]))
lash_groups = [g for g in GROUP if "eyelashes" in g]
lashes = part_from_group(lash_groups, "lashes", lambda t: t)    # remapped below, own material

for p in parts:
    p["V"] = VM[p["base"]].copy()
    p["D"] = {n: SHAPES[n][p["base"]] for n in TARGET_NAMES}
    p["J"], p["W"] = top4(RW[p["base"]])

# teeth: a touch narrower and deeper, so their corners stay inside the slimmer lower face when she smiles
for p in parts:
    if p["name"] in ("teethU", "teethD"):
        c_ = p["V"].mean(0)
        p["V"][:, 0] = c_[0] + (p["V"][:, 0] - c_[0]) * 0.86
        p["V"][:, 2] -= 0.0025
        for n_ in p["D"]: p["D"][n_][:, 0] *= 0.86

# skirt: shorten to mid-thigh, flare, pleats
sk = skirt
hip_y = B["upperleg01.L"]["head"][1]
hem_y = hip_y - 0.30
P = sk["V"]
keep_v = P[:, 1] > hem_y - 0.02
keep_f = keep_v[sk["F"]].all(1)
def subset(p, fmask):
    used = np.unique(p["F"][fmask]); remap = -np.ones(len(p["V"]), int); remap[used] = np.arange(len(used))
    p["F"] = remap[p["F"][fmask]]
    for k in ("V", "uv", "base", "J", "W", "uvsrc"): p[k] = p[k][used]
    p["D"] = {n: d[used] for n, d in p["D"].items()}
subset(sk, keep_f)
P = sk["V"]; cx, cz = 0.0, B["root"]["head"][2]
t = np.clip((hip_y + 0.02 - P[:, 1]) / 0.32, 0, 1)
ang = np.arctan2(P[:, 0] - cx, P[:, 2] - cz)
rad = np.hypot(P[:, 0] - cx, P[:, 2] - cz)
rad2 = rad * (1 + 0.32 * t ** 1.3) + 0.004 * t * np.sign(np.sin(ang * 14)) * np.abs(np.sin(ang * 14)) ** 0.6
P[:, 0] = cx + rad2 * np.sin(ang); P[:, 2] = cz + rad2 * np.cos(ang)
sk["pleat"] = np.sin(ang * 14)

# eyes (high-poly proxy) — weighted to the eye bones
eT, eTU = tris_from(EYE_F, EYE_FU)
_ea = np.asarray(Image.open(f"{MH}/eyes/materials/brown_eye.png"))[..., 3]
_c = EYE_VT[eTU].mean(1); _keep = _ea[((1 - _c[:, 1]) * _ea.shape[0]).astype(int).clip(0, _ea.shape[0] - 1), (_c[:, 0] * _ea.shape[1]).astype(int).clip(0, _ea.shape[1] - 1)] > 128
eT, eTU = eT[_keep], eTU[_keep]
key = eT.astype(np.int64) * 100000 + eTU; uniq, inv = np.unique(key.ravel(), return_inverse=True)
vi = uniq // 100000; ui = uniq % 100000
eye = dict(name="eyes", base=None, F=inv.reshape(-1, 3), V=EYE_VM[vi].copy(), uvsrc=EYE_VT[ui])
eye["uv"] = np.stack([(TILES["eye"][0] + 4 + EYE_VT[ui][:, 0] * (TILES["eye"][2] - 8)) / 4096,
                      (TILES["eye"][1] + 4 + (1 - EYE_VT[ui][:, 1]) * (TILES["eye"][3] - 8)) / 4096], 1)
eye["D"] = {n: EYE_SHAPES[n][vi] for n in TARGET_NAMES}
side = (eye["V"][:, 0] > 0)
eye["J"] = np.zeros((len(vi), 4), int); eye["J"][:, 0] = np.where(side, RBI["LeftEye"], RBI["RightEye"])
eye["W"] = np.zeros((len(vi), 4)); eye["W"][:, 0] = 1
parts.append(eye)

# ------------------------------------------------------------------ 6. clothing: inflate the body where the outfit is
bp = body; Pb = bp["V"]; dom = DOM[bp["base"]]
def dom_is(*names): return np.array([any(d.startswith(n) for n in names) for d in dom])
neck_y = B["neck01"]["head"][1]
collar_y = neck_y + 0.012
_sk_top = VM[verts_of("helper-skirt"), 1].max()
blazer_hem = _sk_top - 0.035
wristL, wristR = B["wrist.L"]["head"], B["wrist.R"]["head"]
elbowL = B["lowerarm01.L"]["head"]; elbowR = B["lowerarm01.R"]["head"]
ankle_y = (B["foot.L"]["head"][1] + B["foot.R"]["head"][1]) / 2
def along(P, a, b): ab = b - a; return ((P - a) @ ab) / (ab @ ab)
armL = dom_is("upperarm", "lowerarm", "shoulder", "clavicle") & (Pb[:, 0] > 0)
armR = dom_is("upperarm", "lowerarm", "shoulder", "clavicle") & (Pb[:, 0] < 0)
hand = dom_is("wrist", "metacarpal", "finger")
# sleeve ends 2.5 cm before the wrist joint
sl = np.zeros(len(Pb))
sl[Pb[:, 0] > 0] = ss(1.02, 0.95, along(Pb[Pb[:, 0] > 0], elbowL, wristL))
sl[Pb[:, 0] < 0] = ss(1.02, 0.95, along(Pb[Pb[:, 0] < 0], elbowR, wristR))
arm_dom = dom_is("upperarm", "lowerarm")
not_head = ~dom_is("head", "jaw", "neck", "levator", "oris", "orbicularis", "risorius", "special", "temporalis", "tongue", "eye")
torso_w = (ss(blazer_hem - 0.005, blazer_hem + 0.005, Pb[:, 1]) * ss(collar_y + 0.006, collar_y - 0.006, Pb[:, 1]) * not_head
           * ~dom_is("upperleg", "lowerleg", "foot", "toe"))
W_BLAZER = np.where(hand, 0.0, np.where(arm_dom, sl, torso_w))
W_SHOE = ss(ankle_y + 0.035, ankle_y + 0.02, Pb[:, 1]) * dom_is("foot", "toe", "lowerleg").astype(float)
W_LEGS = dom_is("upperleg", "lowerleg", "pelvis", "foot", "toe").astype(float) * (1 - W_SHOE)
# smooth the region weights over the mesh so the inflation has no cracks
nb = len(Pb)
E = np.concatenate([bp["F"][:, [0, 1]], bp["F"][:, [1, 2]], bp["F"][:, [2, 0]]]); E = np.concatenate([E, E[:, ::-1]])
A = sparse.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nb, nb)).tocsr(); A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
# vertices sharing a position (uv seams) must move together: weld by base index
def weld_avg(x):
    s = np.zeros(NB); c = np.zeros(NB); np.add.at(s, bp["base"], x); np.add.at(c, bp["base"], 1)
    return (s / np.maximum(c, 1))[bp["base"]]
def smooth(x, it):
    for _ in range(it): x = weld_avg(0.5 * x + 0.5 * (A @ x) / deg)
    return x
Wb_s = smooth(W_BLAZER, 3); Ws_s = smooth(W_SHOE, 3)
Nb_ = vertex_normals(Pb, bp["F"]); Nb_ = np.stack([weld_avg(Nb_[:, i]) for i in range(3)], 1); Nb_ /= norm(Nb_)[:, None]
off = 0.007 * Wb_s + 0.0045 * Ws_s
Pi = Pb + Nb_ * off[:, None]
wsm = np.clip(np.maximum(Wb_s, Ws_s), 0, 1)
# the clothed surface IS the body surface (no skin underneath), so it can simply be low-pass filtered:
# Taubin smoothing removes centimetre-scale anatomy (nipples, navel, ribs) and keeps the overall silhouette
def lap_of(P): return np.stack([weld_avg((A @ P[:, i]) / deg) for i in range(3)], 1) - P
wcl = np.clip(Wb_s * 1.2, 0, 1)[:, None]
for it in range(160):
    Pi = Pi + 0.62 * wcl * lap_of(Pi)
    Pi = Pi - 0.66 * wcl * lap_of(Pi)
for it in range(25):
    Pi = Pi + 0.5 * (np.clip(Ws_s, 0, 1)[:, None]) * lap_of(Pi)     # shoes: smooth out toes
# bust under the blazer: find each breast's most forward point and relax a 4 cm patch around it into a soft dome
chest_y = (B["spine02"]["head"][1] + collar_y) / 2 - 0.06
for sgn in (1, -1):
    cand = (np.sign(Pb[:, 0]) == sgn) & (np.abs(Pb[:, 0]) > 0.03) & (np.abs(Pb[:, 0]) < 0.14) & (np.abs(Pb[:, 1] - chest_y) < 0.10) & (wcl[:, 0] > 0.5)
    if not cand.any(): continue
    ci = np.where(cand)[0]; top = ci[np.argsort(-Pb[ci, 2])[:60]]
    tip = Pb[top].mean(0)
    wl = (np.clip(1 - norm(Pb - tip) / 0.075, 0, 1) ** 1.2 * wcl[:, 0])[:, None]
    for it in range(200):
        Pi = Pi + 0.5 * wl * lap_of(Pi)
    # dense vertex cluster at the old nipple: rebuild its height from a smooth quadratic fitted to the surrounding ring
    rxy = np.hypot(Pi[:, 0] - tip[0], Pi[:, 1] - tip[1]); front = Pi[:, 2] > tip[2] - 0.04
    inner = front & (rxy < 0.034); ring = front & (rxy >= 0.034) & (rxy < 0.06)
    xx_, yy_ = Pi[ring, 0] - tip[0], Pi[ring, 1] - tip[1]
    M_ = np.stack([np.ones_like(xx_), xx_, yy_, xx_ ** 2, xx_ * yy_, yy_ ** 2], 1)
    coef, *_ = np.linalg.lstsq(M_, Pi[ring, 2], rcond=None)
    xi, yi = Pi[inner, 0] - tip[0], Pi[inner, 1] - tip[1]
    zfit = np.stack([np.ones_like(xi), xi, yi, xi ** 2, xi * yi, yi ** 2], 1) @ coef
    blend = ss(0.034, 0.008, rxy[inner])
    Pi[inner, 2] = Pi[inner, 2] * (1 - blend) + zfit * blend
    log("bust relaxed around", tip.round(3), "patch", int(inner.sum()), "verts")
# whole front of the torso under the blazer: heavy low-pass so the fabric reads as one smooth shell
spz = B["spine02"]["head"][2]
fr = (np.clip(Wb_s, 0, 1) * ss(spz - 0.01, spz + 0.04, Pb[:, 2]) * ss(blazer_hem + 0.01, blazer_hem + 0.05, Pb[:, 1])
      * ss(collar_y - 0.02, collar_y - 0.07, Pb[:, 1]) * ~dom_is("upperarm", "lowerarm"))[:, None]
for it in range(300):
    Pi = Pi + 0.6 * fr * lap_of(Pi)
    Pi = Pi - 0.62 * fr * lap_of(Pi)
for it in range(60):
    Pi = Pi + 0.5 * fr * lap_of(Pi)
# keep the fabric outside the skin everywhere
# (not on the smoothed front: there the fabric replaces the skin, and pushing it out would bring the anatomy back)
dd = ((Pi - Pb) * Nb_).sum(1); Pi = Pi + Nb_ * (np.maximum(0, 0.003 * np.clip(Wb_s, 0, 1) - dd) * (1 - fr[:, 0]))[:, None]
bp["V"] = Pi
bp["W_BLAZER"], bp["W_SHOE"], bp["W_LEGS"] = W_BLAZER, W_SHOE, W_LEGS
bp["FR"] = fr[:, 0]

# ------------------------------------------------------------------ 7. hair: chin-length bob with straight bangs
log("hair")
head_mask = dom_is("head", "jaw", "levator", "oris", "orbicularis", "risorius", "special", "temporalis", "neck")
HP = bp["V"][head_mask]; HN = Nb_[head_mask]
tree = cKDTree(HP)
def head_sdf(P):
    d, i = tree.query(P); s = np.sign(((P - HP[i]) * HN[i]).sum(1)); return d * s
scalp = VM[verts_of("helper-hair")]
SC = (scalp.max(0) + scalp.min(0)) / 2; SR = (scalp.max(0) - scalp.min(0)) / 2
SC[1] = scalp[:, 1].max() - SR[1] * 1.0
brow_y = EYE_L[1] + 0.024
face_front = HP[:, 2].max()
def sd_ell(P, c, r):
    q = (P - c) / r; k0 = norm(q); k1 = norm((P - c) / (r * r)); return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1); return b * (1 - h) + a * h - k * h * (1 - h)
def smax(a, b, k): return -smin(-a, -b, k)
CHIN_Y = MOUTH_Y - 0.040
SHOULDER_Y = (B["clavicle.L"]["head"][1] + B["upperarm01.L"]["head"][1]) / 2
HEM_Y = SHOULDER_Y - 0.27                     # hair ends at mid-back
BODY_V = bp["V"][~head_mask]; BODY_N = Nb_[~head_mask]; body_tree = cKDTree(BODY_V)
def body_sdf(P):
    d, i = body_tree.query(P); s = np.sign(((P - BODY_V[i]) * BODY_N[i]).sum(1)); return d * s
_hw = np.abs(HP[np.abs(HP[:, 1] - EYE_L[1]) < 0.01, 0]).max()      # head half-width at eye level
def hair_sdf(P):
    x, y, z = P[:, 0] - SC[0], P[:, 1], P[:, 2]
    hd = head_sdf(P)
    hd = np.where(y < CHIN_Y - 0.03, np.maximum(hd, 0.05), hd)    # its sign is unreliable far from the head
    d = hd - 0.008
    rx, rz = _hw + 0.014, SR[2] * 1.05
    cy = SC[1] - 0.005; ry = (SC[1] + SR[1] + 0.014) - cy
    qy = np.maximum(0, y - cy) / ry
    flare = 1 + 0.10 * np.clip((CHIN_Y - y) / 0.25, 0, 1)
    long_ = (np.sqrt((x / (rx * flare)) ** 2 + ((z - (SC[2] - 0.022)) / (rz * (0.9 + 0.1 * flare))) ** 2 + qy ** 2) - 1) * min(rx, rz)
    # ends slope backwards: the front of each side curtain stops higher, so the hair visibly falls behind the shoulders
    long_ = smax(long_, (SHOULDER_Y + 0.025 + 0.55 * np.clip(z - (SC[2] - 0.045), 0, None)) - y, 0.03)
    d = smin(d, long_, 0.02)
    # face window with straight, slightly wispy bangs
    bang = brow_y + 0.002 + 0.0022 * np.sin(x * 260) + 0.0012 * np.sin(x * 97)
    win = np.maximum(np.maximum(SC[2] + 0.028 - z, np.abs(x) - (_hw - 0.014)), y - bang)
    d = smax(d, -win, 0.007)
    # below the chin keep only the side curtains in front, nothing over the chest centre
    front = np.maximum(np.maximum(SC[2] + 0.005 - z, np.abs(x) - (_hw - 0.006)), y - (CHIN_Y + 0.01))
    d = smax(d, -front, 0.012)
    # below the shoulder line all hair hangs behind the shoulders (no blobs sitting on top of them)
    d = smax(d, -np.maximum((SC[2] - 0.035) - z, y - (SHOULDER_Y + 0.035)), 0.015)
    # layered ends: a soft V at the back, slightly shorter in front
    hem = HEM_Y + 0.05 * np.clip((z - (SC[2] - 0.05)) / 0.12, 0, 1) + 0.010 * np.abs(np.sin(x * 55)) + 0.006 * np.sin(x * 130)
    d = smax(d, hem - y, 0.012)
    # long hair lying on the upper back: a 1 cm sheet over the body surface, behind the shoulders
    low = y < CHIN_Y + 0.01                       # only the lower hair can meet the shoulders and back
    bd = np.full(len(P), 1.0)
    if low.any(): bd[low] = body_sdf(P[low])
    sheet_w = _hw + 0.006 - 0.012 * np.clip((SHOULDER_Y - y) / 0.25, 0, 1)
    hem2 = HEM_Y + 0.07 * (np.clip(np.abs(x) / (_hw + 0.006), 0, 1) ** 2) + 0.006 * np.abs(np.sin(x * 70)) + 0.004 * np.sin(x * 150)
    drape = np.maximum.reduce([np.abs(bd - 0.0125) - 0.0055, z - (SC[2] - 0.045), np.abs(x) - sheet_w,
                               hem2 - y, y - (SHOULDER_Y + 0.06)])
    d = smin(d, drape, 0.012)
    # never inside the body
    d = smax(d, -(bd - 0.007), 0.006)
    return d
lo = np.array([SC[0] - 0.19, HEM_Y - 0.03, SC[2] - 0.16]); hi = np.array([SC[0] + 0.19, SC[1] + SR[1] + 0.04, SC[2] + 0.14])
vox = 0.0026
n = np.ceil((hi - lo) / vox).astype(int) + 1
xs = lo[0] + np.arange(n[0]) * vox; ys = lo[1] + np.arange(n[1]) * vox; zs = lo[2] + np.arange(n[2]) * vox
Yg, Zg = np.meshgrid(ys, zs, indexing="ij"); vol = np.empty(n, np.float32)
for i, xx in enumerate(xs):
    Pq = np.stack([np.full(Yg.size, xx), Yg.ravel(), Zg.ravel()], 1); vol[i] = hair_sdf(Pq).reshape(Yg.shape)
HVv, HF, _, _ = measure.marching_cubes(vol, 0.0, spacing=(vox, vox, vox)); HVv += lo; HF = HF.astype(np.int64)
def signed_volume(P, F): return np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6
if signed_volume(HVv, HF) < 0: HF = HF[:, [0, 2, 1]]
# drop surface buried in the head
c = HVv[HF].mean(1); _bs = np.ones(len(c)); _lo = c[:, 1] < CHIN_Y + 0.01; _bs[_lo] = body_sdf(c[_lo])
_hs = np.ones(len(c)); _hi = c[:, 1] > CHIN_Y - 0.02; _hs[_hi] = head_sdf(c[_hi])     # head test only near the head
HF = HF[(_hs > 0.002) & (_bs > 0.002)]
used = np.unique(HF); rm = -np.ones(len(HVv), int); rm[used] = np.arange(len(used)); HVv = HVv[used]; HF = rm[HF]
from scipy.sparse.csgraph import connected_components
_e = np.concatenate([HF[:, [0, 1]], HF[:, [1, 2]]])
_ncomp, _lab = connected_components(sparse.coo_matrix((np.ones(len(_e)), (_e[:, 0], _e[:, 1])), shape=(len(HVv), len(HVv))), directed=False)
_sizes = np.bincount(_lab[HF[:, 0]], minlength=_ncomp)
HF = HF[_sizes[_lab[HF[:, 0]]] > 0.02 * len(HF)]
used = np.unique(HF); rm = -np.ones(len(HVv), int); rm[used] = np.arange(len(used)); HVv = HVv[used]; HF = rm[HF]
log("hair components kept", int((_sizes > 0.02 * _sizes.sum()).sum()), "of", _ncomp)
def taubin(P, F, it=6, lam=0.5, mu=-0.53):
    n_ = len(P); e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]])
    A_ = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_, n_)).tocsr(); A_.data[:] = 1
    dg = np.asarray(A_.sum(1)).ravel()[:, None]; dg[dg == 0] = 1
    for _ in range(it):
        P = P + lam * (A_ @ P / dg - P); P = P + mu * (A_ @ P / dg - P)
    return P
HVv = taubin(HVv, HF, 6)
HV2, HF2 = fast_simplification.simplify(HVv.astype(np.float32), HF.astype(np.int32), 1 - 4400 / len(HF))
HVv, HF = HV2.astype(float), HF2.astype(np.int64)
if signed_volume(HVv, HF) < 0: HF = HF[:, [0, 2, 1]]
HN = vertex_normals(HVv, HF)
# split into side / top / bottom projections (no pole, no overlap)
fny = HN[HF].mean(1)[:, 1]
for nm, m in (("hair", (fny <= 0.75) & (fny >= -0.6)), ("hair_cap", fny > 0.75), ("hair_bot", fny < -0.6)):
    Fm = HF[m]; used = np.unique(Fm); rm = -np.ones(len(HVv), int); rm[used] = np.arange(len(used))
    Pp = HVv[used]
    if nm == "hair":
        q = Pp - SC; lam = np.arctan2(q[:, 0], q[:, 2]); th = (SC[1] + SR[1] - Pp[:, 1]) / (SC[1] + SR[1] - HEM_Y + 0.02)
        uu = lam / (2 * math.pi) + 0.5
        # seam at the back: duplicate wrapping faces
        Fl = rm[Fm]; uf = uu[Fl]; wrap = (uf.max(1) - uf.min(1)) > 0.5
        extra = {}; newP = []; newU = []; newT = []
        for fi in np.where(wrap)[0]:
            for kk in range(3):
                vi_ = Fl[fi, kk]
                if uu[vi_] < 0.5:
                    if vi_ not in extra: extra[vi_] = len(Pp) + len(newP); newP.append(Pp[vi_]); newU.append(uu[vi_] + 1); newT.append(th[vi_]); used = np.append(used, used[vi_])
                    Fl[fi, kk] = extra[vi_]
        Pp = np.vstack([Pp, newP]) if newP else Pp; uu = np.concatenate([uu, newU]); th = np.concatenate([th, newT])
        x0, y0, w, h = TILES["hair"]
        uv = np.stack([(x0 + 4 + np.clip(uu / 1.03, 0, 1) * (w - 8)) / 4096, (y0 + 4 + np.clip(th, 0, 1) * (h - 8)) / 4096], 1)
        F_ = Fl
    else:
        x0, y0, w, h = TILES[nm]; sc_ = max(np.ptp(Pp[:, 0]), np.ptp(Pp[:, 2]), 1e-6)
        uv = np.stack([(x0 + 4 + (Pp[:, 0] - Pp[:, 0].min()) / sc_ * (w - 8)) / 4096, (y0 + 4 + (Pp[:, 2] - Pp[:, 2].min()) / sc_ * (h - 8)) / 4096], 1)
        F_ = rm[Fm]
    hp = dict(name=nm, base=None, F=F_, V=Pp, uv=uv, D={n_: np.zeros_like(Pp) for n_ in TARGET_NAMES})
    # upper hair follows the head, the ends lying on the shoulders follow the upper back
    wb_ = ss(CHIN_Y + 0.02, SHOULDER_Y - 0.02, Pp[:, 1])          # 0 at the chin -> 1 at the shoulders
    hp["J"] = np.tile([RBI["Head"], RBI["Spine2"], RBI["Neck"], 0], (len(Pp), 1))
    hp["W"] = np.stack([1 - wb_, wb_ * 0.7, wb_ * 0.3, np.zeros(len(Pp))], 1)
    hp["N"] = HN[used] if nm != "hair" else vertex_normals(Pp, F_)
    parts.append(hp)
log("hair tris", len(HF))

# ------------------------------------------------------------------ 8. lashes: own small alpha-masked material
lp = lashes
LASH_W, LASH_H = 512, 256
eyeC = np.where((lp["V"][:, 0] > 0)[:, None], EYE_L, EYE_R)
dist = norm(lp["V"] - eyeC)
uvs = lp["uvsrc"].copy(); new_uv = np.zeros_like(uvs)
# per lash group: find the uv axis that runs root -> tip
gid = np.zeros(len(uvs), int)
for gi, g in enumerate(lash_groups):
    gv = np.unique(np.concatenate([FACES[k] for k in GROUP[g]])); gid[np.isin(lp["base"], gv)] = gi
for gi in range(len(lash_groups)):
    m = gid == gi
    if not m.any(): continue
    uu, vv, dd = uvs[m, 0], uvs[m, 1], dist[m]
    cu, cv = np.corrcoef(uu, dd)[0, 1], np.corrcoef(vv, dd)[0, 1]
    if abs(cu) > abs(cv): along_, across, sgn = uu, vv, np.sign(cu)
    else: along_, across, sgn = vv, uu, np.sign(cv)
    tt = (along_ - along_.min()) / max(np.ptp(along_), 1e-9); tt = tt if sgn > 0 else 1 - tt
    s2 = (across - across.min()) / max(np.ptp(across), 1e-9)
    new_uv[m, 0] = 0.02 + s2 * 0.96; new_uv[m, 1] = 0.02 + tt * 0.96
lp["uv"] = new_uv
_root = new_uv[:, 1] < 0.25
_up = np.zeros(len(new_uv), bool)
for e in (EYE_L, EYE_R):
    m_ = (np.sign(lp["V"][:, 0]) == np.sign(e[0])); _up |= m_ & (lp["V"][:, 1] > e[1])
LASH_UP = cKDTree(lp["V"][_root & _up]); LASH_LO = cKDTree(lp["V"][_root & ~_up])
lash_img = np.zeros((LASH_H, LASH_W, 4), np.float32)
yy, xx = np.mgrid[0:LASH_H, 0:LASH_W] / np.array([LASH_H, LASH_W])[:, None, None]
for k in range(260):
    s0 = rng.random(); L = 0.55 + 0.45 * rng.random(); curl = (rng.random() - 0.5) * 0.04; wdt = 0.0016 + 0.0012 * rng.random()
    xc = s0 + curl * yy ** 2
    a = np.exp(-((xx - xc) / wdt) ** 2) * (yy < L) * (1 - ss(L * 0.6, L, yy) * 0.7)
    lash_img[..., 3] = np.maximum(lash_img[..., 3], a)
lash_img[..., :3] = np.array([0.05, 0.035, 0.03])
LASH_PNG = (np.clip(lash_img, 0, 1) * 255).astype(np.uint8)

# ------------------------------------------------------------------ 9. normals for all parts
for p in parts:
    if "N" not in p:
        N_ = vertex_normals(p["V"], p["F"])
        if p.get("base") is not None:   # weld normals across uv seams
            s3 = np.zeros((NB, 3)); np.add.at(s3, p["base"], N_); N_ = s3[p["base"]]; N_ /= np.maximum(norm(N_)[:, None], 1e-9)
        if p is bp:    # smoothed chest: average normals over ~2 cm so collapsed vertex clusters don't sparkle
            sel = np.where(bp["FR"] > 0.3)[0]
            if len(sel):
                tr_ = cKDTree(p["V"]); _, nn_ = tr_.query(p["V"][sel], k=48)
                Na = N_[nn_].mean(1); Na /= np.maximum(norm(Na)[:, None], 1e-9)
                wN = np.clip((bp["FR"][sel] - 0.3) / 0.4, 0, 1)[:, None]
                N_[sel] = N_[sel] * (1 - wN) + Na * wN; N_[sel] /= np.maximum(norm(N_[sel])[:, None], 1e-9)
        p["N"] = N_

# ------------------------------------------------------------------ 10. texture painting (position maps)
ATL = 4096
log("rasterising position maps")
POS = np.zeros((ATL, ATL, 3), np.float32); NRM = np.zeros((ATL, ATL, 3), np.float32); GRP = -np.ones((ATL, ATL), np.int16)
AUX = np.zeros((ATL, ATL, 4), np.float32)   # blazer, shoe, legs, pleat
main_parts = [p for p in parts if p["name"] != "lashes"]
for gi, p in enumerate(main_parts):
    UVp = p["uv"] * ATL - 0.5
    aux = np.zeros((len(p["V"]), 4))
    if p["name"] == "body": aux[:, 0] = p["W_BLAZER"]; aux[:, 1] = p["W_SHOE"]; aux[:, 2] = p["W_LEGS"]
    if p["name"] == "skirt": aux[:, 3] = p["pleat"]
    for f in p["F"]:
        a, b_, c_ = UVp[f]
        x0 = int(max(0, math.floor(min(a[0], b_[0], c_[0])))); x1 = int(min(ATL - 1, math.ceil(max(a[0], b_[0], c_[0]))))
        y0 = int(max(0, math.floor(min(a[1], b_[1], c_[1])))); y1 = int(min(ATL - 1, math.ceil(max(a[1], b_[1], c_[1]))))
        if x1 < x0 or y1 < y0: continue
        den = (b_[1] - c_[1]) * (a[0] - c_[0]) + (c_[0] - b_[0]) * (a[1] - c_[1])
        if abs(den) < 1e-12: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        l0 = ((b_[1] - c_[1]) * (gx - c_[0]) + (c_[0] - b_[0]) * (gy - c_[1])) / den
        l1 = ((c_[1] - a[1]) * (gx - c_[0]) + (a[0] - c_[0]) * (gy - c_[1])) / den
        l2 = 1 - l0 - l1; m = (l0 >= -0.03) & (l1 >= -0.03) & (l2 >= -0.03)
        if not m.any(): continue
        Lw = np.stack([l0[m], l1[m], l2[m]], 1)
        POS[gy[m], gx[m]] = Lw @ p["V"][f]
        NRM[gy[m], gx[m]] = Lw @ p["N"][f]; AUX[gy[m], gx[m]] = Lw @ aux[f]; GRP[gy[m], gx[m]] = gi
idx = ndimage.distance_transform_edt(GRP < 0, return_distances=False, return_indices=True)
POS = POS[idx[0], idx[1]]; NRM = None; AUX = AUX[idx[0], idx[1]]; GRP = GRP[idx[0], idx[1]]; del idx
GID = {p["name"]: i for i, p in enumerate(main_parts)}
log("painting")
def hexc(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
def noise(scale, octaves=3, seed=0, size=ATL):
    r = np.random.default_rng(seed); out = np.zeros((size, size), np.float32); amp = 1.0
    for _ in range(octaves):
        g_ = r.random((size // scale + 2, size // scale + 2)).astype(np.float32)
        out += amp * ndimage.zoom(g_, scale, order=1)[:size, :size]; amp *= 0.5; scale = max(2, scale // 2)
    return (out - out.min()) / (out.max() - out.min())
N_LO, N_MID, N_HI = noise(128, 4, 1), noise(24, 3, 2), noise(4, 2, 3)
BASE = np.zeros((ATL, ATL, 3), np.float32); ROUGH = np.full((ATL, ATL), 0.55, np.float32); HEIGHT = np.zeros((ATL, ATL), np.float32)
F = float
EL, ER = [F(x) for x in EYE_L], [F(x) for x in EYE_R]; MYf = F(MOUTH_Y); HCz = F(HEAD_C[2])
SCx, SCy, SCz, SRy = F(SC[0]), F(SC[1]), F(SC[2]), F(SR[1])
COLLAR = F(collar_y); SP2z = F(B["spine02"]["head"][2]); BLZ_HEM = F(blazer_hem) + 0.006; SK_HEM = F(hem_y) + 0.02
SKIN = hexc("#E9BB9C"); SKIN_RED = hexc("#DE8F7E")
BLAZER = hexc("#252E45"); SHIRT = hexc("#F5F4F0"); TIE = hexc("#9E2B33"); TIGHTS = hexc("#18181B"); SHOE = hexc("#141416")
LIP = hexc("#C27670")
ex0, ey0, ew, eh = TILES["eye"]
EYE_IMG = np.asarray(Image.open(f"{MH}/eyes/materials/brown_eye.png").convert("RGB").resize((ew - 8, eh - 8)), np.float32) / 255
PAD_R = 24
def blur(m, s): return ndimage.gaussian_filter(m.astype(np.float32), s)
def mix(a, b, w): return a * (1 - w[..., None]) + b * w[..., None]
def paint(r0, r1):
    q0, q1 = max(0, r0 - PAD_R), min(ATL, r1 + PAD_R)
    P_ = POS[q0:q1]; px, py, pz = P_[..., 0], P_[..., 1], P_[..., 2]
    grp = GRP[q0:q1]; aux = AUX[q0:q1]; nlo, nmid, nhi = N_LO[q0:q1], N_MID[q0:q1], N_HI[q0:q1]
    yy, xx = np.mgrid[q0:q1, 0:ATL].astype(np.float32)
    base = np.zeros(px.shape + (3,), np.float32); rough = np.full(px.shape, 0.55, np.float32); height = np.zeros(px.shape, np.float32)
    def g2(cx, cy, sx, sy): return np.exp(-(((px - cx) / sx) ** 2 + ((py - cy) / sy) ** 2))
    is_body = grp == GID["body"]
    col = SKIN * (0.975 + 0.03 * nlo[..., None]) * (0.993 + 0.014 * nmid[..., None])
    ff = is_body & (pz > HCz - 0.05) & (py > MYf - 0.06) & (py < EL[1] + 0.08)
    warm = (g2(0.045, EL[1] - 0.033, 0.022, 0.016) + g2(-0.045, EL[1] - 0.033, 0.022, 0.016)) * ff * 0.6
    warm = warm + g2(0.0, (EL[1] + MYf) / 2 - 0.008, 0.010, 0.012) * ff * 0.45
    col = mix(col, np.broadcast_to(SKIN_RED, col.shape), np.clip(warm, 0, 1) * 0.55)
    for e in (EL, ER):
        sh = np.exp(-(((px - e[0]) / 0.020) ** 2 + ((py - e[1] - 0.010) / 0.009) ** 2)) * ff * (pz > e[2] - 0.01)
        col = mix(col, np.broadcast_to(hexc("#B88270"), col.shape), sh * 0.35)
    for sgn, e in ((1, EL), (-1, ER)):
        t = np.clip((sgn * px - 0.010) / 0.050, 0, 1)
        by = e[1] + 0.0185 + 0.0055 * np.sin(t * math.pi * 0.95) - 0.004 * t
        th = 0.0017 * (1 - 0.6 * t) + 0.0004
        inb = (np.abs(py - by) < th) & (sgn * px > 0.010) & (sgn * px < 0.062) & ff & (pz > e[2] - 0.015)
        col = mix(col, np.broadcast_to(hexc("#33241F"), col.shape), np.clip(blur(inb, 3.0) * 0.85, 0, 1))
    lipzone = ff & (np.abs(py - MYf) < 0.011) & (np.abs(px) < 0.026)
    lipm = blur(lipzone & (((px / 0.0205) ** 2 + ((py - MYf - 0.0004) / 0.0085) ** 2) < 1), 2.5)
    col = mix(col, LIP * (0.93 + 0.1 * nmid[..., None]), np.clip(lipm * 1.2, 0, 1) * 0.85)
    near_eye = ff & ((np.abs(np.abs(px) - abs(EL[0])) < 0.03) & (np.abs(py - EL[1]) < 0.02))
    if near_eye.any():
        pts = P_[near_eye]
        du, _ = LASH_UP.query(pts); dl, _ = LASH_LO.query(pts)
        lash = np.zeros(px.shape, np.float32)
        lash[near_eye] = np.maximum(np.clip(1.4 * np.exp(-(du / 0.0030) ** 2), 0, 1), np.exp(-(dl / 0.0014) ** 2) * 0.55)
        col = mix(col, np.broadcast_to(hexc("#1A1210"), col.shape), lash)
    line = np.exp(-((py - MYf) / 0.0012) ** 2) * (np.abs(px) < 0.024) * ff
    col = mix(col, np.broadcast_to(hexc("#8A4A48"), col.shape), line * 0.45)
    skin_rough = 0.5 - 0.15 * lipm
    # outfit
    wb, wsh, wl = aux[..., 0], aux[..., 1], aux[..., 2]
    twill = 0.5 + 0.5 * np.sin((xx + yy) * 0.35)
    blz = BLAZER * (0.92 + 0.1 * nlo[..., None]) * (0.95 + 0.07 * twill[..., None])
    chest = pz > SP2z + 0.02
    vtop = COLLAR - 0.13
    v_open = chest & (np.abs(px) < (py - vtop) * 0.42) & (py > vtop)
    lapel = chest & (np.abs(np.abs(px) - (py - vtop) * 0.42) < 0.004) & (py > vtop)
    tie = v_open & (np.abs(px) < 0.010 + 0.004 * ss(COLLAR - 0.02, COLLAR - 0.12, py))
    outfit = mix(blz, SHIRT * (0.96 + 0.04 * nmid[..., None]), blur(v_open, 1.5))
    outfit = mix(outfit, TIE * (0.92 + 0.08 * twill[..., None]), tie.astype(np.float32))
    outfit = mix(outfit, outfit * 0.65, blur(lapel, 2))
    for by_ in (COLLAR - 0.16, COLLAR - 0.215):
        btn = chest & ((px - 0.012) ** 2 + (py - by_) ** 2 < 0.0042 ** 2)
        outfit = mix(outfit, np.broadcast_to(hexc("#B8954E"), outfit.shape), blur(btn, 1.0) * 0.85)
    torso_zone = is_body & (np.abs(px) < 0.13) & (py < COLLAR + 0.03) & (py > COLLAR - 0.35)
    wb = np.where(torso_zone, ss(COLLAR + 0.004, COLLAR - 0.004, py), wb)
    torso_px = np.abs(px) < 0.16
    hem_w = ss(BLZ_HEM - 0.0015, BLZ_HEM + 0.0015, py)
    wb = np.where(torso_px & (np.abs(py - BLZ_HEM) < 0.03) & (wb > 0.05), hem_w, wb)
    bc = mix(col, outfit, np.clip(wb * 1.3, 0, 1))
    legs = mix(col, TIGHTS * (0.9 + 0.2 * nmid[..., None]) + col * 0.12, np.clip(wl, 0, 1) * (1 - np.clip(wb, 0, 1)))
    bc = np.where(((wl > 0.01) & (wb < 0.5))[..., None], legs, bc)
    # body under the skirt takes the skirt's base colour, so nothing skin-coloured pokes through the fabric
    under = is_body & torso_px & (py < BLZ_HEM) & (py > SK_HEM) & (wsh < 0.1)
    bc = np.where(under[..., None], hexc("#3E4452") * (0.9 + 0.1 * nmid[..., None]), bc)
    bc = mix(bc, np.broadcast_to(SHOE, bc.shape), np.clip(wsh * 1.2, 0, 1))
    # shirt collar band around the neck, just above the blazer
    collar = is_body & (py > COLLAR - 0.004) & (py < COLLAR + 0.016 - 0.012 * np.clip((HCz - 0.02 - pz) / 0.08, 0, 1)) & (wb < 0.6)
    collar_w = 0 * blur(collar, 1.5)
    bc = mix(bc, SHIRT * (0.95 + 0.05 * nmid[..., None]), np.clip(collar_w, 0, 1))
    base[is_body] = bc[is_body]
    rough[is_body] = np.where(wb > 0.5, 0.82, np.where(wsh > 0.5, 0.28, np.where(wl > 0.5, 0.5, skin_rough)))[is_body]
    height[is_body] = np.where(wb > 0.5, 0.18 * twill, 0.04 * nhi)[is_body]
    m = grp == GID["skirt"]
    # navy-and-grey tartan with a thin red line, shaded by the pleats
    ang_ = np.arctan2(px, pz - SP2z); u_ = ang_ * 0.12; v_ = py
    bandA = (np.abs(((u_ / 0.028) % 1) - 0.5) < 0.17).astype(np.float32)
    bandB = (np.abs(((v_ / 0.028) % 1) - 0.5) < 0.17).astype(np.float32)
    redl = ((np.abs(((u_ / 0.028) % 1) - 0.06) < 0.025) | (np.abs(((v_ / 0.028) % 1) - 0.06) < 0.025)).astype(np.float32)
    skc = hexc("#4A505E") * (1 - 0.32 * np.clip(bandA + bandB, 0, 1.5)[..., None])
    skc = mix(skc, np.broadcast_to(hexc("#22283A"), skc.shape), bandA * bandB * 0.6)
    skc = mix(skc, np.broadcast_to(hexc("#8E2A32"), skc.shape), blur(redl, 0.8) * 0.7)
    skc = skc * (0.82 + 0.24 * (0.5 + 0.5 * aux[..., 3])[..., None])
    base[m] = skc[m]; rough[m] = 0.85; height[m] = (0.15 * twill)[m]
    for nm, c in (("teethU", "#F1ECE2"), ("teethD", "#EDE6DA"), ("tongue", "#C4646E")):
        m = grp == GID[nm]; base[m] = hexc(c); rough[m] = 0.3
    m = (grp == GID["hair"]) | (grp == GID.get("hair_cap", -9)) | (grp == GID.get("hair_bot", -9))
    lam = np.arctan2(px - SCx, pz - SCz)
    strand = 0.5 + 0.22 * np.sin(lam * 180 + 2.5 * np.sin(lam * 23)) + 0.15 * np.sin(lam * 410 + 1.3) + 0.12 * np.sin(lam * 77 + 6 * np.sin(py * 60))
    strand = np.clip(strand + 0.12 * (nmid - 0.5), 0, 1)
    sheen = np.exp(-((py - (SCy + SRy * 0.55)) / 0.025) ** 2)
    hc = hexc("#1C1412") * (0.75 + 0.55 * strand[..., None]) + hexc("#6A4A3A") * (0.6 * sheen * strand)[..., None]
    base[m] = hc[m]; rough[m] = (0.38 + 0.2 * strand)[m]; height[m] = strand[m]
    a, b_ = r0 - q0, r0 - q0 + (r1 - r0)
    BASE[r0:r1] = base[a:b_]; ROUGH[r0:r1] = rough[a:b_]; HEIGHT[r0:r1] = height[a:b_]
for r in range(0, ATL, 512): paint(r, min(ATL, r + 512))
_sat = EYE_IMG.max(-1) - EYE_IMG.min(-1); _ir = np.clip((_sat - 0.08) / 0.2, 0, 1)[..., None]
_lum = EYE_IMG.mean(-1, keepdims=True)
_iris = hexc("#5A3A26") * (0.55 + 1.3 * _lum)
BASE[ey0 + 4:ey0 + eh - 4, ex0 + 4:ex0 + ew - 4] = EYE_IMG * (1 - _ir) + _iris * _ir; ROUGH[ey0:ey0 + eh, ex0:ex0 + ew] = 0.08
base, rough, height = BASE, ROUGH, HEIGHT

# normal map from height
hgt = ndimage.gaussian_filter(height, 0.8)
gx_ = ndimage.sobel(hgt, axis=1); gy_ = ndimage.sobel(hgt, axis=0)
nm_ = np.stack([-gx_ * 1.4, gy_ * 1.4, np.ones_like(gx_)], -1); nm_ /= np.linalg.norm(nm_, axis=-1, keepdims=True)
def jpg(arr, size=None, q=90):
    im = Image.fromarray(arr)
    if size: im = im.resize((size, size), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True); return b.getvalue()
bimg = (np.clip(base, 0, 1) * 255).astype(np.uint8)
nimg = ((nm_ * 0.5 + 0.5) * 255).astype(np.uint8)
orm = np.zeros((ATL, ATL, 3), np.uint8); orm[..., 0] = 255; orm[..., 1] = (np.clip(rough, 0.05, 1) * 255).astype(np.uint8)
os.makedirs(OUT, exist_ok=True)
IMGS = [("T_Jois_BaseColor", "image/jpeg", jpg(bimg, None, 88)), ("T_Jois_Normal", "image/jpeg", jpg(nimg, 2048, 90)),
        ("T_Jois_ORM", "image/jpeg", jpg(orm, 2048, 90))]

Image.fromarray(bimg).resize((1024, 1024)).save(os.path.join(OUT, "atlas_preview.jpg"), quality=85)

# ------------------------------------------------------------------ 11. assemble + glTF
log("assembling")
def assemble(plist):
    Vs, Ns, UVs, Js, Ws, Fs = [], [], [], [], [], []; Ds = {n: [] for n in TARGET_NAMES}; off = 0
    for p in plist:
        Vs.append(p["V"]); Ns.append(p["N"]); UVs.append(p["uv"]); Js.append(p["J"]); Ws.append(p["W"]); Fs.append(p["F"] + off); off += len(p["V"])
        for n in TARGET_NAMES: Ds[n].append(p["D"][n])
    return (np.vstack(Vs).astype(np.float32), np.vstack(Ns).astype(np.float32), np.vstack(UVs).astype(np.float32),
            np.vstack(Js).astype(np.uint8), np.vstack(Ws).astype(np.float32), np.vstack(Fs).astype(np.uint32),
            {n: np.vstack(d).astype(np.float32) for n, d in Ds.items()})
PRIMS = [assemble(main_parts)]
for V_, N_, *_ in PRIMS: N_ /= np.maximum(np.linalg.norm(N_, axis=1, keepdims=True), 1e-9)

class GLB:
    def __init__(s): s.bin = bytearray(); s.views = []; s.acc = []
    def view(s, data, target=None):
        while len(s.bin) % 4: s.bin.append(0)
        o = len(s.bin); s.bin += data; v = {"buffer": 0, "byteOffset": o, "byteLength": len(data)}
        if target: v["target"] = target
        s.views.append(v); return len(s.views) - 1
    def accessor(s, arr, ctype, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr); a = {"bufferView": s.view(arr.tobytes(), target), "componentType": ctype, "count": len(arr), "type": typ}
        if minmax: a["min"] = arr.reshape(len(arr), -1).min(0).tolist(); a["max"] = arr.reshape(len(arr), -1).max(0).tolist()
        s.acc.append(a); return len(s.acc) - 1
    def sparse_vec3(s, d):
        nz = np.where(np.abs(d).max(1) > 1e-7)[0]
        if len(nz) == 0: nz = np.array([0])
        ib = s.view(nz.astype(np.uint16).tobytes()); vb = s.view(np.ascontiguousarray(d[nz]).astype(np.float32).tobytes())
        a = {"componentType": 5126, "count": len(d), "type": "VEC3", "min": d.min(0).tolist(), "max": d.max(0).tolist(),
             "sparse": {"count": int(len(nz)), "indices": {"bufferView": ib, "componentType": 5123}, "values": {"bufferView": vb}}}
        s.acc.append(a); return len(s.acc) - 1
g = GLB()
prims = []
for mi, (V_, N_, UV_, J_, W_, F_, D_) in enumerate(PRIMS):
    assert len(V_) < 65536
    prims.append({"attributes": {"POSITION": g.accessor(V_, 5126, "VEC3", 34962, True), "NORMAL": g.accessor(N_, 5126, "VEC3", 34962),
                                 "TEXCOORD_0": g.accessor(UV_, 5126, "VEC2", 34962), "JOINTS_0": g.accessor(J_, 5121, "VEC4", 34962),
                                 "WEIGHTS_0": g.accessor(W_, 5126, "VEC4", 34962)},
                  "indices": g.accessor(F_.reshape(-1).astype(np.uint16), 5123, "SCALAR", 34963), "material": mi, "mode": 4,
                  "targets": [{"POSITION": g.sparse_vec3(D_[n])} for n in TARGET_NAMES]})
images = [{"name": n, "mimeType": mt, "bufferView": g.view(data)} for n, mt, data in IMGS]
textures = [{"sampler": 0, "source": i, "name": im["name"]} for i, im in enumerate(images)]
nodes = []
for i, (bn, parent, _) in enumerate(RB):
    ppos = RBPOS[parent] if parent else v3(0, 0, 0)
    nodes.append({"name": bn, "translation": (RBPOS[bn] - ppos).tolist()})
for i, (bn, parent, _) in enumerate(RB):
    if parent: nodes[RBI[parent]].setdefault("children", []).append(i)
ibm = np.stack([np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-RBPOS[r[0]][0], -RBPOS[r[0]][1], -RBPOS[r[0]][2], 1]], np.float32) for r in RB])
a_ibm = g.accessor(ibm.reshape(len(RB), 16), 5126, "MAT4")
mesh_node = len(nodes); nodes.append({"name": "Jois_Avatar", "mesh": 0, "skin": 0})
root = len(nodes); nodes.append({"name": "Armature", "children": [RBI["Hips"], mesh_node]})
EMOTIONS = {
    "joy": {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9, "cheekSquintLeft": 0.6, "cheekSquintRight": 0.6, "eyeSquintLeft": 0.45, "eyeSquintRight": 0.45, "browInnerUp": 0.1},
    "surprise": {"browInnerUp": 0.9, "browOuterUpLeft": 0.9, "browOuterUpRight": 0.9, "eyeWideLeft": 0.9, "eyeWideRight": 0.9, "jawOpen": 0.3, "mouthFunnel": 0.3},
    "focus": {"browDownLeft": 0.7, "browDownRight": 0.7, "eyeSquintLeft": 0.45, "eyeSquintRight": 0.45, "mouthPressLeft": 0.5, "mouthPressRight": 0.5},
    "empathy": {"browInnerUp": 0.9, "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4, "eyeSquintLeft": 0.25, "eyeSquintRight": 0.25, "mouthPressLeft": 0.2, "mouthPressRight": 0.2},
}
gltf = {
    "asset": {"version": "2.0", "generator": "Jois Avatar Kit generator 2.0 (MakeHuman CC0 base)",
              "extras": {"jois": {"arkit": ARKIT_ORDER, "visemes": [f"viseme_{v}" for v in VIS], "emotions": EMOTIONS, "visemeRecipes": VISEME_MIX,
                                   "credits": "Base mesh, targets, skeleton, face pose units and eye texture: MakeHuman, CC0"}}},
    "scene": 0, "scenes": [{"name": "Jois", "nodes": [root]}], "nodes": nodes,
    "meshes": [{"name": "SK_Jois_Avatar", "extras": {"targetNames": TARGET_NAMES}, "weights": [0.0] * len(TARGET_NAMES), "primitives": prims}],
    "skins": [{"name": "SKN_Jois", "joints": list(range(len(RB))), "inverseBindMatrices": a_ibm, "skeleton": RBI["Hips"]}],
    "materials": [
        {"name": "M_Jois_Atlas", "doubleSided": True,
         "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 2}, "metallicFactor": 1.0, "roughnessFactor": 1.0},
         "normalTexture": {"index": 1, "scale": 0.5}},
        ],
    "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
    "textures": textures, "images": images, "accessors": g.acc, "bufferViews": g.views,
}
while len(g.bin) % 4: g.bin.append(0)
gltf["buffers"] = [{"byteLength": len(g.bin)}]
js = json.dumps(gltf, separators=(",", ":")).encode()
while len(js) % 4: js += b" "
glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(g.bin)) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(g.bin), 0x004E4942) + bytes(g.bin)
open(os.path.join(OUT, "jois_avatar.glb"), "wb").write(glb)
V_, N_, UV_, J_, W_, F_, D_ = PRIMS[0]
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_build.npz"), V=V_, N=N_, UV=UV_, F=F_,
                    D=np.stack([D_[n] for n in TARGET_NAMES]), names=np.array(TARGET_NAMES),
                    base=np.asarray(Image.fromarray(bimg).resize((2048, 2048))), eye=EYE_L, mouth=MOUTH_Y)
stats = {"triangles": int(sum(len(p[5]) for p in PRIMS)), "vertices": int(sum(len(p[0]) for p in PRIMS)), "morphTargets": len(TARGET_NAMES),
         "bones": len(RB), "drawCalls": 1, "fileMB": round(len(glb) / 1048576, 2), "parts": {p["name"]: int(len(p["F"])) for p in parts},
         "headCenter": HEAD_C.tolist(), "eyeL": EYE_L.tolist(), "mouthY": float(MOUTH_Y)}
json.dump(stats, open(os.path.join(OUT, "avatar_stats.json"), "w"), indent=1)
log("wrote", stats)
