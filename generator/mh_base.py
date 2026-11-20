"""Load the MakeHuman (CC0) base mesh, apply targets, build the default skeleton and face pose units."""
import json, math, os, re
import numpy as np

MH = os.environ.get("MH_DATA", "/tmp/claude-0/mh/mh/makehuman/data")


def load_obj(path=f"{MH}/3dobjs/base.obj"):
    V, VT, faces, fuv, fgroup = [], [], [], [], []
    group = None
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("g "):
            group = line.split()[1].strip()
        elif line.startswith("f "):
            parts = line.split()[1:]
            faces.append([int(p.split("/")[0]) - 1 for p in parts])
            fuv.append([int(p.split("/")[1]) - 1 for p in parts])
            fgroup.append(group)
    return np.array(V), np.array(VT), faces, fuv, fgroup


def load_target(rel):
    idx, d = [], []
    for line in open(f"{MH}/targets/{rel}"):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        idx.append(int(p[0])); d.append([float(x) for x in p[1:4]])
    return np.array(idx, int), np.array(d)


def apply_targets(V, weights):
    V = V.copy()
    for rel, w in weights.items():
        if abs(w) < 1e-6:
            continue
        i, d = load_target(rel)
        if len(i): V[i] += w * d
    return V


# ------------------------------------------------------------------ skeleton
def skeleton(V):
    s = json.load(open(f"{MH}/rigs/default.mhskel"))
    J = {k: V[np.array(v)].mean(0) for k, v in s["joints"].items()}
    bones = {}
    for name, b in s["bones"].items():
        bones[name] = dict(parent=b["parent"], head=J[b["head"]], tail=J[b["tail"]])
    return bones


def load_weights():
    w = json.load(open(f"{MH}/rigs/default_weights.mhw"))["weights"]
    return {b: (np.array([i for i, _ in lst], int), np.array([x for _, x in lst])) for b, lst in w.items()}


# ------------------------------------------------------------------ BVH pose units (Z-up file, MakeHuman is Y-up)
def parse_bvh(path=f"{MH}/poseunits/face-poseunits.bvh"):
    tokens = open(path).read().split()
    i = 0; stack = []; joints = []; order = []
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            name = tokens[i + 1]; joints.append(dict(name=name, parent=stack[-1] if stack else None, channels=[])); i += 2; continue
        if t == "End":
            joints.append(dict(name=None, parent=stack[-1], channels=[])); i += 2; continue
        if t == "{":
            stack.append(joints[-1]["name"]); i += 1; continue
        if t == "}":
            stack.pop(); i += 1; continue
        if t == "CHANNELS":
            n = int(tokens[i + 1]); joints[-1]["channels"] = tokens[i + 2:i + 2 + n]; i += 2 + n; continue
        i += 1
    i += 1
    nframes = int(tokens[i + 1]); i += 2
    i += 3  # Frame Time: x
    vals = np.array([float(x) for x in tokens[i:]]).reshape(nframes, -1)
    out = []; col = 0
    named = [j for j in joints if j["name"]]
    for j in named:
        nc = len(j["channels"]); out.append((j["name"], j["parent"], j["channels"], col)); col += nc
    return out, vals


C_ZUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # z-up -> y-up


def rot_axis(axis, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if axis == "X": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def frame_rotations(joints, vals, frame):
    R = {}
    for name, parent, ch, col in joints:
        M = np.eye(3)
        for k, c in enumerate(ch):
            if c.endswith("rotation"):
                M = M @ rot_axis(c[0], vals[frame, col + k])
        R[name] = C_ZUP @ M @ C_ZUP.T
    return R


def pose_vertices(V, bones, weights, R):
    """BVH semantics: rotations in world-aligned frames at each joint, rest orientation = identity."""
    order = []
    def visit(n):
        if n in order: return
        p = bones[n]["parent"]
        if p: visit(p)
        order.append(n)
    for n in bones: visit(n)
    G = {}
    for n in order:
        b = bones[n]; Rn = R.get(n, np.eye(3))
        if b["parent"]:
            Pr, Pt = G[b["parent"]]
            G[n] = (Pr @ Rn, Pr @ (b["head"] - bones[b["parent"]]["head"]) + Pt)
        else:
            G[n] = (Rn, b["head"].copy())
    out = np.zeros_like(V); wsum = np.zeros(len(V))
    for n, (idx, w) in weights.items():
        if n not in G: continue
        Rg, t = G[n]
        P = (V[idx] - bones[n]["head"]) @ Rg.T + t
        out[idx] += w[:, None] * P; wsum[idx] += w
    miss = wsum < 1e-6
    out[~miss] /= wsum[~miss, None]
    out[miss] = V[miss]
    return out


def face_units(V, bones, weights):
    joints, vals = parse_bvh()
    names = json.load(open(f"{MH}/poseunits/face-poseunits.json"))["framemapping"]
    rest = pose_vertices(V, bones, weights, frame_rotations(joints, vals, 0))
    D = {}
    for f, n in enumerate(names):
        if f == 0: continue
        D[n] = pose_vertices(V, bones, weights, frame_rotations(joints, vals, f)) - rest
    return D, rest
