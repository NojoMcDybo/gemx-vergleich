"""Turn GemX results (SOMA body-model parameters) into joint CSVs for compare.py.

Run with the SOMA environment:
  .venv-soma\\Scripts\\python gemx_joints.py <predimo_dir> [incam|global]   (folders: clapsync.DIRS)
Setup (done 2026-11-21; NVIDIA SOMA-X, Apache-2.0, commercial use allowed per model card):
  py -3.11 -m venv .venv-soma
  .venv-soma\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv-soma\\Scripts\\pip install "py-soma-x==0.3.3" huggingface_hub
  assets: nvidia/soma-x rev v0.3.3 -> assets/soma (SOMA_neutral.npz, SOMA_template_rig.usda,
  SOMA_procedural_transforms.json, MHR/mhr_model_lod6.pt, MHR/base_body_lod6.obj,
  MHR/SOMA_wrap_lod1.obj)
Reads DIRS["gemx"]/<video stem>/hpe_results.pt and writes
DIRS["joints"]/<nr>_gemx.csv (one row per frame of the 30 fps cut video, metres).
GemX stores the MHR identity (45 coeffs) and 69 scales = [global scale 1.0, 68 MHR scales].
"""
import csv, sys
from pathlib import Path

import torch
from soma import SOMALayer, Unit

from clapsync import DIRS

ASSETS = Path(__file__).with_name("assets") / "soma"
# compare.py joint -> SOMA joint (each SOMA joint sits at the proximal end of its segment)
SOMA_JOINTS = {"pelvis": "Hips", "neck": "Neck1", "head": "Head",
               "r_shoulder": "RightArm", "r_elbow": "RightForeArm", "r_wrist": "RightHand",
               "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm", "l_wrist": "LeftHand",
               "r_hip": "RightLeg", "r_knee": "RightShin", "r_ankle": "RightFoot",
               "l_hip": "LeftLeg", "l_knee": "LeftShin", "l_ankle": "LeftFoot"}


def joints_from_params(layer, p):
    n = p["body_pose"].shape[0]
    poses = torch.cat([p["global_orient"][:, None], p["body_pose"].reshape(n, 76, 3)], dim=1)
    scale = p["scale_params"]
    if not torch.all(scale[:, 0] == 1.0):
        raise ValueError("expected scale_params[:, 0] == 1.0 (global scale)")
    ident = p["identity_coeffs"]
    if torch.equal(ident, ident[:1].expand_as(ident)) and torch.equal(scale, scale[:1].expand_as(scale)):
        ident, scale = ident[:1], scale[:1]  # GemX repeats one body for all frames: prepare it once (~40x faster)
    with torch.no_grad():
        layer.prepare_identity(ident, scale_params=scale[:, 1:])
        out = layer.pose(poses, transl=p["transl"], fk_only=True)
    return out.joints  # (n, 77, 3), joints[:, i] = public_joint_names[i + 1]


def make_layer():
    return SOMALayer(data_root=ASSETS, identity_model_type="mhr", low_lod=True, device="cpu",
                     mode="dense", output_unit=Unit.METERS)


def load_joints(layer, pt, space="incam"):
    """hpe_results.pt (path or file object) -> tensor (frames, 15, 3) in SOMA_JOINTS order."""
    res = torch.load(pt, map_location="cpu", weights_only=True)
    names = list(layer.public_joint_names[1:])  # index 0 is the Root dummy
    idx = [names.index(SOMA_JOINTS[n]) for n in SOMA_JOINTS]
    return joints_from_params(layer, res[f"body_params_{space}"])[:, idx]


def main():
    root = Path(sys.argv[1])
    space = sys.argv[2] if len(sys.argv) > 2 else "incam"
    layer = make_layer()
    (root / DIRS["joints"]).mkdir(parents=True, exist_ok=True)
    for pt in sorted((root / DIRS["gemx"]).glob("*/hpe_results.pt")):
        j = load_joints(layer, pt, space)
        out = root / DIRS["joints"] / f"{pt.parent.name[:3]}_gemx.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["frame"] + [f"{n}_{a}" for n in SOMA_JOINTS for a in "xyz"])
            for k, f in enumerate(j.tolist()):
                w.writerow([k] + [round(v, 5) for p in f for v in p])
        print(out.name, len(j), "frames")


if __name__ == "__main__":
    main()
