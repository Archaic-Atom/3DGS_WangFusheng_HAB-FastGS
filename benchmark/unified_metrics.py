"""Unified metric evaluator -- one scoring code path for every method.

Each comparison method trains and renders with its OWN code (its rasterizer is
part of the method under test), but the PSNR / SSIM / LPIPS numbers that end up
in the paper table must come from a single implementation, or the table compares
metric libraries rather than methods. This script is that single implementation:
it is run from the FastGS working tree inside the `habfastgs` env, so it uses
FastGS's own `ssim`, `psnr` and VGG-LPIPS regardless of whose PNGs it is given.

It is deliberately strict rather than forgiving:
  * render/GT pairing is by filename, and a mismatched set is a hard error;
  * a resolution mismatch between a render and its GT is a hard error, never a
    silent resize -- a silent resize would fabricate a comparison.

Usage:
  python unified_metrics.py --renders <dir> --gt <dir> --output <json> [--tag name]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import platform

import torch
import torchvision
import torchvision.transforms.functional as tf
import PIL
from PIL import Image

sys.path.insert(0, os.getcwd())

import lpipsPyTorch  # noqa: E402
from lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402
import utils.image_utils as fastgs_image_utils  # noqa: E402
import utils.loss_utils as fastgs_loss_utils  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".JPG", ".PNG"}


def list_images(directory):
    return sorted(
        p.name for p in Path(directory).iterdir()
        if p.suffix in IMAGE_SUFFIXES
    )


def load(path):
    # [:3] drops any alpha channel; matches FastGS's readImages exactly.
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3, :, :].cuda()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root, suffix=".py"):
    """Hash every implementation file below a package root."""
    root = Path(root).resolve()
    if root.is_file():
        return {str(root): sha256(root)}
    return {
        str(path.resolve()): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix == suffix
    }


def image_manifest(render_dir, gt_dir, render_by_stem, gt_by_stem):
    """Return content-identity evidence for the exact images being scored."""
    rows = []
    for stem in sorted(render_by_stem):
        render_path = Path(render_dir) / render_by_stem[stem]
        gt_path = Path(gt_dir) / gt_by_stem[stem]
        with Image.open(render_path) as render_image:
            render_size = list(render_image.size)
        with Image.open(gt_path) as gt_image:
            gt_size = list(gt_image.size)
        rows.append({
            "stem": stem,
            "render_name": render_path.name,
            "render_bytes": render_path.stat().st_size,
            "render_sha256": sha256(render_path),
            "render_size_wh": render_size,
            "gt_name": gt_path.name,
            "gt_bytes": gt_path.stat().st_size,
            "gt_sha256": sha256(gt_path),
            "gt_size_wh": gt_size,
        })
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return rows, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def unique_by_stem(names, label):
    out = {}
    for name in names:
        stem = Path(name).stem
        if stem in out:
            raise RuntimeError(
                "duplicate {} stem {}: {} and {}".format(
                    label, stem, out[stem], name))
        out[stem] = name
    return out


def main():
    parser = argparse.ArgumentParser(description="Unified PSNR/SSIM/LPIPS evaluator")
    parser.add_argument("--renders", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    torch.cuda.set_device(torch.device("cuda:0"))
    # lpipsPyTorch.lpips() constructs and reloads VGG for every image.  Reuse
    # one identical criterion for the whole run; this changes only evaluator
    # overhead, not the network, weights, preprocessing, or arithmetic.
    lpips_vgg = LPIPS("vgg", "0.1").cuda().eval()

    render_names = list_images(args.renders)
    gt_names = list_images(args.gt)
    if not render_names:
        raise RuntimeError("no images found in {}".format(args.renders))

    # Pair by stem so a .png render can be scored against a .jpg GT, but require
    # exact set equality: silently ignoring extra GTs changes the test split.
    render_by_stem = unique_by_stem(render_names, "render")
    gt_by_stem = unique_by_stem(gt_names, "GT")
    render_stems = set(render_by_stem)
    gt_stems = set(gt_by_stem)
    missing_gt = sorted(render_stems - gt_stems)
    extra_gt = sorted(gt_stems - render_stems)
    if missing_gt or extra_gt:
        raise RuntimeError(
            "render/GT stem sets differ: missing_gt={} extra_gt={}"
            .format(missing_gt[:5], extra_gt[:5]))

    ssims, psnrs, lpipss, per_view = [], [], [], {}
    with torch.no_grad():
        for name in render_names:
            render = load(Path(args.renders) / name)
            gt = load(Path(args.gt) / gt_by_stem[Path(name).stem])
            if render.shape != gt.shape:
                raise RuntimeError(
                    "resolution mismatch on {}: render {} vs gt {}. Refusing to "
                    "resize -- fix the render/eval resolution protocol instead."
                    .format(name, tuple(render.shape), tuple(gt.shape)))
            s = ssim(render, gt).item()
            p = psnr(render, gt).mean().item()
            l = lpips_vgg(render, gt).item()
            ssims.append(s)
            psnrs.append(p)
            lpipss.append(l)
            per_view[name] = {"SSIM": s, "PSNR": p, "LPIPS": l}

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    weight_hashes = {}
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.glob("vgg16*.pth")):
            weight_hashes[str(path.resolve())] = sha256(path)

    lpips_root = Path(lpipsPyTorch.__file__).resolve().parent
    for path in sorted(lpips_root.rglob("*.pth")):
        weight_hashes[str(path.resolve())] = sha256(path)

    content_manifest, content_manifest_sha256 = image_manifest(
        args.renders, args.gt, render_by_stem, gt_by_stem)
    metric_sources = {}
    for module in (fastgs_image_utils, fastgs_loss_utils):
        module_path = Path(module.__file__).resolve()
        metric_sources[str(module_path)] = sha256(module_path)
    lpips_sources = source_hashes(lpips_root)

    result = {
        "tag": args.tag,
        "renders_dir": str(Path(args.renders).resolve()),
        "gt_dir": str(Path(args.gt).resolve()),
        "num_views": len(render_names),
        "evaluator": "unified_metrics.py (FastGS ssim/psnr + VGG LPIPS)",
        "provenance": {
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "lpips_module": str(Path(lpipsPyTorch.__file__).resolve()),
            "lpips_module_sha256": sha256(Path(lpipsPyTorch.__file__).resolve()),
            "fastgs_metric_source_sha256": metric_sources,
            "lpips_source_sha256": lpips_sources,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "weight_sha256": weight_hashes,
            "render_stems_sha256": hashlib.sha256(
                "\n".join(sorted(render_stems)).encode("utf-8")).hexdigest(),
            "image_content_manifest_sha256": content_manifest_sha256,
        },
        "image_content_manifest": content_manifest,
        "SSIM": sum(ssims) / len(ssims),
        "PSNR": sum(psnrs) / len(psnrs),
        "LPIPS": sum(lpipss) / len(lpipss),
        "per_view": per_view,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("UNIFIED n={} PSNR={:.7f} SSIM={:.7f} LPIPS={:.7f}".format(
        result["num_views"], result["PSNR"], result["SSIM"], result["LPIPS"]))
    print("UNIFIED_JSON={}".format(out.resolve()))


if __name__ == "__main__":
    main()
