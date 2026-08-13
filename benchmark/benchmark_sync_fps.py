import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


sys.path.insert(0, os.getcwd())

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from utils.general_utils import safe_state


def load_renderer(renderer_name, sh_degree):
    if renderer_name == "fastgs":
        from gaussian_renderer import GaussianModel, render_fastgs

        gaussians = GaussianModel(sh_degree, optimizer_type="default")

        def render_view(view, pipeline, background, mult):
            return render_fastgs(view, gaussians, pipeline, background, mult)

        return gaussians, render_view

    if renderer_name == "standard":
        from gaussian_renderer import GaussianModel, render

        gaussians = GaussianModel(sh_degree)

        def render_view(view, pipeline, background, mult):
            return render(view, gaussians, pipeline, background)

        return gaussians, render_view

    if renderer_name == "dash":
        from gaussian_renderer import GaussianModel, render

        gaussians = GaussianModel(sh_degree)

        def render_view(view, pipeline, background, mult):
            return render(
                view,
                gaussians,
                pipeline,
                background,
                separate_sh=True,
            )

        return gaussians, render_view

    if renderer_name == "taming":
        from gaussian_renderer import GaussianModel, render

        gaussians = GaussianModel(
            sh_degree,
            optimizer_type="default",
            rendering_mode="abs",
        )

        def render_view(view, pipeline, background, mult):
            return render(view, gaussians, pipeline, background)

        return gaussians, render_view

    raise ValueError("Unsupported renderer: {}".format(renderer_name))


def main():
    parser = argparse.ArgumentParser(description="Synchronized render-only FPS benchmark")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--renderer",
        choices=("fastgs", "standard", "dash", "taming"),
        required=True,
    )
    parser.add_argument("--warmup_repeats", type=int, default=2)
    parser.add_argument("--measure_repeats", type=int, default=10)
    parser.add_argument("--mult", type=float, default=0.7)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipeline_args = pipeline.extract(args)
    gaussians, render_view = load_renderer(args.renderer, dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )
    views = scene.getTestCameras()
    if not views:
        raise RuntimeError("The model has no test cameras; train it with --eval.")

    color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for _ in range(args.warmup_repeats):
            for view in views:
                render_view(view, pipeline_args, background, args.mult)
            torch.cuda.synchronize()

        elapsed_seconds = []
        for repeat in range(args.measure_repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for view in views:
                render_view(view, pipeline_args, background, args.mult)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            elapsed_seconds.append(elapsed)
            print(
                "REPEAT={} ELAPSED_SECONDS={:.9f} FPS={:.6f}".format(
                    repeat + 1, elapsed, len(views) / elapsed
                )
            )

    repeat_fps = [len(views) / elapsed for elapsed in elapsed_seconds]
    result = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "renderer": args.renderer,
        "iteration": scene.loaded_iter,
        "test_views": len(views),
        "warmup_repeats": args.warmup_repeats,
        "measure_repeats": args.measure_repeats,
        "mult": args.mult if args.renderer == "fastgs" else None,
        "fps_mean": statistics.mean(repeat_fps),
        "fps_std": statistics.stdev(repeat_fps) if len(repeat_fps) > 1 else 0.0,
        "fps_median": statistics.median(repeat_fps),
        "fps_min": min(repeat_fps),
        "fps_max": max(repeat_fps),
        "repeat_fps": repeat_fps,
        "elapsed_seconds": elapsed_seconds,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }

    output_path = (
        Path(args.output_json)
        if getattr(args, "output_json", None)
        else Path(dataset.model_path) / "benchmark_sync_fps.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY_JSON={}".format(output_path.resolve()))
    print(
        "FPS_MEAN={:.6f} FPS_STD={:.6f} FPS_MEDIAN={:.6f}".format(
            result["fps_mean"], result["fps_std"], result["fps_median"]
        )
    )


if __name__ == "__main__":
    main()
