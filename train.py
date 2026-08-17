#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import os, random, time, csv
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


def _tensor_summary(tensor):
    if tensor is None:
        return 0.0, 0.0, 0.0
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return 0.0, 0.0, 0.0
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    mean = values.mean().item()
    max_value = values.max().item()
    try:
        p90 = torch.quantile(values, 0.9).item()
    except RuntimeError:
        sorted_values, _ = torch.sort(values)
        p90_index = min(sorted_values.numel() - 1, int(0.9 * sorted_values.numel()))
        p90 = sorted_values[p90_index].item()
    return mean, p90, max_value


def _scalar_value(value):
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


class HABLoadController:
    def __init__(self, opt):
        self.bucket_weight = float(getattr(opt, "hab_load_bucket_weight", 16.0) or 0.0)
        self.ema = float(getattr(opt, "hab_load_ema", 0.90) or 0.0)
        self.ema = min(max(self.ema, 0.0), 0.999)
        self.cost_ema = None
        self.density_ema = None
        self.reference_density = None

    def update(self, render_pkg, gaussian_count, opt):
        num_rendered = _scalar_value(render_pkg.get("num_rendered"))
        num_buckets = _scalar_value(render_pkg.get("num_buckets"))
        cost = float(num_rendered) + self.bucket_weight * float(num_buckets)
        density = cost / max(int(gaussian_count), 1)

        if cost > 0:
            if self.cost_ema is None:
                self.cost_ema = cost
                self.density_ema = density
            else:
                self.cost_ema = self.ema * self.cost_ema + (1.0 - self.ema) * cost
                self.density_ema = self.ema * self.density_ema + (1.0 - self.ema) * density

        opt.hab_current_load_cost = cost
        opt.hab_current_load_density = density
        opt.hab_load_cost_ema = 0.0 if self.cost_ema is None else self.cost_ema
        opt.hab_load_density_ema = 0.0 if self.density_ema is None else self.density_ema
        opt.hab_load_target_cost = 0.0
        opt.hab_load_pressure = 0.0
        return {
            "hab_load_cost": cost,
            "hab_load_density": density,
            "hab_load_cost_ema": opt.hab_load_cost_ema,
            "hab_load_density_ema": opt.hab_load_density_ema,
            "hab_load_target_cost": 0.0,
            "hab_load_pressure": 0.0,
        }

    def configure_event(self, iteration, gaussian_count, opt):
        base_target = int(getattr(opt, "hab_target_gaussians", 0) or 0)
        opt.hab_current_target_gaussians = base_target
        stats = {
            "hab_dynamic_target_gaussians": base_target,
            "hab_load_target_cost": 0.0,
            "hab_load_pressure": 0.0,
        }

        if getattr(opt, "hab_mode", "off") != "load_budget" or base_target <= 0:
            return stats
        if self.density_ema is None or self.density_ema <= 0:
            return stats

        if self.reference_density is None:
            self.reference_density = self.density_ema

        target_ratio = float(getattr(opt, "hab_load_target_ratio", 1.0) or 1.0)
        target_cost = max(self.reference_density * base_target * target_ratio, 1.0)
        current_cost = self.cost_ema if self.cost_ema is not None else 0.0
        pressure = current_cost / target_cost

        gain = float(getattr(opt, "hab_load_gain", 0.25) or 0.0)
        scale = 1.0 / max(1e-6, 1.0 + gain * (pressure - 1.0))
        scale = min(max(scale, float(getattr(opt, "hab_load_min_scale", 0.85) or 0.85)),
                    float(getattr(opt, "hab_load_max_scale", 1.10) or 1.10))

        min_target = int(getattr(opt, "hab_min_target_gaussians", 0) or 0)
        if min_target <= 0:
            min_target = int(base_target * float(getattr(opt, "hab_load_min_scale", 0.85) or 0.85))
        max_target = max(min_target, int(base_target * float(getattr(opt, "hab_load_max_scale", 1.10) or 1.10)))
        dynamic_target = min(max(int(base_target * scale), min_target), max_target)

        opt.hab_current_target_gaussians = dynamic_target
        opt.hab_load_target_cost = target_cost
        opt.hab_load_pressure = pressure
        stats.update({
            "hab_dynamic_target_gaussians": dynamic_target,
            "hab_load_target_cost": target_cost,
            "hab_load_pressure": pressure,
        })
        return stats


class HABStatsLogger:
    def __init__(self, model_path, opt):
        self.interval = int(getattr(opt, "hab_log_interval", 0) or 0)
        self.path = os.path.join(model_path, "hab_stats.csv")
        self.fieldnames = [
            "iteration", "loss", "gaussian_count", "visible_count", "visible_ratio",
            "radii_mean", "radii_p90", "radii_max",
            "radii2_sum",
            "num_rendered", "num_buckets",
            "metric_count_mean", "metric_count_p90", "metric_count_max",
            "importance_mean", "importance_p90", "importance_max",
            "pruning_score_mean", "pruning_score_p90", "pruning_score_max",
            "iter_time_ms", "optim_time_ms",
            "cuda_allocated_mb", "cuda_reserved_mb", "cuda_max_allocated_mb",
            "densify_event", "final_prune_event",
            "event_before_count", "event_after_count",
            "clone_candidates", "split_candidates",
            "baseline_prune_candidates", "baseline_pruned", "hab_budget_pruned",
            "hab_candidate_band_count", "hab_fisher_protected", "hab_fisher_guard_relaxed",
            "budget_floor_spared", "hab_ramp_ceiling",
            "hab_mode", "hab_priority_mode", "hab_budget_schedule",
            "hab_mv_candidate_multiplier", "hab_fisher_protect_quantile",
            "hab_exact_final_count", "hab_target_gaussians", "hab_dynamic_target_gaussians",
            "hab_load_cost", "hab_load_density", "hab_load_cost_ema",
            "hab_load_density_ema", "hab_load_target_cost", "hab_load_pressure"
        ]
        with open(self.path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()

    def should_log(self, iteration, event_stats):
        if self.interval <= 0:
            return bool(event_stats)
        return iteration % self.interval == 0 or bool(event_stats)

    def log(self, iteration, loss, gaussians, visibility_filter, radii, render_pkg,
            iter_time, optim_time, opt, event_stats):
        if not self.should_log(iteration, event_stats):
            return

        gaussian_count = int(gaussians.get_xyz.shape[0])
        visible_count = int(torch.sum(visibility_filter).item())
        visible_ratio = visible_count / max(gaussian_count, 1)
        visible_radii = radii.detach()[radii > 0]
        radii_mean, radii_p90, radii_max = _tensor_summary(visible_radii)
        radii2_sum = torch.sum(visible_radii.float() * visible_radii.float()).item() if visible_radii.numel() else 0.0
        metric_mean, metric_p90, metric_max = _tensor_summary(render_pkg.get("accum_metric_counts"))

        importance_mean, importance_p90, importance_max = _tensor_summary(event_stats.get("importance_score"))
        pruning_mean, pruning_p90, pruning_max = _tensor_summary(event_stats.get("pruning_score"))

        row = {
            "iteration": iteration,
            "loss": float(loss.item()),
            "gaussian_count": gaussian_count,
            "visible_count": visible_count,
            "visible_ratio": visible_ratio,
            "radii_mean": radii_mean,
            "radii_p90": radii_p90,
            "radii_max": radii_max,
            "radii2_sum": radii2_sum,
            "num_rendered": _scalar_value(render_pkg.get("num_rendered")),
            "num_buckets": _scalar_value(render_pkg.get("num_buckets")),
            "metric_count_mean": metric_mean,
            "metric_count_p90": metric_p90,
            "metric_count_max": metric_max,
            "importance_mean": importance_mean,
            "importance_p90": importance_p90,
            "importance_max": importance_max,
            "pruning_score_mean": pruning_mean,
            "pruning_score_p90": pruning_p90,
            "pruning_score_max": pruning_max,
            "iter_time_ms": float(iter_time),
            "optim_time_ms": float(optim_time),
            "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "cuda_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "densify_event": int(event_stats.get("densify_event", False)),
            "final_prune_event": int(event_stats.get("final_prune_event", False)),
            "event_before_count": int(event_stats.get("before_count", 0) or 0),
            "event_after_count": int(event_stats.get("after_count", 0) or 0),
            "clone_candidates": int(event_stats.get("clone_candidates", 0) or 0),
            "split_candidates": int(event_stats.get("split_candidates", 0) or 0),
            "baseline_prune_candidates": int(event_stats.get("baseline_prune_candidates", 0) or 0),
            "baseline_pruned": int(event_stats.get("baseline_pruned", 0) or 0),
            "hab_budget_pruned": int(event_stats.get("hab_budget_pruned", 0) or 0),
            "hab_candidate_band_count": int(event_stats.get("hab_candidate_band_count", 0) or 0),
            "hab_fisher_protected": int(event_stats.get("hab_fisher_protected", 0) or 0),
            "hab_fisher_guard_relaxed": int(event_stats.get("hab_fisher_guard_relaxed", 0) or 0),
            "budget_floor_spared": int(event_stats.get("budget_floor_spared", 0) or 0),
            "hab_ramp_ceiling": int(event_stats.get("hab_ramp_ceiling", 0) or 0),
            "hab_mode": getattr(opt, "hab_mode", "off"),
            "hab_priority_mode": getattr(opt, "hab_priority_mode", "joint"),
            "hab_budget_schedule": getattr(opt, "hab_budget_schedule", "per_event"),
            "hab_mv_candidate_multiplier": float(getattr(opt, "hab_mv_candidate_multiplier", 2.0)),
            "hab_fisher_protect_quantile": float(getattr(opt, "hab_fisher_protect_quantile", 0.90)),
            "hab_exact_final_count": int(bool(getattr(opt, "hab_exact_final_count", False))),
            "hab_target_gaussians": int(getattr(opt, "hab_target_gaussians", 0) or 0),
            "hab_dynamic_target_gaussians": int(event_stats.get(
                "hab_dynamic_target_gaussians",
                getattr(opt, "hab_current_target_gaussians", 0)
            ) or 0),
            "hab_load_cost": float(getattr(opt, "hab_current_load_cost", 0.0) or 0.0),
            "hab_load_density": float(getattr(opt, "hab_current_load_density", 0.0) or 0.0),
            "hab_load_cost_ema": float(getattr(opt, "hab_load_cost_ema", 0.0) or 0.0),
            "hab_load_density_ema": float(getattr(opt, "hab_load_density_ema", 0.0) or 0.0),
            "hab_load_target_cost": float(event_stats.get(
                "hab_load_target_cost",
                getattr(opt, "hab_load_target_cost", 0.0)
            ) or 0.0),
            "hab_load_pressure": float(event_stats.get(
                "hab_load_pressure",
                getattr(opt, "hab_load_pressure", 0.0)
            ) or 0.0),
        }

        with open(self.path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow(row)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    hab_logger = HABStatsLogger(dataset.model_path, opt)
    hab_load_controller = HABLoadController(opt)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if (getattr(opt, "hab_budget_schedule", "per_event") == "at_end"
            and opt.iterations < opt.densify_until_iter):
        raise RuntimeError(
            "at_end schedule requires iterations >= densify_until_iter "
            "so the terminal exact cut cannot invalidate active densification state "
            "({} < {})".format(opt.iterations, opt.densify_until_iter))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        hab_load_controller.update(render_pkg, gaussians.get_xyz.shape[0], opt)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            event_stats = {}

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))

            # AT-END-EXACT: delay the budget ceiling until the final iteration.
            # The cut must happen before
            # scene.save(), otherwise the saved PLY and the logged count diverge.
            if (iteration == opt.iterations
                    and getattr(opt, "hab_budget_schedule", "per_event") == "at_end"
                    and getattr(opt, "hab_mode", "off") != "off"):
                end_target = int(getattr(opt, "hab_target_gaussians", 0) or 0)
                end_before = int(gaussians.get_xyz.shape[0])
                if end_target <= 0:
                    raise RuntimeError("at_end schedule requires a positive HAB target")
                if end_before > end_target:
                    end_cameras = sampling_cameras(scene.getTrainCameras().copy())
                    _, end_score = compute_gaussian_score_fastgs(
                        end_cameras, gaussians, pipe, bg, opt)
                    end_cut, end_keep_mask = gaussians._hab_prune_to_budget(
                        end_target, 1.0, end_score, 0.1,
                        getattr(opt, "hab_priority_mode", "joint"),
                        getattr(opt, "hab_fisher_protect_quantile", 0.90),
                        getattr(opt, "hab_mv_candidate_multiplier", 2.0))
                    if end_keep_mask is not None:
                        if end_score.reshape(-1).shape[0] != end_keep_mask.shape[0]:
                            raise RuntimeError(
                                "at_end score/mask length mismatch: {} vs {}".format(
                                    end_score.reshape(-1).shape[0], end_keep_mask.shape[0]))
                        end_score = end_score.reshape(-1)[end_keep_mask]
                    event_stats = {
                        "before_count": end_before,
                        "after_count": int(gaussians.get_xyz.shape[0]),
                        "baseline_prune_candidates": 0,
                        "baseline_pruned": 0,
                        "hab_budget_pruned": int(end_cut),
                        "final_prune_event": True,
                        "pruning_score": end_score,
                    }
                    event_stats.update(gaussians.last_hab_priority_stats)
                    if event_stats["after_count"] != end_target:
                        raise RuntimeError(
                            "at_end exact budget failed: {} != {}".format(
                                event_stats["after_count"], end_target))
                    gaussians.last_hab_stats = event_stats
                    print("[at_end] budget prune removed {} -> {} (target {}) @ iter {}".format(
                        end_cut, event_stats["after_count"], end_target, iteration))
                elif end_before < end_target:
                    raise RuntimeError(
                        "at_end count {} is already below target {}; count match is impossible"
                        .format(end_before, end_target))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    # The multiview consistent densification of fastgs
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    load_event_stats = hab_load_controller.configure_event(iteration, gaussians.get_xyz.shape[0], opt)
                    event_stats = gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold,
                                                min_opacity = 0.005,
                                                extent = scene.cameras_extent,
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score,
                                                current_iteration = iteration)
                    event_stats.update(load_event_stats)
                    event_stats["densify_event"] = True
                    event_stats["importance_score"] = importance_score
                    event_stats["pruning_score"] = pruning_score

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # FastGS multi-view-consistent pruning runs every 3k iterations after 15k.
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)                    
                if (getattr(opt, "hab_budget_schedule", "per_event") == "ramp"
                        and getattr(opt, "hab_mode", "off") != "off"
                        and not hasattr(opt, "_hab_ramp_start_count")):
                    opt._hab_ramp_start_count = int(gaussians.get_xyz.shape[0])
                load_event_stats = hab_load_controller.configure_event(iteration, gaussians.get_xyz.shape[0], opt)
                floor = 0
                if (getattr(opt, "hab_exact_final_count", False)
                        and getattr(opt, "hab_mode", "off") != "off"
                        and getattr(opt, "hab_budget_schedule", "per_event") != "final_only"):
                    floor = int(getattr(opt, "hab_current_target_gaussians", 0)
                                or getattr(opt, "hab_target_gaussians", 0) or 0)
                event_stats, final_keep_mask = gaussians.final_prune_fastgs(
                    min_opacity = 0.1,
                    pruning_score = pruning_score,
                    min_keep = floor,
                    return_keep_mask = True)
                flat_pruning_score = pruning_score.reshape(-1)
                if flat_pruning_score.shape[0] != final_keep_mask.shape[0]:
                    raise RuntimeError(
                        "final-prune score/mask length mismatch: {} vs {}".format(
                            flat_pruning_score.shape[0], final_keep_mask.shape[0]))
                pruning_score = flat_pruning_score[final_keep_mask]
                event_stats.update(load_event_stats)
                event_stats["final_prune_event"] = True

                # RAMP: spread only the residual post-densification ceiling cut
                # over the four FastGS prune events (18k/21k/24k/27k).  The
                # priority, target, and total removal are unchanged; each partial
                # cut receives its own recovery window.  There is one fixed ramp
                # definition and no tunable slope/grid.
                if (getattr(opt, "hab_budget_schedule", "per_event") == "ramp"
                        and getattr(opt, "hab_mode", "off") != "off"):
                    ramp_target = int(getattr(opt, "hab_target_gaussians", 0) or 0)
                    ramp_start = int(getattr(opt, "_hab_ramp_start_count", 0) or 0)
                    ramp_first = 18_000
                    ramp_last = ((min(opt.iterations, 30_000) - 1) // 3000) * 3000
                    ramp_events = ((ramp_last - ramp_first) // 3000) + 1
                    ramp_index = ((iteration - ramp_first) // 3000) + 1
                    if ramp_target <= 0 or ramp_start <= 0 or ramp_events <= 0:
                        raise RuntimeError("invalid fixed ramp configuration")
                    ramp_span = max(0, ramp_start - ramp_target)
                    ramp_ceiling = int(round(
                        ramp_start - ramp_span * (float(ramp_index) / ramp_events)))
                    ramp_ceiling = max(ramp_target, ramp_ceiling)
                    event_stats["hab_ramp_ceiling"] = ramp_ceiling
                    if gaussians.get_xyz.shape[0] > ramp_ceiling:
                        ramp_cut, ramp_keep_mask = gaussians._hab_prune_to_budget(
                            ramp_ceiling, 1.0, pruning_score, 0.1,
                            getattr(opt, "hab_priority_mode", "joint"),
                            getattr(opt, "hab_fisher_protect_quantile", 0.90),
                            getattr(opt, "hab_mv_candidate_multiplier", 2.0))
                        if ramp_keep_mask is not None:
                            if pruning_score.shape[0] != ramp_keep_mask.shape[0]:
                                raise RuntimeError(
                                    "ramp score/mask length mismatch: {} vs {}".format(
                                        pruning_score.shape[0], ramp_keep_mask.shape[0]))
                            pruning_score = pruning_score[ramp_keep_mask]
                        event_stats["hab_budget_pruned"] = int(
                            event_stats.get("hab_budget_pruned", 0)) + int(ramp_cut)
                        event_stats["after_count"] = int(gaussians.get_xyz.shape[0])
                        event_stats.update(gaussians.last_hab_priority_stats)
                        print("[ramp] event {}/{} @ iter {}: ceiling {} removed {} -> {}"
                              .format(ramp_index, ramp_events, iteration, ramp_ceiling,
                                      ramp_cut, event_stats["after_count"]))

                # LAST-EVENT-EXACT: enforce the exact budget at the last FastGS
                # prune event so subsequent optimization can adapt to the retained
                # set. Densification has already ended, and no later prune event
                # changes the count saved at the final iteration.
                last_prune_event = ((min(opt.iterations, 30_000) - 1) // 3000) * 3000
                if iteration == last_prune_event and getattr(opt, "hab_mode", "off") != "off":
                    budget_target = int(getattr(opt, "hab_target_gaussians", 0) or 0)
                    schedule_mode = getattr(opt, "hab_budget_schedule", "per_event")
                    wants_exact = ((schedule_mode == "final_only"
                                    or getattr(opt, "hab_exact_final_count", False))
                                   and schedule_mode != "at_end")
                    current_n = gaussians.get_xyz.shape[0]
                    if budget_target > 0 and wants_exact and current_n > budget_target:
                        n_cut, exact_keep_mask = gaussians._hab_prune_to_budget(
                            budget_target, 1.0, pruning_score, 0.1,
                            getattr(opt, "hab_priority_mode", "joint"),
                            getattr(opt, "hab_fisher_protect_quantile", 0.90),
                            getattr(opt, "hab_mv_candidate_multiplier", 2.0))
                        if exact_keep_mask is not None:
                            if pruning_score.shape[0] != exact_keep_mask.shape[0]:
                                raise RuntimeError(
                                    "exact-budget score/mask length mismatch: {} vs {}".format(
                                        pruning_score.shape[0], exact_keep_mask.shape[0]))
                            pruning_score = pruning_score[exact_keep_mask]
                        event_stats["hab_budget_pruned"] = int(
                            event_stats.get("hab_budget_pruned", 0)) + int(n_cut)
                        event_stats["after_count"] = int(gaussians.get_xyz.shape[0])
                        event_stats.update(gaussians.last_hab_priority_stats)
                        label = "final_only" if schedule_mode == "final_only" else "exact_budget"
                        print("[{}] budget prune removed {} -> {} (target {}) @ iter {}".format(
                            label, n_cut, gaussians.get_xyz.shape[0], budget_target, iteration))
                    elif budget_target > 0 and wants_exact and current_n < budget_target:
                        print("[exact_budget] WARNING count {} < target {} @ iter {}: "
                              "cannot reach budget, arm is not count-matched".format(
                                  current_n, budget_target, iteration))

                accounted_after = (int(event_stats.get("before_count", 0))
                                   - int(event_stats.get("baseline_pruned", 0))
                                   - int(event_stats.get("hab_budget_pruned", 0)))
                if int(event_stats.get("after_count", 0)) != accounted_after:
                    raise RuntimeError(
                        "prune accounting mismatch at {}: after={} accounted={}".format(
                            iteration, event_stats.get("after_count"), accounted_after))

                event_stats["pruning_score"] = pruning_score
                gaussians.last_hab_stats = event_stats

            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3
            hab_logger.log(iteration, loss, gaussians, visibility_filter, radii, render_pkg,
                           iter_time, optim_time, opt, event_stats)

    # scene.save(iteration)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random/numpy/torch. safe_state() hardcoded 0, "
                             "which made multi-seed runs impossible; this exposes it so "
                             "run-to-run variance can be measured.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # safe_state() seeds everything to a hardcoded 0. Re-seed from --seed so
    # that independent runs are actually independent and reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("Using RNG seed: {}".format(args.seed))

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
