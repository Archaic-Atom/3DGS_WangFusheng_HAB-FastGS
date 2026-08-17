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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.hab_priority_utils import select_guarded_prune_indices

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.last_hab_stats = {}
        self.last_hab_priority_stats = {}
        # Pruning may be invoked outside a densification event (for example with
        # the at-end schedule), so the optional radii cache must
        # exist from construction rather than being created lazily.
        self.tmp_radii = None
        self.setup_functions()

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
        else:
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def _filter_hab_vector(self, value, keep_mask):
        if value is None or keep_mask is None:
            return value
        value = value.reshape(-1)
        if value.shape[0] != keep_mask.shape[0]:
            return value
        return value[keep_mask]

    def _hab_empirical_fisher_proxy(self):
        """Return a low-overhead diagonal empirical-Fisher sensitivity proxy.

        Adam already keeps an EMA of squared gradients.  We aggregate the
        position and log-scale second moments after robust, per-group
        normalization.  This is intentionally described as a proxy rather than
        the block Gauss-Newton/Fisher score used by post-hoc pruning methods.
        """
        if self.optimizer is None:
            return None
        current_count = int(self.get_xyz.shape[0])
        contributions = []
        for group in self.optimizer.param_groups:
            if group.get("name") not in ("xyz", "scaling") or not group.get("params"):
                continue
            state = self.optimizer.state.get(group["params"][0], None)
            if not state or "exp_avg_sq" not in state:
                continue
            second_moment = state["exp_avg_sq"].detach().float()
            if second_moment.shape[0] != current_count:
                continue
            diagonal = torch.nan_to_num(
                second_moment.reshape(current_count, -1).mean(dim=1),
                nan=0.0, posinf=torch.finfo(torch.float32).max, neginf=0.0)
            positive = diagonal[diagonal > 0]
            if positive.numel() == 0:
                continue
            scale = torch.quantile(positive, 0.5).clamp_min(1e-30)
            contributions.append(torch.log1p(diagonal / scale))
        if not contributions:
            return None
        return torch.stack(contributions, dim=0).mean(dim=0)

    def _hab_prune_to_budget(self, target_count, max_prune_fraction, pruning_score, min_opacity,
                             priority_mode="joint", fisher_protect_quantile=0.90,
                             mv_candidate_multiplier=2.0):
        """Prune the Gaussian set down towards ``target_count``.

        ``priority_mode`` selects which signals form the pruning priority. The
        default ``joint`` mode is the HAB-FastGS priority
        ``p_i = q_hat_i + 0.5(1-alpha_i) + 0.25 r_hat_i``. The remaining modes
        use a single signal or random choice while keeping the budget, prune
        fraction, and trigger schedule unchanged.
        """
        current_count = self.get_xyz.shape[0]
        self.last_hab_priority_stats = {
            "hab_candidate_band_count": 0,
            "hab_fisher_protected": 0,
            "hab_fisher_guard_relaxed": 0,
        }
        excess = current_count - target_count
        if target_count <= 0 or excess <= 0 or max_prune_fraction <= 0:
            return 0, None

        prune_budget = min(excess, max(1, int(current_count * max_prune_fraction)))
        device = self.get_xyz.device
        priority = torch.zeros((current_count), dtype=torch.float32, device=device)
        eligible = torch.ones((current_count), dtype=torch.bool, device=device)

        use_score = priority_mode in ("joint", "score_only")
        use_opacity = priority_mode in ("joint", "opacity_only")
        use_radii = priority_mode in ("joint", "radii_only")

        # Eligibility is derived from the pruning score in every mode so all
        # ranking policies draw from the same candidate pool.
        valid_count = 0
        if pruning_score is not None and pruning_score.numel() > 0:
            score = pruning_score.detach().float().reshape(-1).to(device)
            valid_count = min(score.shape[0], current_count)
            valid_score = torch.nan_to_num(score[:valid_count], nan=0.0, posinf=1.0, neginf=0.0)
            score_range = torch.max(valid_score) - torch.min(valid_score)
            if use_score and score_range.item() > 1e-8:
                priority[:valid_count] += (valid_score - torch.min(valid_score)) / score_range

            eligible[:] = False
            eligible[:valid_count] = True
            if valid_count < prune_budget:
                eligible[:] = True

        opacities = self.get_opacity.detach().float().reshape(-1)
        if opacities.shape[0] == current_count:
            if use_opacity:
                priority += 0.5 * (1.0 - torch.clamp(opacities, 0.0, 1.0))
            eligible = torch.logical_or(eligible, opacities < min_opacity)

        if use_radii and self.max_radii2D.shape[0] == current_count:
            radii_cost = self.max_radii2D.detach().float()
            max_radius = torch.max(radii_cost)
            if max_radius.item() > 0:
                priority += 0.25 * torch.clamp(radii_cost / (max_radius + 1e-6), 0.0, 1.0)

        if priority_mode == "random":
            # Uniform random ranking over the same eligible pool. Draws from the
            # global torch RNG, which train.py seeds explicitly.
            priority = torch.rand((current_count), dtype=torch.float32, device=device)

        guarded_modes = {
            "opacity_mv_band",
            "opacity_fisher_guard",
            "opacity_mv_fisher_guard",
        }
        if priority_mode in guarded_modes:
            fisher_proxy = None
            if priority_mode in ("opacity_fisher_guard", "opacity_mv_fisher_guard"):
                fisher_proxy = self._hab_empirical_fisher_proxy()
            selected_indices, diagnostics = select_guarded_prune_indices(
                prune_budget=prune_budget,
                eligible=eligible,
                opacities=opacities,
                pruning_score=pruning_score,
                fisher_proxy=fisher_proxy,
                mode=priority_mode,
                candidate_multiplier=mv_candidate_multiplier,
                fisher_protect_quantile=fisher_protect_quantile,
            )
            self.last_hab_priority_stats = diagnostics
            if selected_indices.numel() == 0:
                return 0, None
            budget_prune_mask = torch.zeros(
                (current_count), dtype=torch.bool, device=device)
            budget_prune_mask[selected_indices] = True
            pruned = int(torch.sum(budget_prune_mask).item())
            keep_mask = ~budget_prune_mask
            self.prune_points(budget_prune_mask)
            return pruned, keep_mask

        priority[~eligible] = -float("inf")
        available = int(torch.isfinite(priority).sum().item())
        if available <= 0:
            return 0, None

        prune_budget = min(prune_budget, available)
        selected_indices = torch.topk(priority, prune_budget, largest=True).indices
        budget_prune_mask = torch.zeros((current_count), dtype=torch.bool, device=device)
        budget_prune_mask[selected_indices] = True
        pruned = int(torch.sum(budget_prune_mask).item())
        keep_mask = ~budget_prune_mask
        self.prune_points(budget_prune_mask)
        return pruned, keep_mask

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None, current_iteration = None):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is the FastGS metric-guided densification path.
            3.  Finally, gaussians with low opacity or very large size are pruned.
        '''
        stats = {
            "before_count": int(self.get_xyz.shape[0]),
            "after_count": int(self.get_xyz.shape[0]),
            "clone_candidates": 0,
            "split_candidates": 0,
            "baseline_prune_candidates": 0,
            "baseline_pruned": 0,
            "hab_budget_pruned": 0,
        }

        self.tmp_radii = radii
        hab_mode = getattr(args, "hab_mode", "off")
        target_count = int(getattr(args, "hab_current_target_gaussians", 0) or 0)
        if target_count <= 0:
            target_count = int(getattr(args, "hab_target_gaussians", 0) or 0)
        budget_start_iter = int(getattr(args, "hab_budget_start_iter", 0) or 0)
        max_prune_fraction = float(getattr(args, "hab_max_prune_fraction", 0.0) or 0.0)
        priority_mode = getattr(args, "hab_priority_mode", "joint")
        fisher_protect_quantile = float(getattr(
            args, "hab_fisher_protect_quantile", 0.90))
        mv_candidate_multiplier = float(getattr(
            args, "hab_mv_candidate_multiplier", 2.0))
        placement = getattr(args, "hab_prune_placement", "pre_densify")
        schedule = getattr(args, "hab_budget_schedule", "per_event")

        # "final_only" defers budget enforcement to one pass at the last native
        # pruning event.
        budget_active = (
            hab_mode != "off"
            and target_count > 0
            and schedule not in ("final_only", "at_end")
            and (current_iteration is None or current_iteration >= budget_start_iter)
        )

        if budget_active and placement == "pre_densify":
            hab_pruned, keep_mask = self._hab_prune_to_budget(
                target_count, max_prune_fraction, pruning_score, min_opacity, priority_mode,
                fisher_protect_quantile, mv_candidate_multiplier
            )
            stats["hab_budget_pruned"] = hab_pruned
            stats.update(self.last_hab_priority_stats)
            importance_score = self._filter_hab_vector(importance_score, keep_mask)
            pruning_score = self._filter_hab_vector(pruning_score, keep_mask)
            if keep_mask is not None and self.tmp_radii is not None:
                radii = self.tmp_radii

        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # Use the multi-view-consistent metric to filter densification candidates.
        if importance_score is None:
            metric_mask = torch.zeros_like(all_clones)
        else:
            metric_mask = torch.zeros_like(all_clones)
            valid_count = min(importance_score.reshape(-1).shape[0], metric_mask.shape[0])
            metric_mask[:valid_count] = importance_score.reshape(-1)[:valid_count] > 5

        stats["clone_candidates"] = int(torch.sum(torch.logical_and(metric_mask, all_clones)).item())
        stats["split_candidates"] = int(torch.sum(torch.logical_and(metric_mask, all_splits)).item())

        self.densify_and_clone_fastgs(metric_mask, all_clones)
        self.densify_and_split_fastgs(metric_mask, all_splits)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if pruning_score is None:
            scores = torch.zeros((self.get_xyz.shape[0]), dtype=torch.float32, device=self.get_xyz.device)
        else:
            score = torch.nan_to_num(pruning_score.detach().reshape(-1).to(self.get_xyz.device), nan=0.0, posinf=1.0, neginf=0.0)
            scores = 1 - score
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)
        stats["baseline_prune_candidates"] = int(to_remove.item())

        # Apply the native FastGS stochastic pruning to eligible candidates.
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32, device=self.get_xyz.device)
            valid_count = min(scores.reshape(-1).shape[0], n_init_points)
            padded_importance[:valid_count] = 1 / (1e-6 + scores.reshape(-1)[:valid_count])
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            if torch.sum(padded_importance).item() > 0:
                sampled_indices = torch.multinomial(padded_importance, min(remove_budget, n_init_points), replacement=False)
                selected_pts_mask[sampled_indices] = True
                final_prune = torch.logical_and(prune_mask, selected_pts_mask)
                stats["baseline_pruned"] = int(torch.sum(final_prune).item())
                self.prune_points(final_prune)

        # The optional post-densify placement applies the same budget after
        # clone/split. The pruning-score vector may be shorter than the expanded
        # Gaussian set, so _hab_prune_to_budget restricts eligibility safely.
        if budget_active and placement == "post_densify":
            hab_pruned, _ = self._hab_prune_to_budget(
                target_count, max_prune_fraction, pruning_score, min_opacity, priority_mode,
                fisher_protect_quantile, mv_candidate_multiplier
            )
            stats["hab_budget_pruned"] = hab_pruned
            stats.update(self.last_hab_priority_stats)

        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
        stats["after_count"] = int(self.get_xyz.shape[0])
        self.last_hab_stats = stats
        return stats

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def final_prune_fastgs(self, min_opacity, pruning_score = None, min_keep = 0,
                           return_keep_mask = False):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        before_count = int(self.get_xyz.shape[0])
        prune_mask = (self.get_opacity < min_opacity).squeeze() 
        scores_mask = torch.zeros_like(prune_mask)
        if pruning_score is not None and pruning_score.numel() > 0:
            score = torch.nan_to_num(pruning_score.detach().reshape(-1).to(prune_mask.device), nan=0.0, posinf=1.0, neginf=0.0)
            valid_count = min(score.shape[0], scores_mask.shape[0])
            scores_mask[:valid_count] = score[:valid_count] > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)

        # Budget floor. When min_keep is set, spare the highest-opacity flagged
        # Gaussians so this late-stage quality prune cannot undercut the requested
        # minimum count.
        skipped_for_floor = 0
        if min_keep > 0:
            keep_after = before_count - int(torch.sum(final_prune).item())
            if keep_after < min_keep:
                cand_idx = torch.nonzero(final_prune, as_tuple=False).squeeze(-1)
                if cand_idx.numel() > 0:
                    deficit = min(min_keep - keep_after, int(cand_idx.numel()))
                    cand_op = self.get_opacity.detach().reshape(-1)[cand_idx]
                    spare = cand_idx[torch.argsort(cand_op, descending=True)[:deficit]]
                    final_prune[spare] = False
                    skipped_for_floor = int(spare.numel())

        pruned = int(torch.sum(final_prune).item())
        keep_mask = ~final_prune
        self.prune_points(final_prune)
        stats = {
            "before_count": before_count,
            "after_count": int(self.get_xyz.shape[0]),
            "baseline_prune_candidates": pruned + skipped_for_floor,
            "baseline_pruned": pruned,
            "hab_budget_pruned": 0,
            "budget_floor_spared": skipped_for_floor,
        }
        self.last_hab_stats = stats
        if return_keep_mask:
            return stats, keep_mask
        return stats
