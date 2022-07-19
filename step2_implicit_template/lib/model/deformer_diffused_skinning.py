import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .broyden import broyden

class ForwardDeformerDiffusedSkinning(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self, subject, cpose_smpl_mesh_path, cpose_weight_grid_path):
        super().__init__()

        self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19]

        ### NOTE query grid data
        self.subject = subject
        self.bbox_grid_extend = None
        self.bbox_grid_center = None
        self.weight_grid = None
        
        if self.bbox_grid_extend is None or self.bbox_grid_center is None or self.weight_grid is None:
            cpose_smpl_mesh = trimesh.load(cpose_smpl_mesh_path, process=False)
            cpose_verts = torch.from_numpy(np.array(cpose_smpl_mesh.vertices)).float().cuda()[:, :3]
            bbox_data_min = cpose_verts.min(0).values
            bbox_data_max = cpose_verts.max(0).values
            bbox_data_extend = (bbox_data_max - bbox_data_min).max()
            bbox_grid_extend = bbox_data_extend * 1.1
            center = (bbox_data_min + bbox_data_max) / 2
            
            grid_pt = torch.from_numpy(np.load(cpose_weight_grid_path)).float().cuda()

            self.bbox_grid_extend = bbox_grid_extend
            self.bbox_grid_center = center
            self.weight_grid = grid_pt


    def forward(self, xd, cond, tfs, eval_mode=False):
        """Given deformed point return its caonical correspondence

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
        xc_init = self.init(xd, tfs)

        xc_opt, others = self.search(xd, xc_init, cond, tfs, eval_mode=eval_mode)

        if eval_mode:
            return xc_opt, others

        # compute correction term for implicit differentiation during training

        # do not back-prop through broyden
        xc_opt = xc_opt.detach()

        # reshape to [B,?,D] for network query
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_opt = xc_opt.reshape((n_batch, n_point * n_init, n_dim))

        xd_opt = self.forward_skinning(xc_opt, cond, tfs)

        grad_inv = self.gradient(xc_opt, cond, tfs).inverse()

        correction = xd_opt - xd_opt.detach()
        correction = torch.einsum("bnij,bnj->bni", -grad_inv.detach(), correction)

        # trick for implicit diff with autodiff:
        # xc = xc_opt + 0 and xc' = correction'
        xc = xc_opt + correction

        # reshape back to [B,N,I,D]
        xc = xc.reshape(xc_init.shape)

        return xc, others

    def init(self, xd, tfs):
        """Transform xd to canonical space for initialization

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc_init (tensor): gradients. shape: [B, N, I, D]
        """
        n_batch, n_point, _ = xd.shape
        _, n_joint, _, _ = tfs.shape

        xc_init = []
        for i in self.init_bones:
            w = torch.zeros((n_batch, n_point, n_joint), device=xd.device)
            w[:, :, i] = 1
            xc_init.append(skinning(xd, w, tfs, inverse=True))

        xc_init = torch.stack(xc_init, dim=2)

        return xc_init

    def search(self, xd, xc_init, cond, tfs, eval_mode=False):
        """Search correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """
        # reshape to [B,?,D] for other functions
        n_batch, n_point, n_init, n_dim = xc_init.shape
        xc_init = xc_init.reshape(n_batch, n_point * n_init, n_dim)
        xd_tgt = xd.repeat_interleave(n_init, dim=1)

        # compute init jacobians
        if not eval_mode:
            J_inv_init = self.gradient(xc_init, cond, tfs).inverse()
        else:
            w = self.query_weights(xc_init, cond, mask=None)
            # ic(einsum("bpn,bnij->bpij", w, tfs)[:, :, :3, :3][0, 0])
            # ic(w.shape, tfs.shape)
            J_inv_init = torch.einsum("bpn,bnij->bpij", w, tfs)[:, :, :3, :3].inverse()
            # J_inv_init = torch.pinverse(einsum("bpn,bnij->bpij", w, tfs)[:, :, :3, :3])

        # reshape init to [?,D,...] for boryden
        xc_init = xc_init.reshape(-1, n_dim, 1)
        J_inv_init = J_inv_init.flatten(0, 1)

        # construct function for root finding
        def _func(xc_opt, mask=None):
            # reshape to [B,?,D] for other functions
            xc_opt = xc_opt.reshape(n_batch, n_point * n_init, n_dim)
            xd_opt = self.forward_skinning(xc_opt, cond, tfs, mask=mask)
            error = xd_opt - xd_tgt
            # reshape to [?,D,1] for boryden
            error = error.flatten(0, 1)[mask].unsqueeze(-1)
            return error

        # run broyden without grad
        with torch.no_grad():
            result = broyden(_func, xc_init, J_inv_init)

        # reshape back to [B,N,I,D]
        xc_opt = result["result"].reshape(n_batch, n_point, n_init, n_dim)
        result["valid_ids"] = result["valid_ids"].reshape(n_batch, n_point, n_init)

        return xc_opt, result

    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        w = self.query_weights(xc, cond, mask=mask)
        xd = skinning(xc, w, tfs, inverse=False)
        return xd

    def query_weights(self, xc, cond, mask=None):
        """Get skinning weights in canonical space

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): valid indices. shape: [B, N]

        Returns:
            w (tensor): skinning weights. shape: [B, N, J]
        """

        def get_w(p_xc, p_mask, p_grid):
            n_batch, n_point, n_dim = p_xc.shape

            if n_batch * n_point == 0:
                return p_xc

            # reshape to [N,?]
            p_xc = p_xc.reshape(n_batch * n_point, n_dim)
            if p_mask is not None:
                p_xc = p_xc[p_mask]   # (n_b*n_p, n_dim)

            x = F.grid_sample(p_grid[None],
                              p_xc[None, None, None],
                              align_corners=False,
                              padding_mode='border')[0, :, 0, 0].T  # [Nv, 24]

            # add placeholder for masked prediction
            if p_mask is not None:
                x_full = torch.zeros(n_batch * n_point, x.shape[-1], device=x.device)
                x_full[p_mask] = x
            else:
                x_full = x

            return x_full.reshape(n_batch, n_point, -1)
        
        def inv_transform_v(v, scale_grid, transl):
            """
            v: [b, n, 3]
            """
            v = v - transl[None, None]
            v = v / scale_grid
            v = v * 2

            return v



        v_cano_in_grid_coords = inv_transform_v(xc, self.bbox_grid_extend, self.bbox_grid_center)

        out = get_w(v_cano_in_grid_coords, mask, self.weight_grid)
        # out = F.grid_sample(grid_pt[None], v_cano_in_grid_coords[None, None], align_corners=False, padding_mode='border')[0, :, 0, 0].T  # [Nv, 24]
        w = out

        # ic(xc.shape, w.shape)
        # ic(w.sum(-1).max(), w.sum(-1).min())
        return w

    def gradient(self, xc, cond, tfs):
        """Get gradients df/dx

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            grad (tensor): gradients. shape: [B, N, D, D]
        """
        xc.requires_grad_(True)

        xd = self.forward_skinning(xc, cond, tfs)

        grads = []
        for i in range(xd.shape[-1]):
            d_out = torch.zeros_like(xd, requires_grad=False, device=xd.device)
            d_out[:, :, i] = 1
            grad = torch.autograd.grad(
                outputs=xd,
                inputs=xc,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad)

        return torch.stack(grads, dim=-2)


def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning

    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

    return x_h[:, :, :3]
