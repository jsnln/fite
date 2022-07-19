import torch
import torch.nn as nn

import numpy as np

from ..lib.model.smpl import SMPLServer
from ..lib.model.network import ImplicitNetwork
from ..lib.model.metrics import calculate_iou
from ..lib.utils.meshing import generate_mesh
from ..lib.model.helpers import masked_softmax
from .model.deformer_diffused_skinning import ForwardDeformerDiffusedSkinning, skinning
from ..lib.utils.render import render_trimesh, render_joint, weights2colors

import kaolin

class SNARFModelDiffusedSkinning(nn.Module):

    def __init__(self, soft_blend, smpl_model_path, pose_conditioning, opt_network, subject, cpose_smpl_mesh_path, cpose_weight_grid_path, meta_info, data_processor=None):
        super().__init__()


        self.soft_blend = soft_blend
        self.pose_conditioning = pose_conditioning

        self.network = ImplicitNetwork(**opt_network)
        self.deformer = ForwardDeformerDiffusedSkinning(subject, cpose_smpl_mesh_path, cpose_weight_grid_path)

        print(self.network)
        print(self.deformer)

        gender      = str(meta_info['gender'])
        betas       = meta_info['betas'] if 'betas' in meta_info else None
        v_template  = meta_info['v_template'] if 'v_template' in meta_info else None

        self.smpl_server = SMPLServer(smpl_model_path=smpl_model_path, gender=gender, betas=betas, v_template=v_template)
        self.data_processor = data_processor

    def forward(self, pts_d, smpl_tfs, smpl_thetas, eval_mode=True):
        
        # rectify rest pose
        smpl_tfs = torch.einsum('bnij,njk->bnik', smpl_tfs, self.smpl_server.tfs_c_inv)

        ### NOTE cond_zero is not used now
        cond = {'smpl': smpl_thetas[:,3:]/np.pi}
        cond_zero = {'smpl': torch.zeros_like(smpl_thetas[:,3:], device=smpl_thetas.device)}

        batch_points = 60000

        accum_pred = []
        # split to prevent out of memory
        for pts_d_split in torch.split(pts_d, batch_points, dim=1):

            # compute canonical correspondences
            if not self.pose_conditioning:
                pts_c, intermediates = self.deformer(pts_d_split, cond_zero, smpl_tfs, eval_mode=eval_mode)
            else:
                pts_c, intermediates = self.deformer(pts_d_split, cond, smpl_tfs, eval_mode=eval_mode)
            
            ### NOTE removed hands (smpl verts are not batched)
            intermediates = self.remove_hands(pts_c, self.smpl_server.verts_c, intermediates)
            
            # query occuancy in canonical space
            num_batch, num_point, num_init, num_dim = pts_c.shape
            pts_c = pts_c.reshape(num_batch, num_point * num_init, num_dim)
            
            if not self.pose_conditioning:
                occ_pd = self.network(pts_c, cond_zero).reshape(num_batch, num_point, num_init)
            else:
                occ_pd = self.network(pts_c, cond).reshape(num_batch, num_point, num_init)

            # aggregate occupancy probablities
            mask = intermediates['valid_ids']
            if eval_mode:
                occ_pd = masked_softmax(occ_pd, mask, dim=-1, mode='max')
            else:
                occ_pd = masked_softmax(occ_pd, mask, dim=-1, mode='softmax', soft_blend=self.soft_blend)

            accum_pred.append(occ_pd)
        # input()
        accum_pred = torch.cat(accum_pred, 1)   

        return accum_pred

    ## NOTE this marks original points near hand as invalid 
    def remove_hands(self, cano_scan_points, cano_body_points, intermediates):
        """
        cano_scan_points: [num_batch, num_point, num_init, num_dim]
        cano_body_points: [1, num_body_point, num_dim]
        intermediates: dict containing bool tensor keyed 'valid_ids',
                 which is [n_batch, n_point, n_init]
        """
        # cano_points = torch.zeros_like(cano_scan_points)
        # cano_occ_gt = torch.zeros_like(cano_scan_occ_gt)

        left_hand_vertex_index = 2005 # palm center
        right_hand_vertex_index = 5509 # palm center
        
        # cut_offset_x = 0.03
        cut_offset_x = 0.2
        cut_offset_y = 0.08
        cut_offset_z = 0.1

        for batch_index in range(cano_scan_points.shape[0]):
            ### NOTE not batched,
            left_hand_x = cano_body_points[0, left_hand_vertex_index, 0]
            right_hand_x = cano_body_points[0, right_hand_vertex_index, 0]
            left_hand_y = cano_body_points[0, left_hand_vertex_index, 1]
            right_hand_y = cano_body_points[0, right_hand_vertex_index, 1]
            left_hand_z = cano_body_points[0, left_hand_vertex_index, 2]
            right_hand_z = cano_body_points[0, right_hand_vertex_index, 2]

            ### NOTE differently initialized correspondences
            for init_index in range(cano_scan_points.shape[2]):
                cano_scan_mask_x_left = (cano_scan_points[batch_index, :, init_index, 0] > left_hand_x) &  (cano_scan_points[batch_index, :, init_index, 0] < left_hand_x+cut_offset_x)
                cano_scan_mask_x_right = (cano_scan_points[batch_index, :, init_index, 0] < right_hand_x) & (cano_scan_points[batch_index, :, init_index, 0] > right_hand_x-cut_offset_x)
                cano_scan_mask_y = (cano_scan_points[batch_index, :, init_index, 1] < left_hand_y+cut_offset_y) & (cano_scan_points[batch_index, :, init_index, 1] > left_hand_y-cut_offset_y)
                cano_scan_mask_z = (cano_scan_points[batch_index, :, init_index, 2] < left_hand_z+cut_offset_z) & (cano_scan_points[batch_index, :, init_index, 1] > left_hand_z-cut_offset_z)

                cano_scan_mask = (cano_scan_mask_x_left | cano_scan_mask_x_right) & cano_scan_mask_y & cano_scan_mask_z

                # body_hands_mask = (cano_body_points[batch_index, 0, :] > left_hand_x-cut_offset) | (cano_body_points[batch_index, 0, :] < right_hand_x+cut_offset)

                ### NOTE set their validity to false
                intermediates['valid_ids'][batch_index, cano_scan_mask, init_index] = 0
            
        return intermediates

    def get_points_near_hands(self, cano_body_points, cano_body_faces, cano_body_normals, near_hand_sigma=0.005):
        """
        cano_scan_points: [num_batch, num_point, num_init, num_dim]
        cano_body_points: [1, num_body_point, num_dim]
        intermediates: dict containing bool tensor keyed 'valid_ids',
                 which is [n_batch, n_point, n_init]
        """
        # cano_points = torch.zeros_like(cano_scan_points)
        # cano_occ_gt = torch.zeros_like(cano_scan_occ_gt)

        left_hand_vertex_index = 2005 # palm center
        right_hand_vertex_index = 5509 # palm center
        
        cut_offset = 0.03
        # cut_offset_x = 0.2
        # cut_offset_y = 0.08
        # cut_offset_z = 0.1

        left_hand_x = cano_body_points[0, left_hand_vertex_index, 0]
        right_hand_x = cano_body_points[0, right_hand_vertex_index, 0]
        left_hand_y = cano_body_points[0, left_hand_vertex_index, 1]
        right_hand_y = cano_body_points[0, right_hand_vertex_index, 1]
        left_hand_z = cano_body_points[0, left_hand_vertex_index, 2]
        right_hand_z = cano_body_points[0, right_hand_vertex_index, 2]

        points_on_hand_indices = (cano_body_points[0, :, 0] > left_hand_x - cut_offset) | (cano_body_points[0, :, 0] < right_hand_x + cut_offset)
        points_on_hand = cano_body_points[0, points_on_hand_indices].reshape(1, -1, 3)
        normals_on_hand = cano_body_normals[0, points_on_hand_indices].reshape(1, -1, 3)

        points_near_hand = points_on_hand + torch.randn(1, points_on_hand.shape[1], 3, device=points_on_hand.device) * normals_on_hand * near_hand_sigma

        near_hand_occ_gt = kaolin.ops.mesh.check_sign(cano_body_points, cano_body_faces, points_near_hand).float()

        return points_near_hand, near_hand_occ_gt

    # def get_points_near_feet(self, cano_body_points, cano_body_faces, cano_body_normals, near_feet_sigma=0.005):
    #     """
    #     cano_scan_points: [num_batch, num_point, num_init, num_dim]
    #     cano_body_points: [1, num_body_point, num_dim]
    #     intermediates: dict containing bool tensor keyed 'valid_ids',
    #              which is [n_batch, n_point, n_init]
    #     """
    #     # cano_points = torch.zeros_like(cano_scan_points)
    #     # cano_occ_gt = torch.zeros_like(cano_scan_occ_gt)

    #     left_foot_vertex_index = 3392 # ankle, ~5 cm above the ground
    #     right_foot_vertex_index = 6730 # ankle, ~5 cm above the ground
        
    #     cut_offset = 0.03
    #     # cut_offset_x = 0.2
    #     # cut_offset_y = 0.08
    #     # cut_offset_z = 0.1

    #     left_foot_x = cano_body_points[0, left_foot_vertex_index, 0]
    #     right_foot_x = cano_body_points[0, right_foot_vertex_index, 0]
    #     left_foot_y = cano_body_points[0, left_foot_vertex_index, 1]
    #     right_foot_y = cano_body_points[0, right_foot_vertex_index, 1]
    #     left_foot_z = cano_body_points[0, left_foot_vertex_index, 2]
    #     right_foot_z = cano_body_points[0, right_foot_vertex_index, 2]

    #     points_on_foot_indices = (cano_body_points[0, :, 0] > left_foot_x - cut_offset) | (cano_body_points[0, :, 0] < right_foot_x + cut_offset)
    #     points_on_foot = cano_body_points[0, points_on_foot_indices].reshape(1, -1, 3)
    #     normals_on_foot = cano_body_normals[0, points_on_foot_indices].reshape(1, -1, 3)

    #     points_near_foot = points_on_foot + torch.randn(1, points_on_foot.shape[1], 3, device=points_on_foot.device) * normals_on_foot * near_feet_sigma

    #     near_foot_occ_gt = kaolin.ops.mesh.check_sign(cano_body_points, cano_body_faces, points_near_foot).float()

    #     return points_near_foot, near_foot_occ_gt

    def training_step(self, data, data_idx):

        # Data prep
        if self.data_processor is not None:
            data = self.data_processor.process(data)

        # BCE loss
        occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_thetas'], eval_mode=False)
        # ic(occ_pd.shape, data['occ_gt'].shape)

        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(occ_pd, data['occ_gt'])
        # self.log('train_bce', loss_bce)
        loss = loss_bce

        ### NOTE near hand part
        num_batch = data['pts_d'].shape[0]
        cond = {'smpl': data['smpl_thetas'][:,3:]/np.pi}
        cond_zero = {'smpl': torch.zeros_like(data['smpl_thetas'][0:1,3:], device=data['smpl_thetas'].device)}

        points_near_hand, occ_gt_near_hand = self.get_points_near_hands(self.smpl_server.verts_c, self.smpl_server.faces_c, self.smpl_server.vnormals_c)
        occ_pd_near_hand = self.network(points_near_hand.expand(num_batch, -1, -1), cond)   # [bs, 1440, 1]
        # ic(occ_pd_near_hand.shape, occ_gt_near_hand.shape)
        occ_gt_near_hand = occ_gt_near_hand[..., None].expand(num_batch, -1, -1)  # [bs, 1440, 1]
        # ic(points_near_hand.shape, occ_gt_near_hand.shape, occ_pd_near_hand.shape, occ_gt_near_hand.dtype)
        loss_near_hand = torch.nn.functional.binary_cross_entropy_with_logits(occ_pd_near_hand, occ_gt_near_hand)

        ### NOTE reweighting
        num_obs_points = occ_pd.shape[1]
        num_hand_points = occ_pd_near_hand.shape[1]
        w_obs = num_obs_points / (num_obs_points + num_hand_points)
        w_hand = num_hand_points / (num_obs_points + num_hand_points)
        # ic(w_obs, w_hand)

        # loss = w_obs * loss + w_hand * loss_near_hand
        
        ### NOTE no reweighting
        loss = loss + loss_near_hand


        # ic(loss)
        return loss
    
    def validation_step(self, data, data_idx):

        if self.data_processor is not None:
            data = self.data_processor.process(data)

        with torch.no_grad():
            if data_idx == 0:
                img_all = self.plot(data)['img_all']
                # ic(img_all.shape)
                # self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
                
            occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_thetas'], eval_mode=True)

            _, num_point, _ = data['occ_gt'].shape
            bbox_iou = calculate_iou(data['occ_gt'][:,:num_point//2]>0.5, occ_pd[:,:num_point//2]>0)
            surf_iou = calculate_iou(data['occ_gt'][:,num_point//2:]>0.5, occ_pd[:,num_point//2:]>0)

        return {'bbox_iou':bbox_iou, 'surf_iou':surf_iou, 'img_all': img_all}

    def validation_epoch_end(self, validation_step_outputs):

        bbox_ious, surf_ious = [], []
        for output in validation_step_outputs:
            bbox_ious.append(output['bbox_iou'])
            surf_ious.append(output['surf_iou'])
        
        # self.log('valid_bbox_iou', torch.stack(bbox_ious).mean())
        # self.log('valid_surf_iou', torch.stack(surf_ious).mean())

    def test_step(self, data, data_idx):

        with torch.no_grad():

            occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_thetas'], eval_mode=True)

            _, num_point, _ = data['occ_gt'].shape
            bbox_iou = calculate_iou(data['occ_gt'][:,:num_point//2]>0.5, occ_pd[:,:num_point//2]>0)
            surf_iou = calculate_iou(data['occ_gt'][:,num_point//2:]>0.5, occ_pd[:,num_point//2:]>0)

        return {'bbox_iou':bbox_iou, 'surf_iou':surf_iou}
            
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def plot(self, data, res=128, verbose=True, fast_mode=False):

        res_up = np.log2(res//32)

        if verbose and fast_mode:
            surf_pred_cano, weights = self.extract_mesh(self.smpl_server.verts_c, data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=True, with_weights=True)
            smpl_verts = self.smpl_server.verts_c if fast_mode else data['smpl_verts'][[0]]
            surf_pred_def, _ = self.extract_mesh(smpl_verts, data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=False, with_weights=False, fast_mode=fast_mode)
            # surf_pred_def = self.extract_mesh(data['smpl_verts'][[0]], data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=False, with_weights=False)

            img_pred_cano = render_trimesh(surf_pred_cano)
            img_pred_def  = render_trimesh(surf_pred_def)
            
            img_joint = render_joint(data['smpl_jnts'].data.cpu().numpy()[0],self.smpl_server.bone_ids)
            img_pred_def[1024:,:,:3] = 255
            img_pred_def[1024:-512,:, :3] = img_joint
            img_pred_def[1024:-512,:, -1] = 255

            results = {
                'img_all': np.concatenate([img_pred_cano, img_pred_def], axis=1),
                'mesh_cano': surf_pred_cano,
                'mesh_def' : surf_pred_def,
                'weights_cano': weights,
            }
        elif verbose:
            surf_pred_cano, weights = self.extract_mesh(self.smpl_server.verts_c, data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=True, with_weights=True)
            surf_pred_def, _ = self.extract_mesh(data['smpl_verts'][[0]], data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=False, with_weights=False)

            ### NOTE bug
            img_pred_cano = render_trimesh(surf_pred_cano)
            img_pred_def  = render_trimesh(surf_pred_def)
            
            img_joint = render_joint(data['smpl_jnts'].data.cpu().numpy()[0],self.smpl_server.bone_ids)
            img_pred_def[1024:,:,:3] = 255
            img_pred_def[1024:-512,:, :3] = img_joint
            img_pred_def[1024:-512,:, -1] = 255

            results = {
                'img_all': np.concatenate([img_pred_cano, img_pred_def], axis=1),
                'mesh_cano': surf_pred_cano,
                'mesh_def' : surf_pred_def,
                'weights_cano': weights,
            }
        else:
            smpl_verts = self.smpl_server.verts_c if fast_mode else data['smpl_verts'][[0]]

            surf_pred_def, _ = self.extract_mesh(smpl_verts, data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=False, with_weights=False, fast_mode=fast_mode)
                        
            img_pred_def  = render_trimesh(surf_pred_def, mode='p')
            results = {
                'img_all': img_pred_def,
                'mesh_def' : surf_pred_def,
            }

        return results

    def extract_mesh(self, smpl_verts, smpl_tfs, smpl_thetas, canonical=False, with_weights=False, res_up=2, fast_mode=False):
        '''
        In fast mode, we extract canonical mesh and then forward skin it to posed space.
        This is faster as it bypasses root finding.
        However, it's not deforming the continuous field, but the discrete mesh.
        '''
        if canonical or fast_mode:
            cond = {'smpl': smpl_thetas[:,3:]/np.pi}
            cond_zero = {'smpl': torch.zeros_like(smpl_thetas[:,3:], device=smpl_thetas.device) }
            
            if not self.pose_conditioning:
                occ_func = lambda x: self.network(x, cond_zero).reshape(-1, 1)
            else:
                occ_func = lambda x: self.network(x, cond).reshape(-1, 1)
        else:
            occ_func = lambda x: self.forward(x, smpl_tfs, smpl_thetas, eval_mode=True).reshape(-1, 1)
            
        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up)

        weights = None
        if fast_mode:
            verts  = torch.tensor(mesh.vertices).type_as(smpl_verts)
            weights = self.deformer.query_weights(verts[None], None).clamp(0,1)[0]

            smpl_tfs = torch.einsum('bnij,njk->bnik', smpl_tfs, self.smpl_server.tfs_c_inv)
            
            verts_mesh_deformed = skinning(verts.unsqueeze(0), weights.unsqueeze(0), smpl_tfs).data.cpu().numpy()[0]
            mesh.vertices = verts_mesh_deformed

        if with_weights:
            verts  = torch.tensor(mesh.vertices).cuda().float()
            weights = self.deformer.query_weights(verts[None], None).clamp(0,1)[0]
            mesh.visual.vertex_colors = weights2colors(weights.data.cpu().numpy())

        return mesh, weights

