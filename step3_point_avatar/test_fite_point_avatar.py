import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
from os.path import join


from .lib.fite_model import FITEModel
from .lib.dataset import FITEPosmapDataset, FITECanoDataRepository
from .lib.losses import normal_loss, chamfer_loss_separate
from .lib.lbs import lbs, inv_lbs
from .lib.utils import save_result_examples


if __name__ == '__main__':

    opt = {}
    with open(join('configs', 'common.yaml'), 'r') as common_opt_f:
        common_opt = yaml.safe_load(common_opt_f)
        opt.update(common_opt)
    with open(join('configs', f'step3.yaml'), 'r') as step_opt_f:
        step_opt = yaml.safe_load(step_opt_f)
        opt.update(step_opt)
    with open(join('configs', f'{opt["expname"]}_subject_list.yaml'), 'r') as subject_list_f:
        subject_list = yaml.safe_load(subject_list_f)
        opt['subject_list'] = subject_list
    with open(join('configs', f'{opt["expname"]}_projection_list.yaml'), 'r') as projection_list_f:
        projection_list = yaml.safe_load(projection_list_f)
        opt['projection_list'] = projection_list

    ### NOTE exp folders
    CHECKPOINTS_PATH = join(opt['result_folder'], opt['expname'], 'step3-checkpoints')
    TESTPCD_PATH = join(opt['result_folder'], opt['expname'], 'step3-test-pcds')

    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)
    if not os.path.exists(TESTPCD_PATH):
        os.makedirs(TESTPCD_PATH)

    ### NOTE add subject ids
    for subject_id_single in range(len(opt['subject_list'])):
        opt['subject_list'][subject_id_single]['id'] = subject_id_single


    cano_data_repo = FITECanoDataRepository(
            subject_list=opt['subject_list'],
            projection_list=opt['projection_list'],
            root_cano_data=opt['data_templates_path'],
            channels_geom_feat=opt['c_geom'],
            n_points_cano_data=opt['n_cano_points'],
            cano_pose_leg_angle=opt['leg_angle'],
            smpl_model_path=opt['smpl_model_path']
    ).set_device('cuda')

    test_dataset = FITEPosmapDataset(
            subject_list=opt['subject_list'],
            projection_list=opt['projection_list'],
            split='test',
            root_rendered=opt['data_posmaps_path'],
            root_packed=opt['data_scans_path'],
            remove_hand_foot_pose=False,
            data_spacing=opt['data_spacing'],
            selected_subjects=opt['selected_subjects'],
    )


    # batch_size = args.batch_size
    n_test_samples = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=False, drop_last=False)

    ### NOTE model preparation
    # Y_SHIFT=0.3
    model = FITEModel(projection_list=opt['projection_list'],
                      input_nc=3,
                      hsize=opt['hsize'],
                      nf=opt['nf'],
                      c_geom=opt['c_geom'],
                      c_pose=opt['c_pose'],
                      up_mode=opt['up_mode'],
                      use_dropout=opt['use_dropout']
                      ).cuda()
    
    # model.load_state_dict(torch.load('checkpoints/checkpoint-1000.pt'))
    if opt['load_epoch'] is None:
        model.load_state_dict(torch.load(join(CHECKPOINTS_PATH, f'checkpoint-latest.pt')))
        cano_data_repo.geom_feats = torch.load(join(CHECKPOINTS_PATH, f'geom-feats-latest.pt'))
    else:
        model.load_state_dict(torch.load(join(CHECKPOINTS_PATH, f'checkpoint-{opt["load_epoch"]:03d}.pt')))
        cano_data_repo.geom_feats = torch.load(join(CHECKPOINTS_PATH, f'geom-feats-{opt["load_epoch"]:03d}.pt'))
    model.eval()
    cano_data_repo.geom_feats.requires_grad_(False)
    

    test_s2m, test_m2s, test_lnormal, test_rgl, test_latent_rgl = 0, 0, 0, 0, 0


    # def tqdm(x):
    #     return x
    with torch.no_grad():
        n_tested_samples = 0
        test_bar = tqdm(enumerate(test_loader))
        for i, batch in test_bar:

            # -------------------------------------------------------
            # ------------ load batch data and reshaping ------------
            
            if opt['eval_use_gt']:
                points_gt = batch['points'].cuda()
                normals_gt = batch['normals'].cuda()
            pose = batch['pose'].cuda()
            transl = batch['transl'].cuda() # NOTE already removed from points, needed only for debug
            subject_id = batch['subject_id'].cuda()

            
            posmaps_batch = {}
            # posmap_weights_batch = {}
            projected_pts_batch = {}
            # ic(projected_pts_batch)
            for proj_id in range(len(model.projection_list)):
                proj_direction = model.projection_list[proj_id]['dirc']
                posmaps_batch[proj_direction] = batch[f'posmap_{proj_direction}'].cuda()
                # posmap_weights_batch[proj_direction] = cano_data_repo.posmap_weights[proj_direction][subject_id]
                projected_pts_batch[proj_direction] = cano_data_repo.projected_points[proj_direction][subject_id]
                # ic(cano_data_repo.projected_points.shape, projected_pts_batch[proj_direction].shape)

            ### NOTE get transformations
            # verts_cano_batch = cano_data_repo.verts_downsampled[subject_id]
            geom_feats_batch = cano_data_repo.geom_feats[subject_id]
            basepoints_batch = cano_data_repo.verts_downsampled[subject_id]
            normals_cano_batch = cano_data_repo.normals_downsampled[subject_id]
            weights_cano_batch = cano_data_repo.weights_downsampled[subject_id]
            cano_smpl_param_batch = cano_data_repo.cano_pose_param[subject_id]
            joints_tpose_batch = cano_data_repo.joints_tpose[subject_id]

            if opt['predeform']:
                predeform_offsets = model.predeformer(geom_feats_batch).permute(0,2,1) * opt['predeform_scaling']  # [b, n_pts, 3]
                out_unposed = inv_lbs(basepoints_batch + predeform_offsets, joints_tpose_batch, cano_smpl_param_batch, cano_data_repo.smpl_parents, weights_cano_batch, return_tfs=True)
            else:
                out_unposed = inv_lbs(basepoints_batch, joints_tpose_batch, cano_smpl_param_batch, cano_data_repo.smpl_parents, weights_cano_batch, return_tfs=True)
            out = lbs(out_unposed['v_unposed'], joints_tpose_batch, pose, cano_data_repo.smpl_parents, lbs_weights=weights_cano_batch, return_tfs=True)
            posed_verts = out['v_posed']

            unposing_tfs = out_unposed['v_tfs_inv'] # [1, 85722, 4, 4]
            posing_tfs = out['v_tfs']               # same as above

            cano_normals_transformed = torch.einsum('bvrc,bvc->bvr', unposing_tfs[:, :, :3, :3], normals_cano_batch)
            cano_normals_transformed = torch.einsum('bvrc,bvc->bvr', posing_tfs[:, :, :3, :3], cano_normals_transformed)
            cano_normals_transformed = F.normalize(cano_normals_transformed, dim=-1)

            residuals, normals = model.forward(geom_feats_batch, projected_pts_batch, basepoints_batch, posmaps_batch)

            ### NOTE transform outputs (rotation only)
            residuals = residuals.permute(0, 2, 1)  # [bs, n_pts, 3]
            normals = normals.permute(0, 2, 1)      # [bs, n_pts, 3]
            
            residuals = torch.einsum('bvrc,bvc->bvr', unposing_tfs[:, :, :3, :3], residuals)    
            residuals = torch.einsum('bvrc,bvc->bvr', posing_tfs[:, :, :3, :3], residuals)
            residuals = residuals * opt['residual_scaling']

            normals = torch.einsum('bvrc,bvc->bvr', unposing_tfs[:, :, :3, :3], normals)
            normals = torch.einsum('bvrc,bvc->bvr', posing_tfs[:, :, :3, :3], normals)
            normals = F.normalize(normals, dim=-1)

            offset_verts = posed_verts + residuals

            offset_verts = offset_verts.contiguous()
            
            if opt['eval_use_gt']:
                points_gt = points_gt.contiguous()
                normals = normals.contiguous()
                normals_gt = normals_gt.contiguous()


                # --------------------------------
                # ------------ losses ------------
                bs = points_gt.shape[0]

                m2s_real, s2m, idx_closest_gt, _ = chamfer_loss_separate(offset_verts, points_gt) #idx1: [#pred points]
                s2m = s2m.mean(1)
                lnormal, closest_target_normals = normal_loss(normals, normals_gt, idx_closest_gt, phase='test')
                nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
                target_points_chosen = torch.gather(points_gt, dim=1, index=nearest_idx)
                pc_diff = target_points_chosen - offset_verts # vectors from prediction to its closest point in gt pcl
                m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
                m2s = torch.mean(m2s**2, 1) # the length (squared) is the approx. pred point to scan surface dist.

                # m2s = m2s_real.mean(1)

                rgl_len = torch.mean((residuals ** 2).reshape(bs, -1), 1)
                rgl_latent = torch.mean(geom_feats_batch**2)
                # rgl_latent = torch.zeros(1).cuda()

                # ------------------------------------------
                # ------------ accumulate stats ------------

                test_m2s += torch.sum(m2s)
                test_s2m += torch.sum(s2m)
                test_lnormal += torch.sum(lnormal)
                test_rgl += torch.sum(rgl_len)
                test_latent_rgl += rgl_latent
            

            save_spacing = 1

            for j in range(offset_verts.shape[0])[::save_spacing]:
                ### NOTE save pred
                if not os.path.exists(join(TESTPCD_PATH, batch['subject_name'][j])):
                    os.makedirs(join(TESTPCD_PATH, batch['subject_name'][j]))
                save_result_examples(join(TESTPCD_PATH, batch['subject_name'][j]), opt["expname"], batch['basename'][j],
                                    points=offset_verts[j]+transl[j][None], normals=normals[j])
                save_result_examples(join(TESTPCD_PATH, batch['subject_name'][j]), opt["expname"], batch['basename'][j] + '_base',
                                    points=posed_verts[j]+transl[j][None], normals=cano_normals_transformed[j])
                                    
                ### NOTE save_gt
                if opt['save_cano'] and opt['args.predeform']:
                    save_result_examples(TESTPCD_PATH, opt["expname"], batch['basename'][j] + '_cano',
                            points=basepoints_batch[j], normals=normals[j])
                    save_result_examples(TESTPCD_PATH, opt["expname"], batch['basename'][j] + '_cano_predef',
                            points=basepoints_batch[j]+predeform_offsets[j], normals=normals[j])
            n_tested_samples += offset_verts.shape[0]
            test_bar.set_description(f'[LOG] tested {n_tested_samples}/{n_test_samples} samples')

    if opt['eval_use_gt']:

        test_m2s /= n_test_samples
        test_s2m /= n_test_samples
        test_lnormal /= n_test_samples
        test_rgl /= n_test_samples
        test_latent_rgl /= n_test_samples

        test_s2m, test_m2s, test_lnormal, test_rgl, test_latent_rgl = list(map(lambda x: x.detach().cpu().numpy(), [test_s2m, test_m2s, test_lnormal, test_rgl, test_latent_rgl]))

        print("model2scan dist: {:.3e}, scan2model dist: {:.3e}, normal loss: {:.3e}"
            " rgl term: {:.3e}, latent rgl term:{:.3e},".format(test_m2s.item(), test_s2m.item(), test_lnormal.item(),
                                                                test_rgl.item(), test_latent_rgl.item()))
