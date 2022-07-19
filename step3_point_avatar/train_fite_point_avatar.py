import yaml
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
from os.path import join

from .lib.fite_model import FITEModel
from .lib.dataset import FITEPosmapDataset, FITECanoDataRepository
from .lib.losses import normal_loss, chamfer_loss_separate
from .lib.utils import save_result_examples, adjust_loss_weights
from .lib.lbs import lbs, inv_lbs

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
    with open(join('configs', f'projection_list.yaml'), 'r') as projection_list_f:
        projection_list = yaml.safe_load(projection_list_f)
        opt['projection_list'] = projection_list

    ### NOTE exp folders
    CHECKPOINTS_PATH = join(opt['result_folder'], opt['expname'], 'step3-checkpoints')
    TRAINPCD_PATH = join(opt['result_folder'], opt['expname'], 'step3-train-pcds')

    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)
    if not os.path.exists(TRAINPCD_PATH):
        os.makedirs(TRAINPCD_PATH)


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
            smpl_model_path=opt['smpl_model_path'],
    ).set_device('cuda')

    train_dataset = FITEPosmapDataset(
            subject_list=opt['subject_list'],
            projection_list=opt['projection_list'],
            split='train',
            root_rendered=opt['data_posmaps_path'],
            root_packed=opt['data_scans_path'],
            remove_hand_foot_pose=False,
            data_spacing=opt['data_spacing'],
            selected_subjects=opt['selected_subjects'],
    )

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, drop_last=True, num_workers=opt['num_workers'])

    ### NOTE model
    model = FITEModel(projection_list=opt['projection_list'],
                      input_nc=3,
                      hsize=opt['hsize'],
                      nf=opt['nf'],
                      c_geom=opt['c_geom'],
                      c_pose=opt['c_pose'],
                      up_mode=opt['up_mode'],
                      use_dropout=opt['use_dropout']
                      ).cuda()

    optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": opt['lr']},
            {"params": cano_data_repo.geom_feats, "lr": opt['lr_geomfeat']}
        ])

    total_iters = 0
    for epoch in range(opt['epochs']):
        train_bar_per_epoch = tqdm.tqdm(enumerate(train_loader))

        wdecay_rgl = adjust_loss_weights(opt['w_rgl'], epoch, mode='decay', start=opt['decay_start'], every=opt['decay_every'])
        if opt['train_normals_from_start']:
            wrise_normal = opt['w_normal']
        else:
            wrise_normal = adjust_loss_weights(opt['w_normal'], epoch,  mode='rise', start=opt['rise_start'], every=opt['rise_every'])
        loss_weights = torch.tensor([opt['w_s2m'], opt['w_m2s'], wrise_normal, wdecay_rgl, opt['w_latent_rgl']])

        for i, batch in train_bar_per_epoch:
            points_gt = batch['points'].cuda()
            normals_gt = batch['normals'].cuda()
            pose = batch['pose'].cuda()
            transl = batch['transl'].cuda() # NOTE already removed from points, needed only for debug
            subject_id = batch['subject_id'].cuda()

            
            posmaps_batch = {}
            projected_pts_batch = {}
            for proj_id in range(len(model.projection_list)):
                proj_direction = model.projection_list[proj_id]['dirc']
                posmaps_batch[proj_direction] = batch[f'posmap_{proj_direction}'].cuda()
                projected_pts_batch[proj_direction] = cano_data_repo.projected_points[proj_direction][subject_id]

            ### NOTE get transformations
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
            points_gt = points_gt.contiguous()
            normals = normals.contiguous()
            normals_gt = normals_gt.contiguous()

            ### NOTE original
            m2s, s2m, idx_closest_gt, _ = chamfer_loss_separate(offset_verts, points_gt) #idx1: [#pred points]
            s2m = torch.mean(s2m)

            # normal loss
            lnormal, closest_target_normals = normal_loss(normals, normals_gt, idx_closest_gt)
            
            # dist from the predicted points to their respective closest point on the GT, projected by
            # the normal of these GT points, to appxoimate the point-to-surface distance
            nearest_idx = idx_closest_gt.expand(3, -1, -1).permute([1,2,0]).long() # [batch, N] --> [batch, N, 3], repeat for the last dim
            target_points_chosen = torch.gather(points_gt, dim=1, index=nearest_idx)
            pc_diff = target_points_chosen - offset_verts # vectors from prediction to its closest point in gt pcl
            m2s = torch.sum(pc_diff * closest_target_normals, dim=-1) # project on direction of the normal of these gt points
            m2s = torch.mean(m2s**2) # the length (squared) is the approx. pred point to scan surface dist.

            rgl_len = torch.mean(residuals ** 2)
            rgl_latent = torch.mean(geom_feats_batch**2)
            if opt['predeform']:
                rgl_predef = torch.mean(predeform_offsets ** 2)

            w_s2m, w_m2s, w_normal, w_rgl, w_latent_rgl = loss_weights
            if opt['predeform']:
                w_predef = w_rgl.clone() / 5
                loss = s2m*w_s2m + m2s*w_m2s + lnormal* w_normal + rgl_len*w_rgl + rgl_latent*w_latent_rgl + rgl_predef*w_predef
            else:
                loss = s2m*w_s2m + m2s*w_m2s + lnormal* w_normal + rgl_len*w_rgl + rgl_latent*w_latent_rgl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if opt['predeform']:
                train_bar_per_epoch.set_description(f"[{total_iters}/{epoch}/{opt['epochs']}] m2s: {m2s:.4e}, s2m: {s2m:.4e}, normal: {lnormal:.4e}, rgl_len: {rgl_len:.4e}, rgl_predef: {rgl_predef:.4e}, rgl_latent: {rgl_latent:.4e}")
            else:
                train_bar_per_epoch.set_description(f"[{total_iters}/{epoch}/{opt['epochs']}] m2s: {m2s:.4e}, s2m: {s2m:.4e}, normal: {lnormal:.4e}, rgl_len: {rgl_len:.4e}, rgl_latent: {rgl_latent:.4e}")

            save_spacing = 1
            if total_iters % opt['save_pcd_every'] == 0:
                with torch.no_grad():
                    debug_pcd_posed_offset = torch.cat([offset_verts[0] + transl[0][None], normals[0]], 1).detach().cpu().numpy()
                    for j in range(offset_verts.shape[0])[::save_spacing]:
                        save_result_examples(TRAINPCD_PATH, f'{opt["expname"]}_epoch{epoch:05d}', batch['basename'][j],
                                    points=offset_verts[j]+transl[j][None], normals=normals[j])
                        if opt['save_cano'] and opt['predeform']:
                            save_result_examples(TRAINPCD_PATH, f'{opt["expname"]}_epoch{epoch:05d}', batch['basename'][j] + '_cano',
                                    points=basepoints_batch[j], normals=normals[j])
                            save_result_examples(TRAINPCD_PATH, f'{opt["expname"]}_epoch{epoch:05d}', batch['basename'][j] + '_cano_predef',
                                    points=basepoints_batch[j]+predeform_offsets[j], normals=normals[j])

            total_iters += 1

        if (epoch + 1) % opt['save_ckpt_every'] == 0:
            torch.save(model.state_dict(), join(CHECKPOINTS_PATH, f'checkpoint-{epoch+1:03d}.pt'))
            torch.save(cano_data_repo.geom_feats, join(CHECKPOINTS_PATH, f'geom-feats-{epoch+1:03d}.pt'))
            torch.save(model.state_dict(), join(CHECKPOINTS_PATH, f'checkpoint-latest.pt'))
            torch.save(cano_data_repo.geom_feats, join(CHECKPOINTS_PATH, f'geom-feats-latest.pt'))
