import os
import glob
import torch
import numpy as np

import yaml
import trimesh
from os.path import join

from pytorch3d.ops import sample_farthest_points

from .lib.snarf_model_diffused_skinning import SNARFModelDiffusedSkinning
from .lib.model.helpers import rectify_pose

if __name__ == '__main__':
    opt = {}
    with open(join('configs', 'common.yaml'), 'r') as common_opt_f:
        common_opt = yaml.safe_load(common_opt_f)
        opt.update(common_opt)
    with open(join('configs', f'step2.yaml'), 'r') as step_opt_f:
        step_opt = yaml.safe_load(step_opt_f)
        opt.update(step_opt)

    exp_folder = join(opt['result_folder'], opt['expname'], 'step2-results')
    checkpoint_folder = join(opt['result_folder'], opt['expname'], 'step2-checkpoints')
    checkpoint_path = join(checkpoint_folder, 'checkpoint-latest.pt')

    # set subject info
    subject = opt['datamodule']['subject']
    minimal_body_path = os.path.join(opt['data_templates_path'], subject, f'{subject}_minimal_tpose.ply')
    v_template = np.array(trimesh.load(minimal_body_path, process=False).vertices)
    with open(join(opt['data_templates_path'], 'gender_list.yaml') ,'r') as f:
        gender = yaml.safe_load(f)[subject]
    meta_info = {'v_template': v_template.copy(), 'gender': gender}
    # meta_info = np.load('meta_info.npz')

    model = SNARFModelDiffusedSkinning(opt['model']['soft_blend'],
                                       opt['smpl_model_path'],
                                       opt['model']['pose_conditioning'],
                                       opt['model']['network'],
                                       subject=opt['datamodule']['subject'],
                                       cpose_smpl_mesh_path=join(opt['data_templates_path'], opt['datamodule']['subject'], opt['datamodule']['subject'] + '_minimal_cpose.ply'),
                                       cpose_weight_grid_path=join(opt['data_templates_path'], opt['datamodule']['subject'], opt['datamodule']['subject'] + '_cano_lbs_weights_grid_float32.npy'),
                                       meta_info=meta_info,
                                       data_processor=None).to('cuda')

    model.load_state_dict(torch.load(checkpoint_path))
    model.deformer.init_bones = np.arange(24)

    # pose format conversion
    smplx_to_smpl = list(range(66)) + [72, 73, 74, 117, 118, 119]  # SMPLH to SMPL

    # load motion sequence
    motion_path = join(opt['data_scans_path'], opt['datamodule']['subject'], 'train')
    if os.path.isdir(motion_path):
        motion_files = sorted(glob.glob(os.path.join(motion_path, '*.npz')))
        smpl_params_all = []
        for f in motion_files:
            f = np.load(f)
            smpl_params = np.zeros(86)
            smpl_params[0], smpl_params[4:76] = 1, f['pose']
            smpl_params = torch.tensor(smpl_params).float().cuda()
            smpl_params_all.append(smpl_params)
        smpl_params_all = torch.stack(smpl_params_all)

    elif '.npz' in motion_path:
        f = np.load(motion_path)
        smpl_params_all = np.zeros( (f['poses'].shape[0], 86) )
        smpl_params_all[:,0] = 1
        if f['poses'].shape[-1] == 72:
            smpl_params_all[:, 4:76] = f['poses']
        elif f['poses'].shape[-1] == 156:
            smpl_params_all[:, 4:76] = f['poses'][:,smplx_to_smpl]

        root_abs = smpl_params_all[0, 4:7].copy()
        for i in range(smpl_params_all.shape[0]):
            smpl_params_all[i, 4:7] = rectify_pose(smpl_params_all[i, 4:7], root_abs)

        smpl_params_all = torch.tensor(smpl_params_all).float().cuda()

    ### NOTE choose only min l1-norm pose
    l1_norm_pose = smpl_params_all.abs().sum(-1)    # [n_poses,]
    min_pose_id = l1_norm_pose.argmin()
    min_pose_id = 0    
    smpl_params_all = smpl_params_all[min_pose_id][None]

    # generate data batch
    smpl_params = smpl_params_all
    data = model.smpl_server.forward(smpl_params, absolute=True)
    data['smpl_thetas'] = smpl_params[:, 4:76]

    print(f'[LOG] extracting implicit templates')
    # low resolution mesh
    results_lowres = model.plot(data, res=opt['extraction']['resolution_low'])
    results_lowres['mesh_cano'].export(join(opt['data_templates_path'], subject, f'{subject}_cano_mesh_{opt["extraction"]["resolution_low"]}.ply'))
    print(f'[LOG] extracted mesh at resolution {opt["extraction"]["resolution_low"]}. saving it at ' + join(opt['data_templates_path'], subject, f'{subject}_cano_mesh_{opt["extraction"]["resolution_low"]}.ply'))

    # high resolution mesh
    results_highres = model.plot(data, res=opt['extraction']['resolution_high'])
    results_highres['mesh_cano'].export(join(opt['data_templates_path'], subject, f'{subject}_cano_mesh_{opt["extraction"]["resolution_high"]}.ply'))
    print(f'[LOG] extracted mesh at resolution {opt["extraction"]["resolution_high"]}. saving it at ' + join(opt['data_templates_path'], subject, f'{subject}_cano_mesh_{opt["extraction"]["resolution_high"]}.ply'))


    print(f'[LOG] downsampling the high resolution mesh and packing the clothed template')
    verts_mesh = results_lowres['mesh_cano'].vertices
    faces_mesh = results_lowres['mesh_cano'].faces
    weights_mesh = results_lowres['weights_cano'].cpu().numpy()

    points = torch.from_numpy(results_highres['mesh_cano'].vertices).float()[None].cuda()
    downsampled_points, downsampled_indices = sample_farthest_points(points, K=opt['n_cano_points'])
    
    verts_downsampled = downsampled_points[0].cpu().numpy()
    normals_downsampled = results_highres['mesh_cano'].vertex_normals[downsampled_indices[0].cpu().numpy()]
    weights_downsampled = results_highres['weights_cano'].cpu().numpy()[downsampled_indices[0].cpu().numpy()]

    verts_tpose = v_template

    np.savez(join(opt['data_templates_path'], f'{subject}_clothed_template.npz'),
            verts_mesh=verts_mesh,
            faces_mesh=faces_mesh,
            weights_mesh=weights_mesh,
            verts_downsampled=verts_downsampled,
            normals_downsampled=normals_downsampled,
            weights_downsampled=weights_downsampled,
            verts_tpose=verts_tpose)

    print(f'[LOG] clothed template packed at ' + join(opt['data_templates_path'], f'{subject}_clothed_template.npz'))
