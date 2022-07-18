import os
from os.path import join
import math
import numpy as np
import trimesh
import torch
import yaml
from glob import glob

import array
import tqdm

import argparse
import smplx
from fiteutils.lbs_and_render import lbs, inv_lbs, vertices2joints
import torch.nn.functional as F

from .parser import parse_args


if __name__ == '__main__':

    ### NOTE useful options
    opt = {}
    with open(join('configs', 'common.yaml'), 'r') as common_opt_f:
        common_opt = yaml.safe_load(common_opt_f)
        opt.update(common_opt)
    with open(join('configs', f'step1.yaml'), 'r') as step_opt_f:
        step_opt = yaml.safe_load(step_opt_f)
        opt.update(step_opt)

    data_templates_path = opt['data_templates_path']
    subject = opt['subject']
    smpl_model_path = opt['smpl_model_path']
    num_joints = opt['num_joints']
    leg_angle = opt['leg_angle']
    point_interpolant_exe = opt['point_interpolant_exe']
    skinning_grid_depth = opt['skinning_grid_depth']
    lbs_surf_grad_exe = opt['lbs_surf_grad_exe']
    ask_before_os_system = bool(opt['ask_before_os_system'])

    tmp_folder_constraints = opt['tmp_folder_constraints']
    tmp_folder_skinning_grid = opt['tmp_folder_skinning_grid']

    if not os.path.exists(tmp_folder_constraints):
        os.makedirs(tmp_folder_constraints)
    if not os.path.exists(tmp_folder_skinning_grid):
        os.makedirs(tmp_folder_skinning_grid)



    # ### NOTE get a canonical-pose SMPL template
    smpl_tpose_mesh_path = join(data_templates_path, subject, f'{subject}_minimal_tpose.ply')
    with open(join(data_templates_path, 'gender_list.yaml') ,'r') as f:
        gender = yaml.safe_load(f)[subject]

    cpose_param = torch.zeros(1, 72)
    cpose_param[:, 5] =  leg_angle / 180 * math.pi
    cpose_param[:, 8] = -leg_angle / 180 * math.pi

    tpose_mesh = trimesh.load(smpl_tpose_mesh_path, process=False)
    smpl_model = smplx.create(smpl_model_path, model_type='smpl', gender=gender)

    tpose_verts = torch.from_numpy(tpose_mesh.vertices).float()[None]
    tpose_joints = vertices2joints(smpl_model.J_regressor, tpose_verts)

    out = lbs(tpose_verts, tpose_joints, cpose_param, smpl_model.parents, smpl_model.lbs_weights[None], return_tfs=False)
    cpose_verts = out['v_posed'][0].cpu().numpy()

    # np.savetxt('cano_data_grad_constraints.xyz', out['v_posed'][0], fmt="%.8f")
    cpose_mesh = trimesh.Trimesh(cpose_verts, smpl_model.faces, process=False)
    cpose_mesh.export(join(data_templates_path, subject, f'{subject}_minimal_cpose.obj'))
    cpose_mesh.export(join(data_templates_path, subject, f'{subject}_minimal_cpose.ply'))
    np.savetxt(join(data_templates_path, subject, f'{subject}_lbs_weights.txt'), smpl_model.lbs_weights.numpy(), fmt="%.8f")

    ### NOTE compute the along-surface gradients of skinning
    cmd = f'{lbs_surf_grad_exe} ' + \
          f'{join(data_templates_path, subject, subject + "_minimal_cpose.obj")} ' + \
          f'{join(data_templates_path, subject, subject + "_lbs_weights.txt")} ' + \
          f'{join(data_templates_path, subject, subject + "_cpose_lbs_grads.txt")} '
    
    if ask_before_os_system:
        go_on = input(f'\n[WILL EXECUTE with os.system] {cmd}\nContinue? (y/n)')
    else:
        go_on = 'y'
    if go_on == 'y':
        os.system(cmd)

    ### NOTE reorganize data
    data = np.loadtxt(join(data_templates_path, subject, subject + "_cpose_lbs_grads.txt"))

    position = data[:, 0:3]
    normals = data[:, 3:6]
    tx = data[:, 6:9]
    ty = data[:, 9:12]
    lbs_w = data[:, 12:36]
    lbs_tx = data[:, 36:60]
    lbs_ty = data[:, 60:84]

    if not os.path.exists(tmp_folder_constraints):
        os.mkdir(tmp_folder_constraints)

    for jid in tqdm.tqdm(range(num_joints)):
        out_fn_grad = os.path.join(tmp_folder_constraints, f"cano_data_lbs_grad_{jid:02d}.xyz")
        out_fn_val = os.path.join(tmp_folder_constraints, f"cano_data_lbs_val_{jid:02d}.xyz")

        grad_field = lbs_tx[:, jid:jid+1] * tx + lbs_ty[:, jid:jid+1] * ty

        out_data_grad = np.concatenate([position, grad_field], 1)
        out_data_val = np.concatenate([position, lbs_w[:, jid:jid+1]], 1)
        np.savetxt(out_fn_grad, out_data_grad, fmt="%.8f")
        np.savetxt(out_fn_val, out_data_val, fmt="%.8f")


    ### NOTE solve for the diffused skinning fields
    for jid in range(num_joints):
        cmd = f'{point_interpolant_exe} ' + \
            f'--inValues {join(tmp_folder_constraints, f"cano_data_lbs_val_{jid:02d}.xyz")} ' + \
            f'--inGradients {join(tmp_folder_constraints, f"cano_data_lbs_grad_{jid:02d}.xyz")} ' + \
            f'--gradientWeight 0.05 --dim 3 --verbose ' + \
            f'--grid {join(tmp_folder_skinning_grid, f"grid_{jid:02d}.grd")} ' + \
            f'--depth {skinning_grid_depth} '
        
        if ask_before_os_system:
            go_on = input(f'\n[WILL EXECUTE with os.system] {cmd}\nContinue? (y/n)')
        else:
            go_on = 'y'
        if go_on == 'y':
            os.system(cmd)

    ### NOTE concatenate all grids
    fn_list = sorted(list(glob(join(tmp_folder_skinning_grid, 'grid_*.grd'))))
    print(fn_list)

    grids = []
    for fn in fn_list:
        with open(fn, 'rb') as f:
            bytes = f.read()
        grid_res = 2 ** skinning_grid_depth
        grid_header_len = len(bytes) - grid_res ** 3 * 8
        grid_np = np.array(array.array('d', bytes[grid_header_len:])).reshape(256, 256, 256)
        grids.append(grid_np)


    grids_all = np.stack(grids, 0)
    grids_all = np.clip(grids_all, 0.0, 1.0)
    grids_all = grids_all / grids_all.sum(0)[None]
    np.save(join(data_templates_path, subject, subject + '_cano_lbs_weights_grid_float32.npy'), grids_all.astype(np.float32))


    ### NOTE query the smpl surface skinning weight using this grid
    # def inv_transform_v(v, scale_data, scale_grid, grid_res, transl):
    #     v = v - transl[None]
    #     v = v / scale_grid
    #     v = v * 2
    #     return v

    # def transform_v(v, scale_data, scale_grid, grid_res, transl):
    #     v = v / (grid_res - 1)
    #     v = v - 0.5
    #     v = v * scale_grid
    #     v = v + transl[None]
    #     v[:, ::2] = np.flip(v[:, ::2], -1)
    #     return v

    # cano_verts = cpose_verts[:, :3]
    # bbox_data_min = cano_verts.min(0)
    # bbox_data_max = cano_verts.max(0)

    # bbox_data_extend = (bbox_data_max - bbox_data_min).max()
    # bbox_grid_extend = bbox_data_extend * 1.1

    # center = (bbox_data_min + bbox_data_max) / 2

    # ### NOTE querying
    # # query_verts = cano_verts
    # clothed_mesh = trimesh.load('cano_data_smplmesh.obj', process=False)  # for smpl
    # query_verts = clothed_mesh.vertices

    # v_cano_in_grid_coords = inv_transform_v(query_verts, bbox_data_extend, bbox_grid_extend, 256, center)
    # v_cano_in_grid_coords_pt = torch.from_numpy(v_cano_in_grid_coords).float()
    # grid_pt = torch.from_numpy(grids_all).float()

    # out = F.grid_sample(grid_pt[None], v_cano_in_grid_coords_pt[None, None, None], align_corners=False)[0, :, 0, 0].T  # [24, 6890]
    # np.savez('clothed_diffused_skinning.npz', verts=query_verts, faces=clothed_mesh.faces, weights=out.numpy())
