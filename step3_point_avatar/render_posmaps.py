import glob
import math
import os
from os.path import join, basename, splitext
import torch
import torch.nn
import numpy as np
import trimesh
from tqdm import tqdm
import glob
import yaml

import smplx

import math
import glm

import numpy as np
from OpenGL.GL import *
import glfw

from smplx.lbs import vertices2joints
from .lib.lbs import lbs, inv_lbs
from .lib.shader_utils import createProgram, loadShader

### NOTE for debugging, get the posed point cloud
def posmap2pcd(img):
    color = img[..., :3]
    mask = img[..., 3:4].astype(bool)

    mask = np.concatenate([mask, mask, mask], -1)

    pcd = color[mask].reshape(-1, 3)
    return pcd

def reindex_verts(p_verts, p_faces):
    reindexed_verts = []
    for fid in range(len(p_faces)):
        vid = p_faces[fid]
        reindexed_verts.append(p_verts[vid].reshape(-1))
    return np.concatenate(reindexed_verts, 0)

def shift(p_verts, y_shift):
    p_verts = p_verts.reshape(-1, 3)
    p_verts = p_verts + np.array([[0, y_shift, 0]], dtype=np.float32)
    # p_verts[:, 2] *= 0.1
    p_verts = p_verts.reshape(-1)
    return p_verts

@torch.no_grad()
def load_data(fn, cano_data, parents, unposed_joints, cano_pose_param, remove_root_pose):
    smpl_params = np.load(fn)
    transl = smpl_params['transl']
    pose = smpl_params['pose']

    verts = cano_data['verts_mesh']
    weights = cano_data['weights_mesh']

    verts = torch.from_numpy(verts).float()[None].cuda()
    weights = torch.from_numpy(weights).float()[None].cuda()
    pose = torch.from_numpy(pose).float()[None].cuda()

    ### NOTE remove root pose
    if remove_root_pose:
        pose[:, :3] = 0
    
    out_unposed = inv_lbs(verts, unposed_joints, cano_pose_param, parents, weights)

    # NOTE use pose correctives or not (no use anyway)
    out = lbs(out_unposed['v_unposed'], unposed_joints, pose, parents, lbs_weights=weights)

    mesh = trimesh.Trimesh(out['v_posed'][0].cpu().numpy(), cano_data['faces_mesh'], process=False)
    mesh_cano = trimesh.Trimesh(cano_data['verts_mesh'], cano_data['faces_mesh'], process=False)

    # NOTE now preprocess verts data
    vertices_cano = reindex_verts(cano_data['verts_mesh'], cano_data['faces_mesh']).astype(np.float32)
    vertices_posed = reindex_verts(mesh.vertices, mesh.faces).astype(np.float32)
    feats = vertices_posed.copy()

    return vertices_cano, feats, smpl_params, mesh
        

def get_proj_mat(y_rotate_deg, x_rotate_deg, y_shift, x_stretch=None):
    """
    y_rotate_deg: rotation angle in degrees
    """
    model_y_rot = glm.rotate(y_rotate_deg / 180 * math.pi, [0.0, 1.0, 0.0])
    model_x_rot = glm.rotate(x_rotate_deg / 180 * math.pi, [1.0, 0.0, 0.0])
    ortho = glm.ortho(-1.0, 1.0, -1.0-y_shift, 1.0-y_shift)
    if x_stretch is not None:
        stretch_mat = glm.mat4(1)
        stretch_mat[0,0] = x_stretch
        return stretch_mat * ortho * model_x_rot * model_y_rot

    return ortho * model_x_rot * model_y_rot
    

def render_one(verts, feats, frame_buffer):
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)
    
    v = np.array(verts, dtype = np.float32)
    c = np.array(feats, dtype = np.float32)

    SIZE_OF_FLOAT = 4

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * SIZE_OF_FLOAT, v)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * SIZE_OF_FLOAT, c)
    glEnableVertexAttribArray(1)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDrawArrays(GL_TRIANGLES, 0, len(v) // 3)
    glfw.swap_buffers(window)
    
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    data = glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, outputType=None)
    rgb = data.reshape(height, width, -1)
    rgb = np.flip(rgb, 0)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return rgb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('subject', type=str)
parser.add_argument('split', type=str)
args = parser.parse_args()

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

    width = opt['posmap_size']
    height = opt['posmap_size']

    # NOTE init window and context
    glfw.init()
    window = glfw.create_window(width, height, "LBS and render", None, None)
    glfw.set_window_pos(window, 600, 300)
    glfw.make_context_current(window)

    # NOTE generate and bind alternative buffers
    frame_buffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)

    color_buffer = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_buffer)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_buffer, 0)
    
    # Configure depth texture map to render to
    depth_buffer = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_buffer)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_buffer, 0)

    attachments = []
    attachments.append(GL_COLOR_ATTACHMENT0)
    glDrawBuffers(1, attachments)
    # NOTE bind it here so you can read it later
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)
    
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glDepthRange(-1.0, 1.0)
    glDisable(GL_CULL_FACE)

    glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE)
    glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE)
    glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE)

    print(glCheckFramebufferStatus(GL_FRAMEBUFFER))

    # NOTE load shaders and set shaders (but not used yet)
    vertex_shader = loadShader(GL_VERTEX_SHADER, 'step3_point_avatar/posmap_shaders/v.glsl')
    fragment_shader = loadShader(GL_FRAGMENT_SHADER, 'step3_point_avatar/posmap_shaders/f.glsl')
    shader_program = createProgram([vertex_shader, fragment_shader])

    glUseProgram(shader_program)

    proj_list = opt['projection_list']

    ### NOTE loading shared cano data
    SUBJECT = args.subject
    SPLIT = args.split

    FN_CANO_DATA = join(opt['data_templates_path'], f'{SUBJECT}_clothed_template.npz')   # should have keys 'faces', 'verts', 'weights'
    # FN_ZERO_POSE_MINIMAL = join(opt['data_templates_path'], SUBJECT, f'{SUBJECT}_minimal_cpose.ply')
    
    FN_GENDER_LIST = join(opt['data_templates_path'], 'gender_list.yaml')
    with open(FN_GENDER_LIST, 'r') as f_gender_list:
        gender_list = yaml.safe_load(f_gender_list)
        GENDER = gender_list[SUBJECT]

    n_joints = opt['num_joints']

    smpl_model = smplx.create(opt['smpl_model_path'], model_type='smpl', gender=GENDER)
    smpl_parents = smpl_model.parents.clone().cuda()

    cano_data = np.load(FN_CANO_DATA)
    cano_pose_param = torch.zeros(1, 72).cuda()
    cano_pose_param[:, 5] =  opt['leg_angle'] / 180*math.pi
    cano_pose_param[:, 8] = -opt['leg_angle'] / 180*math.pi

    tpose_verts = torch.from_numpy(cano_data['verts_tpose']).float()[None]
    tpose_joints = vertices2joints(smpl_model.J_regressor, tpose_verts).cuda()

    if not os.path.exists(join(opt['data_posmaps_path'], SUBJECT, SPLIT)):
        os.makedirs(join(opt['data_posmaps_path'], SUBJECT, SPLIT), exist_ok=True)

    FN_FOLDER = join('data_scans', SUBJECT, SPLIT)   # packed snarf data
    fn_list = sorted(list(glob.glob(join(FN_FOLDER, '*.npz'))))
    fn_list = tqdm(fn_list)
    for fn in fn_list:
        ### NOTE load separate poses
        fn_list.set_description(fn)
        vertices_cano, feats, smpl_params, mesh = load_data(fn, cano_data,
                                                                smpl_parents,
                                                                tpose_joints,
                                                                cano_pose_param,
                                                                remove_root_pose=True)
        rgb_dict_this_pose = {}
        for proj_id in range(len(proj_list)):
            proj_direction = proj_list[proj_id]['dirc']
            # NOTE now starts passing data and render
            proj_params = proj_list[proj_id]

            proj_mat = get_proj_mat(proj_params['y_rot'], proj_params['x_rot'], proj_params['y_shift'], proj_params['x_stretch'])
            glUniformMatrix4fv(glGetUniformLocation(shader_program, 'projection'), 1, GL_FALSE, proj_mat.to_bytes())

            rgb = render_one(vertices_cano, feats, frame_buffer)
            pcd = posmap2pcd(rgb)
            rgb_dict_this_pose[proj_direction] = rgb.copy()
        
        npz_savename = join(opt['data_posmaps_path'], SUBJECT, SPLIT, splitext(basename(fn))[0] + f'_posmap')
        np.savez(npz_savename, **rgb_dict_this_pose)
        
