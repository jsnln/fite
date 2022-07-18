import matplotlib.pyplot as plt
import glob
import math
import os
from os.path import join, basename, splitext
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import trimesh
import tqdm

from icecream import ic

# from shader_utils import loadShader

def batch_rodrigues(
    rot_vecs,
    epsilon: float = 1e-8,
):
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def vertices2joints(J_regressor, vertices):
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def batch_rigid_transform(
    rot_mats,
    joints,
    parents,
    dtype=torch.float32
):
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

def lbs(v_shaped, j_shaped, pose, parents, lbs_weights, body_lbs_weights=None, posedirs=None, return_tfs=True):
    """
    weights: [bs, n_pts, 24]
    body_lbs_weights: [6890, 24], used for similarity weighting between cloth and smpl verts
    posedirs: will add pose correctives if not None, shape: [207, 20670]
    """
    batch_size = max(v_shaped.shape[0], pose.shape[0])
    device, dtype = v_shaped.device, v_shaped.dtype

    if body_lbs_weights is not None and posedirs is not None:
        ident = torch.eye(3, dtype=dtype, device=device)
        if True:    # if pose2rot:
            rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
                [batch_size, -1, 3, 3])

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets_smplverts = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)   # [bs, 6890, 3]
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)
        ic('using pose correctives')
        ### NOTE what I want is smpl_pose_offsets x smpl_weights x cloth_weights => cloth_offsets,
        #  i.e. [bs, 6890, 3] x [6890, 24] x [bs, n_pts, 24]
        joint_offsets = torch.einsum('bmc,mj->bcj', pose_offsets_smplverts, body_lbs_weights.sqrt())
        pose_offsets_clothverts = torch.einsum('bnj,bcj->bnc', lbs_weights.sqrt(), joint_offsets) * 0.05
        
        # smpl_cloth_weights = torch.einsum('bnj,mj->bnm', lbs_weights, body_lbs_weights)
        # # smpl_cloth_weights /= smpl_cloth_weights.sum(-1, keepdim=True)
        # # ic(smpl_cloth_weights.max(), smpl_cloth_weights.min(), smpl_cloth_weights[0, 3720])
        # smpl_cloth_weights = F.softmax(smpl_cloth_weights * 5, dim=-1)
        # pose_offsets_clothverts = torch.einsum('bnm,bmc->bnc', smpl_cloth_weights, pose_offsets_smplverts)
        
        v_posed = pose_offsets_clothverts + v_shaped
        # pass
    else:
        v_posed = v_shaped  # [N, 6890, 3]
    J = j_shaped  # [N, 24, 3]
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])   # [N, 24, 3, 3]
    # from icecream import ic
    # ic(rot_mats.shape, J.shape, parents.shape)
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    ### NOTE Do skinning:
    W = lbs_weights   # [N, 6890, 24]
    T = torch.einsum('bvj,bjrc->bvrc', W, A)

    homogen_coord = torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device) # [N, 6890, 1], all ones
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)   # [N, 6890, 4]
    v_homo = torch.einsum('bvrc,bvc->bvr', T, v_posed_homo)

    verts = v_homo[:, :, :3]# / v_homo[:, :, 3:]

    if return_tfs:
        return {'v_posed': verts, 'j_posed': J_transformed, 'v_tfs': T}
    else:
        return {'v_posed': verts, 'j_posed': J_transformed}


def inv_lbs(v_posed, j_unposed, pose, parents, lbs_weights, return_tfs=True):

    batch_size = max(v_posed.shape[0], pose.shape[0])
    device, dtype = v_posed.device, v_posed.dtype

    v_posed = v_posed  # [N, 6890, 3]
    J = j_unposed  # [N, 24, 3]
    # from icecream import ic
    # ic(pose.shape)
    # ic(batch_rodrigues(pose.reshape(-1, 3)))
    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape([batch_size, -1, 3, 3])   # [N, 24, 3, 3]
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    ### NOTE Do skinning:
    W = lbs_weights   # [N, 6890, 24]
    T = torch.einsum('bvj,bjrc->bvrc', W, A)
    T_inv = torch.linalg.inv(T)

    homogen_coord = torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device) # [N, 6890, 1], all ones
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)   # [N, 6890, 4]
    v_homo = torch.einsum('bvrc,bvc->bvr', T_inv, v_posed_homo)

    verts = v_homo[:, :, :3] # / v_homo[:, :, 3:]

    if return_tfs:
        return {'v_unposed': verts, 'j_posed': J_transformed, 'v_tfs_inv': T_inv}
    else:
        return {'v_unposed': verts, 'j_posed': J_transformed}


def save_skeleton(fn, joints, parents):
    """
    joints: [55, 3]
    parents: [55,]
    """
    assert joints.shape[0] == parents.shape[0]

    with open(fn, 'w') as f:
        for i in range(joints.shape[0]):
            f.write(f"{i} {joints[i][0]:.6f} {joints[i][1]:.6f} {joints[i][2]:.6f} {parents[i]}\n")
