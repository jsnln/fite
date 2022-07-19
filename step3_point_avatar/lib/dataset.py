from os.path import join, basename
import glob
from tqdm import tqdm
import math
import numpy as np
import smplx
import torch
from torch.utils.data import Dataset
import glm

from smplx.lbs import vertices2joints

class FITEPosmapDataset(Dataset):
    @torch.no_grad()
    def __init__(self,
                 subject_list,    # dict (key: subject name; val: subject gender and maybe other attributes)
                 projection_list,
                 split,
                 root_rendered,
                 root_packed,
                 data_spacing,
                 remove_hand_foot_pose,
                 data_shift=0,
                 selected_subjects=None,
                 ):
        super().__init__()

        self.subject_list = subject_list
        self.projection_list = projection_list
        self.split = split
        self.data_root_rendered = root_rendered
        self.data_root_packed = root_packed
        self.data_spacing = data_spacing
        self.data_shift = data_shift

        # dict with subject names as keys
        print(f'[LOG] Found data for these subjects:')
        self.data_len = 0

        self.basenames = []
        self.subject_ids = []
        for subject_id in range(len(self.subject_list)):
            subject_name = self.subject_list[subject_id]['name']
            if selected_subjects is not None and subject_name not in selected_subjects:
                continue
            # ic(join(self.data_root_rendered, subject_name, split, '*_posmap.npz'))
            subject_basename_list = sorted(list(glob.glob(join(self.data_root_rendered, subject_name, split, '*_posmap.npz'))))[self.data_shift:][::self.data_spacing]
            # ic(subject_basename_list)
            self.data_len += len(subject_basename_list)

            self.basenames = self.basenames + [basename(name)[:-11] for name in subject_basename_list]
            self.subject_ids = self.subject_ids + [subject_id] * len(subject_basename_list)

            print(f'[LOG]     {subject_name.ljust(25)}: {len(subject_basename_list)}')

        assert self.data_len == len(self.basenames)
        assert self.data_len == len(self.subject_ids)
        print(f'[LOG] Found {self.data_len} data in total. Loading...')
        
        ### NOTE load posmaps
        print(f'[LOG] Loading posmaps...')
        self.posmaps = {}
        for proj_id in range(len(self.projection_list)):
            proj_direction = self.projection_list[proj_id]['dirc']
            self.posmaps[proj_direction] = []
        for i in tqdm(range(len(self.basenames))):
            fn_base = self.basenames[i]
            subject_id = self.subject_ids[i]
            subject_name = self.subject_list[subject_id]['name']
            posmap_file = np.load(join(self.data_root_rendered, subject_name, split, fn_base + '_posmap.npz'))
            for proj_id in range(len(self.projection_list)):
                proj_direction = self.projection_list[proj_id]['dirc']
                posmap = torch.from_numpy(posmap_file[proj_direction]).float()[..., :3].permute(2,0,1)
                self.posmaps[proj_direction].append(posmap)

        ### NOTE load packed points
        print(f'[LOG] Loading packed point clouds and smpl data...')
        self.point_list = []
        self.normal_list = []
        self.pose_list = []
        self.transl_list = []
        for i in tqdm(range(len(self.basenames))):
            fn_base = self.basenames[i]
            subject_id = self.subject_ids[i]
            subject_name = self.subject_list[subject_id]['name']
            packed_pcd_file = np.load(join(self.data_root_packed, subject_name, split, fn_base + '.npz'))
            if self.split == 'train':
                self.point_list.append(torch.from_numpy(packed_pcd_file['scan_pc']).float())
                self.normal_list.append(torch.from_numpy(packed_pcd_file['scan_n']).float())
            self.pose_list.append(torch.from_numpy(packed_pcd_file['pose']).float())
            self.transl_list.append(torch.from_numpy(packed_pcd_file['transl']).float())

            if remove_hand_foot_pose:
                self.pose_list[-1][30:36] = 0 # feet
                self.pose_list[-1][66:72] = 0 # fingers
                # self.pose_list[-1][21:27] = 0 # ankles
                # self.pose_list[-1][60:66] = 0 # wrists
            
    def __getitem__(self, index):
        if self.split == 'train':
            ret = {
                'points': self.point_list[index] - self.transl_list[index][None],
                'normals': self.normal_list[index],
                'pose': self.pose_list[index],
                'transl': self.transl_list[index], # note that this is already removed from points
            }
        else:
            ret = {
                'pose': self.pose_list[index],
                'transl': self.transl_list[index],
            }
        for proj_id in range(len(self.projection_list)):
            proj_direction = self.projection_list[proj_id]['dirc']
            ret[f'posmap_{proj_direction}'] = self.posmaps[proj_direction][index]
        ret['basename'] = self.basenames[index]
        ret['subject_id'] = self.subject_ids[index]
        ret['subject_name'] = self.subject_list[self.subject_ids[index]]['name']

        return ret

    def __len__(self):
        return self.data_len



class FITECanoDataRepository:
    def __init__(self,
                 subject_list,
                 projection_list,
                 root_cano_data,
                 channels_geom_feat,
                 n_points_cano_data,
                 cano_pose_leg_angle,
                 smpl_model_path):
    
        self.subject_list = subject_list
        self.projection_list = projection_list
        self.root_cano_data = root_cano_data
        self.channels_geom_feat = channels_geom_feat
        self.cano_pose_leg_angle = cano_pose_leg_angle

        n_subjects = len(self.subject_list)
        self.geom_feats = torch.ones(n_subjects, channels_geom_feat, n_points_cano_data).normal_(mean=0., std=0.01)

        self.cano_data_list = {
            'verts_mesh': [],
            'faces_mesh': [],
            'weights_mesh': [],
            'verts_downsampled': [],
            'normals_downsampled': [],
            'weights_downsampled': [],
            'verts_tpose': [],
        }

        ### NOTE load cano data
        print(f'[LOG] Loading canonical data...')
        for subject_id in range(len(self.subject_list)):
            subject_name = self.subject_list[subject_id]['name']
            cano_data = np.load(join(root_cano_data, f'{subject_name}_clothed_template.npz'))
            for key in cano_data.files:
                if 'verts' in key or 'normals' in key or 'weights' in key:
                    data_tmp = torch.from_numpy(cano_data[key]).float()
                else:
                    data_tmp = torch.from_numpy(cano_data[key]).int()
                self.cano_data_list[key].append(data_tmp)

        for key in self.cano_data_list.keys():
            if 'downsampled' in key or 'tpose' in key:
                setattr(self, key, torch.stack(self.cano_data_list[key], dim=0))

        ### NOTE compute unposed joints
        self.smpl_model_male = smplx.create(model_path=smpl_model_path, model_type='smpl', gender='male')
        self.smpl_model_female = smplx.create(model_path=smpl_model_path, model_type='smpl', gender='female')
        self.smpl_model_neutral = smplx.create(model_path=smpl_model_path, model_type='smpl', gender='neutral')
        
        joints_tpose = []
        for subject_id in range(len(self.subject_list)):
            subject_name = self.subject_list[subject_id]['name']
            subject_gender = self.subject_list[subject_id]['gender']
            if subject_gender == 'male':
                joints_tpose.append(vertices2joints(self.smpl_model_male.J_regressor, self.verts_tpose[[subject_id]]))
            elif subject_gender == 'female':
                joints_tpose.append(vertices2joints(self.smpl_model_female.J_regressor, self.verts_tpose[[subject_id]]))
            elif subject_gender == 'neutral':
                joints_tpose.append(vertices2joints(self.smpl_model_neutral.J_regressor, self.verts_tpose[[subject_id]]))
            else:
                print(f'[ERROR] Unknown gender type: {subject_gender}')
        joints_tpose = torch.cat(joints_tpose, 0)
        setattr(self, 'joints_tpose', joints_tpose)

        ### NOTE precompute projected points
        self.projected_points = {}
        tensor_ones_tmp = torch.ones(self.verts_downsampled.shape[0], self.verts_downsampled.shape[1], 1)
        for proj_id in range(len(self.projection_list)):
            proj_config = self.projection_list[proj_id]
            proj_mat_glm = self.get_proj_mat(proj_config['y_rot'], proj_config['x_rot'], proj_config['y_shift'], proj_config['x_stretch'])
            proj_mat = torch.from_numpy(np.array(proj_mat_glm.to_list()).astype(np.float32))    # [4, 4], for this view only
            ### NOTE/IMPORTANT projection matric from glm needs to be transposed 
            proj_mat = proj_mat.t()
            
            points_homo = torch.cat([self.verts_downsampled, tensor_ones_tmp], dim=-1)
            self.projected_points[proj_config['dirc']] = torch.einsum('rc,bnc->bnr', proj_mat, points_homo)[..., :2] # take the projected coords only
            ### NOTE/IMPORTANT projected y coords need to be invertible for pytorch interpolation convention 
            self.projected_points[proj_config['dirc']][..., 1] *= -1

        ### NOTE cano pose param
        self.cano_pose_param = torch.zeros(1, 72)
        self.cano_pose_param[:, 5] = self.cano_pose_leg_angle/180*math.pi
        self.cano_pose_param[:, 8] = -self.cano_pose_leg_angle/180*math.pi
        self.cano_pose_param = self.cano_pose_param.expand(len(self.subject_list), -1)

        self.smpl_parents = self.smpl_model_male.parents.clone()

    def get_proj_mat(self, y_rotate_deg, x_rotate_deg, y_shift, x_stretch=None):
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
    
    def set_device(self, device):
        for key in self.__dict__.keys():
            if isinstance(getattr(self, key), torch.Tensor):
                setattr(self, key, getattr(self, key).to(device))
            if isinstance(getattr(self, key), dict):
                for key_of_attr in getattr(self, key).keys():
                    if isinstance(getattr(self, key)[key_of_attr], torch.Tensor):
                        getattr(self, key)[key_of_attr] = getattr(self, key)[key_of_attr].to(device)
        return self
