import os
from os.path import splitext
import torch

from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from os.path import join
import glob
import yaml
import trimesh

import kaolin
from ...lib.model.smpl import SMPLServer
from ...lib.model.sample import PointInSpace

from icecream import ic

class FITEDataSet(Dataset):

    def __init__(self, dataset_path, data_templ_path, opt, subject, clothing, split):

        ic(dataset_path)

        self.regstr_list = glob.glob(join(dataset_path, subject, split, clothing+'_**.npz'), recursive=True)
        ic(join(dataset_path, subject, clothing+'_**.npz'))
        ic(len(self.regstr_list))
        # ic(os.path.join(dataset_path, 'cape_release', 'sequences', '%05d'%subject, clothing))

        with open(join(data_templ_path, 'gender_list.yaml') ,'r') as f:
            self.gender = yaml.safe_load(f)[subject]

        minimal_body_path = os.path.join(data_templ_path, subject, f'{subject}_minimal_tpose.ply')
        self.v_template = np.array(trimesh.load(minimal_body_path, process=False).vertices)
        self.meta_info = {'v_template': self.v_template, 'gender': self.gender}

        self.max_verts = 40000
        self.points_per_frame = opt['processor']['points_per_frame']

        self.sampler = PointInSpace(**opt['processor']['sampler'])


    def __getitem__(self, index):

        data = {}

        while True:
            try:
                regstr = np.load(self.regstr_list[index])
                poses = regstr['pose']
                break
            except:
                index = np.random.randint(self.__len__())
                print('corrupted npz')

        verts = regstr['scan_v'] - regstr['transl'][None,:]
        verts = torch.tensor(verts).float()

        faces = torch.tensor(regstr['scan_f']).long()

        ### NOTE remove foot poses
        if 'felice' in self.regstr_list[index]:
            poses[21:27] = 0    # foot
            poses[30:36] = 0    # toes

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = 1
        smpl_params[4:76] = torch.tensor(poses).float()

        # data['scan_verts'] = verts
        data['smpl_params'] = smpl_params
        data['smpl_thetas'] = smpl_params[4:76]
        data['smpl_betas'] = smpl_params[76:]

        num_verts, num_dim = verts.shape
        random_idx = torch.randint(0, num_verts, [self.points_per_frame, 1], device=verts.device)
        random_pts = torch.gather(verts, 0, random_idx.expand(-1, num_dim))
        data['pts_d']  = self.sampler.get_points(random_pts[None])[0]
        data['occ_gt'] = kaolin.ops.mesh.check_sign(verts[None], faces, data['pts_d'][None]).float()[0].unsqueeze(-1)
        # ic(faces)

        return data

    def __len__(self):
        return len(self.regstr_list)

''' Used to generate groud-truth occupancy and bone transformations in batchs during training '''
class FITEDataProcessor():

    def __init__(self, opt, smpl_model_path, meta_info, **kwargs):

        self.opt = opt
        self.gender = meta_info['gender']
        self.v_template = meta_info['v_template']

        self.smpl_server = SMPLServer(smpl_model_path=smpl_model_path, gender=self.gender, v_template=self.v_template)
        self.smpl_faces = torch.tensor(self.smpl_server.smpl.faces.astype('int')).unsqueeze(0).cuda()
        
        self.sampler = PointInSpace(**opt['sampler'])

    def process(self, data):

        smpl_output = self.smpl_server(data['smpl_params'], absolute=True)
        data.update(smpl_output)
        return data
