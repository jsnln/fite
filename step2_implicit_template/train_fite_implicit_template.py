import yaml
from tqdm import tqdm
import cv2
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import os
from os.path import join

from .lib.dataset.fite import FITEDataSet, FITEDataProcessor
from .lib.snarf_model_diffused_skinning import SNARFModelDiffusedSkinning

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
    
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    ### NOTE dataset
    dataset_train = FITEDataSet(dataset_path=opt['data_scans_path'],
                                   data_templ_path=opt['data_templates_path'],
                                   opt=opt['datamodule'],
                                   subject=opt['datamodule']['subject'],
                                   clothing=opt['datamodule']['clothing'],
                                   split='train')

    dataloader = DataLoader(dataset_train,
                            batch_size=opt['datamodule']['batch_size'],
                            num_workers=opt['datamodule']['num_workers'], 
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    ### NOTE data processor
    data_processor = FITEDataProcessor(opt['datamodule']['processor'],
                                          smpl_model_path=opt['smpl_model_path'],
                                          meta_info=dataset_train.meta_info)


    model = SNARFModelDiffusedSkinning(opt['model']['soft_blend'],
                                        opt['smpl_model_path'],
                                        opt['model']['pose_conditioning'],
                                        opt['model']['network'],
                                        subject=opt['datamodule']['subject'],
                                        cpose_smpl_mesh_path=join(opt['data_templates_path'], opt['datamodule']['subject'], opt['datamodule']['subject'] + '_minimal_cpose.ply'),
                                        cpose_weight_grid_path=join(opt['data_templates_path'], opt['datamodule']['subject'], opt['datamodule']['subject'] + '_cano_lbs_weights_grid_float32.npy'),
                                        meta_info=dataset_train.meta_info,
                                        data_processor=data_processor).to('cuda')


    optimizer = torch.optim.Adam(model.parameters(), lr=opt['model']['optim']['lr'])

    loader = dataloader
    max_steps = opt['trainer']['max_steps']

    total_steps = 0
    total_epochs = 0
    while total_steps < max_steps:
        tloader = tqdm(dataloader)
        for batch in tloader:
            for data_key in batch:
                batch[data_key] = batch[data_key].to('cuda')
            loss = model.training_step(batch, 0)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=opt['trainer']['gradient_clip_val'])
            optimizer.step()

            tloader.set_description(f'[Epoch: {total_epochs:03d}; Step: {total_steps:05d}/{max_steps:05d}] loss_bce = {loss.item():.4e}')

            if total_steps % opt['trainer']['save_vis_every_n_iters'] == 0:
                with torch.no_grad():
                    img_all = model.validation_step(batch, 0)['img_all']
                cv2.imwrite(join(exp_folder, f'vis-step-{total_steps:05d}.png'), img_all[..., :3][..., ::-1])

            if total_steps % opt['trainer']['save_ckpt_every_n_iters'] == 0:
                torch.save(model.state_dict(), join(checkpoint_folder, 'checkpoint-latest.pt'))
                torch.save(model.state_dict(), join(checkpoint_folder, f'checkpoint-{total_steps:05d}.pt'))

            total_steps += 1
        total_epochs += 1
