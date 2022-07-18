import torch
import torch.nn as nn
import torch.nn.functional as F 


from icecream import ic

from .modules import UnetNoCond5DS, UnetNoCond6DS, UnetNoCond7DS, GeomConvLayers, GaussianSmoothingLayers, GeomConvBottleneckLayers, ShapeDecoder, PreDeformer

class FITEModel(nn.Module):
    def __init__(
                self, 
                projection_list,
                input_nc=3, # num channels of the unet input
                c_geom=64, # channels of the geometric features
                c_pose=64, # channels of the pose features
                nf=64, # num filters for the unet
                posmap_size=256, # size of UV positional map (pose conditioning), i.e. the input to the pose unet
                hsize=256, # hidden layer size of the ShapeDecoder MLP
                up_mode='upconv', # upconv or upsample for the upsampling layers in the pose feature UNet
                use_dropout=False, # whether use dropout in the pose feature UNet
                ):

        super().__init__()

        self.posmap_size = posmap_size
        self.projection_list = projection_list
        
        unets = {32: UnetNoCond5DS, 64: UnetNoCond6DS, 128: UnetNoCond7DS, 256: UnetNoCond7DS}
        unet_loaded = unets[self.posmap_size]

        # U-net: for extracting pixel-aligned pose features from the input UV positional maps
        n_proj_directions = len(projection_list)
        self.poseencode_unets = []
        for proj_id in range(n_proj_directions):
            self.poseencode_unets.append(unet_loaded(input_nc, c_pose//n_proj_directions, nf, up_mode=up_mode, use_dropout=use_dropout))
        self.poseencode_unets = nn.ModuleList(self.poseencode_unets)
        
        self.decoder = ShapeDecoder(in_size=c_pose + c_geom + 3, hsize=hsize, actv_fn='softplus')
        self.predeformer = PreDeformer(in_size=c_geom)

    def query_feats(self, projected_pts, pose_featmaps):
        query_grid = projected_pts[:, :, None] # [bs, n_pts, 1, 3]
        queried_feats = F.grid_sample(pose_featmaps, query_grid)[..., 0]   # [bs, n_feat, n_pts]

        # ic(queried_feats_front.shape, queried_feats_back.shape)
        return queried_feats

    def forward(self, geom_feats_batch, projected_pts_batch, basepoints_batch, posmaps_batch, posmap_weights_batch=None, return_featmap=False):

        queried_feats_batch = []
        for proj_id in range(len(self.projection_list)):
            proj_direction = self.projection_list[proj_id]['dirc']
            # projected_pts_batch = projected_pts_all[proj_direction][subject_id]
            pose_featmap = self.poseencode_unets[proj_id](posmaps_batch[proj_direction])
            
            # ic(projected_pts_batch[proj_direction].shape, pose_featmap.shape)
            queried_feats = self.query_feats(projected_pts_batch[proj_direction], pose_featmap)
            if posmap_weights_batch is not None:
                queried_feats_batch.append(queried_feats * posmap_weights_batch[proj_direction][:, None])
            else:
                queried_feats_batch.append(queried_feats)

        pose_feat_final = torch.cat(queried_feats_batch + [geom_feats_batch, basepoints_batch.permute(0, 2, 1)], 1)

        residuals, normals = self.decoder(pose_feat_final)
        
        if return_featmap:
            return residuals, normals, queried_feats_batch
        return residuals, normals        
