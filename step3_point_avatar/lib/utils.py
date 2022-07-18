import numpy as np
import torch

def tensor2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

def vertex_normal_2_vertex_color(vertex_normal):
    # Normalize vertex normal
    import torch
    if torch.is_tensor(vertex_normal):
        vertex_normal = vertex_normal.detach().cpu().numpy()
    normal_length = ((vertex_normal**2).sum(1))**0.5
    normal_length = normal_length.reshape(-1, 1)
    vertex_normal /= normal_length
    # Convert normal to color:
    color = vertex_normal * 255/2.0 + 128
    return color.astype(np.ubyte)

def export_ply_with_vquality(filename, v_array=None, f_array=None, vq_array=None):
    """
    v_array: vertex array
    vq_array: vertex quality array
    """
    
    Nv = v_array.shape[0] if v_array is not None else 0
    Nf = f_array.shape[0] if f_array is not None else 0

    with open(filename, 'w') as plyfile:
        plyfile.write(f'ply\n')
        plyfile.write(f'format ascii 1.0\n')
        plyfile.write(f'comment trisst custom\n')
        plyfile.write(f'element vertex {Nv}\n')
        plyfile.write(f'property float x\n')
        plyfile.write(f'property float y\n')
        plyfile.write(f'property float z\n')
        if vq_array is not None:
            plyfile.write(f'property float quality\n')
        plyfile.write(f'element face {Nf}\n')
        plyfile.write(f'property list uchar int vertex_indices\n')
        plyfile.write(f'end_header\n')
        for i in range(Nv):
            plyfile.write(f'{v_array[i][0]} {v_array[i][1]} {v_array[i][2]} ')
            
            if vq_array is None:
                plyfile.write('\n')
                continue
            
            plyfile.write(f'{vq_array[i]} ')
            plyfile.write('\n')
            continue
            
        for i in range(Nf):
            plyfile.write(f'3 {f_array[i][0]} {f_array[i][1]} {f_array[i][2]}\n')
            


def customized_export_ply(outfile_name, v, f = None, v_n = None, v_c = None, f_c = None, e = None):
    '''
    Author: Jinlong Yang, jyang@tue.mpg.de

    Exports a point cloud / mesh to a .ply file
    supports vertex normal and color export
    such that the saved file will be correctly displayed in MeshLab

    # v: Vertex position, N_v x 3 float numpy array
    # f: Face, N_f x 3 int numpy array
    # v_n: Vertex normal, N_v x 3 float numpy array
    # v_c: Vertex color, N_v x (3 or 4) uchar numpy array
    # f_n: Face normal, N_f x 3 float numpy array
    # f_c: Face color, N_f x (3 or 4) uchar numpy array
    # e: Edge, N_e x 2 int numpy array
    # mode: ascii or binary ply file. Value is {'ascii', 'binary'}
    '''

    v_n_flag=False
    v_c_flag=False
    f_c_flag=False

    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        if type(v_n) == 'torch.Tensor':
            v_n = v_n.detach().cpu().numpy()
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            # warnings.warn("Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype = np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                # warnings.warn("Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype = np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n'%(N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n'%(N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n'%(N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2], \
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_n[i,0], v_n[i,1], v_n[i,2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n'%\
                    (v[i,0], v[i,1], v[i,2],\
                    v_c[i,0], v_c[i,1], v_c[i,2], v_c[i,3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n'%\
                    (v[i,0], v[i,1], v[i,2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2],\
                    f_c[i,0], f_c[i,1], f_c[i,2], f_c[i,3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n'%\
                    (f[i,0], f[i,1], f[i,2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n'%(e[i,0], e[i,1]))

def save_result_examples(save_dir, model_name, result_name, 
                         points, normals=None, patch_color=None, 
                         texture=None, coarse_pts=None,
                         gt=None, epoch=None):
    # works on single pcl, i.e. [#num_pts, 3], no batch dimension
    from os.path import join
    import numpy as np

    if epoch is None:
        normal_fn = '{}_{}_pred.ply'.format(model_name, result_name)
    else:
        normal_fn = '{}_epoch{}_{}_pred.ply'.format(model_name, str(epoch).zfill(4), result_name)
    normal_fn = join(save_dir, normal_fn)
    points = tensor2numpy(points)

    if normals is not None:
        normals = tensor2numpy(normals)
        color_normal = vertex_normal_2_vertex_color(normals)
        customized_export_ply(normal_fn, v=points, v_n=normals, v_c=color_normal)

    if patch_color is not None:
        patch_color = tensor2numpy(patch_color)
        if patch_color.max() < 1.1:
            patch_color = (patch_color*255.).astype(np.ubyte)
        pcolor_fn = normal_fn.replace('pred.ply', 'pred_patchcolor.ply')
        customized_export_ply(pcolor_fn, v=points, v_c=patch_color)
    
    if texture is not None:
        texture = tensor2numpy(texture)
        if texture.max() < 1.1:
            texture = (texture*255.).astype(np.ubyte)
        texture_fn = normal_fn.replace('pred.ply', 'pred_texture.ply')
        customized_export_ply(texture_fn, v=points, v_c=texture)

    if coarse_pts is not None:
        coarse_pts = tensor2numpy(coarse_pts)
        coarse_fn = normal_fn.replace('pred.ply', 'interm.ply')
        customized_export_ply(coarse_fn, v=coarse_pts)

    if gt is not None: 
        gt = tensor2numpy(gt)
        gt_fn = normal_fn.replace('pred.ply', 'gt.ply')
        customized_export_ply(gt_fn, v=gt)


def adjust_loss_weights(init_weight, current_epoch, mode='decay', start=400, every=20):
    # decay or rise the loss weights according to the given policy and current epoch
    # mode: decay, rise or binary

    if mode != 'binary':
        if current_epoch < start:
            if mode == 'rise':
                weight = init_weight * 1e-6 # use a very small weight for the normal loss in the beginning until the chamfer dist stabalizes
            else:
                weight = init_weight
        else:
            if every == 0:
                weight = init_weight # don't rise, keep const
            else:
                if mode == 'rise':
                    weight = init_weight * (1.05 ** ((current_epoch - start) // every))
                else:
                    weight = init_weight * (0.85 ** ((current_epoch - start) // every))

    return weight