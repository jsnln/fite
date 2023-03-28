import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pymeshlab
import argparse
from scipy.spatial import cKDTree as KDTree
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('gt_dataset', type=str)
parser.add_argument('recon_dataset', type=str)
args = parser.parse_args()


def angular_error(gt_mesh_name, gen_mesh_name, sample_num):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """
    gt_mesh = trimesh.load_mesh(gt_mesh_name)
    gen_mesh = trimesh.load_mesh(gen_mesh_name)

    gt_points, gt_face_index = trimesh.sample.sample_surface(gt_mesh, sample_num)
    gen_points, gen_face_index = trimesh.sample.sample_surface(gen_mesh, sample_num)

    gt_normals = gt_mesh.face_normals[gt_face_index]
    gen_normals = gen_mesh.face_normals[gen_face_index]
    
    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    gt2gen_dist, gt2gen_vert_ids = gen_points_kd_tree.query(gt_points)
    gt2gen_closest_normals_on_gen = gen_normals[gt2gen_vert_ids]
    gt2gen_cos_sim = np.mean(np.einsum('nk,nk->n', gt_normals, gt2gen_closest_normals_on_gen))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    gen2gt_dist, gen2gt_vert_ids = gt_points_kd_tree.query(gen_points)
    gen2gt_closest_normals_on_gen = gt_normals[gen2gt_vert_ids]
    gen2gt_cos_sim = np.mean(np.einsum('nk,nk->n', gen_normals, gen2gt_closest_normals_on_gen))
    cos_sim = (np.abs(gt2gen_cos_sim) + np.abs(gen2gt_cos_sim)) / 2
    
    str_ang = f"angle:          {gt2gen_cos_sim:.6f}      {gen2gt_cos_sim:.6f}      {cos_sim:.6f}\n"

    return str_ang, cos_sim

def print_matching(list_a, list_b):
    counter = 0
    for a, b in zip(list_a, list_b):
        counter += 1
        print(f'Matched {a} and {b}')
    print(f'Matched {counter} of {len(list_a)} and {len(list_b)}')


def res2str(name_a, name_b, res_a2b, res_b2a, ms):
    """
    this normalizes the results by bounding box diagonal
    and put into a new dict
    """

    # error field extraction and normalization
    a2b_error_field = ms.mesh(3).vertex_quality_array() # float64, (100000,)
    b2a_error_field = ms.mesh(5).vertex_quality_array() # float64, (100000,)
    a2b_error_field /= res_a2b['diag_mesh_0']
    b2a_error_field /= res_b2a['diag_mesh_0']

    dist_Haus_a2b = a2b_error_field.max()
    dist_Haus_b2a = b2a_error_field.max()
    dist_symHausd = max(dist_Haus_a2b, dist_Haus_b2a)

    dist_Cham_a2b = (a2b_error_field ** 2).mean()
    dist_Cham_b2a = (b2a_error_field ** 2).mean()
    dist_symChamf = (dist_Cham_a2b + dist_Cham_b2a) / 2

    str_nma = f"name_a: {name_a}\n"
    str_nmb = f"name_b: {name_b}\n"
    str_itm = f"----            a2b               b2a               sym\n"
    str_hau = f"haus:           {dist_Haus_a2b:.6e}      {dist_Haus_b2a:.6e}      {dist_symHausd:.6e}\n"
    str_chm = f"chamfer:        {dist_Cham_a2b:.6e}      {dist_Cham_b2a:.6e}      {dist_symChamf:.6e}\n"
    str_dg0 = f"diag a:         {res_a2b['diag_mesh_0']:.6e}\n"
    str_dg1 = f"diag b:         {res_a2b['diag_mesh_1']:.6e}\n"
    str_num = f"n_samples:      {res_a2b['n_samples']}\n"

    str_all = str_nma + str_nmb + str_itm + str_hau + str_chm + str_dg0 + str_dg1 + str_num
    return str_all, dist_symHausd, dist_Haus_a2b, dist_Haus_b2a, dist_symChamf, dist_Cham_a2b, dist_Cham_b2a

def compare_meshes(meshfile_a, meshfile_b, sample_num):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(meshfile_a)
    ms.load_new_mesh(meshfile_b)
    
    res_a2b = ms.hausdorff_distance(
        sampledmesh=0,
        targetmesh=1,
        savesample=True,
        samplevert=False,
        sampleedge=False,
        samplefauxedge=False,
        sampleface=True,
        samplenum=sample_num
    )

    # 2 is closest from a to b (on b)
    # 3 is sampled from a to b (on a)

    res_b2a = ms.hausdorff_distance(
        sampledmesh=1,
        targetmesh=0,
        savesample=True,
        samplevert=False,
        sampleedge=False,
        samplefauxedge=False,
        sampleface=True,
        samplenum=sample_num
    )

    # 4 is closest from b to a (on a)
    # 5 is sampled from b to a (on b)
    
    str_res, d_haus, d_haus_a2b, d_haus_b2a, d_cham, d_cham_a2b, d_cham_b2a = res2str(meshfile_a, meshfile_b, res_a2b, res_b2a, ms)

    del ms
    return str_res, d_haus, d_cham

if __name__ == '__main__':
    folder_GT = args.gt_dataset
    folder_recon = args.recon_dataset

    list_GT = sorted(list(glob(f'{folder_GT}/*_poisson.ply')))
    list_recon = sorted(list(glob(f'{folder_recon}/*_poisson.ply')))

    log_file = f'{folder_recon}/log_file_ver2.txt'
    log_file_stream = open(log_file, 'w')

    csv_file = f'{folder_recon}/hausandchamfer.csv'
    csv_file_stream = open(csv_file, 'w')

    assert len(list_GT) == len(list_recon), f'you have {len(list_GT)} GT and {len(list_recon)} recon files'
    print_matching(list_GT, list_recon)

    for i in tqdm(range(len(list_GT))):
        print(f'Trying to compare \n    {list_GT[i]}\n    {list_recon[i]}')

        sample_num = 200000
            
        str_res, d_haus, d_cham = compare_meshes(list_GT[i], list_recon[i], sample_num)
        str_ang, cos_sim = angular_error(list_GT[i], list_recon[i], sample_num)

        log_file_stream.write(str_res + str_ang + '---------------\n')
        print(str_res + str_ang + '---------------\n')
        csv_file_stream.write(f'{list_recon[i]},{d_haus},{d_cham},{cos_sim}\n')

    log_file_stream.close()
    csv_file_stream.close()
