# Learning Implicit Templates for Point-Based Clothed Human Modeling (ECCV 2022)

[[project page]](https://jsnln.github.io/fite) 

A **First-Implicit-Then-Explicit (FITE)** pipeline for modeling humans in clothing.

### Overview

Our code consists of three main steps:

1. Given a minimal T-pose SMPL template, compute the diffused skinning field, stored as a 256^3 voxel grid.
2. Learn a canonical implicit template using SNARF, but with our diffused skinning field.
3. Render position maps (posmaps for short) and train the point-based avatars.

You can get a quick start with our pretrained models, or follow the instructions below to go through the whole pipeline.

### Quick Start

To use our pretrained models (download from [here](https://cloud.tsinghua.edu.cn/d/8a6fe3fa9af341fdae06/)), only the test script of step 3 needs to be executed.

First, unzip the downloaded pretrained data, put `{subject_name}_clothed_template.npz` in the folder `data_templates`, and put `checkpoint-400.pt` and `geom-feats-400.pt` in `results/{resynth,cape}_pretrained/step3-checkpoints`. Rename the check points to `*-latest.pt`

To animate a certain subject, say `rp_carla_posed_004`, prepare a pose sequence (`.npz` files containing the `pose` and `transl` parameters) under the directory `data_scans/rp_carla_posed_004/test/`, and then run (assuming the project directory is the working directory):

```bash
python -m step3_point_avatar.render_posmaps rp_carla_posed_004 test
```

This will render the posmaps to `data_posmaps/rp_carla_posed_004/test`. Then, run

```bash
python -m step3_point_avatar.test_fite_point_avatar
```

The animated point clouds will be saved at `results/resynth_pretrained/step3-test-pcds`.

### The Whole Pipeline

Prior to running the pipeline, set `expname` in `configs/common.yaml` to the name of your experiment and keep it fixed. 

#### 1. Dependencies

We have tested this code on Ubuntu 20.04 with Python 3.8.10 and CUDA 11.1.

To run the whole pipeline, the user needs to install the following dependencies.

1. The `PointInterpolant` executable from https://github.com/mkazhdan/PoissonRecon. After successfully building `PointInterpolant`, set `point_interpolant_exe` in the config file `configs/step1.yaml` as the path to the executable.

   ```bash
   git clone https://github.com/mkazhdan/PoissonRecon
   cd PoissonRecon
   make pointinterpolant
   cd ..
   ```

2. Git clone https://github.com/NVIDIAGameWorks/kaolin.git to the FITE project directory and build it.

   ```bash
   git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
   cd kaolin
   git checkout v0.11.0     # optional, other versions should also work
   python setup.py install 	# use --user if needed
   cd ..
   ```


3. Download chamfer distance from this implementation: https://github.com/krrish94/chamferdist/tree/97051583f6fe72d5d4a855696dbfda0ea9b73a6a and build it.

   ```bash
   cd chamferdist
   python setup.py install		# use --user if needed
   cd ..
   ```

4. Download the SMPL models from https://smpl.is.tue.mpg.de/ (v1.1.0). Put them in some folder structured as:

   ```
   smpl_models/
     smpl/
       SMPL_MALE.pkl
       SMPL_FEMALE.pkl
       SMPL_NEUTRAL.pkl
   ```

    Then set `smpl_model_path` in `configs/common.yaml` as the path to this folder.

5. Install these python packages:

   ```bash
   pyyaml   # 6.0
   tqdm     # 4.62.3
   smplx    # 0.1.28
   torch    # 1.10.0+cu111
   trimesh  # 3.10.2
   numpy    # 1.22.3
   opencv-python  # 4.5.5
   scikit-image   # 0.18.1
   pytorch3d      # 0.6.1
   pyglm    # 2.5.7
   pyopengl # 3.1.0
   glfw     # 2.1.0
   scipy    # 1.8.0
   ```

#### 2. Data Preparation for Training

To train on your own data, you need to prepare the scans (which must be closed meshes) as `.npz` files containing the following items:

```
'pose':		of shape (72,), SMPL pose parameters
'transl':	of shape (3,), translation of the scan
'scan_f':	of shape (N_faces, 3), triangle faces of the scan mesh
'scan_v':	of shape (N_vertices, 3), vertices of the scan mesh
'scan_pc':	of shape (N_points, 3), points uniformly sampled on the scan
'scan_n':	of shape (N_points, 3), normals of the sampled points (unit length)
```

Note that the number of points of `scan_pc` and `scan_n` must be the same across different scans (for tensor batching), while the number of vertices and faces can be different across scans. These `.npz` files should be placed at `data_scans/{subject_name}/train`. 

A minimal SMPL body matching the scans is also needed. Prepare the T-pose SMPL mesh as `{subject_name}_minimal_tpose.ply`, and put it in `data_templates/{subject_name}/` . Finally, add the gender of the subject to `data_templates/gender_list.yaml`.

#### 3. Diffused Skinning

This step computes a voxel grid for diffused skinning. First, run

```bash
cd step1_diffused_skinning
sh compile_lbs_surf_grad.sh
cd ..
```

to compile the c++ program for computing surface gradient of LBS weights. Then, change the `subject` option in `configs/step1.yaml` to the name of your subject, and run

```bash
python -m step1_diffused_skinning.compute_diffused_skinning
```

After it finishes, a file named `{subject_name}_cano_lbs_weights_grid_float32.npy` will be placed at `data_templates/{subject_name}/`. You can optionally delete the intermediate files in `data_tmp_constraints` and `data_tmp_skinning_grid` if they take up too much space.


#### 4. Implicit Templates

```bash
cd step2_implicit_templates
python setup.py install    # --user if needed
cd ..
```

Change `datamodule.subject` in `configs/step2.yaml` to the name of the subject to train. Then run

```bash
python -m step2_implicit_template.train_fite_implicit_template
```

Intermediate visualizations and checkpoints can be found at `results/{expname}/step2-results` and `results/{expname}/step2-results`. After the training is done, run

```bash
python -m step2_implicit_template.extract_fite_implicit_template
```

This extracts the implicit templates in canonical poses to `data_templates`.

#### 5. Point Avatar

Prior to train the point avatar(s), the user needs to render the posmaps:

```bash
python -m step3_point_avatar.render_posmaps {subject_name} {dataset_split}
```

Recall that our point avatar part is a multi-subject model. If you do use multiple subjects, each should go through all the steps above. After that, collect the names and genders of all subjects in a config file `configs/{expname}_subject_list.yaml`. Then, run

```bash
python -m step3_point_avatar.train_fite_point_avatar
```

After training is complete, prepare test poses (only `pose` and `transl` are needed) in `data_scans/{subject_name}/test/` and render posmaps same as above (test split). Then, run

```bash
python -m step3_point_avatar.test_fite_point_avatar
```

The outputted point clouds can be found at `results/{expname}/step3-test-pcds/`.

### Acknowledgements & A Note on the License

This code is partly based on [SNARF](https://github.com/xuchen-ethz/snarf) and [POP](https://github.com/qianlim/POP). We thank those authors for making their code publicly available. Note that the code in `step2_implicit_template` is inherited from [SNARF](https://github.com/xuchen-ethz/snarf), while that in `step3_point_avatar` is herited from [POP](https://github.com/qianlim/POP). Please follow their original licenses if you intend to use them.



