# Learning Implicit Templates for Point-Based Clothed Human Modeling (ECCV 2022)

[[project page]](https://jsnln.github.io/fite) 

A **First-Implicit-Then-Explicit (FITE)** pipeline for modeling humans in clothing.

### Note

Still doing some tests. This code may undergo frequent updates in the next few days.

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

### Running the Whole Pipeline

Detailed instructions for running other parts of the code are coming soon. 

### Acknowledgements

This code is partly based on [SNARF](https://github.com/xuchen-ethz/snarf) and [POP](https://github.com/qianlim/POP). We thank those authors for making their code publicly available.



