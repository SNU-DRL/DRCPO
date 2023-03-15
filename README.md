# DR.CPO: Diversified and Realistic 3D Augmentation via Iterative Construction, Random Placement, and HPR Occlusion(AAAI-2023)

This repository provides a Official PyTorch implementation of our AAAI 2023 paper ***DR.CPO: Diversified and Realistic 3D Augmentation via Iterative Construction, Random Placement, and HPR Occlusion.***

* AAAI 2023 paper(will be made available)
* arXiv version(will be made available)

The all implementation for the baseline networks as well as the augmentation methods is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

Our contributions are included in all three files below.

* pcdet/datasets/augmentor/iter_occ_augmentor_utils.py
* pcdet/datasets/augmentor/iter_occ_database_sampler.py
* pcdet/datasets/kitti/kitti_create_iter_candidates.py


-------------------

# Abstract

In autonomous driving, data augmentation is commonly used for improving 3D object detection. The most basic methods include insertion of copied objects and rotation and scaling of the entire training frame. Numerous variants have been developed as well. The existing methods, however, are con- siderably limited when compared to the variety of the real world possibilities. In this work, we develop a diversified and realistic augmentation method that can flexibly construct a whole-body object, freely locate and rotate the object, and apply self-occlusion and external-occlusion accordingly. To improve the diversity of the whole-body object construction, we develop an iterative method that stochastically combines multiple objects observed from the real world into a single object. Unlike the existing augmentation methods, the con- structed objects can be randomly located and rotated in the training frame because proper occlusions can be reflected to the whole-body objects in the final step. Finally, proper self- occlusion at each local object level and external-occlusion at the global frame level are applied using the Hidden Point Removal (HPR) algorithm that is computationally efficient. HPR is also used for adaptively controlling the point density of each object according to the object’s distance from the LiDAR. Experiment results show that the proposed DR.CPO algorithm is data-efficient and model-agnostic without incurring any computational overhead. DR.CPO can improve mAP performance by 2.08% when compared to the best 3D detection result known for KITTI dataset.

# Our contributions:

* We introduce DR.CPO, a data augmentation algorithm specifically developed for 3D point clouds, to provide di-
  versified and realistic views at the level of individual objects and at the level of entire frame.
* We develop an iterative method for constructing realistic whole-body objects where only real observations are
  used.
* We present a computationally light occlusion method based on HPR. Our method can implement self-occlusion, external-occlusion, and density adjustment.
* We show that DR.CPO is data efficient, model-agnostic, and computationally efficient.
* We achieve state-of-the-art performance on the KITTI dataset. DR.CPO improves mAP performance by 2.08%

# Installation

### Requirements
The codes are tested in the following environment:

* Ubuntu 18.04
* Python 3.7+
* PyTorch 1.7.1
* CUDA 10.2
* spconv v2.x
  * You could also install the latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).

### Install DR.CPO

* Install virtualenv

```
virtualenv -p /usr/bin/python3.7 ~/.venv/drcpo
source ~/.venv/drcpo/bin/activate
```

* Install this `pcdet` library and its dependent libraries by running the following command:
```
python setup.py develop
```

# Getting Started



## Dataset Preparation

* Please organize the data as follows

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI training set
* NOTE: if you already have the data infos from `pcdet v0.1`, you can choose to use the old infos and set the DATABASE_WITH_FAKELIDAR option in tools/cfgs/dataset_configs/kitti_dataset.yaml as True. The second choice is that you can create the infos and gt database again and leave the config unchanged.

```
drcpo
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset_drcpo.yaml
```
* Generate the candiates infos by running the following command: 

```
python pcdet/datasets/kitti/kitti_create_iter_candidates.py
```



## Training

### Train a model

You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters.

* Train with multiple GPUs or multiple machines

```
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:

```shell script
python train.py --cfg_file ${CONFIG_FILE}
```

## Testing

* Test and evaluate the pretrained models

* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```



# Citation

```
@inproceedings{currently not available
}
```



# Thanks to 

We would like to express our gratitude to the creators and contributors of OpenPCDet, an open-source framework for 3D point cloud detection. 

# Contact

Please contact the author if you have any questions about our repository/paper: Jungwook Shin(jungwook.shin@snu.ac.kr)
