import pickle

import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils
from functools import partial
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.datasets.augmentor.iter_occ_augmentor_utils import ExternalHiddenPointRemove

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
            
        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)
        self.reduced_info = sampler_cfg.get('reduced_info', None)

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)

                if self.reduced_info is not None:
                    reduced_info_path = self.root_path.resolve() / self.reduced_info['path']
                    samples_name = self.reduced_info['name']
                    with open(reduced_info_path, 'rb') as f:
                        reduced_idx = pickle.load(f)[samples_name]
                    for cur_class in class_names:
                        for info in infos[cur_class]:
                            if info['image_idx'] in reduced_idx:
                                self.db_infos[cur_class].extend([info])
                else:
                    [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)
        
        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        self.object_transform = []

        if 'USE_EXT_OCCLUSION' in sampler_cfg:
            self.use_ext_occlusion = sampler_cfg.USE_EXT_OCCLUSION
            self.radius = sampler_cfg.OBJECT_AUG_LIST.EXT_OCC[0]
            self.ext_z_basis = sampler_cfg.OBJECT_AUG_LIST.EXT_OCC[1]

            self.ext_occ = ExternalHiddenPointRemove(radius=self.radius, z_basis=self.ext_z_basis)
        else:
            self.use_ext_occlusion = False

        self.bev_ranges = []

        self.enable_near_trans = False
        self.enable_far_trans = False
        self.enable_obj_aug = False

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def objects_augmentation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_objects_augmentation, config=config)

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)
            
        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        if 'gt_boxes_mask' in data_dict.keys():
            gt_boxes_mask = data_dict['gt_boxes_mask']
            gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
            gt_names = data_dict['gt_names'][gt_boxes_mask]
        else:
            gt_boxes = data_dict['gt_boxes']
            gt_names = data_dict['gt_names']

        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )

        obj_points_list, obj_mask_list = [], []
        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            if 'NORMALIZE' in self.sampler_cfg:
                if not self.sampler_cfg['NORMALIZE']:
                    obj_points[:, :3] += info['box3d_lidar'][:3]
                else:
                    angle = info['box3d_lidar_org'][6]
                    obj_points = np.concatenate([self.object_rotate_points_along_z(obj_points[:,:3], -angle), obj_points[:,3:]], axis=1)
            else:
                obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.enable_obj_aug:
                new_dict = self.random_objects_augmentation(dict(points=obj_points,bbox=info['box3d_lidar'],name=info['name']))
                obj_points = new_dict['points']
                obj_mask = new_dict['mask']
                obj_mask_list.append(obj_mask)
            else:
                obj_mask = True
            #extreme_box = new_dict['fit_box']
            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            if obj_mask:
                obj_points_list.append(obj_points)
        if len(obj_mask_list) > 0:
            obj_mask = np.array(obj_mask_list)
        else:
            obj_mask = np.ones(len(total_valid_sampled_dict),dtype=bool)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])[obj_mask]
        sampled_gt_boxes = sampled_gt_boxes[obj_mask]

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def object_rotate_points_along_z(self, points, angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:

        """

        cosa = np.cos(angle)
        sina = np.sin(angle)
        zeros = np.zeros_like(angle)
        ones = np.ones_like(angle)
        rot_matrix = np.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), axis=0).reshape(3, 3)
        points_rot = np.matmul(points[:, 0:3], rot_matrix)
        return points_rot

    def iou_base_valid_mask(self, sampled_boxes, existed_boxes):
        iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
        iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
        iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
        iou1 = iou1 if iou1.shape[1] > 0 else iou2
        valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]

        return valid_mask

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """

        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)

        if 'gt_boxes_mask' in data_dict.keys():
            gt_boxes_mask = data_dict['gt_boxes_mask']
            gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
            gt_names = data_dict['gt_names'][gt_boxes_mask]

        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                n = sampled_boxes.shape[0]
                if self.enable_near_trans:
                    noise_translate = np.array(
                        [
                            [np.random.uniform(-sampled_boxes[i,3], sampled_boxes[i,3], 1),
                             np.random.uniform(-sampled_boxes[i,4], sampled_boxes[i,4], 1)] for i in range(n)
                        ]).reshape(n, -1)
                    #translated_sampled_boxes = sampled_boxes[sampled_boxes]
                    sampled_boxes[:, 0:2] += noise_translate
                elif self.enable_far_trans:
                    noise_translate = np.array(
                        [
                            np.random.uniform(self.translate_range[0], self.translate_range[2], n),
                            np.random.uniform(self.translate_range[1], self.translate_range[3], n),
                        ]).T
                    sampled_boxes[:, 0:2] = noise_translate
                    noise_angle = np.random.uniform(-np.pi, np.pi, n)
                    for i in range(n):
                        sampled_boxes[i, 6] = noise_angle[i]

                valid_mask = self.iou_base_valid_mask(sampled_boxes, existed_boxes)
                valid_sampled_dict = []

                for x in valid_mask:
                    if 'NORMALIZE' in self.sampler_cfg and self.sampler_cfg['NORMALIZE']:
                        sampled_dict[x]['box3d_lidar_org'] = copy.deepcopy(sampled_dict[x]['box3d_lidar'])
                    sampled_dict[x]['box3d_lidar'] = sampled_boxes[x]
                    valid_sampled_dict.append(sampled_dict[x])

                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)
            if self.use_ext_occlusion:
                data_dict = self.global_hiden_point_removal(data_dict)

        if 'gt_boxes_mask' in data_dict:
            data_dict.pop('gt_boxes_mask')

        return data_dict

    def global_hiden_point_removal(self, data_dict):
        b = self.ext_occ(data_dict)
        mask = []
        for i in range(data_dict['gt_boxes'].shape[0]):
            c = np.sum(roiaware_pool3d_utils.points_in_boxes_cpu(data_dict['points'][:, 0:3], data_dict['gt_boxes'][i].reshape(1, -1)))
            if c < 5:
                mask.append(False)
            else:
                mask.append(True)
        data_dict['gt_boxes'] = data_dict['gt_boxes'][np.array(mask)]
        data_dict['gt_names'] = data_dict['gt_names'][np.array(mask)]
        return data_dict

    def random_objects_augmentation(self, data_dict=None):

        transform_sel = np.random.randint(0, len(self.object_transform))
        data_dict = self.object_transform[transform_sel](data_dict)
        # from tools.visualizer.visualize_mayavi import draw_scenes
        # draw_scenes(data_dict['points'],gt_boxes=data_dict['gt_boxes'].reshape(1,-1))
        return data_dict

