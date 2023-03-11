import copy
import pickle
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils

from functools import partial
from ..augmentor import iter_occ_augmentor_utils as d_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from torchvision import transforms
from pcdet.datasets.augmentor.iter_occ_augmentor_utils import ExternalHiddenPointRemove
from pcdet.datasets.augmentor.iter_occ_augmentor_utils import RandomPlacement

class IterOcclusionDataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        #self.class_names = ['Pedestrian','Cyclist']

        self.reduced_info = sampler_cfg.get('reduced_info', None)
        self.iterative_mode = sampler_cfg.get('ITERATIVE', False)
        for class_name in class_names:
            self.db_infos[class_name] = []

        #self.filter_by_min_class_names = sampler_cfg.PREPARE.filter_by_min_class_names
        self.mask_min_points = dict()
        for name_num in sampler_cfg.OBJECT_AUG_LIST.MASK_MIN_POINTS:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            self.mask_min_points[name] = min_num

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        self.object_transform = []
        if 'USE_EXT_OCCLUSION' in sampler_cfg:
            self.use_ext_occlusion = sampler_cfg.USE_EXT_OCCLUSION
        else:
            self.use_ext_occlusion = False

        if 'OBJECT_AUG_LIST' in sampler_cfg:
            self.enable_near_trans = sampler_cfg.OBJECT_AUG_LIST.NEAR_TRANS_VARIANCE
            self.enable_far_trans = sampler_cfg.OBJECT_AUG_LIST.FAR_TRANS_VARIANCE
            self.enable_obj_aug = sampler_cfg.OBJECT_AUG_LIST.OBJECT_AUGMENTATION
            self.translate_range = sampler_cfg.OBJECT_AUG_LIST.TRANS_RANGE
            if 'LIMIT_ANGLE' in sampler_cfg.OBJECT_AUG_LIST:
                self.limit_angle = sampler_cfg.OBJECT_AUG_LIST.LIMIT_ANGLE
            else:
                self.limit_angle = 1
            if 'FIX_ANGLE' in sampler_cfg.OBJECT_AUG_LIST:
                self.fix_angle = sampler_cfg.OBJECT_AUG_LIST.FIX_ANGLE
            else:
                self.fix_angle = False

            if 'CAND_TOP_K' in sampler_cfg.OBJECT_AUG_LIST:
                self.candidates_k = sampler_cfg.OBJECT_AUG_LIST.CAND_TOP_K
            else:
                self.candidates_k = 160
            for i, augs in enumerate(sampler_cfg.OBJECT_AUG_LIST.AUG_LIST):
                local_augment_list = [d_utils.PointcloudToTensor()]
                if 's-HPR' in augs:
                    self.m = sampler_cfg.OBJECT_AUG_LIST.OCC[0]
                    self.z_basis = sampler_cfg.OBJECT_AUG_LIST.OCC[1]
                    self.random = sampler_cfg.OBJECT_AUG_LIST.RANDOM
                    self.mirror_info = sampler_cfg.OBJECT_AUG_LIST.MRR
                    self.iterative_construction = d_utils.IterativePointMirrorMixup(
                        root_path =self.root_path,
                        mirror_info=self.mirror_info,
                        db_infos=self.db_infos,
                        best_top_k=self.candidates_k
                    )

                    local_augment_list.append(
                        d_utils.SelfHiddenPointRemove(
                            m=self.m, z_basis=self.z_basis,
                            random=self.random))

                local_augment_list.append(d_utils.PointcloudLocalRangeFilter())
                self.object_transform.append(transforms.Compose(local_augment_list))
        else:
            self.enable_near_trans = False
            self.enable_far_trans = False
            self.enable_obj_aug = False

        self.random_placement = RandomPlacement(translate_range = self.translate_range,
                                                enable_near_trans=self.enable_near_trans,
                                                enable_far_trans=self.enable_far_trans,
                                                fix_angle=self.fix_angle)

        self.ext_radius = sampler_cfg.OBJECT_AUG_LIST.EXT_OCC[0]
        self.ext_z_basis = sampler_cfg.OBJECT_AUG_LIST.EXT_OCC[1]
        self.ext_occ = ExternalHiddenPointRemove(radius=self.ext_radius,
                                                 z_basis=self.ext_z_basis)
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

    def objects_augmentation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_objects_augmentation, config=config)

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

    # Section 3.2 : Random Placement
    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        if 'gt_boxes_mask' in data_dict.keys():
            gt_boxes_mask = data_dict['gt_boxes_mask']
            gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
            gt_names = data_dict['gt_names'][gt_boxes_mask]
        else:
            gt_boxes = data_dict['gt_boxes']
            gt_names = data_dict['gt_names']

        points = data_dict['points']

        obj_points_list = []
        adjusted_sampled_gt_boxes = []
        mask_idx = np.ones(len(total_valid_sampled_dict),dtype=bool)
        for idx, info in enumerate(total_valid_sampled_dict):

            # Section 3.1 : Iterative construction
            complete_dict = self.iterative_construction(data_dict=info)
            adjusted_sampled_gt_boxes.append(complete_dict['bbox'].reshape(1,-1))

            # Section 3.3 : HPR Occlusion pipeline (s-HPR)
            new_dict = self.random_objects_augmentation(complete_dict)
            new_points = new_dict['points']

            if new_points.shape[0] < self.mask_min_points[info['name']]:
                mask_idx[idx] = False
                continue

            obj_points_list.append(new_points)

        if len(obj_points_list) < 2:
            return data_dict

        obj_points_all = np.concatenate(obj_points_list, axis=0)
        adjusted_sampled_gt_boxes = np.concatenate(adjusted_sampled_gt_boxes, axis=0)
        total_valid_sampled_dict = np.array(total_valid_sampled_dict)[mask_idx].tolist()

        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])
        sampled_gt_boxes = adjusted_sampled_gt_boxes[mask_idx]

        points = box_utils.remove_points_in_boxes3d(points, sampled_gt_boxes[:, 0:7])
        points = np.concatenate([obj_points_all, points], axis=0)

        if gt_names.shape[0] == 0:
            gt_names = sampled_gt_names
        else:
            gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)

        if gt_boxes.shape[0] == 0:
            gt_boxes = sampled_gt_boxes
        else:
            gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        assert gt_boxes.shape[0] == gt_names.shape[0]

        # add fitting extreme mask boxes
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    # Section 3.1 & 3 : HPR Occlusion(e-HPR)
    def global_hiden_point_removal(self, data_dict):

        data_dict = self.ext_occ(data_dict)
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
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)

            if int(sample_group['sample_num']) > 0:
                valid_sampled_dict = []

                sampled_dict = copy.deepcopy(self.sample_with_fixed_number(class_name, sample_group))
                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                #Section 3.2 : Create the information of Random Placement
                sampled_boxes = self.random_placement(sampled_boxes)

                #Section 3.2 : Collision avoidance
                if existed_boxes.shape[0] == 0:
                    valid_mask = np.arange(sampled_boxes.shape[0])
                else:
                    valid_mask = self.iou_base_valid_mask(sampled_boxes[:,:7], existed_boxes)

                for x in valid_mask:
                    sampled_dict[x]['box3d_lidar'] = sampled_boxes[x]
                    valid_sampled_dict.append(sampled_dict[x])

                valid_sampled_boxes = sampled_boxes[valid_mask]
                if existed_boxes.shape[0] == 0:
                    existed_boxes = copy.deepcopy(valid_sampled_boxes[:,:7])
                else:
                    existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:,:7]), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        if total_valid_sampled_dict.__len__() > 0:
            # Section 3.2 : Random Placement
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

            # Section 3.3 : HPR occlusion(e-HPR)
            if self.use_ext_occlusion:
                data_dict = self.global_hiden_point_removal(data_dict)

        if 'gt_boxes_mask' in data_dict.keys():
            data_dict.pop('gt_boxes_mask')
        return data_dict

    # Section 3.3 :HPR Occlusion(s-HPR)
    def random_objects_augmentation(self, data_dict=None):

        transform_sel = np.random.randint(0, len(self.object_transform))
        data_dict = self.object_transform[transform_sel](data_dict)
        return data_dict
