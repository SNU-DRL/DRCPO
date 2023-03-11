import torch
import numpy as np
from ...utils import common_utils
import open3d as o3d
import math


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

class PointcloudToTensor(object):
    def __call__(self, data_dict):
        points = data_dict['points']
        data_dict['aug_info'] = []
        data_dict['points']  = torch.from_numpy(points).float()
        return data_dict

class PointcloudLocalRangeFilter(object):
    def __call__(self, data_dict):
        points = data_dict['points']
        if not 'box_range' in data_dict:
            return data_dict

        if points.shape[0] == 0:
            return data_dict

        box_range = data_dict['box_range']

        in_range_flags = ((points[:, 0] > box_range[0])
                          & (points[:, 1] > box_range[1])
                          & (points[:, 2] > box_range[2])
                          & (points[:, 0] < box_range[3])
                          & (points[:, 1] < box_range[4])
                          & (points[:, 2] < box_range[5]))

        new_points = points[in_range_flags]
        data_dict['points'] = new_points

        return data_dict

"""
3.1 Iterative Construction : Self-Mirroring for Constructing a Whole-body Object
"""
class PointMirror(object):
    def __init__(self, mirror_info):
        self.mirror_info = dict()
        for info in mirror_info:
            self.mirror_info.update({info['NAME']:info['AXIS']})

    def __call__(self, data_dict):

        points = data_dict['points']
        name = data_dict['name']
        if points.shape[0] == 0:
            data_dict['mask'] = False
            return data_dict

        if name in self.mirror_info.keys():
            mirror_axis = self.mirror_info[name]
            points_list = [points]
            for cur_axis in mirror_axis:
                assert cur_axis in ['x', 'y']
                if cur_axis == 'x':
                    mirrored_points = np.concatenate([points[..., 0:1], -points[..., 1:2], points[..., 2:3], points[..., 3:4]], axis=-1)
                if cur_axis == 'y':
                    mirrored_points = np.concatenate([-points[..., 0:1], points[..., 1:2], points[..., 2:3], points[..., 3:4]], axis=-1)
                points_list.append(mirrored_points)
            data_dict['points'] = np.concatenate(points_list, axis=0)
        else:
            data_dict['mask'] = True
            data_dict['points'] = points
        data_dict['mask'] = True
        return data_dict


"""
3.1 Iterative Construction : Constructing a Whole-body Object
"""
class IterativePointMirrorMixup(object):
    def __init__(self, root_path, mirror_info, db_infos,
                 occ_threshold=0.84, max_inter=10, num_point_features=4,
                 best_top_k=160, mean_creteria=True, lamda=500):

        self.root_path = root_path
        self.max_inter = max_inter
        self.db_infos = db_infos
        self.num_point_features=num_point_features
        self.occ_threshold = occ_threshold
        self.do_mirror = PointMirror(mirror_info=mirror_info)
        self.best_top_k = best_top_k
        self.lamda = lamda
        self.mean_creteria = mean_creteria

    def __call__(self, data_dict):
        name = data_dict['name']
        file_path = self.root_path / data_dict['path']
        obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
            [-1, self.num_point_features])

        angle = data_dict['box3d_lidar'][7]
        data_dict['points'] = np.concatenate([self.object_rotate_points_along_z(obj_points[:, :3], -angle), obj_points[:, 3:]],
                                    axis=1)

        mixed_ponints = self.do_mirror(data_dict)['points']

        top_k = min(self.best_top_k, len(data_dict['mix_candidates_idx']))
        self.max_inter = min(self.max_inter, top_k)
        rand_choices = np.random.choice(data_dict['mix_candidates_idx'][:top_k], size=self.max_inter, replace=False)
        part_ranges = data_dict['part_ranges']
        threshold = np.mean(data_dict['occupancy'][data_dict['occupancy'] > 0.0])

        for choice in rand_choices:
            mix_infos = self.db_infos[name][choice]
            file_path = self.root_path / mix_infos['path']
            mix_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.num_point_features])
            mix_dict = dict(name=name,points=mix_points)
            angle = mix_infos['box3d_lidar'][6]
            mix_dict['points'] = np.concatenate(
                [self.object_rotate_points_along_z(mix_points[:, :3], -angle), mix_points[:, 3:]],
                axis=1)

            mrr_tgt_points = self.do_mirror(mix_dict)['points']

            occupancy_rate = []
            mixed_ponints = np.concatenate([mixed_ponints, mrr_tgt_points], axis=0)

            if self.mean_creteria:
                new_occmap = data_dict['occupancy'] + mix_infos['occupancy']
                fill_part_rate = np.sum(new_occmap > threshold)/len(part_ranges)
                data_dict['occupancy'] = new_occmap
            else:
                threshold = np.mean(mix_infos['occupancy']) * self.lamda / mixed_ponints.shape[0]
                for part in part_ranges:
                    occupancy_rate.append(np.sum(self.in_range_3d(mixed_ponints, part)) / mixed_ponints.shape[0])
                occupancy_rate = np.array(occupancy_rate)
                fill_part_rate = np.sum(occupancy_rate > threshold) / len(part_ranges)

            if fill_part_rate > self.occ_threshold:
                break

        point_size = np.max(mixed_ponints, axis=0)[:3] - np.min(mixed_ponints, axis=0)[:3]
        new_size = np.max(np.stack([data_dict['box3d_lidar'][3:6], np.ceil(point_size * 100) / 100], axis=0), axis=0)
        data_dict['box3d_lidar'][3:6] = new_size

        data_dict = dict(points=mixed_ponints, bbox=data_dict['box3d_lidar'][:7], name=name)

        return data_dict

    def in_range_3d(self, points, box_range):
        in_range_flags = ((points[:, 0] > box_range[0])
                          & (points[:, 1] > box_range[1])
                          & (points[:, 2] > box_range[2])
                          & (points[:, 0] < box_range[3])
                          & (points[:, 1] < box_range[4])
                          & (points[:, 2] < box_range[5]))

        return in_range_flags

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

"""
3.2 Random Placement 
"""
class RandomPlacement(object):
    def __init__(self, translate_range, enable_near_trans, enable_far_trans, fix_angle):
        self.translate_range = translate_range
        self.enable_near_trans = enable_near_trans
        self.enable_far_trans = enable_far_trans
        self.fix_angle = fix_angle

    def __call__(self, sampled_boxes):
        n = sampled_boxes.shape[0]
        if self.enable_near_trans:
            noise_translate = np.array(
                [
                    [np.random.uniform(-sampled_boxes[i, 3], sampled_boxes[i, 3], 1),
                     np.random.uniform(-sampled_boxes[i, 4], sampled_boxes[i, 4], 1)] for i in range(n)
                ]).reshape(n, -1)
            sampled_boxes[:, 0:2] += noise_translate
        elif self.enable_far_trans:
            noise_translate = np.array(
                [
                    np.random.uniform(self.translate_range[0], self.translate_range[2], n),
                    np.random.uniform(self.translate_range[1], self.translate_range[3], n),
                ]).T
            sampled_boxes[:, 0:2] = noise_translate
        if self.fix_angle == False:
            rand_angle = np.random.uniform(-np.pi, np.pi, n)
            sampled_boxes = np.concatenate([sampled_boxes, sampled_boxes[:, 6].reshape(n, -1)], axis=1)
            for i in range(n):
                sampled_boxes[i, 6] = rand_angle[i]
        else:
            sampled_boxes = np.concatenate([sampled_boxes, sampled_boxes[:, 6].reshape(n, -1)], axis=1)

        return sampled_boxes

"""
3.3 HPR Occlusion : s-HPR method
"""
class SelfHiddenPointRemove(object):
    def __init__(self, m=200, min_num_points=10, z_basis=2,
                 random=False):
        self.m = m
        self.min_num_points = min_num_points
        self.pcd = o3d.geometry.PointCloud()
        self.z_basis = z_basis
        self.random = random
        self.max_distance = math.sqrt(70.4 ** 2 + 40 ** 2)

    def __call__(self, data_dict):

        if not type(data_dict['points']) is np.ndarray:
            points = data_dict['points'].numpy()
        else:
            points = data_dict['points']

        if points.shape[0] < self.min_num_points:
            data_dict['mask'] = False
            return data_dict

        r_angle = np.array([data_dict['bbox'][6]])
        r_points = common_utils.rotate_points_along_z(points[np.newaxis,:,:], r_angle)[0]

        self.pcd.points = o3d.utility.Vector3dVector(r_points[:, :3])
        coord_max = np.max(points[:, :3], axis=0)
        coord_min = np.min(points[:, :3], axis=0)
        coord_dis = coord_max - coord_min

        if self.random:
            z_basis = np.random.choice([0,1])
            m = np.random.choice([200,400])
        else:
            z_basis = self.z_basis
            m = self.m

        camera_loc = np.concatenate([-data_dict['bbox'][0:2],np.array([z_basis])],axis=0).tolist()
        radius = math.sqrt(coord_dis[0]**2 + coord_dis[1]**2)*m

        _, pt_map = self.pcd.hidden_point_removal(camera_loc, radius)
        occluded_points = r_points[np.array(pt_map),:]

        occluded_points[:, :3] += data_dict['bbox'][:3]

        data_dict['points'] = torch.from_numpy(occluded_points).float()

        if data_dict['points'].shape[0] < int(self.min_num_points):
            data_dict['mask'] = False
        else:
            data_dict['mask'] = True

        return data_dict

"""
3.3 HPR Occlusion : e-HPR method
"""
class ExternalHiddenPointRemove(object):
    def __init__(self, radius=100000, box_regeneration=False, box_margin=0.001, min_num_points=5,
                 x_basis=0.0,y_basis=0.0,z_basis=0.0, z_random=False):
        self.radius = radius
        self.margin = box_margin
        self.min_num_points = min_num_points
        self.box_regeneration = box_regeneration
        self.pcd = o3d.geometry.PointCloud()
        self.x_basis = x_basis
        self.y_basis = y_basis
        self.z_basis = z_basis
        self.z_random = z_random


    def __call__(self, data_dict):
        if not type(data_dict['points']) is np.ndarray:
            points = data_dict['points'].numpy()
        else:
            points = data_dict['points']


        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        coord_max = np.max(points[:, :3], axis=0)
        coord_min = np.min(points[:, :3], axis=0)
        coord_dis = coord_max - coord_min

        if self.z_random:
            camera_loc = [0, 0, np.random.uniform(0,3)]
        else:
            camera_loc = [0, 0, self.z_basis]

        _, pt_map = self.pcd.hidden_point_removal(camera_loc, self.radius)
        occluded_points = points[np.array(pt_map),:]

        data_dict['points'] = torch.from_numpy(occluded_points).float()
        return data_dict
