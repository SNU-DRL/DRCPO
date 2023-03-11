import copy
import pickle
import os
import torch

from pathlib import Path
import numpy as np

from tqdm import tqdm
import pandas as pd

from pcdet.ops.iou3d_nms import iou3d_nms_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_POINT_FEATURES = 4

def filter_by_difficulty(db_infos, removed_difficulty):
    new_db_infos = [
       info for info in db_infos
       if info['difficulty'] not in removed_difficulty
    ]
    return new_db_infos

def filter_by_min_points(db_infos, min_num):
    min_num = int(min_num)
    filtered_infos = []
    for info in db_infos:
        if info['num_points_in_gt'] >= min_num:
            filtered_infos.append(info)

    return filtered_infos

def get_normalized_cloud(obj_pnt_fpath, gt_box):
    pnts = np.fromfile(str(obj_pnt_fpath), dtype=np.float32).reshape([-1,4])
    pnts = np.concatenate([single_rotate_points_along_z(pnts[:,:3], -gt_box[6]), pnts[:,3:]], axis=1)
    return pnts

def remove_bottom(pnts, gt_box, bottom):
    if bottom == 0.0:
        return pnts
    zthresh =  - gt_box[5] / 2 + bottom
    keep_bool = pnts[:, 2] > zthresh
    return pnts[keep_bool]


def save_pnts_box(pnts, box, name, path):
    template = {
        "name": name,
        "points": pnts,
        "box": box
    }
    with open(path, 'wb') as f:
        pickle.dump(template, f)

def coords3inds(coords, ny, nx):
    gperm1 = nx * ny
    gperm2 = nx
    zdim = coords[:, 2] * gperm1
    ydim = coords[:, 1] * gperm2
    xdim = coords[:, 0]
    inds = zdim + ydim + xdim
    return inds.astype(np.int32)

def mirror(pnts):
    mirror_pnts = np.concatenate([pnts[...,0:1], -pnts[...,1:2], pnts[...,2:3], pnts[...,3:4]], axis=-1)
    return np.concatenate([pnts, mirror_pnts], axis=0)

def single_rotate_points_along_z(points, angle):
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
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=0).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    return points_rot

def get_iou(box_tensor, box_dims_list):
    limit = len(box_dims_list)
    start = 0
    iou3d_lst = []
    for i in range(11):
        end = min(start + limit // 10, limit)
        iou3d = iou3d_nms_utils.boxes_iou3d_gpu(box_tensor[start:end, :], box_tensor)
        iou3d_lst.append(iou3d)
        start = end
    iou3d = torch.cat(iou3d_lst, dim=0)
    print("iou3d", iou3d.shape)
    return iou3d

def toTensor(sample):
    return torch.from_numpy(sample).float().to("cuda")

def in_range_3d(points, box_range):
    in_range_flags = ((points[:, 0] > box_range[0])
                          & (points[:, 1] > box_range[1])
                          & (points[:, 2] > box_range[2])
                          & (points[:, 0] < box_range[3])
                          & (points[:, 1] < box_range[4])
                          & (points[:, 2] < box_range[5]))
    return in_range_flags

def bbox_generation(points, margin=0.0):
    coord_min = np.min(points[:,:3], axis=0) - margin
    coord_max = np.max(points[:,:3], axis=0) + margin

    new_box = np.zeros(7)
    new_box[:3] = (coord_min + coord_max) / 2
    new_box[3:6] = coord_max - coord_min

    return new_box

def remove_outofbox(pnts, box):
    dim = box[3:6]
    point_in_box_mask = np.logical_and(pnts[...,:3] <= dim * 0.5, pnts [...,:3]>= -dim * 0.5)

    # N, M
    point_in_box_mask = np.prod(point_in_box_mask.astype(np.int8), axis=-1, dtype=bool)
    return pnts[point_in_box_mask, :]

def generate_aug_patch_cube(patch_range, div, type='Car'):
    xrange_min, yrange_min, zrange_min, xrange_max, yrange_max, zrange_max = patch_range
    div_x, div_y, div_z = div

    x_min = (np.linspace(xrange_min, xrange_max, div_x + 1))[:div_x]
    x_max = (np.linspace(xrange_min, xrange_max, div_x + 1))[1:div_x + 1]
    y_min = (np.linspace(yrange_min, yrange_max, div_y + 1))[:div_y]
    y_max = (np.linspace(yrange_min, yrange_max, div_y + 1))[1:div_y + 1]
    z_min = (np.linspace(zrange_min, zrange_max, div_z + 1))[:div_z]
    z_max = (np.linspace(zrange_min, zrange_max, div_z + 1))[1:div_z + 1]

    box_ranges =[]
    strong_interest_idx = []
    idx = 0
    for i in range(len(z_max)):
        for j in range(len(x_max)):
            for k in range(len(y_max)):
                if j == 0 or j == len(x_max)-1:
                    strong_interest_idx.append(idx)
                box_range = [x_min[j], y_min[k], z_min[i], x_max[j], y_max[k], z_max[i]]
                box_ranges.append(box_range)
                idx += 1

    return box_ranges, strong_interest_idx

def mrr_and_extract_vox_occupation(root_path=None, splits=['train','val'],
                                   type='Car', apply_mirror=True, dataset_name='kitti'):
    min_points = dict(Car=5, Pedestrian=5, Cyclist=5)
    difficulty = [-1]
    if type == 'Car':
        div = [10,4,3]
    elif type == 'Cyclist':
        div = [4,2,3]
    elif type == 'Pedestrian':
        div = [2,2,3]

    db_info_list, box_dims_list, point_list, mirrored_point_list = [],[],[],[]
    occupancy_list,sparsity_list=[],[]

    for split in splits:
        db_info_save_path = Path(root_path) / ('{}_dbinfos_{}.pkl'.format(dataset_name, split))
        with open(db_info_save_path, 'rb') as f:
            all_db_infos = pickle.load(f)[type]
            all_db_infos = filter_by_difficulty(all_db_infos, difficulty)
            min_num = min_points[type]
            all_db_infos = filter_by_min_points(all_db_infos, min_num)
            all_db_pd = pd.DataFrame(all_db_infos)
        max_points_in_gt = all_db_pd['num_points_in_gt'].max()

        for k in range(len(all_db_infos)):
            info = all_db_infos[k]
            class_name = info['name']
            if class_name != type:
                continue
            gt_box = info['box3d_lidar']
            box_dims_list.append(np.concatenate([np.zeros_like(gt_box[0:3]), np.array(gt_box[3:6]), np.zeros_like(gt_box[6:7])], axis=-1))
            db_info_list.append(info)

            obj_pnt_fpath = root_path / info['path']
            obj_pnts = get_normalized_cloud(str(obj_pnt_fpath), gt_box)
            mirrored_obj_pnts = mirror(obj_pnts)
            point_list.append(obj_pnts)

            box_range = [-gt_box[3] / 2, -gt_box[4] / 2, -gt_box[5] / 2,
                         gt_box[3] / 2, gt_box[4] / 2, gt_box[5] / 2]
            part_ranges, strong_interest_idx = generate_aug_patch_cube(box_range, div)

            occupancy_rate = []
            for part in part_ranges:
                occupancy_rate.append(np.sum(in_range_3d(mirrored_obj_pnts, part)) / max_points_in_gt)
            occupancy_rate = np.array(occupancy_rate)
            all_db_infos[k]['occupancy'] = occupancy_rate
            all_db_infos[k]['part_ranges'] = part_ranges

            occupancy_list.append(occupancy_rate)
            if apply_mirror:
                n = mirrored_obj_pnts.shape[0]
                mirrored_point_list.append(mirrored_obj_pnts)
            else:
                n = obj_pnts.shape[0]
                mirrored_point_list.append(obj_pnts)

    return db_info_list, box_dims_list, point_list, mirrored_point_list, occupancy_list#, sparsity_list


def create_iterative_candidates_db(root_path, topk_min, class_names, alpha=0.8):

        class_names = class_names
        apply_mirror_lst = [True, False, True]
        dataset_name = 'kitti'
        root_path = root_path

        mixup_database_dict = {}
        for i, class_name in enumerate(class_names):
            # points and mirroring
            db_info_list, box_dims_list, point_list, mirrored_point_list, occupancy_list = mrr_and_extract_vox_occupation(
                root_path=root_path, splits=['train'], type=class_name, apply_mirror=apply_mirror_lst[i], dataset_name=dataset_name
            )
            box_tensor = torch.as_tensor(box_dims_list, device="cuda", dtype=torch.float32)
            iou3d = get_iou(box_tensor, box_dims_list)
            iou3d = iou3d.cuda()
            sorted_iou, best_iou_indices = torch.topk(iou3d, min(topk_min[i], len(iou3d)), dim=-1, sorted=True, largest=True)
            del iou3d
            torch.cuda.empty_cache()
            best_occupancy = np.array(occupancy_list)[np.array(best_iou_indices.cpu())]

            for c in tqdm(range(len(db_info_list))):
                cur_occ = db_info_list[c]['occupancy']
                cur_occ_candidates = best_occupancy[c]

                interest_map = cur_occ < np.mean(cur_occ)*alpha
                candidate_interest_map = cur_occ_candidates[:, interest_map]
                overlap = np.sum(candidate_interest_map, axis=1)

                cand, best_occ_over_indices= torch.topk(torch.Tensor(overlap), int(topk_min[i]/2), dim=0, sorted=True, largest=True)
                cand_indices = best_iou_indices[c][best_occ_over_indices].cpu().numpy()
                db_info_list[c]['mix_candidates_idx'] = cand_indices

            torch.cuda.empty_cache()
            mixup_database_dict.update({class_name:db_info_list})
        filepath = root_path / '{}_candidates_dbinfos_train.pkl'.format(dataset_name)
        with open(filepath, 'wb') as f:
            pickle.dump(mixup_database_dict, f)

# Section 3.1 : Indexing Candidate Objects (Pre-processing)
if __name__ == '__main__':
    topk_min = [800, 800, 400]
    obj_types = ['Car', 'Pedestrian', 'Cyclist']
    apply_mirror_lst = [True, False, True]
    dataset_name = 'kitti'
    ROOT_DIR = (Path(__file__).resolve().parent / '/home/svcapp/userdata').resolve()
    path = ROOT_DIR / 'data' / dataset_name

    create_iterative_candidates_db(root_path=path,
                                   topk_min=topk_min,
                                   class_names=obj_types)



