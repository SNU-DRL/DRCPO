from .detector3d_template import Detector3DTemplate
import torch

class PVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        if torch.sum(batch_dict['rois'][batch_dict['rois'] != 0.0]) == 0:
            print('after roi_head assign target : roi is not existed')
        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if torch.sum(batch_dict['rois'][batch_dict['rois'] != 0.0]) == 0:
            print('after roi_head assign target : roi is not existed')
        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)

            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]

        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            disp_dict.update(dict(Car=torch.sum(batch_dict['gt_boxes'][:,:,7]==1.0).item()/batch_dict['gt_boxes'].shape[0],
                                  Pedestrian=torch.sum(batch_dict['gt_boxes'][:,:,7]==2.0).item()/batch_dict['gt_boxes'].shape[0],
                                  Cyclist=torch.sum(batch_dict['gt_boxes'][:,:,7]==3.0).item()/batch_dict['gt_boxes'].shape[0]))
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
