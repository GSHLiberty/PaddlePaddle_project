import paddle
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask


class Adapter(BaseAdapter):

    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def bbox_proc(self, bbox):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1]
            )
        return paddle.concat(x=(bbox, area.unsqueeze(axis=2)), axis=-1)

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = paddle.nn.Linear(in_features=5, out_features
                =__C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = paddle.nn.Linear(in_features=imgfeat_linear_size,
            out_features=__C.HIDDEN_SIZE)

    def clevr_init(self, __C):
        self.grid_linear = paddle.nn.Linear(in_features=__C.FEAT_SIZE[
            'clevr']['GRID_FEAT_SIZE'][1], out_features=__C.HIDDEN_SIZE)

    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        img_feat_mask = make_mask(frcn_feat)
        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = paddle.concat(x=(frcn_feat, bbox_feat), axis=-1)
        img_feat = self.frcn_linear(frcn_feat)
        return img_feat, img_feat_mask

    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']
        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)
        return img_feat, img_feat_mask
