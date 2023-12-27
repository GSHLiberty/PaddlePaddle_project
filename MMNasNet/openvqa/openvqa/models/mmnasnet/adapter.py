import paddle
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask


class Adapter(BaseAdapter):

    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def relation_embedding(self, f_g):
        x_min, y_min, x_max, y_max = paddle.chunk(x=f_g, chunks=4, axis=2)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = x_max - x_min + 1.0
        h = y_max - y_min + 1.0
        x = cx
        perm_5 = list(range(x.ndim))
        perm_5[-1] = -2
        perm_5[-2] = -1
        delta_x = cx - x.transpose(perm=perm_5)
        delta_x = paddle.clip(x=paddle.abs(x=delta_x / w), min=0.001)
        delta_x = paddle.log(x=delta_x)
        x = cy
        perm_6 = list(range(x.ndim))
        perm_6[-1] = -2
        perm_6[-2] = -1
        delta_y = cy - x.transpose(perm=perm_6)
        delta_y = paddle.clip(x=paddle.abs(x=delta_y / h), min=0.001)
        delta_y = paddle.log(x=delta_y)
        x = w
        perm_7 = list(range(x.ndim))
        perm_7[-1] = -2
        perm_7[-2] = -1
        delta_w = paddle.log(x=w / x.transpose(perm=perm_7))
        x = h
        perm_8 = list(range(x.ndim))
        perm_8[-1] = -2
        perm_8[-2] = -1
        delta_h = paddle.log(x=h / x.transpose(perm=perm_8))
        size = delta_h.shape
        delta_x = paddle.reshape(delta_x, shape=[size[0], size[1], size[2], 1])
        delta_y = paddle.reshape(delta_y, shape=[size[0], size[1], size[2], 1])
        delta_w = paddle.reshape(delta_w, shape=[size[0], size[1], size[2], 1])
        delta_h = paddle.reshape(delta_h, shape=[size[0], size[1], size[2], 1])
        position_mat = paddle.concat(x=(delta_x, delta_y, delta_w, delta_h),
            axis=-1)
        return position_mat

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = paddle.nn.Linear(in_features=5, out_features
                =__C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = paddle.nn.Linear(in_features=imgfeat_linear_size,
            out_features=__C.HIDDEN_SIZE)

    def gqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = paddle.nn.Linear(in_features=5, out_features
                =__C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = paddle.nn.Linear(in_features=imgfeat_linear_size,
            out_features=__C.HIDDEN_SIZE)
        if __C.USE_AUX_FEAT:
            self.grid_linear = paddle.nn.Linear(in_features=__C.FEAT_SIZE[
                'gqa']['GRID_FEAT_SIZE'][1], out_features=__C.HIDDEN_SIZE)

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
        rel_embed = self.relation_embedding(bbox_feat)
        return img_feat, rel_embed, img_feat_mask

    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']
        img_feat_mask = make_mask(frcn_feat)
        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = paddle.concat(x=(frcn_feat, bbox_feat), axis=-1)
        img_feat = self.frcn_linear(frcn_feat)
        if self.__C.USE_AUX_FEAT:
            grid_feat_mask = make_mask(grid_feat)
            img_feat_mask = paddle.concat(x=(img_feat_mask, grid_feat_mask),
                axis=-1)
            grid_feat = self.grid_linear(grid_feat)
            img_feat = paddle.concat(x=(img_feat, grid_feat), axis=1)
        rel_embed = self.relation_embedding(bbox_feat)
        return img_feat, rel_embed, img_feat_mask

    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']
        img_feat_mask = make_mask(grid_feat)
        img_feat = self.grid_linear(grid_feat)
        rel_embed = self.relation_embedding(bbox_feat)
        return img_feat, rel_embed, img_feat_mask
