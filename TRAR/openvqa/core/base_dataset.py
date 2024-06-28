import paddle
import numpy as np
import glob, json, random
from openvqa.utils.feat_filter import feat_filter


class BaseDataSet(paddle.io.Dataset):

    def __init__(self):
        self.token_to_ix = None
        self.pretrained_emb = None
        self.ans_to_ix = None
        self.ix_to_ans = None
        self.data_size = None
        self.token_size = None
        self.ans_size = None

    def load_ques_ans(self, idx):
        raise NotImplementedError()

    def load_img_feats(self, idx, iid):
        raise NotImplementedError()

    def __getitem__(self, idx):
        ques_ix_iter, ans_iter, iid = self.load_ques_ans(idx)
        frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(
            idx, iid)
        return paddle.to_tensor(data=frcn_feat_iter), paddle.to_tensor(data
            =grid_feat_iter), paddle.to_tensor(data=bbox_feat_iter
            ), paddle.to_tensor(data=ques_ix_iter), paddle.to_tensor(data=
            ans_iter)

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)


class BaseAdapter(paddle.nn.Layer):

    def __init__(self, __C):
        super(BaseAdapter, self).__init__()
        self.__C = __C
        if self.__C.DATASET in ['vqa']:
            self.vqa_init(__C)
        elif self.__C.DATASET in ['clevr']:
            self.clevr_init(__C)
        else:
            exit(-1)

    def vqa_init(self, __C):
        raise NotImplementedError()

    def clevr_init(self, __C):
        raise NotImplementedError()

    def forward(self, frcn_feat, grid_feat, bbox_feat):
        feat_dict = feat_filter(self.__C.DATASET, frcn_feat, grid_feat,
            bbox_feat)
        if self.__C.DATASET in ['vqa']:
            return self.vqa_forward(feat_dict)
        elif self.__C.DATASET in ['clevr']:
            return self.clevr_forward(feat_dict)
        else:
            exit(-1)

    def vqa_forward(self, feat_dict):
        raise NotImplementedError()

    def clevr_forward(self, feat_dict):
        raise NotImplementedError()
