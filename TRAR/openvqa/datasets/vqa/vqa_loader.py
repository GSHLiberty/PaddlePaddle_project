import numpy as np
import glob, json, re, en_vectors_web_lg
from openvqa.core.base_dataset import BaseDataSet
from openvqa.utils.ans_punct import prep_ans


class DataSet(BaseDataSet):

    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C
        frcn_feat_path_list = glob.glob(__C.FEATS_PATH[__C.DATASET]['train'
            ] + '/*.npy') + glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] +
            '/*.npy') + glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] +
            '/*.npy')
        stat_ques_list = json.load(open(__C.RAW_PATH[__C.DATASET]['train'],
            'r'))['questions'] + json.load(open(__C.RAW_PATH[__C.DATASET][
            'val'], 'r'))['questions'] + json.load(open(__C.RAW_PATH[__C.
            DATASET]['test'], 'r'))['questions'] + json.load(open(__C.
            RAW_PATH[__C.DATASET]['vg'], 'r'))['questions']
        self.ques_list = []
        self.ans_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.RAW_PATH[__C.DATASET][
                split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.RAW_PATH[__C.DATASET][
                    split + '-anno'], 'r'))['annotations']
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()
        print(' ========== Dataset size:', self.data_size)
        self.iid_to_frcn_feat_path = self.img_feat_path_load(
            frcn_feat_path_list)
        self.qid_to_ques = self.ques_load(self.ques_list)
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_ques_list,
            __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)
        self.ans_to_ix, self.ix_to_ans = self.ans_stat(
            'openvqa/datasets/vqa/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size (occur more than {} times):'
            .format(8), self.ans_size)
        print('Finished!')
        print('')

    def img_feat_path_load(self, path_list):
        iid_to_path = {}
        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            iid_to_path[iid] = path
        return iid_to_path

    def ques_load(self, ques_list):
        qid_to_ques = {}
        for ques in ques_list:
            qid = str(ques['question_id'])
            qid_to_ques[qid] = ques
        return qid_to_ques

    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {'PAD': 0, 'UNK': 1, 'CLS': 2}
        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)
        for ques in stat_ques_list:
            words = re.sub('([.,\'!?\\"()*#:;])', '', ques['question'].lower()
                ).replace('-', ' ').replace('/', ' ').split()
            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)
        pretrained_emb = np.array(pretrained_emb)
        return token_to_ix, pretrained_emb

    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))
        return ans_to_ix, ix_to_ans

    def load_ques_ans(self, idx):
        if self.__C.RUN_MODE in ['train']:
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            iid = str(ans['image_id'])
            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
            ans_iter = self.proc_ans(ans, self.ans_to_ix)
            return ques_ix_iter, ans_iter, iid
        else:
            ques = self.ques_list[idx]
            iid = str(ques['image_id'])
            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)
            return ques_ix_iter, np.zeros(1), iid

    def load_img_feats(self, idx, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat = frcn_feat.astype(np.float32)
        frcn_feat_iter = self.proc_img_feat(frcn_feat, img_feat_pad_size=
            self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
        bbox_feat_iter = np.zeros(1)
        grid_feat_iter = np.zeros(1)
        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]
        img_feat = np.pad(img_feat, ((0, img_feat_pad_size - img_feat.shape
            [0]), (0, 0)), mode='constant', constant_values=0)
        return img_feat

    def proc_bbox_feat(self, bbox, img_shape):
        if self.__C.BBOX_NORMALIZE:
            bbox_nm = np.zeros((bbox.shape[0], 4), dtype=np.float32)
            bbox_nm[:, 0] = bbox[:, 0] / float(img_shape[1])
            bbox_nm[:, 1] = bbox[:, 1] / float(img_shape[0])
            bbox_nm[:, 2] = bbox[:, 2] / float(img_shape[1])
            bbox_nm[:, 3] = bbox[:, 3] / float(img_shape[0])
            return bbox_nm
        return bbox

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
        words = re.sub('([.,\'!?\\"()*#:;])', '', ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()
        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']
            if ix + 1 == max_token:
                break
        return ques_ix

    def get_score(self, occur):
        if occur == 0:
            return 0.0
        elif occur == 1:
            return 0.3
        elif occur == 2:
            return 0.6
        elif occur == 3:
            return 0.9
        else:
            return 1.0

    def proc_ans(self, ans, ans_to_ix):
        ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
        ans_prob_dict = {}
        for ans_ in ans['answers']:
            ans_proc = prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1
        if self.__C.LOSS_FUNC in ['kld']:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = ans_prob_dict[ans_] / 10.0
        else:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = self.get_score(ans_prob_dict
                        [ans_])
        return ans_score
