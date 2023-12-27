import paddle
from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans
import numpy as np
import glob, json, time


class DataSet(paddle.io.Dataset):

    def __init__(self, __C):
        self.__C = __C
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')
        self.stat_ques_list = json.load(open(__C.QUESTION_PATH['train'], 'r'))[
            'questions'] + json.load(open(__C.QUESTION_PATH['val'], 'r'))[
            'questions'] + json.load(open(__C.QUESTION_PATH['test'], 'r'))[
            'questions'] + json.load(open(__C.QUESTION_PATH['vg'], 'r'))[
            'questions']
        self.ques_list = []
        self.ans_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()
        print('== Dataset size:', self.data_size)
        if self.__C.PRELOAD:
            print('==== Pre-Loading features ...')
            time_start = time.time()
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print('==== Finished in {}s'.format(int(time_end - time_start)))
        else:
            self.iid_to_img_feat_path = img_feat_path_load(self.
                img_feat_path_list)
        self.qid_to_ques = ques_load(self.ques_list)
        self.token_to_ix, self.pretrained_emb = tokenize(self.
            stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8),
            self.ans_size)
        print('Finished!')
        print('')

    def __getitem__(self, idx):
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)
        if self.__C.RUN_MODE in ['train']:
            # import ipdb
            # ipdb.set_trace()
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)
            ans_iter = proc_ans(ans, self.ans_to_ix)
        else:
            ques = self.ques_list[idx]
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)
        paddle.disable_static()
        return paddle.to_tensor(data=img_feat_iter), paddle.to_tensor(data=
            ques_ix_iter), paddle.to_tensor(data=ans_iter)

    def __len__(self):
        return self.data_size
