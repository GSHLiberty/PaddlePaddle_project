import paddle
import paddle.nn
from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED


class AttFlat(paddle.nn.Layer):

    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C
        self.mlp = MLP(in_size=__C.HIDDEN_SIZE, mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES, dropout_r=__C.DROPOUT_R, use_relu=True)
        self.linear_merge = paddle.nn.Linear(in_features=__C.HIDDEN_SIZE *
            __C.FLAT_GLIMPSES, out_features=__C.FLAT_OUT_SIZE)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = paddle.where(x_mask.squeeze(axis=1).squeeze(axis=1).unsqueeze(axis=2), paddle.full_like(att, -1e9), att)
        att = paddle.nn.functional.softmax(x=att, axis=1)
        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(paddle.sum(x=att[:, :, i:i + 1] * x, axis=1))
        x_atted = paddle.concat(x=att_list, axis=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


class Net(paddle.nn.Layer):

    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.embedding = paddle.nn.Embedding(num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE)
        if __C.USE_GLOVE:
            paddle.assign(paddle.to_tensor(data=pretrained_emb), output=
                self.embedding.weight)
        self.lstm = paddle.nn.LSTM(input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE, num_layers=1, time_major=not True,
            direction='forward')
        self.img_feat_linear = paddle.nn.Linear(in_features=__C.
            IMG_FEAT_SIZE, out_features=__C.HIDDEN_SIZE)
        self.backbone = MCA_ED(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = paddle.nn.Linear(in_features=__C.FLAT_OUT_SIZE,
            out_features=answer_size)

    def forward(self, img_feat, ques_ix):
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(axis=2))
        img_feat_mask = self.make_mask(img_feat)
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)
        img_feat = self.img_feat_linear(img_feat)
        lang_feat, img_feat = self.backbone(lang_feat, img_feat, lang_feat_mask, img_feat_mask)
        lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)
        img_feat = self.attflat_img(img_feat, img_feat_mask)
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = paddle.nn.functional.sigmoid(x=self.proj(proj_feat))
        return proj_feat

    def make_mask(self, feature):
        return (paddle.sum(x=paddle.abs(x=feature), axis=-1) == 0).unsqueeze(axis=1).unsqueeze(axis=2)
