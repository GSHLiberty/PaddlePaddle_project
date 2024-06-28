import paddle
from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.TRAR.trar import TRAR_ED
from openvqa.models.TRAR.adapter import Adapter


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
        # Ensure x_mask is properly squeezed and unsqueezed without errors, and matches the dimensions of att
        x_mask = x_mask.squeeze(axis=1).squeeze(axis=1).unsqueeze(axis=2)

        # Ensure that the scalar used with paddle.where matches the dtype of att
        negative_infinity = paddle.full_like(att, fill_value=-1000000000.0)

        # Use paddle.where to apply the mask
        att = paddle.where(x_mask, att, negative_infinity)

        # Applying softmax to normalize the attention scores
        att = paddle.nn.functional.softmax(att, axis=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            # Multiply attention scores with the input tensor and sum over the axis
            att_list.append(paddle.sum(att[:, :, i:i + 1] * x, axis=1))

        # Concatenate the attended feature vectors
        x_atted = paddle.concat(att_list, axis=1)

        # Pass the concatenated features through a linear layer
        x_atted = self.linear_merge(x_atted)
        
        return x_atted



class Net(paddle.nn.Layer):

    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C
        self.embedding = paddle.nn.Embedding(num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE)
        if __C.USE_GLOVE:
            paddle.assign(paddle.to_tensor(data=pretrained_emb), output=
                self.embedding.weight)
        self.lstm = paddle.nn.LSTM(input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE, num_layers=1, time_major=not True,
            direction='forward')
        self.adapter = Adapter(__C)
        self.backbone = TRAR_ED(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = paddle.nn.Linear(in_features=__C.FLAT_OUT_SIZE,
            out_features=answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        lang_feat_mask = make_mask(ques_ix.unsqueeze(axis=2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)
        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        lang_feat, img_feat = self.backbone(lang_feat, img_feat,
            lang_feat_mask, img_feat_mask)
        lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)
        img_feat = self.attflat_img(img_feat, img_feat_mask)
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)
        return proj_feat
