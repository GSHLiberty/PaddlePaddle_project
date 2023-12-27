import sys
sys.path.append('/data2/gsh/paddlepaddle/Guyu/utils')
import paddle_aux
import paddle


class LabelSmoothing(paddle.nn.Layer):
    """Implement label smoothing."""

    def __init__(self, device, size, padding_idx, label_smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        self.size = size
        self.device = device
        self.smoothing_value = label_smoothing / (size - 2)
        # self.one_hot = paddle.full(shape=(1, size), fill_value=self.smoothing_value).to(device)
        self.one_hot = paddle.full(shape=(1, size), fill_value=self.smoothing_value)
        # self.one_hot = paddle.to_tensor(self.one_hot, place=paddle.CUDAPlace(0))
        self.one_hot = paddle.to_tensor(self.one_hot, place=paddle.CPUPlace())
        self.one_hot[0, self.padding_idx] = 0
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        real_size = output.shape[1]
        if real_size > self.size:
            real_size -= self.size
        else:
            real_size = 0
        model_prob = paddle.tile(self.one_hot, repeat_times=[target.shape[0], 1])
        # model_prob = self.one_hot.repeat(target.shape[0], 1)
        if real_size > 0:
            ext_zeros = paddle.full(shape=(model_prob.shape[0], real_size),
                fill_value=self.smoothing_value).to(self.device)
            model_prob = paddle.concat(x=(model_prob, ext_zeros), axis=-1)
        model_prob.put_along_axis_(axis=1, indices=target, values=self.
            confidence)
        mask = target == self.padding_idx  # Create a mask where the condition is true
        model_prob = paddle.where(mask, paddle.zeros_like(model_prob), model_prob)  # Apply the mask
        return paddle.nn.functional.kl_div(input=output, label=model_prob, reduction='sum')
