import paddle


def make_mask(feature):
    return (paddle.sum(x=paddle.abs(x=feature), axis=-1) == 0).unsqueeze(axis=1
        ).unsqueeze(axis=2)
