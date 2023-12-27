import paddle
"""
Basic or helper implementation.
"""


def apply_nd(fn, input):
    """
    Apply fn whose output only depends on the last dimension values
    to an arbitrary n-dimensional input.
    It flattens dimensions except the last one, applies fn, and then
    restores the original size.
    """
    x_size = input.shape
    x_flat = paddle.reshape(input, shape=[-1, x_size[-1]])
    output_flat = fn(x_flat)
    output_size = x_size[:-1] + (output_flat.shape[-1],)
    return paddle.reshape(output_flat, shape=output_size)


def affine_nd(input, weight, bias):
    """
    An helper function to make applying the "wx + b" operation for
    n-dimensional x easier.
    :param input: (Tensor) An arbitrary input data, whose size is
                  (d0, d1, ..., dn, input_dim)
    :param weight: (Tensor) A matrix of size (output_dim, input_dim)
    :param bias: (Tensor) A bias vector of size (output_dim,)
    :returns: The result of size (d0, ..., dn, output_dim)
    """
    input_size = input.shape
    input_flat = paddle.reshape(input, shape=[-1, input_size[-1]])
    bias_expand = bias.unsqueeze(axis=0).expand(shape=[input_flat.shape[0],
        bias.shape[0]])
    output_flat = paddle.addmm(input=bias_expand, x=input_flat, y=weight)
    output_size = input_size[:-1] + (weight.shape[1],)
    output = paddle.reshape(output_flat, shape=output_size)
    return output


def dot_nd(query, candidates):
    """
    Perform a dot product between a query and n-dimensional candidates.
    :param query: (Tensor) A vector to query, whose size is
                  (query_dim,)
    :param candidates: (Tensor) A n-dimensional tensor to be multiplied
                       by query, whose size is (d0, d1, ..., dn, query_dim)
    :returns: The result of the dot product, whose size is
              (d0, d1, ..., dn)
    """
    cands_size = candidates.shape
    cands_flat = paddle.reshape(candidates, shape=[-1, cands_size[-1]])
    output_flat = paddle.mv(x=cands_flat, vec=query)
    output = paddle.reshape(output_flat, shape=cands_size[:-1])
    return output


def convert_to_one_hot(indices, num_classes):
    """
    :param indices: (Tensor) A vector containing indices,
                    whose size is (batch_size,).
    :param num_classes: (Tensor) The number of classes, which would be
                        the second dimension of the resulting one-hot matrix.
    :returns: The one-hot matrix of size (batch_size, num_classes).
    """
    batch_size = indices.shape[0]
    indices = indices.unsqueeze(axis=1)
    one_hot = indices.data.new(batch_size, num_classes).zero_(
        ).put_along_axis_(axis=1, indices=indices.data, values=1)
    return one_hot


def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = paddle.nn.functional.softmax(x=logits, axis=1)
    if mask is not None:
        mask = mask.astype(dtype='float32')
        probs = probs * mask + eps
        probs = probs / probs.sum(axis=1, keepdim=True)
    return probs


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=(probs.max(axis=1), probs.argmax(
        axis=1))[1], num_classes=logits.shape[1])
    return one_hot


def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.
    :param logits: (Tensor) A un-normalized probability values,
                   which has the size (batch_size, num_classes)
    :param temperature: (float) A temperature parameter. The higher
                        the value is, the smoother the distribution is.
    :param mask: (Tensor, optional) If given, it masks the softmax
                 so that indices of '0' mask values are not selected.
                 The size is (batch_size, num_classes).
    :returns: The sampled output, which has the property explained above.
    """
    eps = 1e-20
    u = logits.data.new(*logits.shape).uniform_()
    gumbel_noise = -paddle.log(x=-paddle.log(x=u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = (y.max(axis=1), y.argmax(axis=1))[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.shape[1]
        ).astype(dtype='float32')
    y = (y_hard - y).detach() + y
    return y


def sequence_mask(seq_length, max_length=None):
    if max_length is None:
        max_length = seq_length.data.max()
    batch_size = seq_length.shape[0]
    seq_range = paddle.arange(start=0, end=max_length).astype(dtype='int64')
    seq_range_expand = seq_range.unsqueeze(axis=0).expand(shape=[batch_size,
        max_length])
    if 'gpu' in str(seq_length.place):
        seq_range_expand = seq_range_expand
    seq_length_expand = seq_length.unsqueeze(axis=1).expand_as(y=
        seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """
    Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    :param inputs: (Tensor) padded batch of variable length sequences.
    :param lengths: (list[int]) list of sequence lengths
    :param batch_first: (bool, optional) if True, inputs should be B x T x *.
    :returns: A Tensor with the same size as inputs, but with each sequence
              reversed according to its length.
    """
    if not batch_first:
        x = inputs
        perm_21 = list(range(x.ndim))
        perm_21[0] = 1
        perm_21[1] = 0
        inputs = x.transpose(perm=perm_21)
    if inputs.shape[0] != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.shape[1])) for _ in range(inputs.
        shape[0])]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length - 1::-1]
    reversed_indices = paddle.to_tensor(data=reversed_indices, dtype='int64'
        ).unsqueeze(axis=2).expand_as(y=inputs)
    if 'gpu' in str(inputs.place):
        device = inputs.place.gpu_device_id()
        reversed_indices = reversed_indices
    reversed_inputs = paddle.take_along_axis(arr=inputs, axis=1, indices=
        reversed_indices)
    if not batch_first:
        x = reversed_inputs
        perm_22 = list(range(x.ndim))
        perm_22[0] = 1
        perm_22[1] = 0
        reversed_inputs = x.transpose(perm=perm_22)
    return reversed_inputs


if __name__ == '__main__':
    print('reverse_padded_sequence')
    inputs = paddle.to_tensor(data=[[[1], [2], [3], [0]], [[4], [4], [0], [
        0]], [[3], [5], [6], [8]]], dtype='int64')
    lengths = [3, 2, 4]
    batch_first = True
    result = reverse_padded_sequence(inputs, lengths, batch_first)
    print(result)
    print('masked_softmax')
    logits = paddle.to_tensor(data=[[1, 2], [3, 2], [1, 5]], dtype='float32')
    mask = paddle.to_tensor(data=[[1, 1], [1, 0], [0, 0]], dtype='int64')
    result = masked_softmax(logits, mask)
    print(result)
    print('sequence_mask')
    seq_length = paddle.to_tensor(data=[2, 3, 5, 4, 1], dtype='int64')
    max_length = 4
    print(sequence_mask(seq_length, max_length))
    print('st_gumbel_softmax')
    logits = paddle.to_tensor(data=[[1, 2, 5, 2, 3], [3, 2, 8, 1, 1], [1, 5,
        9, 3, 1]], dtype='float32')
    print(st_gumbel_softmax(logits, temperature=1.0, mask=None))
    print('greedy_select')
    logits = paddle.to_tensor(data=[[10, 2, 5, 2, 3], [3, 2, 8, 1, 11], [1,
        5, 9, 30, 1]], dtype='float32')
    print(greedy_select(logits, mask=None))
