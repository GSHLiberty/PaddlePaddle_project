import paddle
import numpy as np
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[(i), :] for i, c in
        enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.
        int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(y=mx)
    return mx


def accuracy(output, labels):
    preds = (output.max(axis=1), output.argmax(axis=1))[1].astype(dtype=
        labels.dtype)
    correct = preds.equal(y=labels).astype(dtype='float64')
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = paddle.to_tensor(data=np.vstack((sparse_mx.row, sparse_mx.col))
        ).astype(dtype='int64')
    values = paddle.to_tensor(data=sparse_mx.data)
    shape = list(sparse_mx.shape)
    return paddle.sparse.sparse_coo_tensor(indices=indices, values=values, shape=shape)
