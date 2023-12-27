import paddle
import math
import numpy as np


def my_loss(y_input, y_target):
    return paddle.nn.functional.nll_loss(input=y_input, label=y_target)


def OpenAITransformer_loss(lm_criterion, clf_criterion, lm_coef, X, Y, M,
    clf_logits, lm_logits=None):
    if lm_logits is not None:
        X_shifted = paddle.slice(X, axes=[2], starts=[1], ends=[None])
        # 将结果展平为一维张量
        x_shifted = paddle.reshape(X_shifted, shape=[-1])
        M = paddle.reshape(M, shape=[-1, M.shape[2]])
        lm_losses = lm_criterion(lm_logits, x_shifted)
        lm_losses = paddle.reshape(lm_losses, shape=[X.shape[0] * X.shape[1], X.shape[2] - 1])
        lm_losses = lm_losses * M[:, 1:]
        lm_losses = lm_losses.sum(axis=1) / paddle.sum(x=M[:, 1:], axis=1)
    clf_losses = clf_criterion(clf_logits, Y)
    if lm_coef > 0 and lm_logits is not None:
        train_loss = clf_losses.sum() + lm_coef * lm_losses.sum()
    else:
        train_loss = clf_losses.sum()
    return train_loss


class Loss(object):
    """ Base class for encapsulation of the loss functions.

    This class defines interfaces that are commonly used with loss functions
    in training and inferencing.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.

    Attributes:
        name (str): name of the loss function used by logging messages.
        criterion (torch.nn._Loss): one of PyTorch's loss function.  Refer
            to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.  Implementation depends on individual
            sub-classes.
        acc_loss (int or torcn.nn.Tensor): variable that stores accumulated loss.
        norm_term (float): normalization term that can be used to calculate
            the loss of multiple batches.  Implementation depends on individual
            sub-classes.
    """

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        self.acc_loss = 0
        self.norm_term = 0

    def reset(self):
        """ Reset the accumulated loss. """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """ Get the loss.

        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.

        Returns:
            loss (float): value of the loss.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """ Evaluate and accumulate loss given outputs and expected results.

        This method is called after each batch with the batch outputs and
        the target (expected) results.  The loss and normalization term are
        accumulated in this method.  Override it to define your own accumulation
        method.

        Args:
            outputs (torch.Tensor): outputs of a batch.
            target (torch.Tensor): expected output of a batch.
        """
        raise NotImplementedError

    def cuda(self):
        self.criterion

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError('No loss to back propagate.')
        self.acc_loss.backward()


class NLLLoss(Loss):
    """ Batch averaged negative log-likelihood loss.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """
    _NAME = 'Avg NLLLoss'

    def __init__(self, weight=None, mask=None, size_average=True):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError('Must provide weight with a mask.')
            weight[mask] = 0
        super(NLLLoss, self).__init__(self._NAME, paddle.nn.NLLLoss(weight=
            weight))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.data.item()
        if self.size_average:
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1


class Perplexity(NLLLoss):
    """ Language model perplexity loss.

    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """
    _NAME = 'Perplexity'
    _MAX_EXP = 100

    def __init__(self, weight=None, mask=None):
        super(Perplexity, self).__init__(weight=weight, mask=mask,
            size_average=False)

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        if self.mask is None:
            self.norm_term += np.prod(target.shape)
        else:
            self.norm_term += target.data.ne(self.mask).sum()

    def get_loss(self):
        nll = super(Perplexity, self).get_loss()
        nll /= self.norm_term.item()
        if nll > Perplexity._MAX_EXP:
            print('WARNING: Loss exceeded maximum value, capping to e^100')
            return math.exp(Perplexity._MAX_EXP)
        return math.exp(nll)
