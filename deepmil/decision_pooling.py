import threading


import numpy as np


import torch
import torch.nn as nn

thread_lock = threading.Lock()  # lock for threads to protect the instruction that cause randomness and# make them
# thread-safe.

import reproducibility


__all__ = ["WildCatPoolDecision", "ClassWisePooling"]


class WildCatPoolDecision(nn.Module):
    """Compute the score of each class using wildcat pooling strategy.
    Reference to wildcat pooling:
    http://webia.lip6.fr/~cord/pdfs/publis/Durand_WILDCAT_CVPR_2017.pdf
    """
    def __init__(self, kmax=0.5, kmin=None, alpha=1, dropout=0.0):
        """
        Input:
            kmax: int or float scalar. The number of maximum features to consider.
            kmin: int or float scalar. If None, it takes the same value as kmax. The number of minimal features to
            consider.
            alpha: float scalar. A weight , used to compute the final score.
            dropout: float scalar. If not zero, a dropout is performed over the min and max selected features.
        """
        super(WildCatPoolDecision, self).__init__()

        assert isinstance(kmax, (int, float)) and kmax > 0, "kmax must be an integer or a float in ]0, 1]"
        assert kmin is None or (isinstance(kmin, (int, float)) and kmin >= 0), "kmin must be None or the same type " \
                                                                               "as " \
                                                                               "kmax, and it must be >= 0 or None"
        self.kmax = kmax
        self.kmin = kmax if kmin is None else kmin
        self.alpha = alpha
        self.dropout = dropout

        self.dropout_md = nn.Dropout(p=dropout, inplace=False)

    def get_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k == 1 and isinstance(k, float):
            return int(n)
        elif k == 1 and isinstance(k, int):
            return 1
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        Input:
            In the case of K classes:
                x: torch tensor of size (n, c, h, w), where n is the batch size, c is the number of classes,
                h is the height of the feature map, w is its width.
            seed: int, seed for the thread to guarantee reproducibility over a fixed number of gpus.
        Output:
            scores: torch vector of size (k). Contains the wildcat score of each class. A score is a linear combination
            of different features. The class with the highest features is the winner.
        """
        b, c, h, w = x.shape
        activations = x.view(b, c, h * w)

        n = h * w

        sorted_features = torch.sort(activations, dim=-1, descending=True)[0]
        kmax = self.get_k(self.kmax, n)
        kmin = self.get_k(self.kmin, n)

        # dropout
        if self.dropout != 0.:
            if seed is not None:
                thread_lock.acquire()
                assert prngs_cuda is not None, "`prngs_cuda` is expected to not be None. Exiting .... [NOT OK]"
                prng_state = (torch.cuda.get_rng_state().cpu())
                reproducibility.force_seed(seed)
                torch.cuda.set_rng_state(prngs_cuda.cpu())

                sorted_features = self.dropout_md(sorted_features)  # instruction that causes randomness.

                reproducibility.force_seed(seed)
                torch.cuda.set_rng_state(prng_state)
                thread_lock.release()
            else:
                sorted_features = self.dropout_md(sorted_features)

        scores = sorted_features.narrow(-1, 0, kmax).sum(-1).div_(kmax)

        if kmin > 0 and self.alpha != 0.:
            scores.add(sorted_features.narrow(-1, n - kmin, kmin).sum(-1).mul_(self.alpha / kmin)).div_(2.)

        return scores

    def __repr__(self):
        return self.__class__.__name__ + "(kmax={}, kmin={}, alpha={}, dropout={}".format(self.kmax, self.kmin,
                                                                                          self.alpha, self.dropout)


class ClassWisePooling(nn.Module):
    """
    Pull a feature map per class.
    Reference to wildcat:
    http://webia.lip6.fr/~cord/pdfs/publis/Durand_WILDCAT_CVPR_2017.pdf
    """
    def __init__(self, classes, modalities):
        """
        Init. function.
        :param classes: int, number of classes.
        :param modalities: int, number of modalities per class.
        """
        super(ClassWisePooling, self).__init__()

        self.C = classes
        self.M = modalities

    def forward(self, input):
        N, C, H, W = input.size()
        assert C == self.C * self.M, 'Wrong number of channels, expected {} channels but got {}'.format(
            self.C * self.M, C)
        return torch.mean(input.view(N, self.C, self.M, -1), dim=2).view(N, self.C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(classes={}, modalities={})'.format(self.C, self.M)


if __name__ == "__main__":
    b, c = 10, 2
    reproducibility.force_seed(0)
    funcs = [WildCatPoolDecision(dropout=0.5)]
    x = torch.randn(b, c, 12, 12)
    for func in funcs:
        out = func(x)
        print(func.__class__.__name__, '->', out.size(), out)

