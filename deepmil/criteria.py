import sys

import torch
import torch.nn as nn

sys.path.append("..")

import constants

from reproducibility import force_seed

from shared import announce_msg

__all__ = ["TrainLoss", "KLUniformLoss", "NegativeEntropy", "Metrics"]


class KLUniformLoss(nn.Module):
    """
    KL loss KL(q, p) where q is a uniform distribution.
    This amounts to = -1/c . sum_i log2 p_i.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(KLUniformLoss, self).__init__()

        self.softmax = nn.Softmax(dim=1)  # The log-softmax.

    def forward(self, scores):
        """
        Forward function
        :param scores: unormalized scores (batch_size, nbr_classes)
        :return: loss. scalar.
        """
        logsoftmax = torch.log2(self.softmax(scores))
        loss = (-logsoftmax).mean(dim=1).mean()
        return loss


class NegativeEntropy(nn.Module):
    """
    Negative entropy loss.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(NegativeEntropy, self).__init__()

        self.sftmax = nn.Softmax(dim=1)

    def forward(self, scores):
        """
        Forward function.
        :param scores: unormalized scores.
        :return: loss. scalar.
        """
        probs = self.sftmax(scores)
        loss = ((probs * torch.log2(probs)).sum(dim=1)).mean()

        return loss


class TrainLoss(nn.Module):
    """
    Loss.
    """
    def __init__(self,
                 use_reg=False,
                 reg_loss=constants.NoneLoss,
                 use_size_const=False,
                 init_t=1.,
                 max_t=10.,
                 mulcoef=1.01,
                 normalize_sz=False,
                 epsilon=0.,
                 lambda_neg=1e-3
                 ):
        """
        Init. function.
        :param use_reg: bool. whether or not use background regularization.
        :param reg_loss: str. name of the background regularization loss.
        :param use_size_const: whether or not use the size constraint over
        the background.
        :param init_t: float > 0. initial t for ELB for the size constraint.
        :param max_t: float > 0. max t for ELB for the size constraint.
        :param mulcoef: float > 0. increasing factor for ELB for the size
        constraint.
        :param normalize_sz: bool. if true, the size of the mask is normalized.
        :param epsilon: float >= 0. small constant used for the size constraint.
        :param lambda_neg: float >= 0. lambda for the regularization term.
        """
        super(TrainLoss, self).__init__()

        self.CE = nn.CrossEntropyLoss(reduction="mean")  # The cross entropy
        # loss.

        self.reg_loss = None
        self.lambda_neg = lambda_neg
        if use_reg:
            msg = "'reg' must be in {}".format(constants.reg_losses)
            assert reg_loss in constants.reg_losses, msg
            self.reg_loss =  sys.modules[__name__].__dict__[reg_loss]()

        # elb for size constraint.
        self.elb = None
        self.use_size_const = use_size_const
        self.normalize_sz = normalize_sz
        self.epsilon = epsilon
        self.t_tracker = []  # track `t` of ELB if there is any.
        self.register_buffer(
            "zero", torch.tensor([0.], requires_grad=False).float())
        if use_size_const:
            self.elb = _LossExtendedLB(init_t=init_t,
                                       max_t=max_t,
                                       mulcoef=mulcoef
                                       )

    def size_const(self, masks_pred):
        """
        Compute the loss over the size of the masks.
        :param masks_pred: foreground predicted mask. shape: (bs, 1, h, w).
        :return: ELB loss. a scalar that is the sum of the losses over bs.
        """
        assert masks_pred.ndim == 4, "Expected 4 dims, found {}.".format(
            masks_pred.ndim)

        msg = "nbr masks must be 1. found {}.".format(masks_pred.shape[1])
        assert masks_pred.shape[1] == 1, msg

        # background
        backgmsk = 1. - masks_pred
        bsz = backgmsk.shape[0]
        h = backgmsk.shape[2]
        w = backgmsk.shape[3]
        l1 = torch.abs(backgmsk.contiguous().view(bsz, -1)).sum(dim=1)
        if self.normalize_sz:
            l1 = l1 / float(h * w)

        l1 = l1 - self.epsilon
        loss_back = self.elb(-l1)

        # foreground
        l1_fg = torch.abs(masks_pred.contiguous().view(bsz, -1)).sum(dim=1)
        if self.normalize_sz:
            l1_fg = l1_fg / float(h * w)

        l1_fg = l1_fg - self.epsilon
        loss_fg = self.elb(-l1_fg)

        loss = loss_back + loss_fg

        return loss

    def update_t(self):
        """
        Update the value of `t` of the ELB method.
        :return:
        """
        if self.elb is not None:
            self.t_tracker.append(self.elb.t_lb.item())
            self.elb.update_t()

    def get_t(self):
        """
        Returns the value of 't_lb' of the ELB method.
        """
        if self.elb is not None:
            return self.elb.get_t()
        else:
            return self.zero


    def forward(self,
                scores_pos,
                sc_cl_se,
                labels,
                masks_pred,
                scores_neg=None
                ):
        """
        Performs forward function: computes the losses.
        """
        # classification loss over the localizer
        loss_cl_seg = self.CE(sc_cl_se, labels)

        total_loss = loss_cl_seg
        # classification loss: positive regions
        loss_pos = self.CE(scores_pos, labels)

        total_loss = total_loss + loss_pos

        # regularization: loss over negative regions.
        loss_neg = torch.tensor([0.])
        if self.reg_loss is not None:
            assert scores_neg is not None, "ERROR"
            loss_neg = self.reg_loss(scores_neg)
            total_loss = total_loss + self.lambda_neg * loss_neg

        # constraint on background size.
        loss_sz_con = torch.tensor([0.])
        bsz = float(scores_pos.shape[0])
        if self.use_size_const:
            loss_sz_con = self.size_const(masks_pred=masks_pred) / bsz
            total_loss = total_loss + loss_sz_con


        return total_loss, loss_pos, loss_neg, loss_cl_seg

    def __str__(self):
        return "{}()".format(self.__class__.__name__,)

class _LossExtendedLB(nn.Module):
    """
    Extended log-barrier loss (ELB).
    Optimize inequality constraint : f(x) <= 0.

    Refs:
    1. Kervadec, H., Dolz, J., Yuan, J., Desrosiers, C., Granger, E., and Ben
     Ayed, I. (2019b). Constrained deep networks:Lagrangian optimization
     via log-barrier extensions.CoRR, abs/1904.04205
    2. S. Belharbi, I. Ben Ayed, L. McCaffrey and E. Granger,
    “Deep Ordinal Classification with Inequality Constraints”, CoRR,
    abs/1911.10720, 2019.
    """
    def __init__(self,
                 init_t=1.,
                 max_t=10.,
                 mulcoef=1.01
                 ):
        """
        Init. function.

        :param init_t: float > 0. The initial value of `t`.
        :param max_t: float > 0. The maximum allowed value of `t`.
        :param mulcoef: float > 0.. The coefficient used to update `t` in the
        form: t = t * mulcoef.
        """
        super(_LossExtendedLB, self).__init__()

        msg = "`mulcoef` must be a float. You provided {} ....[NOT OK]".format(
            type(mulcoef))
        assert isinstance(mulcoef, float), msg
        msg = "`mulcoef` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(mulcoef)
        assert mulcoef > 0., msg

        msg = "`init_t` must be a float. You provided {} ....[NOT OK]".format(
            type(init_t))
        assert isinstance(init_t, float), msg
        msg = "`init_t` must be > 0. float. You provided {} " \
              "....[NOT OK]".format(init_t)
        assert init_t > 0., msg

        msg = "`max_t` must be a float. You provided {} ....[NOT OK]".format(
            type(max_t))
        assert isinstance(max_t, float), msg
        msg = "`max_t` must be > `init_t`. float. You provided {} " \
              "....[NOT OK]".format(max_t)
        assert max_t > init_t, msg

        self.init_t = init_t

        self.register_buffer(
            "mulcoef", torch.tensor([mulcoef], requires_grad=False).float())
        # create `t`.
        self.register_buffer(
            "t_lb", torch.tensor([init_t], requires_grad=False).float())

        self.register_buffer(
            "max_t", torch.tensor([max_t], requires_grad=False).float())

    def set_t(self, val):
        """
        Set the value of `t`, the hyper-parameter of the log-barrier method.
        :param val: float > 0. new value of `t`.
        :return:
        """
        msg = "`t` must be a float. You provided {} ....[NOT OK]".format(
            type(val))
        assert isinstance(val, float) or (isinstance(val, torch.Tensor) and
                                          val.ndim == 1 and
                                          val.dtype == torch.float), msg
        msg = "`t` must be > 0. float. You provided {} ....[NOT OK]".format(val)
        assert val > 0., msg

        if isinstance(val, float):
            self.register_buffer(
                "t_lb", torch.tensor([val], requires_grad=False).float()).to(
                self.t_lb.device
            )
        elif isinstance(val, torch.Tensor):
            self.register_buffer("t_lb", val.float().requires_grad_(False))

    def get_t(self):
        """
        Returns the value of 't_lb'.
        """
        return self.t_lb

    def update_t(self):
        """
        Update the value of `t`.
        :return:
        """
        self.set_t(torch.min(self.t_lb * self.mulcoef, self.max_t))

    def forward(self, fx):
        """
        The forward function.
        :param fx: pytorch tensor. a vector.
        :return: real value extended-log-barrier-based loss.
        """
        assert fx.ndim == 1, "fx.ndim must be 1. found {}.".format(fx.ndim)

        loss_fx = fx * 0.

        # vals <= -1/(t**2).
        ct = - (1. / (self.t_lb**2))

        idx_less = ((fx < ct) | (fx == ct)).nonzero().squeeze()
        if idx_less.numel() > 0:
            val_less = fx[idx_less]
            loss_less = - (1. / self.t_lb) * torch.log(- val_less)
            loss_fx[idx_less] = loss_less

        # vals > -1/(t**2).
        idx_great = (fx > ct).nonzero().squeeze()
        if idx_great.numel() > 0:
            val_great = fx[idx_great]
            loss_great = self.t_lb * val_great - (1. / self.t_lb) * \
                torch.log((1. / (self.t_lb**2))) + (1. / self.t_lb)
            loss_fx[idx_great] = loss_great

        loss = loss_fx.sum()

        return loss

    def __str__(self):
        return "{}(): extended-log-barrier-based method.".format(
            self.__class__.__name__)


class Dice(nn.Module):
    """
    Computes Dice index for binary classes.
    """
    def __init__(self):
        """
        Init. function.
        """
        super(Dice, self).__init__()

    def forward(self, pred_m, true_m):
        """
        Forward function.
        Computes Dice index [0, 1] for binary classes.
        :param pred_m: predicted mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :return vector of size (n) contains Dice index of each sample. values
        are in [0, 1].
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        pflat = pred_m
        tflat = true_m
        intersection = (pflat * tflat).sum(dim=1)

        return (2. * intersection) / (pflat.sum(dim=1) + tflat.sum(dim=1))

    def __str__(self):
        return "{}(): Dice index.".format(self.__class__.__name__)


class IOU(nn.Module):
    """
    Computes the IOU (intersection over union) metric for one class.
    """
    def __init__(self, smooth=1.):
        """
        Init. function.
        :param smooth: float > 0. smoothing value.
        """
        super(IOU, self).__init__()

        assert smooth > 0., "'smooth' must be > 0. found {}.".format(smooth)
        msg = "'smooth' type must be float, found {}.".format(type(smooth))
        assert isinstance(smooth, float), msg
        self.smooth = smooth

    def forward(self, pred_m, true_m):
        """
        Forward function.
        :param pred_m: predicted mask ([0, 1], float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :param true_m: true mask (binary, float32) of shape (n, m) where n
        is the batch size and m is the number of elements (pixels).
        :return vector of size (n) contains IOU metric of each sample.
        values are in [0, 1] where 0 is the best.
        """
        assert pred_m.ndim == 2, "'pred_m.ndim' = {}. must be {}.".format(
            pred_m.ndim, 2)

        assert true_m.ndim == 2, "'true_m.ndim' = {}. must be {}.".format(
            true_m.ndim, 2)

        assert true_m.shape == pred_m.shape, "size mismatches: {}, {}.".format(
            true_m.shape, pred_m.shape
        )
        msg = "'pred_m' dtype required is torch.float. found {}.".format(
            pred_m.dtype)
        assert pred_m.dtype == torch.float, msg
        msg = "'true_m' dtype required is torch.float. found {}.".format(
            true_m.dtype)
        assert true_m.dtype == torch.float, msg

        pflat = pred_m
        tflat = true_m
        intersection = (pflat * tflat).sum(dim=1)
        union = pflat.sum(dim=1) + tflat.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou

    def __str__(self):
        return "{}(): IOU metric for one class. " \
               "".format(self.__class__.__name__)


class Metrics(nn.Module):
    """
    Compute some metrics.

    1. ACC: Classification accuracy. in [0, 1]. 1 is the best. [if avg=True].
    2. Dice index.
    3. mIOU: mean intersection over union.

    Note: 2 and 3 are for binary segmentation.
    """
    def __init__(self, threshold=0.5):
        """
        Init. function.
        :param threshold: float. threshold in [0., 1.].
        """
        super(Metrics, self).__init__()
        msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
        assert 0 <= threshold <= 1., msg
        msg = "'threshold' type must be float. found {}.".format(
            type(threshold))
        assert isinstance(threshold, float), msg

        self.threshold = threshold
        self.dice = Dice()
        self.iou = IOU()

    def predict_label(self, scores):
        """
        Predict the output label based on the scores or probabilities for
        global classification.

        :param scores: matrix (n, nbr_c) of unormalized-scores or probabilities.
        :return: vector of long integer. The predicted label(s).
        """
        return scores.argmax(dim=1, keepdim=False)

    def forward(self,
                scores,
                labels,
                masks_pred,
                masks_trg,
                avg=False,
                threshold=None
                ):
        """
        The forward function.

        :param scores: matrix (n, nbr_c) of unormalized-scores or probabilities.
        :param labels: vector of Log integers. The ground truth labels.
        :param masks_pred: torch tensor. predicted masks (for seg.).
        normalized scores. shape: (n, m) where n is the batch size.
        :param masks_trg: torch tensor. target mask (for seg). shape: (n, m)
        where n is the batch size and m is the number of pixels in the mask.
        :param avg: bool If True, the metrics are averaged
        by dividing by the total number of samples.
        :param threshold: float. threshold in [0., 1.] or None. if None,
        we use self.threshold. otherwise, we us this threshold.
        :return:
            acc: scalar (torch.tensor of size 1). classification
            accuracy (avg or sum).
            dice_index: scalar (torch.tensor of size 1). Dice index (avg or
            sum).
            iou: scalar (torch.tensor of size 1). Mean IOU over classes (
            binary) [sum or average over samples].
        """
        msg = "`scores` must be a matrix with size (h, w) where `h` is the " \
              "number of samples, and `w` is the number of classes. We found," \
              " `scores.ndim`={}, and `inputs.shape`={} .... " \
              "[NOT OK]".format(scores.ndim, scores.shape)
        assert scores.ndim == 2, msg
        assert scores.shape[1] > 1, "Number of classes must be > 1 ....[NOT OK]"

        assert labels.ndim == 1, "`labels` must be a vector ....[NOT OK]"
        msg = "`labels` and `scores` dimension mismatch....[NOT OK]"
        assert labels.numel() == scores.shape[0], msg

        msg = "'masks_pred.ndim' = {}. must be {}.".format(masks_pred.ndim, 2)
        assert masks_pred.ndim == 2, msg

        msg = "'masks_trg.ndim' = {}. must be {}.".format(masks_trg.ndim, 2)
        assert masks_trg.ndim == 2, msg

        msg = "size mismatches: {}, {}.".format(
            masks_trg.shape, masks_pred.shape
        )
        assert masks_trg.shape == masks_pred.shape, msg
        msg = "'masks_pred' dtype required is torch.float. found {}.".format(
            masks_pred.dtype)
        assert masks_pred.dtype == torch.float, msg
        msg = "'masks_trg' dtype required is torch.float. found {}.".format(
            masks_trg.dtype)
        assert masks_trg.dtype == torch.float, msg

        n, c = scores.shape
        msg = "batch size mismatches. scores {}, masks_pred {}, " \
              "masks_trg {}".format(n, masks_pred.shape[0], masks_trg.shape[0])
        assert n == masks_pred.shape[0] == masks_trg.shape[0], msg

        cur_threshold = self.threshold

        if threshold is not None:
            msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
            assert 0. <= threshold <= 1., msg
            msg = "'threshold' type must be float. found {}.".format(
                type(threshold))
            assert isinstance(threshold, float), msg
            cur_threshold = threshold

        # This class should not be included in any gradient computation.
        with torch.no_grad():
            plabels = self.predict_label(scores)  # predicted labels
            # 1. ACC in [0, 1]
            acc = ((plabels - labels) == 0.).float().sum()

            # 2. Dice index in [0, 1]
            ppixels = self.get_binary_mask(
                pred_m=masks_pred, threshold=cur_threshold)
            dice_forg = self.dice(pred_m=ppixels,
                                  true_m=masks_trg
                                  ).sum()
            dice_back = self.dice(pred_m=1. - ppixels,
                                  true_m=1. - masks_trg
                                  ).sum()

            # 3. mIOU:
            # foreground
            iou_fgr = self.iou(pred_m=ppixels, true_m=masks_trg)
            # background
            iou_bgr = self.iou(pred_m=1.-ppixels, true_m=1-masks_trg)
            iou = (iou_fgr + iou_bgr) / 2.  # avg. over classes (2)
            iou = iou.sum()

            if avg:
                acc = acc / float(n)
                dice_forg = dice_forg / float(n)
                dice_back = dice_back / float(n)
                iou = iou / float(n)
        return acc, dice_forg, dice_back, iou

    def binarize_mask(self, masks_pred, threshold):
        """
        Predict the binary mask for segmentation.

        :param masks_pred: tensor (n, whatever-dims) of normalized-scores.
        :param threshold: float. threshold in [0., 1.]
        :return: tensor of same shape as `masks_pred`. Contains the binary
        mask, thresholded at `threshold`. dtype: float.
        """
        msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
        assert 0. <= threshold <= 1., msg
        msg = "'threshold' type must be float. found {}.".format(
            type(threshold))
        assert isinstance(threshold, float), msg


        return (masks_pred >= threshold).float()

    def get_binary_mask(self, pred_m, threshold=None):
        """
        Get binary mask by thresholding.
        :param pred_m: torch tensor of shape (n, what-ever-dim)
        :param threshold: float. threshold in [0., 1.] or None. if None,
        we use self.threshold. otherwise, we us this threshold.
        :return:
        """
        cur_threshold = self.threshold

        if threshold is not None:
            msg = "'threshold' must be in [0, 1]. found {}.".format(threshold)
            assert 0. <= threshold <= 1., msg
            msg = "'threshold' type must be float. found {}.".format(
                type(threshold))
            assert isinstance(threshold, float), msg
            cur_threshold = threshold

        return self.binarize_mask(pred_m, threshold=cur_threshold)


    def __str__(self):
        return "{}(): computes ACC, Dice index metrics.".format(
            self.__class__.__name__)

# ====================== TEST =========================================

def test_TrainLoss():
    loss = TrainLoss()
    print("Testing {}".format(loss))
    cuda = 0
    DEVICE = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    loss.to(DEVICE)
    num_classes = 2
    b, h, w = 16, 200, 200
    masks = torch.rand(b, 1, h, w).to(DEVICE)
    out_pos = (torch.rand(b, num_classes).to(DEVICE),
               torch.rand(b, num_classes, h, w).to(DEVICE))
    out_neg = (torch.rand(b, num_classes).to(DEVICE),
               torch.rand(b, num_classes, h, w).to(DEVICE))
    scores_seg = (torch.rand(b, num_classes)).to(DEVICE)
    maps_seg = (torch.rand(b, num_classes, 10, 10)).to(DEVICE)
    netoutput = (out_pos, out_neg, masks, scores_seg, maps_seg)
    labels = torch.empty(b, dtype=torch.long).random_(2).to(DEVICE)

    print("Loss class at head: {}".format(
        loss.loss_class_head_seg(scores_seg, labels)))
    losses = loss(netoutput, labels)
    for l in losses:
        print(l, l.size())


def test__LossExtendedLB():
    force_seed(0, check_cudnn=False)
    instance = _LossExtendedLB(init_t=1., max_t=10., mulcoef=1.01)
    announce_msg("Testing {}".format(instance))

    cuda = 1
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(cuda))
    instance.to(DEVICE)

    b = 16
    fx = (torch.rand(b)).to(DEVICE)

    out = instance(fx)
    for r in range(10):
        instance.update_t()
        print("epoch {}. t: {}.".format(r, instance.t_lb))
    print("Loss ELB.sum(): {}".format(out))

if __name__ == "__main__":
    # test_TrainLoss()
    test__LossExtendedLB()

