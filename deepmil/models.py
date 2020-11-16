# All credits of Synchronized BN go to Tamaki Kojima(tamakoji@gmail.com) (https://github.com/tamakoji/pytorch-syncbn)
# DeeplabV3:  L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam.  Re-
# thinking  atrous  convolution  for  semantic  image  segmenta-
# tion. arXiv preprint arXiv:1706.05587, 2017..

# Source based: https://github.com/speedinghzl/pytorch-segmentation-toolbox
# BN: https://github.com/mapillary/inplace_abn

# PSPNet:  H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia.  Pyramid scene
# parsing network. In CVPR, pages 2881–2890, 2017. https://arxiv.org/abs/1612.01105


# Other stuff: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# Deeplab:
# https://github.com/speedinghzl/pytorch-segmentation-toolbox
# https://github.com/speedinghzl/Pytorch-Deeplab
# https://github.com/kazuto1011/deeplab-pytorch
# https://github.com/isht7/pytorch-deeplab-resnet
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/CSAILVision/semantic-segmentation-pytorch


# Pretrained:
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/28aab5849db391138881e3c16f9d6482e8b4ab38/dataset.py
# https://github.com/CSAILVision/sceneparsing
# https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/28aab5849db391138881e3c16f9d6482e8b4ab38
# Input normalization (natural images):
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/28aab5849db391138881e3c16f9d6482e8b4ab38/dataset.py

import threading
import sys
import math
import os
import collections
import numbers

from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.nn import functional as F

from tools import check_if_allow_multgpu_mode, announce_msg

sys.path.append("..")

from deepmil.decision_pooling import WildCatPoolDecision, ClassWisePooling
thread_lock = threading.Lock()  # lock for threads to protect the instruction that cause randomness and make them
# thread-safe.

import reproducibility

ACTIVATE_SYNC_BN = True
# Override ACTIVATE_SYNC_BN using variable environment in Bash:
# $ export ACTIVATE_SYNC_BN="True"   ----> Activate
# $ export ACTIVATE_SYNC_BN="False"   ----> Deactivate

if "ACTIVATE_SYNC_BN" in os.environ.keys():
    ACTIVATE_SYNC_BN = (os.environ['ACTIVATE_SYNC_BN'] == "True")

announce_msg("ACTIVATE_SYNC_BN was set to {}".format(ACTIVATE_SYNC_BN))

if check_if_allow_multgpu_mode() and ACTIVATE_SYNC_BN:  # Activate Synch-BN.
    from deepmil.syncbn import nn as NN_Sync_BN
    BatchNorm2d = NN_Sync_BN.BatchNorm2d
    announce_msg("Synchronized BN has been activated. \n"
                 "MultiGPU mode has been activated. {} GPUs".format(torch.cuda.device_count()))
else:
    BatchNorm2d = nn.BatchNorm2d
    if check_if_allow_multgpu_mode():
        announce_msg("Synchronized BN has been deactivated.\n"
                     "MultiGPU mode has been activated. {} GPUs".format(torch.cuda.device_count()))
    else:
        announce_msg("Synchronized BN has been deactivated.\n"
                     "MultiGPU mode has been deactivated. {} GPUs".format(torch.cuda.device_count()))

# DEFAULT SEGMENTATION PARAMETERS ###########################

INNER_FEATURES = 256  # ASPPModule
OUT_FEATURES = 512  # ASPPModule, PSPModule,

#
# ###########################################################

ALIGN_CORNERS = True

__all__ = ['resnet18', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WildCatClassifierHead(nn.Module):
    """
    A WILDCAT type classifier head.
    """
    def __init__(self, inplans, modalities, num_classes, kmax=0.5, kmin=None, alpha=0.6, dropout=0.0):
        super(WildCatClassifierHead, self).__init__()

        self.to_modalities = nn.Conv2d(inplans, num_classes * modalities, kernel_size=1, bias=True)
        self.to_maps = ClassWisePooling(num_classes, modalities)
        self.wildcat = WildCatPoolDecision(kmax=kmax, kmin=kmin, alpha=alpha, dropout=dropout)

    def forward(self, x, seed=None, prngs_cuda=None):

        modalities = self.to_modalities(x)
        maps = self.to_maps(modalities)
        scores = self.wildcat(x=maps, seed=seed, prngs_cuda=prngs_cuda)

        return scores, maps


class MaskHead(nn.Module):
    """
    Class that pulls the mask from feature maps.
    """
    def __init__(self, inplans, modalities, nbr_masks):
        """

        :param inplans: int. number of input features.
        :param modalities: int. number of modalities.
        :param nbr_masks: int. number of masks to pull.
        """
        super(MaskHead, self).__init__()

        self.to_modalities = nn.Conv2d(inplans,
                                       nbr_masks * modalities,
                                       kernel_size=1,
                                       bias=True
                                       )
        self.to_masks = ClassWisePooling(nbr_masks, modalities)

    def forward(self, x):
        """
        The forward function.
        :param x: input tensor fetaure maps.
        :return:
        """
        modalities = self.to_modalities(x)
        masks = self.to_masks(modalities)

        return masks



class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sigma=0.5,
                 w=8,
                 num_classes=2,
                 scale=(0.5, 0.5),
                 modalities=4,
                 kmax=0.5,
                 kmin=None,
                 alpha=0.6,
                 dropout=0.0
                 ):
        """
        Init. function.
        :param block: class of the block.
        :param layers: list of int, number of layers per block.
        :param num_masks: int, number of masks to output. (supports only 1).
        """

        # classifier stuff
        cnd = isinstance(scale, tuple) or isinstance(scale, list)
        cnd = cnd or isinstance(scale, float)
        msg = "`scale` should be a tuple, or a list, or a float with " \
              "values in ]0, 1]. You provided {} .... [NOT " \
              "OK]".format(scale)
        assert cnd, msg

        if isinstance(scale, tuple) or isinstance(scale, list):
            msg = "`scale[0]` (height) should be > 0 and <= 1. " \
                  "You provided `{}` ... [NOT OK]".format(scale[0])
            assert 0 < scale[0] <= 1, msg
            msg = "`scale[1]` (width) should be > 0 and <= 1. " \
                  "You provided `{}` .... [NOT OK]".format(scale[1])
            assert 0 < scale[0] <= 1, msg
        elif isinstance(scale, float):
            msg = "`scale` should be > 0, <= 1. You provided `{}` .... " \
                  "[NOT OK]".format(scale)
            assert 0 < scale <= 1, msg
            scale = (scale, scale)

        self.scale = scale
        self.num_classes = num_classes


        self.inplanes = 128
        super(ResNet, self).__init__()

        # Encoder

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Find out the size of the output.

        if isinstance(self.layer4[-1], Bottleneck):
            in_channel4 = self.layer1[-1].bn3.weight.size()[0]
            in_channel8 = self.layer2[-1].bn3.weight.size()[0]
            in_channel16 = self.layer3[-1].bn3.weight.size()[0]
            in_channel32 = self.layer4[-1].bn3.weight.size()[0]
        elif isinstance(self.layer4[-1], BasicBlock):
            in_channel4 = self.layer1[-1].bn2.weight.size()[0]
            in_channel8 = self.layer2[-1].bn2.weight.size()[0]
            in_channel16 = self.layer3[-1].bn2.weight.size()[0]
            in_channel32 = self.layer4[-1].bn2.weight.size()[0]
        else:
            raise ValueError("Supported class .... [NOT OK]")

        print(in_channel32, in_channel16, in_channel8, in_channel4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # =================  SEGMENTOR =========================================
        self.sigma = sigma

        self.const2 = torch.tensor([w], requires_grad=False).float()
        self.register_buffer("w", self.const2)

        self.mask_head = WildCatClassifierHead(in_channel32,
                                               modalities,
                                               num_classes,
                                               kmax=kmax,
                                               kmin=kmin,
                                               alpha=alpha,
                                               dropout=dropout
                                               )

        print("Num. parameters headmask: {}".format(
            sum([p.numel() for p in self.mask_head.parameters()])))
        # ======================================================================

        # ================================ CLASSIFIER ==========================
        self.cl32 = WildCatClassifierHead(in_channel32,
                                          modalities,
                                          num_classes,
                                          kmax=kmax,
                                          kmin=kmin,
                                          alpha=alpha,
                                          dropout=dropout
                                          )
        # ======================================================================

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, code=None, mask_c=None, seed=None, prngs_cuda=None):
        """
        Forward function.

        MultiGPU issue: During training, we need at some point to call the
        following functions:
        self.get_mask_xpos_xneg(), self.segment(), self.classify().
        Since we wrap the model within torch.nn.DataParallel, the function
         parallel_apply() needs to call the model
        (call self.forward()). Therefore, it makes it complicated to call other
        functions such as self.segment().
        To deal with this, de decide to encode the calling of the sub-functions
        of the model so we can pass every time
        by self.forward().
        Code:
            1. None: standard forward.
            2. "get_mask_xpos_xneg": call self.get_mask_xpos_xneg()
            3. "segment": call self.segment()
            4. "classify": call self.classify()


        :param x: input.
        :param code: None or a string. See above.
        :param mask_c: input for self.self.get_mask_xpos_xneg()
        :param seed: int, a seed for the case of Multigpus to
        guarantee reproducibility for a fixed number of GPUs.
        See  https://discuss.pytorch.org/t/did-anyone-succeed-to-reproduce-
        their-code-when-using-multigpus/47079?u=sbelharbi
        In the case of one GPU, the seed in not necessary (and it will not be
         used); se set it to None.
        :param prngs_cuda: value returned by torch.cuda.get_prng_state().
        :return:
        """
        if code is None:
            # 1. Segment: forward.
            mask, cl_scores_seg = self.segment(x=x,
                                               seed=seed,
                                               prngs_cuda=prngs_cuda
                                               )

            mask, x_pos, x_neg = self.get_mask_xpos_xneg(x, mask)

            scores_pos = self.classify(x=x_pos,
                                    seed=seed,
                                    prngs_cuda=prngs_cuda
                                    )
            scores_neg = self.classify(x=x_neg,
                                    seed=seed,
                                    prngs_cuda=prngs_cuda
                                    )

            return scores_pos, scores_neg, mask, cl_scores_seg

        if code == "get_mask_xpos_xneg":
            return self.get_mask_xpos_xneg(x, mask_c)

        if code == "segment":
            return self.segment(x=x, seed=seed, prngs_cuda=prngs_cuda)

        if code == "classify":
            return self.classify(x=x, seed=seed, prngs_cuda=prngs_cuda)

        raise ValueError("You seem to have figure it out how to use model.forward() using multiGPU. However, "
                         "you provided an unsupported code {}. Please double check. This is the list of supported "
                         "codes: None, 'get_mask_xpos_xneg', 'segment', 'classify'. Exiting .... [NOT OK]")

    def get_mask_xpos_xneg(self, x, mask_c):
        """
        Compute X+, X-.
        :param x: input X.
        :param mask_c: continous mask.
        :return:
        """
        # 2. Prepare the mask for multiplication.
        mask = self.get_pseudo_binary_mask(mask_c)
        x_pos, x_neg = self.apply_mask(x, mask)

        return mask, x_pos, x_neg

    def segment(self, x, seed=None, prngs_cuda=None):
        """
        Forward function.
        Any mask is is composed of two 2D plans:
            1. The first plan represents the background.
            2. The second plan represents the regions of interest (glands).

        :param x: tensor, input image with size (nb_batch, depth, h, w).
        :param seed: int, seed for thread (to guarantee reproducibility over
        a fixed number of multigpus.)
        :return: (out_pos, out_neg, mask):
            x_pos: tensor, the image with the mask applied.
            size (nb_batch, depth, h, w)
            x_neg: tensor, the image with the complementary mask applied.
            size (nb_batch, depth, h, w)
            out_pos: tuple of tensors, the output of the classification of the
            positive regions. (scores,
            wildcat feature maps)
            out_neg: tuple of tensors, the output of the classification of the
            negative regions. (scores,
            wildcat feature maps).
            mask: tensor, the mask of each image in the batch. size
            (batch_size, 1, h, w) if evaluation mode is on or (
            batch_size, 1, h*, w*) where h*, w* is the size of the input after
             downsampling (if the training mode is on).
        """
        b, _, h, w = x.shape

        # x: 1 / 1: [n, 3, 480, 480]
        # Only number of filters change: (18, 50, 101): a/b/c.
        # Down-sample:
        x_0 = self.relu1(self.bn1(self.conv1(x)))  # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1. (
        # downsampled due to the stride=2)
        x_1 = self.relu2(self.bn2(self.conv2(x_0)))  # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1.
        x_2 = self.relu3(self.bn3(self.conv3(x_1)))  # 1 / 2: [2, 128, 240, 240]  --> x2^1 to get back to 1.
        x_3 = self.maxpool(x_2)        # 1 / 4:  [2, 128, 120, 120]         --> x2^2 to get back to 1.
        x_4 = self.layer1(x_3)       # 1 / 4:  [2, 64/256/--, 120, 120]   --> x2^2 to get back to 1.
        x_8 = self.layer2(x_4)     # 1 / 8:  [2, 128/512/--, 60, 60]    --> x2^3 to get back to 1.
        x_16 = self.layer3(x_8)    # 1 / 16: [2, 256/1024/--, 30, 30]   --> x2^4 to get back to 1.
        # x_16 = F.dropout(x_16, p=0.3, training=self.training, inplace=False)
        x_32 = self.layer4(x_16)   # 1 / 32: [n, 512/2048/--, 15, 15]   --> x2^5 to get back to 1.

        scores, maps = self.mask_head(x=x_32,
                                      seed=seed,
                                      prngs_cuda=prngs_cuda
                                      )

        # compute M+
        prob = F.softmax(scores, dim=1)
        mpositive = torch.zeros((b, 1, maps.size()[2], maps.size()[3]),
                                dtype=maps.dtype,
                                layout=maps.layout,
                                device=maps.device
                                )
        for i in range(b):  # for each sample
            for j in range(prob.size()[1]):  # sum the: prob(class) * mask(class)
                mpositive[i] = mpositive[i] + prob[i, j] * maps[i, j, :, :]


        # does not work.
        # mpositive = self.mask_head(x_32)  # todo: try x32, x16, both.

        mpos_inter = F.interpolate(input=mpositive,
                                   size=(h, w),
                                   mode='bilinear',
                                   align_corners=ALIGN_CORNERS
                                   )

        return mpos_inter, scores

    def classify(self, x, seed=None, prngs_cuda=None):
        # Resize the image first.
        _, _, h, w = x.shape
        h_s, w_s = int(h * self.scale[0]), int(w * self.scale[1])
        # reshape

        if seed is not None:
            # Detaching is not important since we do not compute any gradient below this instruction.
            # When using multigpu, detaching seems to help obtain reproducible results.
            # It does not guarantee the reproducibility 100%.
            x = F.interpolate(input=x, size=(h_s, w_s), mode='bilinear', align_corners=ALIGN_CORNERS).detach()
        else:
            # It is ok to use this for one single gpu. The code is 100% reproducible.
            x = F.interpolate(input=x, size=(h_s, w_s), mode='bilinear', align_corners=ALIGN_CORNERS)

        x = self.relu1(self.bn1(self.conv1(x)))  # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1.
        x = self.relu2(self.bn2(self.conv2(x)))  # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1.
        x = self.relu3(self.bn3(self.conv3(x)))  # 1 / 2: [2, 128, 240, 240]  --> x2^1 to get back to 1.
        x = self.maxpool(x)  # 1 / 4:  [2, 128, 120, 120]         --> x2^2 to get back to 1.
        x_4 = self.layer1(x)  # 1 / 4:  [2, 64/256/--, 120, 120]   --> x2^2 to get back to 1.
        x_8 = self.layer2(x_4)  # 1 / 8:  [2, 128/512/--, 60, 60]    --> x2^3 to get back to 1.
        x_16 = self.layer3(x_8)  # 1 / 16: [2, 256/1024/--, 30, 30]   --> x2^4 to get back to 1.
        x_32 = self.layer4(x_16)  # 1 / 32: [n, 512/2048/--, 15, 15]   --> x2^5 to get back to 1.

        # classifier at 32.
        scores32, maps32 = self.cl32(x=x_32, seed=seed, prngs_cuda=prngs_cuda)

        # Final
        scores, maps = scores32, maps32

        return scores

    def get_pseudo_binary_mask(self, x):
        """
        Compute a mask by applying a sigmoid function.
        The mask is not binary but pseudo-binary (values are close to 0/1).

        :param x: tensor of size (batch_size, 1, h, w), contain the feature
         map representing the mask.
        :return: tensor, mask. with size (nbr_batch, 1, h, w).
        """
        x = (x - x.min()) / (x.max() - x.min())
        return torch.sigmoid(self.w * (x - self.sigma))

    def apply_mask(self, x, mask):
        """
        Apply a mask (and its complement) over an image.

        :param x: tensor, input image. [size: (nb_batch, depth, h, w)]
        :param mask: tensor, mask. [size, (nbr_batch, 1, h, w)]
        :return: (x_pos, x_neg)
            x_pos: tensor of size (nb_batch, depth, h, w) where only positive
             regions are shown (the negative regions
            are set to zero).
            x_neg: tensor of size (nb_batch, depth, h, w) where only negative
            regions are shown (the positive regions are set to zero).
        """
        mask_expd = mask.expand_as(x)
        x_pos = x * mask_expd
        x_neg = x * (1 - mask_expd)

        return x_pos, x_neg


def load_url(url, model_dir='../pretrained', map_location=torch.device('cpu')):
    """
    Download pre-trained models.
    :param url: str, url of the pre-trained model.
    :param model_dir: str, path to the temporary folder where the pre-trained models will be saved.
    :param map_location: a function, torch.device, string, or dict specifying how to remap storage locations.
    :return: torch.load() output. Loaded dict state.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def test_resnet():
    model = resnet18(pretrained=True)
    print("Testing {}".format(model.__class__.__name__))
    model.train()
    print("Num. parameters: {}".format(sum([p.numel() for p in model.parameters()])))
    cuda = "1"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # torch.cuda.set_device(int(cuda))
        pass

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)
    x = torch.randn(2, 3, 480, 480)
    x = x.to(DEVICE)
    scores_pos, scores_neg, mask = model(x)
    print(x.size(), mask.size())


if __name__ == "__main__":
    import sys

    test_resnet()
