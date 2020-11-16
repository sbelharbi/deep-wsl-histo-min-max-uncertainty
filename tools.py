import datetime as dt
import time
import sys
import numpy as np
import shutil
import glob
import os
from os.path import join, relpath
import getpass
import multiprocessing
import math
from operator import mul
import functools
import yaml
import matplotlib as mlp
import matplotlib.pyplot as plt
import warnings
from PIL import Image, ImageDraw, ImageFont
import tqdm
import numbers
import copy
import argparse
import pickle as pkl
import fnmatch
import ctypes
from multiprocessing import Process, Lock
from io import BytesIO
import zipfile
from collections import OrderedDict
import subprocess

import torch

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, f1_score
from torchvision import transforms
from scipy import interp
import pydensecrf.densecrf as dcrf


class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


class AverageMeter(object):
    """Compute and stores the average and current value."""
    def __init__(self):
        self.values = []
        self.counter = 0

        self.latest_avg = 0

        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, value):
        self.values.append(value)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


class CRF(object):
    """
    CRF class to perform post-processing when called.
    """
    def __init__(self, nbr_classes, n_iter=5, sxyg=(3, 3), sxyb=(80, 80), srgbb=(13, 13, 13)):
        """
        Init. function.
        :param nbr_classes: int, the total number of classes.
        :param n_iter: int, the number of iterations for inference.
        :param sxyg: tuple of int coefficients (sx, sy) of the Gaussian filter.
        :param sxyb: tuple of int coefficients (sx, sy) of the bilateral filter.
        :param srgbb: tuple of the coefficients (sr, sg, sb) of the bilateral filter.
        """
        self.nbr_classes = nbr_classes
        self.n_iter = n_iter
        self.sxyg = sxyg
        self.sxyb = sxyb
        self.srgbb = srgbb

    def __call__(self, img, softmx):
        """
        Post-process the 2d softmax output maps.
        :param img: numpy.ndarray of shape (h, w, depth) of type unint8. The input image.
        :param softmx: numpy.ndarray of shape (c, h, w) of float32. Contains the probbaility maps for each class of
        the `c` classes.
        :return: smooth_prob: numpy.ndarray, matrix of post-processed probabilities of size (c, h, w) of float32
        where the sum over `c` axis is 1. (probability)
        """
        # img
        assert isinstance(img, np.ndarray), 'img must be instance of numpy.ndarray. You provided {} .... [NOT ' \
                                            'OK]'.format(type(img))
        assert img.dtype == np.uint8, 'img dtype must be numpy.uint8. You provided {} .... [NOT OK]'.format(img.dtype)
        assert img.ndim == 3, 'img must be RGB: 3 dims: : (h, w, 3). You provided {} .... [NOT OK]'.format(img.ndim)
        assert img.shape[2] == 3, 'the last dim of img must be 3: (h, w, 3). You provided {} .... [NOT OK]'.format(
            img.shape[2]
        )

        # softmx
        assert isinstance(softmx, np.ndarray), 'softmx must be instance of numpy.ndarray. You provided {} .... [NOT ' \
                                               'OK]'.format(type(softmx))
        assert softmx.dtype == np.float32, 'softmx dtype must be numpy.float32. You provided {} .... [NOT OK]'.format(
            softmx.dtype)
        assert softmx.ndim == 3, 'softmx must have 3 dims: (c, h, w). You provided {} .... [NOT OK]'.format(
            softmx.ndim)
        assert softmx.shape[0] == self.nbr_classes, 'The first dim of softmx must be {}: ({}, w). You provided {} ' \
                                                    '.... [NOT OK]'.format(self.nbr_classes, self.nbr_classes,
                                                                           softmx.shape[0])
        assert (softmx >= 0).all(), 'softmx must be >=0. .... [NOT OK]'
        assert (softmx.sum(axis=0).squeeze() == np.ones_like(softmx.sum(axis=0).squeeze())).all(), 'softmx must ' \
                                                                                                   'probabilities....' \
                                                                                                   '[NOT OK]'

        # img, softmx
        assert img.shape[0] == softmx.shape[1], 'img height {} must be the same as in the softmax {}. They are not ' \
                                                'the same. .... [NOT OK]'.format(img.shape[0], softmx[1])
        assert img.shape[1] == softmx.shape[2], 'img width {} must be the same as in the softmx {}. They are not the ' \
                                                'same. .... [NOT OK]'.format(img.shape[1], softmx[2])

        assert softmx.shape[0] == self.nbr_classes, 'softmx.shape[0] {} must be the same as self.nbr_classes {}. They' \
                                                    'are different. .... [NOT OK]'.format(softmx.shape[0],
                                                                                          self.nbr_classes)

        h, w, _ = img.shape
        d = dcrf.DenseCRF2D(w, h, self.nbr_classes)
        softmx = softmx.reshape((self.nbr_classes, -1))
        U = -np.log(softmx)

        # Add unary potentials
        d.setUnaryEnergy(U)

        # Add pairwise potentials
        # # Adds the locations feature only (color-independent).
        d.addPairwiseGaussian(sxy=self.sxyg, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # # Add  location and color features: (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=self.sxyb, srgb=self.srgbb, rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        # Inference
        Q = d.inference(self.n_iter)
        smooth_prob = np.array(Q)  # shape (nbr_classes, h * w)
        smooth_prob = smooth_prob.reshape((self.nbr_classes, h, w)).astype(np.float32)

        return smooth_prob


def count_nb_params(model):
    """
    Count the number of parameters within a model.

    :param model: nn.Module.
    :return: int, number of learnable parameters.
    """
    return sum([p.numel() for p in model.parameters()])

# ================================================
# Visualisation of the regions of interest in MIL.
# ================================================


class VisualiseMIL(object):
    def __init__(self,
                 alpha=128,
                 floating=3,
                 height_tag=60,
                 bins=100,
                 rangeh=(0, 1),
                 color_map=mlp.cm.get_cmap("seismic"),
                 height_tag_paper=130
                 ):
        """
        A visualisation tool for MIL predictions.

        :param alpha: the transparency value for the overlapped image.
        :param floating: int, number of decimals to display.
        :param height_tag: int, the height of the tag banner.
        :param bins: int, number of bins. Used when one wants to plot the
        distribution of the scores.
        :param rangeh: tuple, default range of the x-axis for the histograms.
        :param color_map: type of the color map to use.
        """
        super(VisualiseMIL, self).__init__()

        self.color_map = color_map  # default color map.
        self.alpha = alpha

        self.bins = bins
        self.rangeh = rangeh

        # precision of writing the probabilities.
        self.prec = "%." + str(floating) + "f"
        self.height_tag = height_tag
        self.height_tag_paper = height_tag_paper  # for the paper.
        self.y = int(self.height_tag / 4)  # y position of the text inside the
        # tag. (first line)
        self.y2 = self.y * 2  # y position of the text inside the tag.
        # (second line)
        self.y3 = self.y * 3  # y position of the text inside the tag.
        # (3rd line)
        self.dx = 10  # how much space to put between LABELS (not word)
        # inside the tag.
        self.space = 10  # (pixels) how much space to leave between images.

        # Fonts:
        self.font_regular = ImageFont.truetype(
            "./fonts/Inconsolata/Inconsolata-Regular.ttf", size=15)
        self.font_bold = ImageFont.truetype(
            "./fonts/Inconsolata/Inconsolata-Bold.ttf", size=15)

        self.font_bold_paper = ImageFont.truetype(
            "./fonts/Inconsolata/Inconsolata-Bold.ttf", size=120)
        self.font_bold_paper_small = ImageFont.truetype(
            "./fonts/Inconsolata/Inconsolata-Bold.ttf", size=80)

        # Colors:
        self.white = "rgb(255, 255, 255)"
        self.green = "rgb(0, 255, 0)"
        self.red = "rgb(255, 0, 0)"
        self.orange = "rgb(255,165,0)"

        # Margin:
        self.left_margin = 10  # the left margin.

        # dim of the input image.
        self.h = None
        self.w = None

    def convert_mask_into_heatmap(self,
                                  input_img,
                                  mask,
                                  binarize=False
                                  ):
        """
        Convert a mask into a heatmap.

        :param input_img: PIL.Image.Image of type float32. The input image.
        :param mask: 2D numpy.ndarray (binary or continuous in [0, 1]).
        :param binarize: bool. If True, the mask is binarized by thresholding
        (values >=0.5 will be set to 1. ELse, 0).
        :return:
        """
        if binarize:
            mask = ((mask >= 0.5) * 1.).astype(np.float32)

        img_arr = self.color_map((mask * 255).astype(np.uint8))  #
        # --> in [0, 1.], (h, w, 4)

        return self.superpose_two_images_using_alpha(
            input_img.copy(),
            Image.fromarray(np.uint8(img_arr * 255)),
            self.alpha)

    @staticmethod
    def superpose_two_images_using_alpha(back,
                                         forg,
                                         alpha
                                         ):
        """
        Superpose two PIL.Image.Image uint8 images.
        images must have the same size.

        :param back: background image. (RGB)
        :param forg: foreground image (L).
        :param alpha: int, in [0, 255] the alpha value.
        :return:R PIL.Image.Image RGB uint8 image.
        """
        # adjust the alpha
        forg.putalpha(alpha)  # https://pillow.readthedocs.io/en/3.1.x/
        # reference/Image.html#PIL.Image.Image.putalpha
        # back.putalpha(int(alpha/4))
        back.paste(forg, (0, 0), forg)

        return back

    @staticmethod
    def drawonit(draw,
                 x,
                 y,
                 label,
                 fill,
                 font,
                 dx
                 ):
        """
        Draw text on an ImageDraw.new() object.

        :param draw: object, ImageDraw.new()
        :param x: int, x position of top left corner.
        :param y: int, y position of top left corner.
        :param label: str, text message to draw.
        :param fill: color to use.
        :param font: font to use.
        :param dx: int, how much space to use between LABELS (not word).
        Useful to compute the position of the next
        LABEL. (future)
        :return:
            . ImageDraw object with the text drawn on it as requested.
            . The next position x.
        """
        draw.text((x, y), label, fill=fill, font=font)
        x += font.getsize(label)[0] + dx

        return draw, x

    def create_tag_input(self,
                         him,
                         wim,
                         label,
                         name
                         ):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message
        over it.

        Dedicated to the input image.

        Written message: "Input: label  (h) him pix. x (w) wim pix."

        :param him: int, height of the image.
        :param wim: int, the width of the image containing the tag.
        :param label: str, the textual tag.
        :param name: str, name of the input image file.
        :return:
        """
        if label is None:
            label = "unknown"
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "In.cl.:",
                                self.white,
                                self.font_regular,
                                self.dx
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                label,
                                self.white,
                                self.font_bold,
                                self.dx
                                )

        x = self.left_margin
        msg = "(h){}pix.x(w){}pix.".format(him, wim)
        self.drawonit(draw,
                      x,
                      self.y2,
                      msg,
                      self.white,
                      self.font_bold,
                      self.dx
                      )

        return img_tag

    def create_tag_pred_mask(self,
                             wim,
                             label,
                             probability,
                             status,
                             f1pos,
                             f1neg
                             ):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message
        over it.

        Dedicated to the predicted map.

        Written message:
        "Class: label  probability % [correct or wrong] (h) him pix. x (w) wim
        pix. #Patches = "
        :param wim: int, width of the image.
        :param label: str, the class name.
        :param probability: float, the probability of the prediction.
        :param status: str, the status of the prediction: "correct", "wrong",
        None. If None, no display of the status.
        :param dice: float or None, Dice index. (if possible)
        :return:
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "Pred.cl.:",
                                self.white,
                                self.font_regular,
                                self.dx
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                label,
                                self.white,
                                self.font_bold,
                                self.dx
                                )

        # Jump to the second line (helpful when the name of the class is long).
        x = self.left_margin
        draw, x = self.drawonit(draw,
                                x,
                                self.y2,
                                "Prob.: {}%".format(
                                    str(self.prec % (probability * 100))),
                                self.white,
                                self.font_regular,
                                self.dx
                                )

        if status == "correct":
            color = self.green
        elif status == "wrong":
            color = self.red
        elif status is None:
            color = self.orange
            status = "predicted"
        else:
            raise ValueError("Unsupported status `{}` .... "
                             "[NOT OK]".format(status))

        draw, x = self.drawonit(draw,
                                x,
                                self.y2,
                                "Status: [",
                                self.white,
                                self.font_regular,
                                0
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y2,
                                "{}".format(status),
                                color,
                                self.font_bold,
                                0
                                )
        self.drawonit(draw,
                      x,
                      self.y2,
                      "]",
                      self.white,
                      self.font_regular,
                      self.dx
                      )

        x = self.left_margin
        f1posstr = "None" if status is None else str(
            self.prec % (f1pos * 100.)) + "%"
        f1negstr = "None" if status is None else str(
            self.prec % (f1neg * 100.)) + "%"
        draw, x = self.drawonit(draw,
                                x,
                                self.y3,
                                "F1+: {}".format(f1posstr),
                                self.white,
                                self.font_regular,
                                self.dx
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y3,
                                "F1-: {}".format(f1negstr),
                                self.white,
                                self.font_regular,
                                self.dx
                                )

        return img_tag

    def create_tag_true_mask(self, wim, status):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message
        over it.

        Dedicated to the true mask.

        Written message:
        "True mask: [known or unknown]"
        :param wim: int, width of the image.
        :param status: str, the status of the prediction: "correct", "wrong",
        None. If None, no display of the status.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "True mask:",
                                self.white,
                                self.font_regular,
                                self.dx
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "[",
                                self.white,
                                self.font_regular,
                                0
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                status,
                                self.green,
                                self.font_bold,
                                0
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "]",
                                self.white,
                                self.font_regular,
                                self.dx
                                )

        return img_tag

    def create_tag_heatmap_pred_mask(self, wim, iter):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message
        over it.

        Dedicated to the predicted mask.

        Written message:
        "Heatmap pred. mask.       [iter.?/Final]"
        :param wim: int, width of the image.
        :param iter: str, the number of iteration when this mask was draw.
        "final" to indicate that this is final
        prediction.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "Heatmap ped.mask.",
                                self.white,
                                self.font_regular,
                                self.dx
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "[",
                                self.white,
                                self.font_regular,
                                0
                                )
        draw, x = self.drawonit(draw,
                                x,
                                self.y,
                                "iter.{}".format(iter),
                                self.green,
                                self.font_bold,
                                0
                                )
        self.drawonit(draw,
                      x,
                      self.y,
                      "]",
                      self.white,
                      self.font_regular,
                      0
                      )

        return img_tag

    def get_class_name(self, name_classes, i):
        """
        Get the str name of the class based on the integer.

        :param name_classes: dict, {"class_name": int}.
        :param i: int or None, the class ID. None if unknown label.
        :return: str, the class name.
        """
        msg = "'i' must be an integer. Provided: {}, {}".format(i, type(i))
        assert isinstance(i, int) or i is None, msg
        msg = "class ID `{}` does not exist within possible IDs `{}` " \
              ".... [NOT OK]".format(i, list(name_classes.values()))
        assert (i in list(name_classes.values())) or (i is None), msg

        if i is not None:
            return list(name_classes.keys())[
                list(name_classes.values()).index(i)]
        else:
            return "Unknown"

    def convert_array_into_hist_PIL_img_do_roc(self,
                                               mask,
                                               bins,
                                               rangeh,
                                               true_mask
                                               ):
        """
        Compute:
        1. The histogram of a numpy array and plot it, then, convert it
        into a PIL.Image.Image image.
        2. Compute ROC curve (and the area under it), and plot it, then convert
        it into  aPIL.Image.Image.

        :param mask: numpy.ndarray, 2D matrix containing the predicted mask
         (continous).
        :param bins: int, number of bins in the histogram.
        :param rangeh: tuple, range of the histogram.
        :param true_mask: numpy.ndarray, 2D matri containing the true mask
        (binary) where 1 indicates the glands.
        :return: PIL.Image.Image uint8 RGB image.
        """
        floating = 4
        prec = "%." + str(floating) + "f"
        font_sz = 10
        lw = 2

        fig = plt.figure()
        # Histogram.
        fig.add_subplot(221)

        plt.hist(mask.ravel(),
                 bins=bins,
                 weights=np.ones_like(mask.ravel()) / float(mask.size),
                 range=rangeh
                 )
        plt.xlabel("x: mask values")
        plt.ylabel("y: P(x0 <= x <= x1)")

        # ROC
        fig.add_subplot(222)

        tpr, fpr, roc_auc = compute_roc_curve_once(
            true_mask.ravel(),
            mask.ravel().astype(np.float32)
        )
        plt.plot(fpr, tpr, color='darkorange', lw=lw,
                 label='ROC(AUC: {})'.format(prec % roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC. AUC: {}'.format(prec % roc_auc))
        plt.legend(loc='lower right', fancybox=True, shadow=True,
                   prop={'size': font_sz})
        plt.tight_layout()

        # Precision-recall
        fig.add_subplot(223)

        precision, recall, p_r_auc = compute_precision_recall_curve_once(
            true_mask.ravel(), mask.ravel().astype(np.float32))
        plt.plot(recall, precision, color='darkorange', lw=lw,
                 label='Precision-recall(AUC: {})'.format(prec % p_r_auc))
        plt.plot([0, 1], [0.5, 0.5], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-recall. AUC: {}'.format(prec % p_r_auc))
        plt.legend(loc='lower right', fancybox=True, shadow=True,
                   prop={'size': font_sz})
        plt.tight_layout()

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(data, mode="RGB")

        plt.close()

        return img

    def create_hists(self, mask, bins, rangeh, k, true_mask):
        """
        Creates:
         1. Histogram of the heatmap of the predicted mask.
         [it is positioned at the end of the entire image)
         2. Plots the ROC curve and computes the area under it.

        :param mask: numpy.ndarray float32 2D matrix of size (h, w).
        The mask (non-binarized).
        :param bins: int, number of bins in the histogram.
        :param rangeh: tuple, range of the histogram.
        :param k: int, number of the images that will be plotted in the final
        image.
        :param true_mask: numpy.ndarray float 2D matrix of size (h, w). The
        true mask (binary) where 1 indicates the
        glands.
        :return: PIL.Image.Image RGB, uint8 image. an image where all the
        images are left zero except the image
        corresponding to the heatmap of the mask where we plot its histogram.
        """
        img_hist = self.convert_array_into_hist_PIL_img_do_roc(mask,
                                                               bins,
                                                               rangeh,
                                                               true_mask
                                                               )
        w_his, h_his = img_hist.size

        h, w = mask.shape
        # We resize the histogram image into half its original size.
        # img_hist = img_hist.resize((int(w_his / 2), int(h_his / 2)))
        # img_out = Image.new("RGB", (k * w + (k - 1) * self.space, int(h_his / 2)))
        img_out = Image.new("RGB", (k * w + (k - 1) * self.space, h_his))
        img_out.paste(img_hist, ((k - 1) * (w + self.space), 0))

        return img_out

    def __call__(self,
                 input_img,
                 probab,
                 pred_label,
                 pred_mask,
                 f1pos,
                 f1neg,
                 name_classes,
                 iter,
                 pred_mask_bin=None,
                 use_tags=True,
                 label=None,
                 mask=None,
                 show_hists=True,
                 bins=None,
                 rangeh=None,
                 name_file=""
                 ):
        """
        Visualise MIL prediction.

        :param input_img: PIL.Image.Image RGB uint8 image. of size (h, w).
        :param probab: float, the probability of the predicted class.
        :param pred_label: int, the ID of the predicted class. We allow the
        user to provide the prediction.
        Generally, it can be inferred from the scores.
        :param pred_mask: numpy.ndarray, 2D float matrix of size (h, w). The
        predicted mask.
        :param f1pos: float [0, 1]. Dice index over the positive regions.
        :param f1neg: float [0, 1]. Dice index over the negative regions.
        :param name_classes: dict, {"class_name": int}.
        :param iter: str, indicates the iteration when this call happens.
        "Final" to indicate this is the final prediction.
        :param pred_mask_bin: same as pred_mask but binary if available.
        :param use_tags: True/False, if True, additional information will be
        allowed to be displayed.
        :param label: int or None, the the ID of the true class of the
        input_image. None: if the true label is unknown.
        :param mask: numpy.ndarray or None, 2D float matrix of size (h, w).
        The true mask. None if the true mask is
        unknown.
        :param show_hists: True/False. If True, a histogram of the scores in
        each map will be displayed.
        :param bins: int, number of bins in the histogram. If None, self.bins
        will be used.
        :param rangeh: tuple, range of the histogram. If None, self.rangeh will
        be used.
        :param name_file: str, name of the input image file.
        :return: PIL.Image.Image RGB uint8 image.
        """
        msg = "'input_image' type must be `{}`, but we found `{}` " \
              "".format(Image.Image, type(input_img))
        assert isinstance(input_img, Image.Image), msg

        msg = "'probab' must of type `{}` but we found `{}` ".format(
            float, type(probab))
        assert isinstance(probab, float), msg

        msg = "'pred_label' must be of type `{}` but we found `{}` .... [NOT " \
                                            "OK]".format(int, type(pred_label))
        assert isinstance(pred_label, int), msg
        msg = "'label' must be `{}` or None. We found `{}` ".format(
            int, type(label))
        assert (isinstance(label, int) or label is None), msg
        msg = "'pred_mask' must be `{}`, but we found `{}` ".format(
            np.ndarray, type(mask))
        assert isinstance(pred_mask, np.ndarray), msg

        msg = "'mask' must be `{}` or None, but we found `{}`".format(
            np.ndarray, type(mask))
        assert isinstance(mask, np.ndarray) or mask is None, msg

        msg = "'name_classes' must be of type `{}`, but we found `{}` " \
              "".format(dict, type(name_classes))
        assert isinstance(name_classes, dict), msg

        msg = "'use_tags' must be of type `{}`, but we found `{}` ".format(
            bool, type(use_tags))
        assert isinstance(use_tags, bool), msg

        wim, him = input_img.size
        msg = "predicted mask {} and image shape ({}, {}) do not match " \
              "".format(pred_mask.shape, him, wim)
        assert wim == pred_mask.shape[1] and him == pred_mask.shape[0], msg

        # convert masks into images.
        if mask is None:
            true_mask = np.zero((him, wim), dtype=np.float32)
        else:
            true_mask = mask

        mask_img = self.convert_mask_into_heatmap(input_img,
                                                  true_mask,
                                                  binarize=False
                                                  )

        pred_mask_img = self.convert_mask_into_heatmap(input_img,
                                                       pred_mask,
                                                       binarize=False
                                                       )
        pred_mask_bin_img = self.convert_mask_into_heatmap(
            input_img,
            pred_mask if pred_mask_bin is None else pred_mask_bin,
            binarize=True if pred_mask_bin is None else False
        )

        # create tags
        if use_tags:
            input_tag = self.create_tag_input(him,
                                              wim,
                                              self.get_class_name(
                                                  name_classes, label),
                                              name_file
                                              )
            true_mask_tag = self.create_tag_true_mask(
                wim,
                "unknown" if mask is None else "known"
            )
            class_name = self.get_class_name(name_classes, pred_label)
            if label is not None:
                status = "correct" if label == pred_label else "wrong"
            else:
                status = "unknown"
            pred_mask_tag = self.create_tag_pred_mask(wim,
                                                      class_name,
                                                      probab,
                                                      status,
                                                      f1pos,
                                                      f1neg
                                                      )
            heat_pred_mask_tag = self.create_tag_heatmap_pred_mask(wim, iter)

        # creates histograms
        nbr_imgs = 4
        if show_hists:
            histogram = self.create_hists(pred_mask,
                                          bins,
                                          rangeh,
                                          nbr_imgs,
                                          true_mask
                                          )

        img_out = Image.new("RGB",
                            (wim * nbr_imgs + self.space * (nbr_imgs - 1), him)
                            )
        if use_tags:
            img_out = Image.new("RGB",
                                (wim * nbr_imgs + self.space * (nbr_imgs - 1),
                                 him + self.height_tag)
                                )

        list_imgs = [input_img,
                     mask_img,
                     pred_mask_bin_img,
                     pred_mask_img]
        list_tags = [input_tag,
                     true_mask_tag,
                     pred_mask_tag,
                     heat_pred_mask_tag]
        for i, img in enumerate(list_imgs):
            img_out.paste(img, (i * (wim + self.space), 0), None)
            if use_tags:
                img_out.paste(list_tags[i], (i * (wim + self.space), him))

        if show_hists:
            wh, hh = histogram.size
            wnow, hnow = img_out.size
            assert wh == wnow

            img_final = Image.new("RGB", (wnow, hh + hnow))
            img_final.paste(img_out, (0, 0), None)
            img_final.paste(histogram, (0, hnow), None)
        else:
            img_final = img_out

        return img_final


class VisualizePaper(VisualiseMIL):
    """
    Visualize overlapped images for the paper.
    """

    def create_tag_input(self, him, wim, label, name_file):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the input image.

        Written message: "Input: label  (h) him pix. x (w) wim pix."

        :param him: int, height of the image.
        :param wim: int, the width of the image containing the tag.
        :param label: str, the textual tag.
        :param name: str, name of the input image file.
        :return:
        """
        if label is None:
            label = "unknown"
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)

        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, "Input: {} | ".format(name_file), self.white, self.font_regular,
                                self.dx)
        draw, x = self.drawonit(draw, x, self.y, label, self.white, self.font_bold, self.dx)

        # msg = "(h){}pix.x(w){}pix.".format(him, wim)
        # self.drawonit(draw, x, self.y, msg, self.white, self.font_bold, self.dx)

        return img_tag

    def create_tag_pred_mask(self, wim, msg1, msg2):
        """

        :param wim:
        :param msg1:
        :param msg2:
        :return:
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y, msg1, self.white, self.font_regular, self.dx)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, self.y2, msg2, self.white, self.font_regular, self.dx)

        return img_tag

    def create_tag_paper(self, wim, msg, font=None):
        """
        Craeate a VISIBLE tag for the paper.

        :param wim: int, image width.
        :param msg: message (str) to display.
        :return:
        """
        if font is None:
            font = self.font_bold_paper

        img_tag = Image.new("RGB", (wim, self.height_tag_paper), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin
        draw, x = self.drawonit(draw, x, 0, msg, self.white, font, self.dx)

        return img_tag

    def create_tag_true_mask(self, wim):
        """
        Create a PIL.Image.Image of RGB uint8 type. Then, writes a message over it.

        Dedicated to the true mask.

        Written message:
        "True mask: [known or unknown]"
        :param wim: int, width of the image.
        :param status: str, the status of the prediction: "correct", "wrong", None. If None, no display of the status.
        :return: PIL.Image.Image.
        """
        img_tag = Image.new("RGB", (wim, self.height_tag), "black")

        draw = ImageDraw.Draw(img_tag)
        x = self.left_margin

        draw, x = self.drawonit(draw, x, self.y, "True mask", self.white, self.font_regular, self.dx)

        return img_tag

    def __call__(self,
                 name_classes,
                 img,
                 label,
                 name_file,
                 true_mask,
                 per_method,
                 methods,
                 order_methods,
                 show_heat_map=False,
                 show_tags=False,
                 show_tag_paper=False,
                 use_small_font_paper=False
                 ):
        """

        :param img:
        :param name_file:
        :param true_mask:
        :param per_method:
        :param show_heat_map: Bool. If true, we show heat maps. Else, we show binary masks.
        :param show_tags: Bool. If True, we show tags below the images.
        :return:
        """

        assert isinstance(img, Image.Image), "'input_image' type must be `{}`, but we found `{}` .... [NOT OK]" \
                                             "".format(Image.Image, type(img))
        assert isinstance(true_mask, np.ndarray) or true_mask is None, "'mask' must be `{}` or None, but we found `{}` .... [" \
                                                                       "NOT OK]".format(np.ndarray, type(true_mask))
        assert isinstance(name_classes, dict), "'name_classes' must be of type `{}`, but we found `{}` .... [NOT " \
                                               "OK]".format(dict, type(name_classes))

        wim, him = img.size
        assert wim == true_mask.shape[1] and him == true_mask.shape[0], "predicted mask {} and image shape ({}, " \
                                                                        "{}) do not " \
                                                                        "match .... [NOT OK]".format(
            true_mask.shape, him, wim)

        mask_img = self.convert_mask_into_heatmap(img, true_mask, binarize=False)
        true_mask_tag = self.create_tag_true_mask(wim)

        list_imgs = [img, mask_img]
        input_tag = self.create_tag_input(him, wim, self.get_class_name(name_classes, label), name_file)
        list_tags = [input_tag, true_mask_tag]
        for k in order_methods:
            if per_method[k]["pred_label"] is not None:
                pred_label = self.get_class_name(name_classes, int(per_method[k]["pred_label"]))
            else:
                pred_label = "--"

            f1_foreg = per_method[k]["f1_score_forg_avg"]
            f1_back = per_method[k]["f1_score_back_avg"]
            msg1 = "F1+: {}%  F1-: {}% ".format(self.prec % f1_foreg, self.prec % f1_back)
            msg2 = "Prediction: {} (Method: {})".format(pred_label, methods[k])
            list_tags.append(self.create_tag_pred_mask(wim, msg1, msg2))

            if show_heat_map:
                list_imgs.append(self.convert_mask_into_heatmap(img, per_method[k]["pred_mask"], binarize=False))
            else:
                list_imgs.append(self.convert_mask_into_heatmap(img, per_method[k]["binary_mask"], binarize=False))

        nbr_imgs = len(methods.keys()) + 2
        font = self.font_bold_paper
        if use_small_font_paper:
            font = self.font_bold_paper_small

        tag_paper_img = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), self.height_tag_paper))
        list_tags_paper = [self.create_tag_paper(wim, "Input", font), self.create_tag_paper(wim, "True mask", font)]
        for k in order_methods:
            list_tags_paper.append(self.create_tag_paper(wim, methods[k], font))

        img_out = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), him))
        if show_tags:
            img_out = Image.new("RGB", (wim * nbr_imgs + self.space * (nbr_imgs - 1), him + self.height_tag))
        for i, img in enumerate(list_imgs):
            img_out.paste(img, (i * (wim + self.space), 0), None)
            tag_paper_img.paste(list_tags_paper[i], (i * (wim + self.space), 0), None)
            if show_tags:
                img_out.paste(list_tags[i], (i * (wim + self.space), him))

        if show_tag_paper:
            img_out_final = Image.new("RGB", (img_out.size[0], img_out.size[1] + self.height_tag_paper))
            img_out_final.paste(img_out, (0, 0), None)
            img_out_final.paste(tag_paper_img, (0, img_out.size[1]), None)
            img_out = img_out_final

        return img_out, tag_paper_img


class VisualizeImages(VisualizePaper):
    """
    Visualize images from dataset.
    """
    def __call__(self, name_classes, list_images, list_true_masks, list_labels, rows, columns, show_tags=False):
        """

        :param name_classes:
        :param list_images:
        :param list_true_masks:
        :return:
        """
        for i, msk in enumerate(list_true_masks):
            assert isinstance(msk, np.ndarray), "'mask' must be `{}` or None, but we found `{}` .... [" \
                                                "NOT OK]".format(np.ndarray, type(msk))
        for i, img in enumerate(list_images):
            assert isinstance(img, Image.Image), "'input_image' type must be `{}`, but we found `{}` .... [NOT OK]" \
                                                 "".format(Image.Image, type(img))

        assert isinstance(name_classes, dict), "'name_classes' must be of type `{}`, but we found `{}` .... [NOT " \
                                               "OK]".format(dict, type(name_classes))

        assert rows == 1, "We support only 1 row!!!! You asked for {}".format(rows)
        assert len(list_images) == len(list_true_masks), "list_images and list_true_masks must have the same number " \
                                                         "of elements. You provided: len(list_images) = {}," \
                                                         "len(list_true_masks) = {}".format(len(list_images),
                                                                                            len(list_true_masks))

        nbr_imgs = len(list_images)
        extra_w_space = self.space * (nbr_imgs - 1)
        w_out = 0
        max_h = 0
        for im in list_images:
            w_out += im.size[0]
            max_h = max(max_h, im.size[1])

        w_out += extra_w_space
        img_out = Image.new("RGB", (w_out, max_h))
        img_tags = Image.new("RGB", (w_out, self.height_tag_paper))
        i = 0
        p = 0
        for im, msk in zip(list_images, list_true_masks):
            wim = im.size[0]
            tmp = self.convert_mask_into_heatmap(im, msk, binarize=False)
            img_out.paste(tmp, (p + i * self.space, 0), None)
            img_tags.paste(self.create_tag_paper(wim, self.get_class_name(name_classes, list_labels[i])),
                           (p + i * self.space, 0), None)
            p += wim
            i += 1

        if show_tags:
            final_out = Image.new("RGB", (w_out, max_h + self.height_tag_paper))
        else:
            final_out = Image.new("RGB", (w_out, max_h))
        final_out.paste(img_out, (0, 0), None)
        if show_tags:
            final_out.paste(img_tags, (0, max_h), None)

        return final_out


def log(fname, txt):
    with open(fname, 'a') as f:
        f.write(txt + "\n")


def get_exp_name(args):
    """Create the name of the exp based on its configuration.
    Input:
        args: object. Contains the configuration of the exp.
    """
    time.sleep(np.random.randint(1, 5))  # in case many exps start in the same time, ..., wait a little bit.
    # TODO: create this function: add other info in the name of the experience.
    time_exp = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    name = "b-sz-{}-{}-{}-fold-{}-ds-{}".format(
        args.batch_size, args.model["name"], time_exp, args.fold, args.dataset)
    return name


def get_cpu_device():
    """
    Return CPU device.
    :return:
    """
    return torch.device("cpu")


def get_device(args):
    """
    Returns the device on which the computations will be performed.
    Input:
        args: object. Contains the configuration of the exp that has been read from the yaml file.

    Return:
        torch.device() object.
    """
    username = getpass.getuser()
    if username == "sbelharbi":  # LIVIA
        device = torch.device("cuda:" + args.cudaid if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.cudaid))

    return device


def load_pre_pretrained_model(model, path_file, strict):
    """
    Load parameters in path_file into the model.
    The mapping is done on CPU.
    model needs to be on CPU. If it is on GPU, an error is raised. Deal with it on your own.
    If Path_file indicates parameters that are on GPU, they are moved to CPU.

    :param model: instance of torch.nn.Module
    :param path_file: str, path to the file containing the parameters.
    :param strict: Bool. If True, the loading must be strict.
    :return: model, torch.nn.Module with the parameter loaded.
    """
    # check the target is is on CPU:
    if next(model.parameters()).is_cuda:
        raise ValueError("We expected the target model to be on CPU. You need to move to CPU then, load your "
                         "parameters. Exiting .... [NOT OK]")
    if not os.path.exists(path_file):
        raise ValueError("File {} does not exist. Exiting .... [NOT OK]")

    model.load_state_dict(torch.load(path_file, map_location=get_cpu_device()),
                          strict=strict)
    print("Parameters have been loaded successfully from {} .... [OK]".format(path_file))

    return model


def copy_model_params_from_gpu_to_cpu(model_src, model_trg):
    """
    Copies the parameters of the model on GPU to the parameters of the model on CPU.
    :param model_src: model on GPU.
    :param model_trg: model on CPU.
    :return:
    """
    state_dict_src = model_src.state_dict()
    state_dict_trg = model_trg.state_dict()

    for k in state_dict_src.keys():
        state_dict_trg[k] = copy.deepcopy(state_dict_src[k].cpu())  # Expensive operation (move from GPU to CPU).

    model_trg.load_state_dict(state_dict_trg)

    return model_trg


def copy_model_state_dict_from_gpu_to_cpu(model_src_gpu):
    """
    Copies the state dict of the model on GPU to CPU.
    :param model_src_gpu: model on GPU.
    :return: new_state_dict: the model_src_gpu state dict in CPU.
    """
    state_dict_gpu = model_src_gpu.state_dict()
    new_state_dict = OrderedDict()

    for ks, vs in state_dict_gpu.items():
        # Example of name of parameters when using multi-gpus: module.layer4.2.bn3.weight
        # For the same parameter on a single gpu: layer4.2.bn3.weight
        if "module." in ks:
            assert os.environ["ALLOW_MULTIGPUS"] == "True", "The word 'module' is expected in the sub-modules name " \
                                                            "only when using multigpu. We found the word 'module' but" \
                                                            "it does not seem that we are in a a multigpu mode. " \
                                                            "Exiting .... [NOT OK]"
            ks = ks.replace("module.", "")
        new_state_dict[ks] = copy.deepcopy(vs.cpu())  # # to be safe, we use deepcopy.
        # Expensive operation (move from GPU to CPU).

    return new_state_dict


def get_state_dict(model):
    """
    Get a COPY of  the state dict of a model.
    :param model: a model.
    :return:new_state_dict: the state of the model. It is on the same device as the model.
    """
    state_dict_gpu = model.state_dict()
    new_state_dict = OrderedDict()

    for ks, vs in state_dict_gpu.items():
        # Example of name of parameters when using multi-gpus: module.layer4.2.bn3.weight
        # For the same parameter on a single gpu: layer4.2.bn3.weight
        if "module." in ks:
            assert os.environ["ALLOW_MULTIGPUS"] == "True", "The word 'module' is expected in the sub-modules name " \
                                                            "only when using multigpu. We found the word 'module' but" \
                                                            "it does not seem that we are in a a multigpu mode. " \
                                                            "Exiting .... [NOT OK]"
            ks = ks.replace("module.", "")
        new_state_dict[ks] = copy.deepcopy(vs)  # to be safe, we use deepcopy

    return new_state_dict


def get_rootpath_2_dataset(args):
    """
    Returns the root path to the dataset depending on the server.
    :param args: object. Contains the configuration of the exp that has been
    read from the yaml file.
    :return: baseurl, a str. The root path to the dataset independently from
    the host.
    """
    datasetname = args.dataset
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = "{}/datasets".format(os.environ["EXDRIVE"])
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = "{}/datasets".format(os.environ["NEWHOME"])
    elif "CC_CLUSTER" in os.environ.keys():
        if "SLURM_TMPDIR" in os.environ.keys():
            # if we are running within a job use the node disc:  $SLURM_TMPDIR
            baseurl = "{}/datasets".format(os.environ["SLURM_TMPDIR"])
        else:
            # if we are not running within a job, use the scratch.
            # this cate my happen if someone calls this function outside a job.
            baseurl = "{}/datasets".format(os.environ["SCRATCH"])

    msg_unknown_host = "Sorry, it seems we are enable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "So, we recommend you add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    if datasetname == "bach-part-a-2018":
        baseurl = join(baseurl, "ICIAR-2018-BACH-Challenge")
    elif datasetname == "fgnet":
        baseurl = join(baseurl, "FGNET")
    elif datasetname == "afad-lite":
        baseurl = join(baseurl, "tarball-lite")
    elif datasetname == "afad-full":
        baseurl = join(baseurl, "tarball")
    elif datasetname == "Caltech-UCSD-Birds-200-2011":
        baseurl = join(baseurl, "Caltech-UCSD-Birds-200-2011")
    elif datasetname == 'Oxford-flowers-102':
        baseurl = join(baseurl, 'Oxford-flowers-102')
    elif datasetname == 'historical-color-image-decade':
        baseurl = join(baseurl, 'HistoricalColor-ECCV2012')
    elif datasetname == 'cifar-10':
        baseurl = join(baseurl, 'cifar-10')
    elif datasetname == 'cifar-100':
        baseurl = join(baseurl, 'cifar-100')
    elif datasetname == 'svhn':
        baseurl = join(baseurl, 'svhn')
    elif datasetname == 'mnist':
        baseurl = join(baseurl, 'mnist')
    elif datasetname == "glas":
        baseurl = join(baseurl,
                       "GlaS-2015/Warwick QU Dataset (Released 2016_07_08)")

    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def get_abs_path_preprocessed_dir_path(args):
    """
    Get the absolute path of the directory where preprocessed files are stored.
    It depends on the dataset name, the number of classes, and the fold.

    If the folder does not exist, it creates it.

    Inside this folder there is 3 files: train, valid, and test files (the results of the preprocessing).
    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: path: str, absolute path to the directory of the preprocessed data.
    """
    baseurl = get_rootpath_2_dataset(args)
    desc = ""
    if args.preprocessor:
        if "stain" in args.preprocessor.keys():
            desc = args.preprocessor["stain"]["method"]
    relative_fold_path = join(
        "preprocessed_" + args.dataset + "_" + desc, str(len(args.name_classes)) + "_classes", "fold" + str(args.fold)
    )
    abs_path = join(baseurl, relative_fold_path)

    # create the folder is not exist:
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

    return abs_path


def check_prepreprocessing_if_done(args):
    """
    Check if the preprocessing of the data was already done or not.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: True/False: True if the preprocessing has been already done.
    """
    outD_preprocessed = get_abs_path_preprocessed_dir_path(args)

    check = [
        os.path.isfile(join(outD_preprocessed, "train_" + str(args.fold), "metadata.pkl")),
        os.path.isfile(join(outD_preprocessed, "valid_" + str(args.fold), "metadata.pkl")),
        os.path.isfile(join(outD_preprocessed, "test_" + str(args.fold), "metadata.pkl"))
    ]
    return all(check)


def check_target_stain_path(args, train_samples):
    """
    Check if the args has a preprocessing step to do.
    If yes, check if there is a stain normalization.
    If yes, check if the target path (relative) was provided.
    if Yes:
        make the path absolute.
    If not, select a random  path (they are already absolute) from the train samples, then set the target variable to
    the selected path.
    Return the args updated if there has been an update to the target, or return the original args.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :param train_samples: list of str ABSOLUTE path to the train images.
    :return: args: object. Same as input args. See above explanation.
    """
    if args.preprocessor:
        if "stain" in args.preprocessor.keys():
            if args.preprocessor["stain"]["target"]:  # it does exist and it was setup to something other than None, "".
                # set the ABSOLUTE path.
                if args.dataset == "bc18bch":
                    args.preprocessor["stain"]["target"] = join(
                        get_rootpath_2_dataset(args), args.preprocessor["stain"]["target"]
                    )
                    return args
                else:
                    raise ValueError("Dataset {} unknown. Re-check .... [NOT OK]".format(args.dataset))
            else:  # it does not exist. In this case, select randomly an image (absolute path),
                # The image is selected from the train set.
                args.preprocessor["stain"]["target"] = train_samples[np.random.choice(np.arange(len(train_samples)))]
                warnings.warn("You haven't specify the path of the target image from which the stain target will be "
                              "extracted. We choose randomly the following image from the train set as a "
                              "target:".format(args.preprocessor["stain"]["target"]))
                return args

        else:
            return args
    else:
        return args


def copy_code(dest):
    """Copy code to the exp folder for reproducibility.
    Input:
        dest: path to the destination folder (the exp folder).
    """
    # extensions to copy.
    exts = ["py", "sh"]
    flds_files = ["."]
    flds_all = ["deepmil"]

    for fld in flds_all:
        shutil.copytree(fld, join(dest, fld))

    for fld in flds_files:
        for ext in exts:
            files = glob.iglob(os.path.join(fld, "*." + ext))
            for file in files:
                if os.path.isfile(file):
                    shutil.copy(file, dest)


def final_processing(model, dataloader, dataset, dataset_name, test_fn, criterion, device, epoch, callback,
                     log_file, OUTD, args, save_pred_for_later_comp=False):
    """
    Perform the final computations once the training is over.
    And do some final stuff.
    :param model: model.
    :param dataloader: dataloader.
    :param dataset: dataset.
    :param dataset_name: str, name of the set: "Train", "Valid", "Test". (created with `set_for_eval`=True.
    :param test_fn: function, deepmil.train.validate()
    :param criterion: function, evaluation function such as torch.nn.CrossEntropyLoss().
    :param device: torch.device()
    :param epoch: int, epoch where the (best) model was found.
    :param callback: callback.
    :param log_file: the log file.
    :param OUTD: str, output directory of the current experiment.
    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :param save_pred_for_later_comp: Bool. If True, some statistics will be saved for later comparison with other
    methods.
    """
    visualiser = VisualiseMIL(alpha=args.alpha, floating=args.floating, height_tag=args.height_tag, bins=args.bins,
                              rangeh=args.rangeh)

    def draw_regions(datasetx, namex, masksx, predictionsx, probsx, f1posx, f1negx):
        """
        Plot the regions of interest.
        :param datasetx: instance of loader.PhotoDataset.
        :param namex: str, name of the dataset.
        :param masksx: np.ndarray 2D float32 matrix of size (h, w). Mask of the positive region.
        :param predictionsx: np.ndarray, numpy vector of numpy.int64 of predictions. ID of the predicted labels.
        :param probsx: np.ndarray, probability of the predicted classes.
        :param f1posx: np.ndarray, dice of each image over positive regions.
        :param f1negx: np.ndarray, dice of each image over negative regions.
        :return:
        """
        n = len(datasetx)
        outd_data = join(OUTD, namex.lower(), "prediction")  # should already been created.
        if not os.path.exists(outd_data):
            os.makedirs(outd_data)

        print("Saving the prediction masks of the {} dataset. It is going to take some time .... [OK]".format(namex))
        save_pred = dict()

        # Truth
        save_pred["images_path"] = []  # relative path to the image.
        save_pred["masks_path"] = []  # relative path to the mask.
        save_pred["labels"] = []  # contains the true labels (int) to compute the classification error.

        save_pred["pred_masks"] = []  # predicted binary mask (to compute dice index: F1+, F1-).
        save_pred["pred_masks_c"] = []  # predicted continuous mask to be used for CRF post-processing.
        save_pred["pred_labels"] = []  # contains the image label predictions (int). Useful for printing the predicted
        # class.
        save_pred["pred_prob"] = []  # for our method: it contains the prob. of the predicted class. For the
        # other methods, it contains the prob. of each class.

        save_pred["method"] = "Ours" if args.nbr_times_erase == 0 else "Ours + recu.er."
        save_pred["name_classes"] = args.name_classes  # name class: int.
        save_pred["dataset"] = args.dataset  # name of the dataset. Useful to get the full path of the files.

        true_labels = []
        # metrics_ = dict()
        rootpath = get_rootpath_2_dataset(args)
        # zipout = zipfile.ZipFile(join(OUTD, namex.lower(), "prediction.zip"), "w")

        for i in tqdm.tqdm(range(n), ncols=80, total=n):
            img = datasetx.get_original_input_img(i)  # PIL.Image.Image uint8 RGB image.
            label = datasetx.get_original_input_label_int(i)  # int.
            true_labels.append(label)
            true_mask = np.array(datasetx.get_original_input_mask(i))
            true_mask = (true_mask != 0).astype(np.float32)
            mask = copy.deepcopy(masksx[i])
            prediction = int(predictionsx[i])
            prob = probsx[i]
            f1posi = f1posx[i]
            f1negi = f1negx[i]
            # Speed up saving. Del. later.
            # if i < 200:
            img_visu = visualiser(
                img, prob, prediction, mask, f1posi, f1negi, args.name_classes, "Final",
                use_tags=args.use_tags, label=label, mask=true_mask, show_hists=args.show_hists, bins=args.bins,
                rangeh=args.rangeh
            )
            name_file = datasetx.absolute_paths_imgs[i].split(os.sep)[-1].split(".")[0]  # e.g. 'train_13'

            # We use zip files to avoid overloading the disc with files. It is SLIGHTLY faster than writing files
            # in disc (mainly/probably due to compression).
            # Write in file-in-memory
            # fobj = BytesIO()  # this is a file object
            # img_visu.save(fobj, format=args.extension[1])  # save in file
            # object in memory.
            # compress it in memory in the zip file.
            # zipout.writestr(name_file + "." + args.extension[0],
            # fobj.getvalue())
            # run `unzip -o prediction.zip -d prediction` to decompress the file.

            # Expensive operation: DISC I/O.
            img_visu.save(join(outd_data, name_file + "." + args.extension[0]), args.extension[1], optimize=True)

            # # Compute metrics sample per sample!!!
            # tmp = compute_metrics(true_labels=[label], pred_labels=[prediction], true_masks=[true_mask],
            #                       pred_masks=[mask], binarize=True, ignore_roc_pr=True, average=False)
            # for key in tmp.keys():
            #     if key in metrics_.keys():
            #         metrics_[key] += tmp[key]
            #     else:
            #         metrics_[key] = tmp[key]

            # Things to save for later: relative path to the image, mask, and label.
            # Predicted binary mask,

            # Find the relative path
            save_pred["images_path"].append(relpath(datasetx.samples[i][0], rootpath))
            save_pred["masks_path"].append(relpath(datasetx.samples[i][1], rootpath))
            save_pred["labels"].append(label)

            save_pred["pred_masks"].append(mask >= 0.5)  # binary to save RAM space.
            save_pred["pred_masks_c"].append(mask)  # continuous mask. Needed for CRF post-processing.
            save_pred["pred_labels"].append(prediction)
            save_pred["pred_prob"].append(prob)

        # Normalize the metrics.
        # for key in metrics_.keys():
        #     metrics_[key] /= float(n)

        # zipout.close()
        cmdx = ["cd {} ".format(join(OUTD, namex.lower())),
                "tar -cf {}.tar.gz {} ".format("prediction", "prediction"),
                # "rm -r {} ".format("prediction")
                ]
        cmdx = " && ".join(cmdx)
        subprocess.run(cmdx, shell=True, check=True)

        if save_pred_for_later_comp:
            for_later = join(OUTD, namex.lower(), "stats_for_comp_{}.pkl".format(namex.lower()))
            with open(for_later, "wb") as flater:
                pkl.dump(save_pred, flater, protocol=pkl.HIGHEST_PROTOCOL)

        print("Done saving the prediction images for the {} dataset .... [OK]".format(namex))

    # for dataloader, ssample, name, dataset in zip(dataloaders, save_sample, names, datasets):
    log(log_file, "==============================================================================================: \n")
    log(log_file, "\t\t\t\t\t\t {}: \n".format(dataset_name))
    log(log_file, "==============================================================================================: \n")
    log(log_file, "Best Epoch: {} \n".format(epoch))
    stats = init_stats(train=False)
    #
    stats, stats_now, pred = test_fn(model, dataset, dataloader, criterion, device, stats, epoch, callback, log_file,
                                     dataset_name)

    total_loss = stats_now["total_loss"].mean()
    loss_pos = stats_now["loss_pos"].mean()
    loss_neg = stats_now["loss_neg"].mean()
    loss_class_seg = stats_now["loss_class_seg"].mean()
    f1pos = stats_now["f1pos"]
    f1neg = stats_now["f1neg"]
    errors = stats_now["errors"]

    predictions = np.array(pred["predictions"])
    labels = np.array(pred["labels"])
    probs = np.array(pred["probs"])
    masks = pred["masks"]

    draw_regions(dataset, dataset_name, masks, predictions, probs, f1pos, f1neg)

    # # Compute specificity
    # t0 = dt.datetime.now()
    # print("start spec")
    # specificity = compute_specificity_once(stats_now["for_roc"]["y_mask"],
    #                                        ((stats_now["for_roc"]["y_hat_mask"] >= 0.5) * 1.).astype(np.float32))
    # print("Time compute spec {}".format(dt.datetime.now() - t0))
    # # Compute F1 score: foreground, background.
    # t0 = dt.datetime.now()
    # print("start f1 foreg")
    # f1_score_forg = compute_f1_score_once(stats_now["for_roc"]["y_mask"],
    #                                       ((stats_now["for_roc"]["y_hat_mask"] >= 0.5) * 1.).astype(np.float32))
    # print("Time compute f1 foreg {}".format(dt.datetime.now() - t0))
    # t0 = dt.datetime.now()
    # print("start f1 back")
    # f1_score_back = compute_f1_score_once(1 - stats_now["for_roc"]["y_mask"],
    #                                       1 - ((stats_now["for_roc"]["y_hat_mask"] >= 0.5) * 1.).astype(np.float32))
    # print("Time compute f1 back {}".format(dt.datetime.now() - t0))

    # # ROC
    # out_roc_file = join(OUTD, dataset_name.lower(), "roc_{}_{}_FINAL.png".format(dataset_name, epoch))
    # t0 = dt.datetime.now()
    # print("start roc")
    # out_roc, _ = plot_roc_curve(y_mask=stats_now["for_roc"]["y_mask"],
    #                             y_hat_mask=stats_now["for_roc"]["y_hat_mask"], epoch=epoch, path=out_roc_file,
    #                             title="{}. ROC. [FINAL]".format(dataset_name))
    # print("Time compute roc {}".format(dt.datetime.now() - t0))
    # # Precision-recall
    # out_p_r_file = join(OUTD, dataset_name.lower(), "precision_recall_{}_{}_FINAL.png".format(dataset_name, epoch))
    # t0 = dt.datetime.now()
    # print("start pr")
    # out_p_r, _ = plot_precision_recall_curve(y_mask=stats_now["for_roc"]["y_mask"],
    #                                          y_hat_mask=stats_now["for_roc"]["y_hat_mask"], epoch=epoch,
    #                                          path=out_p_r_file,
    #                                          title="{}. Precision-Recall. [FINAL]".format(dataset_name))
    # print("Time compute p-r {}".format(dt.datetime.now() - t0))

    # # Plot the probability dist.
    # out_prob = join(OUTD, dataset_name.lower(), "prob+-_{}_{}_FINAL.png".format(dataset_name, epoch))
    # plot_hist_probs_pos_neg({"probs_pos": pred["probs_pos"],
    #                          "probs_neg": pred["probs_neg"]},
    #                         path=out_prob, epoch=epoch, title="{}. Probs.dist.".format(dataset_name))

    conf_mtx = confusion_matrix(labels, predictions)
    log(log_file, "{}\n".format(args.name_classes))
    log(log_file, "Confusion matrix:\n" + np.array2string(conf_mtx) + "\n\n")
    log(log_file, "Total loss: {} \n".format(total_loss))
    log(log_file, "Loss pos.: {} \n".format(loss_pos))
    log(log_file, "Loss neg.: {} \n".format(loss_neg))
    log(log_file, "Loss classification (Seg.head): {} \n".format(loss_class_seg))
    log(log_file, "********************* FACTORS *********************** \n")
    log(log_file, "Image level: \n")
    log(log_file, "Classification error.: {} \n".format(errors))
    log(log_file, "Pixel level: \n")

    # assert np.mean(loss_dice) * 100 == metrics["dice_avg"], "Something wrong with Dice metric! or NAH"
    log(log_file, "ROC AUC.: {} \n".format(None))
    log(log_file, "Precision-recall AUC.: {} \n".format(None))
    log(log_file, "Specificity: {} \n".format(None))
    log(log_file, "F1 score (foreground): {} \n".format(np.mean(f1pos) * 100.))
    log(log_file, "F1 score (background): {} \n".format(np.mean(f1neg) * 100.))

    # dump the factors for later: average over splits/folds.
    with open(join(OUTD, dataset_name.lower(), "factors_{}_{}_FINAL.pkl".format(dataset_name, epoch)),
              "wb") as off:
        factors_dict = dict()
        factors_dict["roc_auc"] = 0.
        factors_dict["p_r_auc"] = 0.
        factors_dict["dice"] = np.mean(f1pos) * 100.
        factors_dict["classification_error"] = errors
        factors_dict["total_loss"] = total_loss
        factors_dict["specificity"] = None
        factors_dict["f1_score_forg"] = np.mean(f1pos) * 100.
        factors_dict["f1_score_back"] = np.mean(f1neg) * 100.
        pkl.dump(factors_dict, off, protocol=pkl.HIGHEST_PROTOCOL)

    # Not now: it takes space.
    # with open(join(OUTD, dataset_name.lower(), "roc_stuff_{}_{}_FINAL.pkl".format(dataset_name, epoch)), "wb") as ofroc:
    #     pkl.dump(out_roc, ofroc, protocol=pkl.HIGHEST_PROTOCOL)
    #
    # with open(join(OUTD, dataset_name.lower(), "stats_{}_{}_FINAL.pkl".format(dataset_name, epoch)), "wb") as ofstats:
    #     pkl.dump(stats_now, ofstats, protocol=pkl.HIGHEST_PROTOCOL)
    #
    # with open(join(OUTD, dataset_name.lower(), "pred_{}_{}_FINAL.pkl".format(dataset_name, epoch)), "wb") as ofpred:
    #     pkl.dump(pred, ofpred, protocol=pkl.HIGHEST_PROTOCOL)


    # 2. args may have been modified. So, we need to save it.
    if not hasattr(args, "MYSEED"):
        setattr(args, "MYSEED", os.environ["MYSEED"])
        with open(join(OUTD, "code", "final-config.yaml"), "w") as fx:
            yaml.dump(args, fx)


def str2bool(v):
    """
    Read `v`: and returns a boolean value:
    True: if `v== "True"`
    False: if `v=="False"`
    :param v: str.
    :return: bool.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v == "True":
            return True
        elif v == "False":
            return False
        else:
            raise ValueError(
                "Expected value: 'True'/'False'. found {}.".format(v))
    else:
        raise argparse.ArgumentTypeError('String boolean value expected: '
                                         '"True"/"Flse"')


def get_yaml_args(input_args):
    """
    Gets the yaml arguments.

    :param input_args: the output of parser.parse_args().
    :return:
        args: object, where each attribute is an element from parsing the yaml file.
        args_dict: dicts, the same as the previous one, but a dictionary. (see code).
    """
    # list of allowed variables to be override.
    lvars = {""}
    parser = argparse.ArgumentParser()
    with open(join("./config_yaml/", input_args.yaml), 'r') as f:
        args = yaml.load(f)
        args["cudaid"] = input_args.cudaid
        args["yaml"] = input_args.yaml


        # Checking
        if args["dataset"] == "glas":
            # Not sure why I had two vars. with the same value. No time to fix this.
            assert args["nbr_classes"] == 2, "glas has 2 classes. You " \
                                             "provided {}. [NOT OK]".format(
                args["nbr_classes"])
            assert args["model"]["num_classes"] == 2, "glas  2 classes. " \
                                                      "You provided {}. " \
                                                      "[NOT OK]".format(
                args["model"]["num_classes"])
        elif args["dataset"] == "Caltech-UCSD-Birds-200-2011":
            msg = "'Caltech-UCSD-Birds-200-2011' has 200/5 classes. " \
                  "found {}".format(args["nbr_classes"])
            assert args["nbr_classes"] in [200, 5], msg
            assert args["model"]["num_classes"] in [200, 5], msg
        elif args['dataset'] == 'Oxford-flowers-102':
            msg = "'Oxford-flowers-102' has 102 classes. " \
                  "found {}".format(args['num_classes'])
            assert args["nbr_classes"] == 102, msg

        # set the path to model parameters to load them.
        parser.add_argument("--path_pre_trained", type=str, default=None,
                            help="Absolute path to file containing parameters of a "
                                 "model. Use --strict to specify if the  pre-trained "
                                 "model needs to match exactly the current model or not.")
        parser.add_argument("--strict", type=str2bool, default=None,
                            help="If True, the pre-trained model needs to "
                                 "match exactly the current model. Default: "
                                 "True.")
        # Allow the user to override some values in the yaml.
        # This helps changing the hyper-parameters using the command line
        # without changing the yaml file (very
        # helpful during debug!!!!).
        # Create a new parser.
        parser.add_argument("--yaml", type=str,
                            help="yaml file containing the configuration.")
        parser.add_argument("--cudaid", type=str, default="0", help="cuda id.")

        parser.add_argument("--batch_size", type=int, default=None,
                            help="Training batch size (optimizer)")
        parser.add_argument("--lr", type=float, default=None,
                            help="Learning rate (optimizer)")
        parser.add_argument("--momentum", type=float, default=None,
                            help="Momentum (optimizer)")
        parser.add_argument("--weight_decay", type=float, default=None,
                            help="Weight decay (optimizer)")
        parser.add_argument("--step_size", type=int, default=None,
                            help="Step size for lr scheduler.")
        parser.add_argument("--max_epochs", type=int, default=None,
                            help="Max epoch")
        parser.add_argument("--name", type=str, default=None,
                            help="Optimizer name.")
        parser.add_argument("--valid_batch_size", type=str, default=None,
                            help="Batch size for validation.")
        parser.add_argument("--nesterov", type=str2bool, default=None,
                            help="Whether or not to use nesterov.")
        parser.add_argument("--lr_scheduler_name", type=str, default=None,
                            help="Lr scheduler name.")
        parser.add_argument("--use_lr_scheduler", type=str2bool, default=None,
                            help="Whether or not use an lr scheduler.")
        parser.add_argument("--gamma", type=float, default=None,
                            help="gamma for mystep.")
        parser.add_argument("--min_lr", type=float, default=None,
                            help="min_lr for mystep.")

        parser.add_argument("--crop_size", type=int, default=None,
                            help="crop size.")
        parser.add_argument("--up_scale_small_dim_to", type=int, default=None,
                            help="dim to upscale the min to.")


        parser.add_argument("--fold", type=int, default=None, help="Fold")
        parser.add_argument("--split", type=int, default=None, help="Split")

        parser.add_argument("--alpha", type=float, default=None,
                            help="Alpha (classifier, wildcat)")
        parser.add_argument("--kmax", type=float, default=None,
                            help="Kmax (classifier, wildcat)")
        parser.add_argument("--kmin", type=float, default=None,
                            help="Kmin (classifier, wildcat)")
        parser.add_argument("--dropout", type=float, default=None,
                            help="Dropout (classifier, wildcat)")
        parser.add_argument("--modalities", type=int, default=None,
                            help="Number of modalities (classifier, wildcat)")
        parser.add_argument("--pretrained", type=str2bool, default=None,
                            help="True/False (classifier, wildcat)")
        parser.add_argument("--w", type=float, default=None,
                            help="w for thresholding")
        parser.add_argument("--sigma", type=float, default=None,
                            help="sigma for thresholding")
        parser.add_argument("--delta_sigma", type=float, default=None,
                            help="delta sigma for thresholding.")
        parser.add_argument("--max_sigma", type=float, default=None,
                            help="max sigma for thresholding.")
        parser.add_argument("--model_name", type=str, default=None,
                            help="Name of the model: resnet18, resnet50, "
                                 "resnet101")

        parser.add_argument("--use_reg", type=str2bool, default=None,
                            help="whether to use or not a loss regularization "
                                 "over the background.")
        parser.add_argument("--reg_loss", type=str, default=None,
                            help="regularization loss over the background.")
        parser.add_argument("--final_thres", type=float, default=None,
                            help="Segmentation final threshold.")
        parser.add_argument("--scale_in_cl", type=float, default=None,
                            help="Ratio to scale input image for the "
                                 "classifier.")
        parser.add_argument("--padding_size", type=float, default=None,
                            help="Ratio to padd the image.")
        parser.add_argument("--pad_eval", type=str2bool, default=None,
                            help="whether or not pad during evaluation.")
        parser.add_argument("--dataset", type=str, default=None,
                            help="dataset's name.")

        parser.add_argument("--use_size_const", type=str2bool, default=None,
                            help="whether or not use constraint on the size "
                                 "of the background mask.")
        parser.add_argument("--init_t", type=float, default=None,
                            help="init t for ELB for background size "
                                 "constraint.")
        parser.add_argument("--max_t", type=float, default=None,
                            help="max t for ELB for background size "
                                 "constraint.")
        parser.add_argument("--mulcoef", type=float, default=None,
                            help="mult coef for ELB for background size "
                                 "constraint.")
        parser.add_argument("--normalize_sz", type=str2bool, default=None,
                            help="whether or not normalize the size of "
                                 "background mask for ELB constraint.")
        parser.add_argument("--epsilon", type=float, default=None,
                            help="epsilon. used for the ELB constraint over "
                                 "the background size.")
        parser.add_argument("--lambda_neg", type=float, default=None,
                            help="Lambda for the background loss.")


        parser.add_argument(
            "--debug_subfolder", type=str, default=None,
            help="Name of subfolder that is used for debugging. Default: ''.")

        # TODO: finish this overriding!
        input_parser = parser.parse_args()

        def warnit(name, vl_old, vl):
            """
            Warn that the variable with the name 'name' has changed its value from 'vl_old' to 'vl' through command
            line.
            :param name: str, name of the variable.
            :param vl_old: old value.
            :param vl: actual value.
            :return:
            """
            print("{}: {}  -----> {}".format(name, vl_old, vl))

        attributes = input_parser.__dict__.keys()

        for k in attributes:
            val_k = getattr(input_parser, k)
            if k in args.keys():
                if val_k is not None:
                    warnit(k, args[k], val_k)
                    args[k] = val_k
                else:
                    warnit(k, args[k], args[k])

            elif k in args['model'].keys():  # try model
                if val_k is not None:
                    warnit('model.{}'.format(k), args['model'][k], val_k)
                    args['model'][k] = val_k
                else:
                    warnit('model.{}'.format(k), args['model'][k],
                           args['model'][k])

            elif k in args['optimizer'].keys():  # try optimizer 0
                if val_k is not None:
                    warnit(
                        'optimizer.{}'.format(k), args['optimizer'][k], val_k)
                    args['optimizer'][k] = val_k
                else:
                    warnit(
                        'optimizer.{}'.format(k), args['optimizer'][k],
                        args['optimizer'][k]
                    )
            else:
                raise ValueError("Key {} was not found in args. ..."
                                 "[NOT OK]".format(k))

        
        args["MYSEED"] = os.environ['MYSEED']
        args['model']['scale_in_cl'] = (
            args['model']['scale_in_cl'],
            args['model']['scale_in_cl']
        )
        if args['padding_size'] not in [None, 'None']:
            args['padding_size'] = (args['padding_size'], args['padding_size'])
        else:
            args['padding_size'] = None

        args_dict = copy.deepcopy(args)
        args = Dict2Obj(args)

    return args, args_dict


def get_train_transforms_img(args):
    """
    Get the transformation to perform over the images for the train samples.
    All the transformation must perform on PIL.Image.Image and returns a PIL.Image.Image object.

    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return: a torchvision.transforms.Compose() object.
    """

    if args.dataset == "glas":
        # TODO: check values of jittering: https://arxiv.org/pdf/1806.07064.pdf
        return transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    elif args.dataset == "Caltech-UCSD-Birds-200-2011":
        return transforms.Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.00),
            transforms.RandomHorizontalFlip()
        ])
    elif args.dataset == 'Oxford-flowers-102':
        return transforms.Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.00),
            transforms.RandomHorizontalFlip()
        ])
    else:
        raise ValueError("Dataset {} unsupported. Exiting .... [NOT OK]".format(args.dataset))


def get_transforms_tensor(args):
    """
     Return tensor transforms.
    :param args: object. Contains the configuration of the exp that has been read from the yaml file.
    :return:
    """
    if args.dataset == "glas":
        # TODO: check values of jittering: https://arxiv.org/pdf/1806.07064.pdf
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    elif args.dataset == "Caltech-UCSD-Birds-200-2011":
        # Normalization.
        # https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/28aab5849db391138881e3c16f9d6482e8b4ab38/dataset.py
        # [102.9801 / 255., 115.9465 / 255., 122.7717 / 255.],
        #                                  [1., 1., 1.]

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    elif args.dataset == 'Oxford-flowers-102':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    else:
        raise ValueError("Dataset {} unsupported. Exiting .... [NOT OK]".format(args.dataset))


# ===============
# Multiprocessing
# ===============


def shared_array_multi_processes(shape, datatype):
    """
    Form a shared memory numpy array with specific data type for multiprocessing purpose.

    :param shape: a tuple of the shape of the array (h, w, ...). To share a matrix of height `h` and width `w`,
           shape = (h, w).
    :param datatype: ctypes.c_*, data type if the shared array. To share a matrix with `uint8` type, use:
           datatype = ctypes.c_uint8.
    :return shared_array: numpy.ndarray, a share numpy array.

    Reference:
        https://gist.github.com/nfaggian/9755516
        http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-
        python-multiprocessing
    """

    shared_array_base = multiprocessing.Array(datatype, functools.reduce(mul, shape, 1))
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)

    return shared_array


def chunks_into_n(l, n):
    """
    Split iterable l into n chunks (iterables) with the same size.

    :param l: iterable.
    :param n: number of chunks.
    :return: iterable of length n.
    """
    chunksize = int(math.ceil(len(l) / n))
    return (l[i * chunksize:i * chunksize + chunksize] for i in range(n))


def chunk_it(l, n):
    """
    Create chunks with the same size (n) from the iterable l.
    :param l: iterable.
    :param n: int, size of the chunk.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

# ========
# Plotting
# ========


def plot_curve(values, path, title="", x_str="", y_str="", best_iter=-1, plot_avg=True, avg_perd=20, dpi=100):
    """
    Plot a curve.

    :param values: list or numpy.ndarray of values to plot (y)
    :param path: str, path where to save the figure.
    :param title: str, the title of the plot.
    :param x_str: str, the name of the x axis.
    :param y_str: str, the name of the y axis.
    :param best_iter: integer. The epoch of the best iteration.
    :param plot_avg: bool, If true, a moving average if plotted over the original curve.
    :param avg_perd: int, the size of the moving average.
    :param dpi: int, the dpi of the image.
    """
    assert isinstance(values, list) or isinstance(values, np.ndarray), "'values' must be either a list or a " \
                                                                       "numpy.ndarray. You provided `{}` .... [NOT " \
                                                                       "OK]".format(type(values))
    if isinstance(values, list):
        values = np.asarray(values)

    assert isinstance(best_iter, int), "'best_iter' must be an integer. You provided `{}` .... [NOT " \
                                       "OK]".format(type(best_iter))
    assert (0 <= best_iter < values.size) or (best_iter < 0 and abs(best_iter) <= values.size), \
        "'best_iter' = `{}` is greater than the number of available values `{}`.... [NOT OK]".format(best_iter,
                                                                                                     values.size)

    floating = 6
    font_sz = 6
    prec = "%." + str(floating) + "f"
    alpha = 1.
    if plot_avg:
        alpha = 0.2

    best_value = str(prec % np.float(values[best_iter]))
    fig = plt.figure()
    plt.plot(values, label="Best val: {} at {}{}".format(best_value, best_iter, x_str), alpha=alpha)
    if plot_avg:
        signal = np.convolve(values, np.ones((avg_perd,))/avg_perd, mode="valid")
        plt.plot(signal, label="Running avg. {}. Ends at: {} .".format(avg_perd, prec % signal[-1]))
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.title(title, fontsize=font_sz)
    plt.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': font_sz})
    plt.grid(True)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    del fig


def plot_curves(values_dict,
                path,
                title="",
                best_iter=-1,
                plot_avg=True,
                avg_perd=20,
                dpi=100
                ):
    """
    Plot a set of curves using subplots.

    :param values_dict: dict, each key contains a list of values
    to plot (y).
    :param path: str, path where to save the figure.
    :param title: str, the title of the plot.
    :param best_iter: integer. The epoch of the best iteration.
    :param plot_avg: bool, If true, a moving average if plotted over the
    original curve.
    :param avg_perd: int, the size of the moving average.
    :param dpi: int, the dpi of the image.
    """
    msg = "'values' must be dict. You provided `{}` .... [NOT OK]".format(type(
        values_dict))
    assert isinstance(values_dict, dict), msg

    nbr_curves = len(values_dict.keys())
    sz = max([np.array(values_dict[k]).size for k in values_dict.keys()])
    if sz == 0:  # nothing has been recorded yet.
        return 0


    floating = 6
    font_sz = 10
    prec = "%." + str(floating) + "f"
    alpha = 1.
    if plot_avg:
        alpha = 0.2

    f, axes = plt.subplots(nbr_curves,
                           1,
                           sharex=False,
                           figsize=(20, 20))  # (w, h) of the figure in inches.
    for i, k in enumerate(values_dict.keys()):
        vals = np.array(values_dict[k])
        ax = axes[i]
        ax.set_ylabel(k, fontsize=font_sz)
        ax.plot(vals, label="{}".format(k), alpha=alpha)
        ax.grid()
        if plot_avg:
            signal = np.convolve(vals,
                                 np.ones((avg_perd,)) / avg_perd,
                                 mode="valid"
                                 )
            ax.plot(signal, label="Run. avg.")

        ax.legend(loc='upper right',
                  fancybox=True,
                  shadow=True,
                  prop={'size': font_sz}
                  )

        if i < (nbr_curves - 1):  # suppress the labels of the x ticks of all
            # axes except the last one.
            ax.set_xticklabels([])

    ax.set_xlabel("iter. [x axis is NOT shared]", fontsize=font_sz)
    plt.suptitle(title, fontsize=font_sz)

    f.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    del f


def superpose_curves(values_dict, path, epoch, title="Title", label="label", dpi=100, compute_mse=False):
    """
    Superpose a set of curves on the same plot. (subplot 0]
    + superpose their distribution. [subplot 1]

    Note: this function was built mainly for the size constraints. (superpose the true and the predicted sizes).

    :param values_dict: dict, each key contains a numpy.ndarray of values to plot (y).
    :param path: str, path where to save the figure.
    :param epoch: integer. The epoch at which the statistics were taken.
    :param label: str, y label.
    :param title: str, the title of the plot.
    :param dpi: int, the dpi of the image.
    :param compute_mse: bool, if True, meas squared error is computed between the two curves. (Valid only if there are
    ONLY two curves).
    """
    assert isinstance(values_dict, dict), "'values' must be dict. You provided `{}` .... [NOT OK]".format(type(
        values_dict))

    nbr_curves = len(values_dict.keys())
    if compute_mse:
        assert nbr_curves == 2, "You asked to compute the MSE while there are many curves! Option valid only if there" \
                                "2 curves .... [NOT OK]"
        mse = ((values_dict[list(values_dict.keys())[0]] - values_dict[list(values_dict.keys())[1]])**2).mean()

    floating = 7
    prec = "%." + str(floating) + "f"
    font_sz = 20
    alpha = 1.
    nbr_bins = 100  # the higher, the more time is required to plot it!

    f, axes = plt.subplots(nbr_curves + 1, 1, sharex=False, figsize=(20, 20))  # (w, h) of the figure in inches.
    # 1. Superpose the values.
    ax = axes[0]
    ax.set_ylabel(label, fontsize=font_sz)
    ax.grid()
    ax.set_xlabel("Sample [i] ", fontsize=font_sz)
    for i, k in enumerate(values_dict.keys()):
        ax.plot(values_dict[k], label="{}".format(k), alpha=alpha)
        ax.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': font_sz})

    # 2. Superpose the distributions.
    alpha = 0.4

    for i, k in enumerate(values_dict.keys()):
        ax = axes[i+1]
        array = values_dict[k]
        ax.hist(array, label="{}".format(k), alpha=alpha, density=False, bins=nbr_bins,
                weights=np.ones_like(array.ravel()) / float(array.size))
        ax.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': font_sz})
        ax.set_ylabel("Prob. density", fontsize=font_sz)
        ax.grid()

    ax.set_xlabel(label, fontsize=font_sz)
    # 3. Title and save.
    if compute_mse:
        plt.suptitle("{}. Iter.: {}. MSE: {}".format(title, epoch, prec % mse), fontsize=font_sz)
    else:
        plt.suptitle("{}. Iter.: {}.".format(title, epoch), fontsize=font_sz)

    f.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    del f


def plot_hist_probs_pos_neg(values_dict, path, epoch, title="Title", dpi=100):
    """
    Plot the histogram of the probability distribution of each class within the positive and the negative region.
    P(Y=0|X+), P(Y=1|X+),
    P(Y=0|X-), P(Y=1|X-).
    Supports only two classes.
    # TODO: add the class names to the title: what 0 means and what 1 means.

    :param values_dict: dict, each key contains a numpy.ndarray of values to plot (y). Two keys: `probs_pos`,
    `probs_neg`. Each one is a numpy.ndarray of size (n, 2).
    :param path: str, path where to save the figure.
    :param epoch: integer. The epoch at which the statistics were taken.
    :param title: str, the title of the plot.
    :param dpi: int, the dpi of the image.
    """
    assert isinstance(values_dict, dict), "'values' must be dict. You provided `{}` .... [NOT OK]".format(type(
        values_dict))

    nbr_curves = len(values_dict.keys())
    assert "probs_pos" in values_dict.keys(), "`values_dict` must contain the kye `probs_pos`. We did not find it. " \
                                              "...[NOT OK]"
    assert "probs_neg" in values_dict.keys(), "`values_dict` must contain the kye `probs_neg`. We did not find it. " \
                                              "...[NOT OK]"
    assert isinstance(values_dict["probs_pos"], np.ndarray), "`values_dict[probs_pos]` must be a numpy.ndarray type. " \
                                                             "We found {} .... [NOT OK]".format(
        type(values_dict["probs_pos"]))
    assert isinstance(values_dict["probs_neg"], np.ndarray), "`values_dict[probs_neg]` must be a numpy.ndarray type. " \
                                                             "We found {} .... [NOT OK]".format(
        type(values_dict["probs_neg"]))
    assert nbr_curves == 2, "We expect two keys, one for the positive region, and the other for the negative regio. " \
                            "We found {} .... [NOT OK]".format(nbr_curves)

    assert values_dict["probs_pos"].ndim == 2, "We support only two classes. You provided {} .... [NOT OK]".format(
        values_dict["probs_pos"].ndim)
    assert values_dict["probs_neg"].ndim == 2, "We support only two classes. You provided {} .... [NOT OK]".format(
        values_dict["probs_neg"].ndim)

    floating = 3
    prec = "%." + str(floating) + "f"
    font_sz = 20
    alpha = 1.
    nbr_bins = 100  # the higher, the more time is required to plot it!

    f, axes = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(20, 20))  # (w, h) of the figure in inches.

    # Positive
    for i in range(2):
        ax = axes[0][i]
        array = values_dict["probs_pos"][:, i]
        ax.hist(array, label="P(Y={} |X+)".format(i), alpha=alpha, density=False,
                bins=nbr_bins, range=(0, 1), weights=np.ones_like(array.ravel()) / float(array.size))
        ax.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': font_sz})
        ax.set_ylabel("Prob. density", fontsize=font_sz)
        ax.set_xlabel("P(Y={} |X+)".format(i), fontsize=font_sz)
        ax.grid()

    # Negative
    for i in range(2):
        ax = axes[1][i]
        array = values_dict["probs_neg"][:, i]
        ax.hist(array, label="P(Y={} |X-)".format(i), alpha=alpha, density=False,
                bins=nbr_bins, range=(0, 1), weights=np.ones_like(array.ravel()) / float(array.size))
        ax.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': font_sz})
        ax.set_ylabel("Prob. density", fontsize=font_sz)
        ax.set_xlabel("P(Y={} |X-)".format(i), fontsize=font_sz)
        ax.grid()

    # 3. Title and save.
    plt.suptitle("{}. Iter.: {}.".format(title, epoch), fontsize=font_sz)

    f.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    del f


def compute_roc_curve_once(y_mask, y_hat_mask):
    """
    Compute ROC curve for one sample.
    ROC: computed
    using https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

    AUC: computed using https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc

    scikit-learn version: '0.20.2'.

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param y_mask: numpy.ndarray of float32. Vector containing binary values, where 1 indicates foreground. (true value)
    :param y_hat_mask: numpy.ndarray of float32. Vector containing the predicted probability values of a pixel being
    foreground. It has the same size as y_mask. (predicted values)
    :return: tpr, fpr, roc_auc:
        tpr: True positive rate vector. (interpolated)
        fpr: False positive rate vector. (fixed)
        roc_auc: Area under the ROC curve.
    """
    for var, var_n in zip((y_mask, y_hat_mask), ("y_mask", "y_hat_mask")):
        assert isinstance(var, np.ndarray), "`{}` must be of type {}. You provided {} .... [NOT OK]".format(var_n,
            np.ndarray, type(var))
        assert var.dtype == np.float32, "`{}` must be of data type {}. You provided {} .... [NOT OK]".format(var_n,
            np.float32, var.dtype)
        assert var.ndim == 1, "`{}` must have 1 dimension. You provided {} .... [NOT OK]".format(var_n, var.ndim)

    assert y_mask.size == y_hat_mask.size, "`y_mask` and `y_hat_mask` must have the same number of elements. You " \
                                           "provided `y_mak` with {} and `y_hat_mask` with {} .... [" \
                                           "NOT OK]".format(y_mask.size, y_hat_mask.size)

    fpr, tpr, thresholds = roc_curve(y_mask, y_hat_mask, pos_label=1)
    fpr_fixed = np.asarray(np.arange(0, 1., 1e-3).tolist() + [1.])
    tpr_interpolated = interp(fpr_fixed, fpr, tpr)  # the interpolated TPR using the a-axis fpr_fixed.

    roc_auc = auc(fpr_fixed, tpr_interpolated)

    return tpr_interpolated, fpr_fixed, roc_auc


def compute_specificity_once(y_mask, y_hat_mask):
    """
    Compute specificity for one sample: specificity = TNR = TN/(TN + FP).
    The higher, the better.

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param y_mask: numpy.ndarray of float32. Vector containing binary values, where 1 indicates foreground. (true
    values)
    :param y_hat_mask: numpy.ndarray of float32. Vector containing the predicted binary mask values of a pixel being
    a foreground. It has the same size as y_mask. (predicted values)
    :return: specificity: float.
    """
    for var, var_n in zip((y_mask, y_hat_mask), ("y_mask", "y_hat_mask")):
        assert isinstance(var, np.ndarray), "`{}` must be of type {}. You provided {} .... [NOT OK]".format(var_n,
            np.ndarray, type(var))
        assert var.dtype == np.float32, "`{}` must be of data type {}. You provided {} .... [NOT OK]".format(var_n,
            np.float32, var.dtype)
        assert var.ndim == 1, "`{}` must have 1 dimension. You provided {} .... [NOT OK]".format(var_n, var.ndim)

    assert y_mask.size == y_hat_mask.size, "`y_mask` and `y_hat_mask` must have the same number of elements. You " \
                                           "provided `y_mak` with {} and `y_hat_mask` with {} .... [" \
                                           "NOT OK]".format(y_mask.size, y_hat_mask.size)

    # tn = np.sum((1 - y_mask) * (1 - y_hat_mask)).astype(float)
    # fp = np.sum((1 - y_mask) * y_hat_mask).astype(float)
    #
    # specificity = 0.
    # if (tn + fp) != 0.:
    #     specificity = tn / (tn + fp)

    specificity = 0.
    total_n = np.sum((y_mask == 0) * 1.)
    t_n = np.sum(np.logical_and(y_hat_mask == 0,  y_mask == 0) * 1.)
    if total_n != 0.:
        specificity = t_n / total_n

    return specificity


def compute_f1_score_once(y_mask, y_hat_mask):
    """
    Compute F1 score for one sample: specificity = TNR = TN/(TN + FP).
    The higher, the better.
    F1 score:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    https://en.wikipedia.org/wiki/F1_score

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param y_mask: numpy.ndarray of float32. Vector containing binary values, where 1 indicates foreground. (true
    values)
    :param y_hat_mask: numpy.ndarray of float32. Vector containing the predicted binary mask values of a pixel being
    a foreground. It has the same size as y_mask. (predicted values)
    :return: f1: float.
    """
    for var, var_n in zip((y_mask, y_hat_mask), ("y_mask", "y_hat_mask")):
        assert isinstance(var, np.ndarray), "`{}` must be of type {}. You provided {} .... [NOT OK]".format(var_n,
            np.ndarray, type(var))
        assert var.dtype == np.float32, "`{}` must be of data type {}. You provided {} .... [NOT OK]".format(var_n,
            np.float32, var.dtype)
        assert var.ndim == 1, "`{}` must have 1 dimension. You provided {} .... [NOT OK]".format(var_n, var.ndim)

    assert y_mask.size == y_hat_mask.size, "`y_mask` and `y_hat_mask` must have the same number of elements. You " \
                                           "provided `y_mak` with {} and `y_hat_mask` with {} .... [" \
                                           "NOT OK]".format(y_mask.size, y_hat_mask.size)

    f1 = f1_score(y_mask, y_hat_mask, pos_label=1)

    return f1


def plot_roc_curve(y_mask, y_hat_mask, epoch, path="", title="", dpi=100):
    """
    Plot ROC curve using the function compute_roc_curve_once().

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param y_mask: numpy.ndarray of float32. Vector containing binary values, where 1 indicates gland. This vector is
    the concatenation (stacking) of all the 2D true masks of a set.
    :param y_hat_mask: numpy.ndarray of float32. Vector containing the predicted probability values of a pixel being
    a gland. This vector is the concatenation (stacking) of all the 2D predicted probability masks of a set.
    :param path: str, path where to save the figure. If you do not want to save it, set it to "".
    :param epoch: integer. The epoch at which the statistics were taken.
    :param title: str, the title of the plot.
    :param dpi: int, the dpi of the image.
    :return:
    """
    floating = 3
    prec = "%." + str(floating) + "f"
    font_sz = 15
    lw = 2

    fig = plt.figure(figsize=(15, 15))

    tpr, fpr, roc_auc = compute_roc_curve_once(y_mask, y_hat_mask)
    out = {"tpr": copy.deepcopy(tpr),
           "fpr": copy.deepcopy(fpr),
           "roc_auc": copy.deepcopy(roc_auc)}

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve model (AUC = {})'.format(prec % roc_auc))
    plt.plot([0, 1], [0, 1], color="black", linestyle='--', lw=lw, label='ROC curve random guess  (AUC = {})'.format(
        prec % 0.5))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}. Epoch: {}.'.format(title, epoch))
    plt.legend(loc='lower right', fancybox=True, shadow=True, prop={'size': font_sz})

    if path != "":
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')

    return out, fig


def compute_precision_recall_curve_once(y_mask, y_hat_mask):
    """
    Compute precision-recall curve for one sample.
    Precision-recall: computed
    using https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve

    AUC: computed using https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc

    scikit-learn version: '0.20.2'.

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param y_mask: numpy.ndarray of float32. Vector containing binary values, where 1 indicates foreground. (true
    values)
    :param y_hat_mask: numpy.ndarray of float32. Vector containing the predicted probability values of a pixel being
    a foreground. It has the same size as y_mask. (predicted values)
    :return: precision, recall, precison_recall_auc:
        precision: precision rate vector. (interpreted)
        recall: recall vector. (fixed)
        precison_recall_auc: Area under the precison-recall curve.
    """
    for var, var_n in zip((y_mask, y_hat_mask), ("y_mask", "y_hat_mask")):
        assert isinstance(var, np.ndarray), "`{}` must be of type {}. You provided {} .... [NOT OK]".format(var_n,
            np.ndarray, type(var))
        assert var.dtype == np.float32, "`{}` must be of data type {}. You provided {} .... [NOT OK]".format(var_n,
            np.float32, var.dtype)
        assert var.ndim == 1, "`{}` must have 1 dimension. You provided {} .... [NOT OK]".format(var_n, var.ndim)

    assert y_mask.size == y_hat_mask.size, "`y_mask` and `y_hat_mask` must have the same number of elements. You " \
                                           "provided `y_mak` with {} and `y_hat_mask` with {} .... [" \
                                           "NOT OK]".format(y_mask.size, y_hat_mask.size)

    precision, recall, thresholds = precision_recall_curve(y_mask, y_hat_mask, pos_label=1)
    # reverse precision and recall because recall is ordered in decreasing order. Since we want to interpolate the
    # precision later using numpy.interp(), we need the x (recall) to be increasing!!!!.
    precision = precision[::-1]
    recall = recall[::-1]

    recall_fixed = np.asarray(np.arange(0, 1., 1e-3).tolist() + [1.])
    precision_interp = interp(recall_fixed, recall, precision)

    precison_recall_auc = auc(recall_fixed, precision_interp)

    return precision_interp, recall_fixed, precison_recall_auc


def compute_dice_index(y_mask, y_hat_mask):
    """
    Compute dice index.

    :param y_mask: binary vector . (true mask)
    :param y_hat_mask: binary vector. (predicted mask)
    :return: Dice index.
    """
    for var, var_n in zip((y_mask, y_hat_mask), ("y_mask", "y_hat_mask")):
        assert isinstance(var, np.ndarray), "`{}` must be of type {}. You provided {} .... [NOT OK]".format(var_n,
            np.ndarray, type(var))
        assert var.dtype == np.float32, "`{}` must be of data type {}. You provided {} .... [NOT OK]".format(var_n,
            np.float32, var.dtype)
        assert var.ndim == 1, "`{}` must have 1 dimension. You provided {} .... [NOT OK]".format(var_n, var.ndim)

    assert y_mask.size == y_hat_mask.size, "`y_mask` and `y_hat_mask` must have the same number of elements. You " \
                                           "provided `y_mak` with {} and `y_hat_mask` with {} .... [" \
                                           "NOT OK]".format(y_mask.size, y_hat_mask.size)
    # Compute Dice index.
    pflat = y_hat_mask
    tflat = y_mask
    intersection = (pflat * tflat).sum()

    return (2. * intersection) / (pflat.sum() + tflat.sum())


def compute_metrics(true_labels, pred_labels, true_masks, pred_masks, binarize=True, ignore_roc_pr=False, average=True):
    """
    Compute the following metrics:
        1. Image level:
            1.1 Average classification error. (%)
        2. Pixel level:
            2.1 Average Dice index. (%)
            2.2 Average F1 score (foreground). (%)
            2.3 Average F1 score (background). (%)
            2.4 Average specificity (True negative Rate). (%)
            2.4 Average ROC-AUC. (%)
            2.5 Precision-recall AUC. (%)

    Note: ***********************************************************
    *    When using boolean data: F1 score is the same as Dice index.
    *****************************************************************

    :param true_labels: list of true labels (int)
    :param pred_labels: list of predicted labels (int).
    :param true_masks: list of true masks (2D matrix).
    :param pred_masks: list of predicted masks (2D matrix).
    :param binarize: Bool. If True, we binarize the mask to compute F1, Dice indx.
    :param ignore_roc_pr: Bool. If True, we do not compute ROC, Precision-recall curves.
    :param average: Bool, If True, the stats. are averaged. If not, they are just summed. The later case is useful
    when multi-processing.
    :return: the aforementioned metrics.
    """
    nbr = len(true_labels)
    for el in [true_labels, pred_labels, true_masks, pred_masks]:
        assert len(el) == nbr, "One of the args. has different size than {}. Exiting .... [NOT OK]".format(nbr)

    # Avg. classification error
    acc_cl_error = (nbr - np.sum(np.asarray(true_labels) == np.asarray(pred_labels)))

    # Pixel level:
    acc_dice, acc_f1_for, acc_f1_back, acc_roc, acc_pr, acc_spec = 0., 0., 0., 0., 0., 0.

    for msk, msk_ht in tqdm.tqdm(zip(true_masks, pred_masks), ncols=80, total=nbr):
        bin_msk_hat = msk_ht
        if binarize:
            bin_msk_hat = ((msk_ht >= 0.5) * 1.).astype(np.float32)

        # flatten arrays
        msk = np.ravel(msk).astype(np.float32)
        msk_ht = np.ravel(msk_ht).astype(np.float32)
        bin_msk_hat = np.ravel(bin_msk_hat).astype(np.float32)

        # Dice
        acc_dice += compute_dice_index(msk, bin_msk_hat)

        # F1:
        acc_f1_for += compute_f1_score_once(msk, bin_msk_hat)
        acc_f1_back += compute_f1_score_once(1 - msk, 1 - bin_msk_hat)

        # Specificity
        acc_spec += compute_specificity_once(msk, bin_msk_hat)

        # Roc, P-R
        if not ignore_roc_pr:
            acc_roc += compute_roc_curve_once(msk, msk_ht)[2]
            acc_pr += compute_precision_recall_curve_once(msk, msk_ht)[2]

    metrics = dict()

    if not average:
        nbr = 1.

    metrics["cl_error_avg"] = 100. * acc_cl_error / float(nbr)
    metrics["dice_avg"] = 100. * acc_dice / float(nbr)
    metrics["f1_score_forg_avg"] = 100. * acc_f1_for / float(nbr)
    metrics["f1_score_back_avg"] = 100. * acc_f1_back / float(nbr)
    metrics["specificity_avg"] = 100. * acc_spec / float(nbr)
    metrics["roc_auc_avg"] = 100. * acc_roc / float(nbr)
    metrics["p_r_auc_avg"] = 100. * acc_pr / float(nbr)

    return metrics


def metric_worker(iterx, trg, lock):
    """
    A worker that processes a set of samples within the `iter` list.
    :param iterx: lists of inputs for compute_metrics(): true_labels, pred_labels, true_masks, pred_masks, binarize,
    ignore_roc_pr.
    :param trg: numpy.ndarray. Shared array to store the computed stats.
    :param lock: Instance of Lock(), to lock the shared data (trg).
    :return: Write in trg.
    """
    true_labels, pred_labels, true_masks, pred_masks, binarize, ignore_roc_pr = iterx
    metrics = compute_metrics(true_labels=true_labels, pred_labels=pred_labels, true_masks=true_masks,
                              pred_masks=pred_masks, binarize=binarize, ignore_roc_pr=ignore_roc_pr,
                              average=False)
    # Write in the shared space: Add the computed metrics.
    lock.acquire()
    trg[0] += metrics["cl_error_avg"]
    trg[1] += metrics["dice_avg"]
    trg[2] += metrics["f1_score_forg_avg"]
    trg[3] += metrics["f1_score_back_avg"]
    trg[4] += metrics["specificity_avg"]
    trg[5] += metrics["roc_auc_avg"]
    trg[6] += metrics["p_r_auc_avg"]
    lock.release()


def compute_metrics_mp(true_labels, pred_labels, true_masks, pred_masks, binarize=True, ignore_roc_pr=False,
                       nbr_workers=8):
    """
    The same as compute_metrics() but using multi_processing.
    See compute_metrics() for the input description.

    This may cause a teeny-weeny difference compared to the case without multiprocessing. For example:
    No multiprocessing:
        cl_error_avg: 40.0
        dice_avg: 68.77295711899419
        f1_score_forg_avg: 68.77295711899419
        f1_score_back_avg: 30.62561484458003
        specificity_avg: 30.25878691201272
        roc_auc_avg: 0.0
        p_r_auc_avg: 0.0

    With multiprocessing: 1 workers
        cl_error_avg: 40.0
        dice_avg: 68.77295532226563
        f1_score_forg_avg: 68.77295532226563
        f1_score_back_avg: 30.625613403320312
        specificity_avg: 30.258786010742188
        roc_auc_avg: 0.0
        p_r_auc_avg: 0.0
    With ultiprocessing: 8 workers
        cl_error_avg: 40.0
        dice_avg: 68.77296142578125
        f1_score_forg_avg: 68.77296142578125
        f1_score_back_avg: 30.625613403320312
        specificity_avg: 30.2587890625
        roc_auc_avg: 0.0
        p_r_auc_avg: 0.0
    :return:
    """
    # Sanity check
    nbr = len(true_labels)
    for el in [true_labels, pred_labels, true_masks, pred_masks]:
        assert len(el) == nbr, "One of the args. has different size than {}. Exiting .... [NOT OK]".format(nbr)

    # Create a shared memory to store the metrics.
    shared_trg = shared_array_multi_processes(shape=(1, 7), datatype=ctypes.c_float)
    shared_trg = np.ravel(shared_trg)
    shared_trg *= 0.  # Initialize the values to 0.

    # Chunk the data
    c_true_labels = list(chunks_into_n(true_labels, nbr_workers))
    c_pred_labels = list(chunks_into_n(pred_labels, nbr_workers))
    c_true_masks = list(chunks_into_n(true_masks, nbr_workers))
    c_pred_masks = list(chunks_into_n(pred_masks, nbr_workers))
    c_binarize = [binarize for _ in range(nbr_workers)]
    c_ignore_roc_pr = [ignore_roc_pr for _ in range(nbr_workers)]

    # Create a lock
    lock = Lock()
    # Create the processes
    processes = [
        Process(target=metric_worker, args=([
            c_true_labels[pp], c_pred_labels[pp], c_true_masks[pp], c_pred_masks[pp], c_binarize[pp],
            c_ignore_roc_pr[pp]
                                            ], shared_trg, lock)) for pp in range(nbr_workers)
    ]
    # Start the processes
    [p.start() for p in processes]

    # Join the processes
    [p.join() for p in processes]

    # Collect the results
    metrics = dict()
    metrics["cl_error_avg"] = shared_trg[0] / float(nbr)
    metrics["dice_avg"] = shared_trg[1] / float(nbr)
    metrics["f1_score_forg_avg"] = shared_trg[2] / float(nbr)
    metrics["f1_score_back_avg"] = shared_trg[3] / float(nbr)
    metrics["specificity_avg"] = shared_trg[4] / float(nbr)
    metrics["roc_auc_avg"] = shared_trg[5] / float(nbr)
    metrics["p_r_auc_avg"] = shared_trg[6] / float(nbr)

    return metrics


def plot_precision_recall_curve(y_mask, y_hat_mask, epoch, path="", title="", dpi=100):
    """
    Plot precision-recall curve using the function compute_precision_recall_curve_once().

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param y_mask: numpy.ndarray of float32. Vector containing binary values, where 1 indicates gland. This vector is
    the concatenation (stacking) of all the 2D true masks of a set.
    :param y_hat_mask: numpy.ndarray of float32. Vector containing the predicted probability values of a pixel being
    a gland. This vector is the concatenation (stacking) of all the 2D predicted probability masks of a set.
    :param path: str, path where to save the figure. If you do not want to save it, set it to "".
    :param epoch: integer. The epoch at which the statistics were taken.
    :param title: str, the title of the plot.
    :param dpi: int, the dpi of the image.
    :return:
    """
    floating = 3
    prec = "%." + str(floating) + "f"
    font_sz = 15
    lw = 2

    fig = plt.figure(figsize=(15, 15))

    precision, recall, precison_recall_auc = compute_precision_recall_curve_once(y_mask, y_hat_mask)
    out = {"precision": copy.deepcopy(precision),
           "recall": copy.deepcopy(recall),
           "precision_recall_auc": copy.deepcopy(precison_recall_auc)}

    plt.plot(recall, precision, color='darkorange', lw=lw, label='Precision-recall curve model (AUC = {})'.format(
        prec % precison_recall_auc))
    plt.plot([0, 1], [0.5, 0.5], color="black", linestyle='--', lw=lw,
             label='Precision-recall curve random guess (AUC = {})'.format(prec % 0.5))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('{}. Epoch: {}.'.format(title, epoch))
    plt.legend(loc='lower right', fancybox=True, shadow=True, prop={'size': font_sz})

    if path != "":
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')

    return out, fig


def summaries_exps(fd_in, pattern="factors_Test_*_FINAL.pkl"):
    """
    Compute stats over multiple experiments (splits/folds) to estimate the average of some specific stats. Save the
    summary within the input folder `fd_in` in a pickle file.
    Stats:
        * Dice (over pixels)
        * ROC  (over pixels)
        * Precision-recall (over pixels)
        * Classification error (over images).

    :param fd_in: str, input folder where all the experiments live.
    :param pattern: str, pattern to match files.
    :return: dict(), `output` contains all the average stats.
    """
    assert os.path.exists(fd_in), "Folder {} does not exist .... [NOT OK]".format(fd_in)
    files = []
    for r, d, f in os.walk(fd_in):
        for file in f:
            if fnmatch.fnmatch(file, pattern):
                files.append(os.path.join(r, file))

    roc_auc_s, p_r_auc_s, dice_s, cl_error_s = [], [], [], []
    fpr_s, recall_s = [], []
    tpr_s, precision_s = [], []

    specificities = []
    f1_scores_forg, f1_scores_back = [], []

    for i, f in enumerate(files):
        print("Processing stat. file: {} ----> {} / {}".format(f, i, len(files)))
        with open(f, 'rb') as fin:
            stats = pkl.load(fin)
            dice_s.append(stats["dice"])
            cl_error_s.append(stats["classification_error"])
            roc_auc_s.append(stats["roc_auc"])
            p_r_auc_s.append(stats["precision_recall_auc"])
            # x-axis: fpr, recall.
            fpr_s.append(stats["fpr"])
            recall_s.append(stats["recall"])

            # y-axis: tpr, precision.
            tpr_s.append(stats["tpr"])
            precision_s.append(stats["precision"])

            specificities.append(stats["specificity"])
            f1_scores_forg.append(stats["f1_score_forg"])
            f1_scores_back.append(stats["f1_score_back"])

    # Compute average/std: specificity, dice, classification error, roc_auc, p_r_auc
    specificity_avg = {"mean": np.mean(specificities), "std": np.std(specificities)}
    f1_score_forg_avg = {"mean": np.mean(f1_scores_forg), "std": np.std(f1_scores_forg)}
    f1_score_back_avg = {"mean": np.mean(f1_scores_back), "std": np.std(f1_scores_back)}
    dice_avg = {"mean": np.mean(dice_s), "std": np.std(dice_s)}
    cl_error_avg = {"mean": np.mean(cl_error_s), "std": np.std(cl_error_s)}
    roc_auc_avg = {"mean": np.mean(roc_auc_s), "std": np.std(roc_auc_s)}
    p_r_auc_avg = {"mean": np.mean(p_r_auc_s), "std": np.std(p_r_auc_s)}

    # Compute a fixed x-axis: fpr, recall.
    # all_fpr = np.unique(np.concatenate(fpr_s))  # sorted in increasing order.
    # all_recall = np.unique(np.concatenate(recall_s))
    all_fpr = np.asarray(np.arange(0, 1., 1e-3).tolist() + [1.])  # np.unique(np.concatenate(fpr_s))  # sorted in
    # increasing order.
    all_recall = np.asarray(np.arange(0, 1., 1e-3).tolist() + [1.])  # np.unique(np.concatenate(recall_s))

    # Compute the final curves as an average, then compute auc: roc, precision-recall.

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(fpr_s)):
        mean_tpr += interp(all_fpr, fpr_s[i], tpr_s[i])

    mean_tpr /= len(fpr_s)
    final_roc_auc = auc(all_fpr, mean_tpr)

    mean_precision = np.zeros_like(all_recall)
    for i in range(len(recall_s)):
        mean_precision += interp(all_recall, recall_s[i], precision_s[i])

    mean_precision /= len(recall_s)
    final_p_r_auc = auc(all_recall, mean_precision)

    output = dict()
    output["specificity_avg"] = specificity_avg
    output["f1_score_forg_avg"] = f1_score_forg_avg
    output["f1_score_back_avg"] = f1_score_back_avg
    output["dice_avg"] = dice_avg
    output["cl_error_avg"] = cl_error_avg
    output["roc_auc_avg"] = roc_auc_avg
    output["p_r_auc_avg"] = p_r_auc_avg
    output["roc"] = {"mean_tpr": mean_tpr,
                     "all_fpr": all_fpr,
                     "final_roc_auc": final_roc_auc}
    output["p_r"] = {"mean_precision": mean_precision,
                     "all_recall": all_recall,
                     "final_p_r_auc": final_p_r_auc}

    with open(join(fd_in, "summary.pkl"), "wb") as fout:
        pkl.dump(output, fout, pkl.HIGHEST_PROTOCOL)

    return output


def perform_summarization(fd_in):
    """
    Compute stats. summarization using summaries_exps(), then plot what is it necessary, and save into text file the
    final results.
    :param fd_in:
    :return:
    """
    summary = summaries_exps(fd_in)
    with open(join(fd_in, "summary.txt"), "w") as fout:
        fout.write("Summary:\n")
        fout.write("Dice: {} % +- {} \n".format(summary["dice_avg"]["mean"], summary["dice_avg"]["std"]))
        fout.write("Classification error: {} % +- {} \n".format(summary["cl_error_avg"]["mean"],
                                                                summary["cl_error_avg"]["std"]))
        fout.write("ROC AUC: {} +- {} \n".format(summary["roc_auc_avg"]["mean"], summary["roc_auc_avg"]["std"]))
        fout.write("Final ROC AUC: {} \n".format(summary["roc"]["final_roc_auc"]))

        fout.write("Precision-recall AUC: {} +- {} \n".format(summary["p_r_auc_avg"]["mean"], summary["p_r_auc_avg"][
            "std"]))
        fout.write("Final Precision-recall AUC: {} \n".format(summary["p_r"]["final_p_r_auc"]))

    # Plot ROC and save it.
    plot_roc_curve_avg(tpr=summary["roc"]["mean_tpr"], fpr=summary["roc"]["all_fpr"],
                       roc_auc=summary["roc"]["final_roc_auc"], avg_roc_auc=summary["roc_auc_avg"]["mean"],
                       std_roc_auc=summary["roc_auc_avg"]["std"], path=join(fd_in, "roc.eps"), dpi=1000)

    # Plot Precision-recall curve and save it.
    plot_p_r_curve_avg(precision=summary["p_r"]["mean_precision"], recall=summary["p_r"]["all_recall"],
                       p_r_auc=summary["p_r"]["final_p_r_auc"], avg_p_r_auc=summary["p_r_auc_avg"]["mean"],
                       std_p_r_auc=summary["p_r_auc_avg"]["std"], path=join(fd_in, "precision_recall.eps"), dpi=1000)


def plot_roc_curve_avg(tpr, fpr,roc_auc, avg_roc_auc, std_roc_auc, path="", dpi=1000):
    """
    Plot ROC curve and save it in a high quality (*.eps).

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param tpr: numpy array, of the TPR averaged.
    :param fpr: numpy array, of the FPR fixed (unique).
    :param roc_auc: float, the ROC AUC of (fpr, tpr).
    :param avg_roc_auc: float, the average of ROC AUC of all ROC (before interpolation).
    :param std_roc_auc: float, the std of ROC AUC of all ROC (before interpolation).
    :param path: str, path where to save the figure. If you do not want to save it, set it to "".
    :param dpi: int, the dpi of the image. (1000: for high quality)
    :return:
    """
    floating = 3
    prec = "%." + str(floating) + "f"
    font_sz = 15
    lw = 2

    fig = plt.figure(figsize=(15, 15))

    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='Average ROC curve model. AUC = {}. (avg.auc +- std= {} +- {})'.format(
        prec % roc_auc, prec % avg_roc_auc, prec % std_roc_auc))
    plt.plot([0, 1], [0, 1], color="black", linestyle='--', lw=lw, label='ROC curve random guess  (AUC = {})'.format(
        prec % 0.5))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC.')
    plt.legend(loc='lower right', fancybox=True, shadow=True, prop={'size': font_sz})

    if path != "":
        fig.savefig(path, format="eps", bbox_inches='tight', dpi=dpi)
    plt.close('all')

    return fig


def plot_p_r_curve_avg(precision, recall, p_r_auc, avg_p_r_auc, std_p_r_auc, path="", dpi=1000):
    """
    Plot precision recall curve and save it in a high quality (*.eps).

    Note: The positive label (i.e., gland) is coded as 1 while 0 represents non-gland objects.

    :param precision: numpy array, of the precision averaged.
    :param recall: numpy array, of the recall fixed (unique).
    :param p_r_auc: float, the precision-recall AUC of (precision, recall).
    :param avg_p_r_auc: float, the average of precision-recall AUC of all precision-recall curves (before
    interpolation).
    :param std_p_r_auc: float, the std of precision-recall AUC of all precision-recall curves (before
    interpolation).
    :param path: str, path where to save the figure. If you do not want to save it, set it to "".
    :param dpi: int, the dpi of the image. (1000: for high quality)
    :return:
    """
    floating = 3
    prec = "%." + str(floating) + "f"
    font_sz = 15
    lw = 2

    fig = plt.figure(figsize=(15, 15))

    plt.plot(recall, precision, color='darkorange', lw=lw,
             label='Average Precision-recall curve model. AUC = {}. (avg.auc +- std= {} +- {})'.format(
             prec % p_r_auc, prec % avg_p_r_auc, prec % std_p_r_auc))
    plt.plot([0, 1], [0.5, 0.5], color="black", linestyle='--', lw=lw,
             label='Average Precision-recall curve random guess  (AUC = {})'.format(prec % 0.5))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-recall.')
    plt.legend(loc='lower right', fancybox=True, shadow=True, prop={'size': font_sz})

    if path != "":
        fig.savefig(path, format="eps", bbox_inches='tight', dpi=dpi)
    plt.close('all')

    return fig


def announce_msg(msg, upper=True):
    """
    Display sa message in the standard output. Something like this:
    =================================================================
                                message
    =================================================================

    :param msg: str, text message to display.
    :param upper: True/False, if True, the entire message is converted into uppercase. Else, the message is displayed
    as it is.
    :return: str, what was printed in the standard output.
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])

    print(output_msg)

    return output_msg


def init_stats():
    """
    Initialize the stats for tracking values of some dataset.
    :return: dict, with many keys where each one is assigned to track a
    specific statistic.
    """
    # Common keys for training and evaluation.
    out = {
        "total_loss": [],
        "loss_pos": [],
        "loss_neg": [],
        "acc": [],
        "f1pos": [],
        "f1neg": [],
        "miou": []
    }

    return out


def create_folders_for_exp(exp_folder, name):
    """
    Create a set of folder fot the current exp.
    :param exp_folder: str, the path to the current exp.
    :param name: str, name of the dataset (train, validation, test)
    :return: object, where each attribute is a folder. There is the following attributes:
        . folder: the name of the folder that will contain everything about this dataset.
        . size_evol: for the evolution of the the of the positive regions.
        . prediction: for the image prediction.
    """
    l_dirs = dict()

    l_dirs["folder"] = join(exp_folder, name)

    for k in l_dirs:
        if not os.path.exists(l_dirs[k]):
            os.makedirs(l_dirs[k])

    return Dict2Obj(l_dirs)


def check_if_allow_multgpu_mode():
    """
    Check if we can do multigpu.
    If yes, allow multigpu.
    :return: ALLOW_MULTIGPUS: bool. If True, we enter multigpu mode: 1. Computation will be dispatched over the
    AVAILABLE GPUs. 2. Synch-BN is activated.
    """
    if "CC_CLUSTER" in os.environ.keys():
        ALLOW_MULTIGPUS = os.environ["CC_CLUSTER"] in ["beluga", "cedar", "graham"]  # CC.
    else:
        ALLOW_MULTIGPUS = False  # LIVIA

    # ALLOW_MULTIGPUS = True
    os.environ["ALLOW_MULTIGPUS"] = str(ALLOW_MULTIGPUS)
    NBRGPUS = torch.cuda.device_count()
    ALLOW_MULTIGPUS = ALLOW_MULTIGPUS and (NBRGPUS > 1)

    return ALLOW_MULTIGPUS



# ===========================================================================================================
#                                            TEST
# ===========================================================================================================


def test_announce_msg():
    """
    Test announce_msg()
    :return:
    """
    announce_msg("Hello world!!!")


def test_VisualiseMIL():
    """
    test VisualiseMIL().

    :return:
    """
    OUTD = "./data/debug/visualization"
    if not os.path.exists(OUTD):
        os.makedirs(OUTD)
    path_mask = "./data/debug/input/testA_1_anno.bmp"
    input_image = Image.open("./data/debug/input/testA_1.bmp").convert("RGB")
    visualisor = VisualiseMIL(alpha=128, floating=3, height_tag=60, bins=100, rangeh=(0, 1))
    # visualisor.create_tag_input(453, 589, "benign", "").show()

    # visualisor.create_tag_pred_mask(775, "Malignant", 0.687, "wrong", 0.79, 15.17, 14.00, "hayhasyc_tag").show()

    # visualisor.create_tag_true_mask(1024, "known", 12.546).show()
    # visualisor.create_tag_heatmap_pred_mask(1024, "Final").show()

    # mask = np.random.rand(768, 1024)
    # visualisor.create_hists(mask, bins=50, rangeh=(0, 1), k=4).show()

    w, h = input_image.size
    mask = Image.open(path_mask, "r").convert("L")
    mask_np = np.array(mask)
    mask = (mask_np != 0).astype(np.float32)
    # visualisor.convert_mask_into_heatmap(input_image, mask).show()

    label = 0
    name_classes = {'benign': 0, 'malignant': 1}
    probab = 0.7
    w, h = input_image.size
    pred_mask = np.random.rand(h, w)

    t0 = dt.datetime.now()

    img = visualisor(input_image, probab, 1, pred_mask, 0.4, name_classes, "122298", 12.34, 10.78, use_tags=True,
                     label=label, mask=mask, show_hists=True, bins=None, rangeh=None)
    print("Time of visualization: `{}` .... [OK]".format(dt.datetime.now() - t0))
    img.save(join(OUTD, "display.jpeg"), "JPEG")

    print("`{}` was tested successfully .... [OK]".format(visualisor.__class__.__name__))


def test_plot_curve():
    outD = "./data/debug/plots"
    if not os.path.exists(outD):
        os.makedirs(outD)
    values_dict = {"mtx1": np.random.rand(1000),
                   "mtx2": np.random.rand(2000),
                   "mtx3": np.random.rand(1500)}
    base = 1000
    for i in range(3):
        values_dict["mtx_juj_hdy_{}".format(i)] = np.random.rand(base * (i+1))

    t0 = dt.datetime.now()
    plot_curves(values_dict, join(outD, "Fig.png"), "Title")
    print("Plotting succeeded. Time: {} .... [OK]".format(dt.datetime.now() - t0))


def test_superpose_curves():
    outD = "./data/debug/plots"
    if not os.path.exists(outD):
        os.makedirs(outD)
    values_dict = {"mtx1": np.random.rand(2000),
                   "mtx2": np.random.rand(2000)}

    t0 = dt.datetime.now()
    superpose_curves(values_dict, join(outD, "Fig-superpose.png"), 10, "Title", "label!", compute_mse=True)
    print("Plotting succeeded. Time: {} .... [OK]".format(dt.datetime.now() - t0))


def test_plot_hist_probs_pos_neg():
    outD = "./data/debug/plots"
    if not os.path.exists(outD):
        os.makedirs(outD)

    from scipy.special import softmax
    values_dict = {"probs_pos": softmax(np.random.rand(50, 2), axis=1),
                   "probs_neg": softmax(np.random.rand(50, 2), axis=1)}

    t0 = dt.datetime.now()
    plot_hist_probs_pos_neg(values_dict, join(outD, "Fig-probs-pos-neg.png"), 10, "Title")
    print("Plotting succeeded. Time: {} .... [OK]".format(dt.datetime.now() - t0))


def test_compute_roc_curve_once():
    t0 = dt.datetime.now()
    n = 26530911
    y_mask = ((np.random.rand(n) > 0.5) * 1.).astype(np.float32)
    y_hat_mask = np.random.rand(n).astype(np.float32)

    tx = dt.datetime.now()
    tpr, fpr, roc_auc, tpr_interpolated = compute_roc_curve_once(y_mask, y_hat_mask)
    print("ROC. Time: {} .... [OK]".format(dt.datetime.now() - tx))
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic.')
    plt.legend(loc="lower right")
    print("Plotting succeeded. Total test time: {} .... [OK]".format(dt.datetime.now() - t0))
    plt.show()


def test_plot_roc_curve():
    outD = "./data/debug/plots"
    if not os.path.exists(outD):
        os.makedirs(outD)
    t0 = dt.datetime.now()
    n = 200000
    y_mask = ((np.random.rand(n) > 0.5) * 1.).astype(np.float32)
    y_hat_mask = np.random.rand(n).astype(np.float32)
    out, fig = plot_roc_curve(y_mask, y_hat_mask, epoch=10, path="", title="", dpi=100)
    plt.show()

    out, fig = plot_roc_curve(y_mask, y_hat_mask, epoch=10, path=join(outD, "roc.png"), title="", dpi=100)

    print("Plotting succeeded. Time: {} .... [OK]".format(dt.datetime.now() - t0))


def test_plot_precision_recall_curve():
    outD = "./data/debug/plots"
    if not os.path.exists(outD):
        os.makedirs(outD)
    t0 = dt.datetime.now()
    n = 200000
    y_mask = ((np.random.rand(n) > 0.5) * 1.).astype(np.float32)
    y_hat_mask = np.random.rand(n).astype(np.float32)
    out, fig = plot_precision_recall_curve(y_mask, y_hat_mask, epoch=10, path="", title="", dpi=100)
    plt.show()

    out, fig = plot_precision_recall_curve(y_mask, y_hat_mask, epoch=10, path=join(outD, "precision_recall.png"),
                                           title="", dpi=100)

    print("Plotting succeeded. Time: {} .... [OK]".format(dt.datetime.now() - t0))


def test_summaries_exps():
    fd_in = "."
    t0 = dt.datetime.now()
    summaries_exps(fd_in, "*.py")

    print("Summarization succeeded. Time: {} .... [OK]".format(dt.datetime.now() - t0))


def test_perform_summarization():
    fd_in = "./exps-debug/"
    t0 = dt.datetime.now()
    perform_summarization(fd_in)

    print("Summarization perf. succeeded. Time: {} .... [OK]".format(dt.datetime.now() - t0))


def test_compute_metrics_multi_processing():
    np.random.seed(0)
    nbr_samples = 10
    dim_img = 100
    true_labels = np.random.binomial(1, 0.7, nbr_samples).tolist()
    pred_labels = np.random.binomial(1, 0.7, nbr_samples).tolist()
    true_masks = []
    pred_masks = []
    for i in range(nbr_samples):
        true_masks.append(np.random.binomial(1, 0.7, dim_img))
        pred_masks.append(np.random.binomial(1, 0.7, dim_img))

    # Compute the metrics without multiprocessing
    metrics = compute_metrics(true_labels, pred_labels, true_masks, pred_masks, binarize=False, ignore_roc_pr=True,
                              average=True)
    print("No multiprocessing:")
    for k in metrics.keys():
        print("{}: {}".format(k, metrics[k]))

    # Compute the same metrics using multiprocessing
    for i in range(8):
        metrics = compute_metrics_mp(true_labels, pred_labels, true_masks, pred_masks, binarize=False, ignore_roc_pr=True,
                                     nbr_workers=i+1)
        print("With multiprocessing: {} workers".format(i+1))
        for k in metrics.keys():
            print("{}: {}".format(k, metrics[k]))


def test_CRF():
    from PIL import Image
    from scipy.special import softmax
    crf = CRF(2)
    img = Image.open('data/debug/input/testA_1.bmp').convert('RGB')
    print(img.size)
    img = np.array(img)
    print(img.shape, img.dtype)
    h, w, _ = img.shape

    sfmx = np.random.rand(2, h, w)
    sfmx = softmax(sfmx, axis=0).astype(np.float32)
    res = sfmx[0, :, :] * 255 / sfmx.max()
    plt.imshow(res)
    plt.show()

    sfmx_ = crf(img, sfmx)

    res = sfmx_[0, :, :] * 255 / sfmx_.max()
    plt.imshow(res)
    plt.show()


if __name__ == "__main__":

    # test_plot_curve()

    # test_superpose_curves()

    # test_plot_hist_probs_pos_neg()

    # test_VisualiseMIL()

    # test_announce_msg()

    # test_compute_roc_curve_once()

    # test_plot_roc_curve()

    # test_plot_precision_recall_curve()

    # test_summaries_exps()

    # test_perform_summarization()

    # test_compute_metrics_multi_processing()

    test_CRF()


