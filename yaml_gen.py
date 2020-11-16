import datetime as dt
from os.path import join

import yaml


import constants

# TODO: Adjust the exp. generator.

# DEFAULT configuration.
config = {
    # ######################### GENERAL STUFF ##############################
    "MYSEED": 0,  # Seed for reproducibility. [0, 1, 2, ...]
    "dataset": constants.GLAS,  # name of the dataset: glas,
    # Caltech-UCSD-Birds-200-2011,
    # Oxford-flowers-102
    "img_extension": "jpg",  # extension of the images in the dataset.
    "name_classes": {'benign': 0, 'malignant': 1},  # dict. name classes and
    # corresponding int. If dict if too big,
    # you can dump it in the fold folder in a yaml file. We will load it when
    # needed. Use the name of the file.
    "nbr_classes": 2,  # Total number of classes. glas: 2,
    # Caltech-UCSD-Birds-200-2011: 200.
    "split": 0,  # split id.
    "fold": 0,  # folder id.
    "fold_folder": "./folds",  # relative path to the folder of the folds.
    "resize": None,  # PIL format of the image size (w, h). The size to which
    # the original images are resized to.
    "crop_size": (480, 480),  # Size of the patches to be cropped (h, w).
    "up_scale_small_dim_to": 500,  # None # int or None. If int, the images are
    # upscaled to this size while
    # preserving the ratio. See loader.PhotoDataset().
    "padding_size": None,  # float,  # padding ratios for the original image
    # for (top/bottom) and (left/right).
    # Can be
    # applied on both, training/evaluation modes. To be specified in
    # PhotoDataset(). If specified, only training
    # images are padded. To pad evaluation images, you need to set the
    # variable: `pad_eval` to True.
    "pad_eval": False,  # If True, evaluation images are padded in the same
    # way. The final mask is cropped inside the
    # predicted mask (since this last one is bigger due to the padding).
    "padding_mode": "reflect",  # type of padding. Accepted modes:
    # https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.
    # transforms.functional.pad
    "preload": True,  # If True, images are loaded and saved in RAM to avoid
    # disc access.
    "batch_size": 8,  # the batch size for training.
    "valid_batch_size": 1,  # the batch size for validation.
    "num_workers": 8,  # number of workers for dataloader of the trainset.
    "max_epochs": 400,  # number of training epochs.
    # ######################### VISUALISATION OF REGIONS OF INTEREST #######
    "normalize": True,  # If True, maps are normalized using softmax.
    # [NOT USED IN THIS CODE]
    "alpha_plot": 128,  # transparency alpha, used for plotting. In [0, 255].
    # The lower the value, the more transparent
    # the map.
    "floating": 3,  # the number of floating points to print over the maps.
    "height_tag": 50,  # the height of the margin where the tag is written.
    "use_tags": True,  # If True, extra information will be display under the
    # images.
    "show_hists": False,  # If True, histograms of scores will be displayed as
    # density probability.
    "bins": 100,  # int, number of bins in the histogram.
    "rangeh": (0, 1),  # tuple, range of the histogram.
    "extension": ("jpeg", "JPEG"),  # format into which the maps are saved.
    # ######################### Optimizer ##############################
    "optimizer": {  # the optimizer
        # ==================== SGD =======================
        "name": "sgd",  # str name.
        "lr": 0.001,  # Initial learning rate.
        "momentum": 0.9,  # Momentum.
        "dampening": 0.,  # dampening.
        "weight_decay": 1e-5,  # The weight decay (L2) over the parameters.
        "nesterov": True,  # If True, Nesterov algorithm is used.
        # ========== LR scheduler: how to adjust the learning rate. ============
        # ========> torch.optim.lr_scheduler.StepLR
        # "name": "step",  # str name.
        # "step_size": 20,  # Frequency of which to adjust the lr.
        # "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
        # "last_epoch": -1,  # the index of the last epoch where to
        # stop adjusting the LR.
        # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
        "use_lr_scheduler": True,
        "lr_scheduler_name": "mystep",  # str name.
        "step_size": 40,  # Frequency of which to adjust the lr.
        "gamma": 0.5,  # the update coefficient: lr = gamma * lr.
        "last_epoch": -1,  # the index of the last epoch where to stop
        # adjusting the LR.
        "min_lr": 1e-7,  # minimum allowed value for lr.
        # ========> torch.optim.lr_scheduler.MultiStepLR
        # "name": "multistep",  # str name.
        # "milestones": [0, 100],  # milestones.
        # "gamma": 0.1,  # the update coefficient: lr = gamma * lr.
        # "last_epoch": -1  # the index of the last epoch where to stop
        # adjusting the LR.
    },
    # ######################### Model ##############################
    "model": {
        "model_name": "resnet18",  # name of the classifier.
        "pretrained": True,  # use/or not the ImageNet pretrained models.
        "strict": True,  # bool. Must be always be True. if True,
        # the pretrained model has to have the exact architecture as this
        # current model. if not, an error will be raise. if False, we do the
        # best. no error will be raised in case of mismatch.
        "path_pre_trained": None,  # None, `None` or a valid str-path. if str,
        # it is the absolute/relative path to the pretrained model. This can
        # be useful to resume training or to force using a filepath to some
        # pretrained weights.
        # =============================  classifier ==========================
        "num_classes": 2,  # number of output classes. glas: 2,
        # Caltech-UCSD-Birds-200-2011: 200.
        "scale_in_cl": 1.,  # float. will be converted to (scale, scale),
        # ratio used to the input images for the classifier. or 1. if no
        # scale is required.
        "modalities": 5,  # number of modalities (wildcat).
        "kmax": 0.1,  # kmax. (wildcat)
        "kmin": 0.1,  # kmin. (wildcat)
        "alpha": 0.0,  # alpha. (wildcat)
        "dropout": 0.0,  # dropout over the kmin and kmax selected activations.
        # . (wildcat).
        # ===============================  Segmentor ===========================
        "sigma": 0.15,  # simga for the thresholding (init. value).
        "delta_sigma": 0.001,  # how much to increase sigma each epoch.
        "max_sigma": 0.2,   # max value of sigma.
        "w": 5.  # w for the thresholding.
    },
    "use_reg": False,  # perform or not the regularization over the background.
    "reg_loss": constants.KLUniform,  # loss regu. over the background.
    "final_thres": 0.5,  # segm. threshold final.
    "debug_subfolder": '',  # subfolder used for debug. if '', we do not
    # consider it.
    "use_size_const": False,  # use or not size constraint over background.
    "init_t": 5.,  # elb for size cons. over background.
    "max_t": 10.,  # elb for size cons. over background.
    "mulcoef": 1.01,  # elb for size cons. over background.
    "normalize_sz": False,  # normalize or not the size of a background mask.
    "epsilon": 0.,  # elb for size cons. over background.
    "lambda_neg": 1e-7  # lambda for the background loss.
}


fold_yaml = "config_yaml"
fold_bash = "config_bash"
name_config = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
name_config = "glas"

name_yaml = join(fold_yaml, name_config + ".yaml")

with open(name_yaml, 'w') as f:
    yaml.dump(config, f)
