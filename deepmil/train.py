import os
from os.path import join
import pickle as pkl
import subprocess
import datetime as dt
from copy import deepcopy


import tqdm


import numpy as np
from scipy.special import softmax
import torch
from torch.nn import functional as F
import matplotlib as mlp
import matplotlib.pyplot as plt

from tools import log
from tools import announce_msg
from tools import VisualiseMIL

from deepmil.criteria import Metrics
from deepmil.criteria import Dice
from deepmil.criteria import IOU

import reproducibility
import constants
from estimate_thres import EstimateThresSeg
from estimate_thres import EstimateThresCl


def train_one_epoch(model,
                    optimizer,
                    dataloader,
                    criterion,
                    device,
                    tr_stats,
                    args,
                    epoch=0,
                    log_file=None,
                    ALLOW_MULTIGPUS=False,
                    NBRGPUS=1
                    ):
    """
    Perform one epoch of training.
    :param model:
    :param optimizer:
    :param dataloader:
    :param criterion:
    :param device:
    :param epoch:
    :param callback:
    :param log_file:
    :param ALLOW_MULTIGPUS: bool. If True, we are in multiGPU mode.
    :return:
    """
    model.train()

    metrics = Metrics(threshold=args.final_thres).to(device)
    dice_f = Dice()
    iou_f = IOU()
    metrics.eval()

    f1pos_tr, f1neg_tr, miou_tr, acc_tr = 0., 0., 0., 0.
    l_f1pos_tr, l_f1neg_tr, l_miou_tr, l_acc_tr = [], [], [], []
    cnt = 0.

    length = len(dataloader)
    t0 = dt.datetime.now()
    myseed = int(os.environ["MYSEED"])

    for i, (data, masks, labels) in tqdm.tqdm(
            enumerate(dataloader), ncols=80, total=length):
        reproducibility.force_seed(myseed + epoch)

        data = data.to(device)
        labels = labels.to(device)
        masks = torch.stack(masks)
        masks = masks.to(device)

        model.zero_grad()

        t_l, l_p, l_n, l_c_s = 0., 0., 0., 0.
        prngs_cuda = None  # TODO: crack in optimal code.
        bsz = data.shape[0]

        # Optimization:
        # if model.nbr_times_erase == 0:  # no erasing.
        if not ALLOW_MULTIGPUS:
            # TODO: crack in optimal code.
            if "CC_CLUSTER" in os.environ.keys():
                msg = "Something wrong. You deactivated multigpu mode, " \
                      "but we find {} GPUs. This will not guarantee " \
                      "reproducibility. We do not know why you did that. " \
                      "Exiting .... [NOT OK]".format(NBRGPUS)
                assert NBRGPUS <= 1, msg
            seeds_threads = None
        else:
            msg = "Something is wrong. You asked for multigpu mode. " \
                  "But, we found {} GPUs. Exiting " \
                  ".... [NOT OK]".format(NBRGPUS)
            assert NBRGPUS > 1, msg
            # The seeds are generated randomly before calling the threads.
            reproducibility.force_seed(myseed + epoch + i)  # armor.
            seeds_threads = torch.randint(
                0, np.iinfo(np.uint32).max + 1, (NBRGPUS, )).to(device)
            reproducibility.force_seed(myseed + epoch + i)  # armor.
            prngs_cuda = []
            # Create different prng states of cuda before forking.
            for seed in seeds_threads:
                # get the corresponding state of the cuda prng with respect
                # to the seed.
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                prngs_cuda.append(torch.cuda.get_rng_state())
            reproducibility.force_seed(myseed + epoch + i)  # armor.

        # TODO: crack in optimal code.
        if prngs_cuda is not None and prngs_cuda != []:
            prngs_cuda = torch.stack(prngs_cuda)

        reproducibility.force_seed(myseed + epoch + i)  # armor.
        scores_pos, scores_neg, mask_pred, sc_cl_se, cams = model(
            x=data,
            glabels=labels,
            seed=seeds_threads,
            prngs_cuda=prngs_cuda
        )

        reproducibility.force_seed(myseed + epoch + i)  # armor.

        msg = "shape mismatches: pred {}  true {}".format(
            masks.shape, mask_pred.shape)
        assert masks.shape == mask_pred.shape, msg

        t_loss, l_p, l_n, l_seg = criterion(scores_pos,
                                            sc_cl_se,
                                            labels,
                                            mask_pred,
                                            scores_neg,
                                            cams
                                            )

        t_loss.backward()

        # Update params.
        optimizer.step()

        # End optimization.
        cnt += bsz

        if args.dataset == constants.GLAS:
            cl_scores = scores_pos  # sc_cl_se
            acc, dice_forg, dice_back, miou = metrics(
                scores=cl_scores,
                labels=labels,
                masks_pred=mask_pred.contiguous().view(bsz, -1),
                masks_trg=masks.contiguous().view(bsz, -1),
                avg=True
                )

            # tracking
            f1pos_tr += dice_forg
            f1neg_tr += dice_back
            miou_tr += miou
        elif args.dataset == constants.CAMELYON16P512:
            cl_scores = sc_cl_se  # sc_cl_se, scores_pos
            ppixels = metrics.get_binary_mask(
                mask_pred.contiguous().view(bsz, -1), metrics.threshold)
            masks_trg = masks.contiguous().view(bsz, -1)

            dice_forg_, dice_back_, miou_ = [], [], []

            plabels = metrics.predict_label(cl_scores)
            acc = ((plabels - labels) == 0.).float().sum()

            for zz in range(bsz):
                if masks_trg[zz].sum() > 0:
                    l_f1pos_tr.append(
                        dice_f(ppixels[zz].view(1, -1),
                               masks_trg[zz].view(1, -1)
                               ))
                    dice_forg_.append(l_f1pos_tr[-1])

                    if (1. - masks_trg[zz]).sum() > 0:
                        tmp = iou_f(ppixels[zz].view(1, -1),
                                    masks_trg[zz].view(1, -1))
                        tmp = tmp + iou_f(1. - ppixels[zz].view(1, -1),
                                          1. - masks_trg[zz].view(1, -1))
                        l_miou_tr.append(tmp / 2.)
                        miou_.append(l_miou_tr[-1])

                if (1. - masks_trg[zz]).sum() > 0:
                    l_f1neg_tr.append(dice_f(1. - ppixels[zz].view(1, -1),
                                             1. - masks_trg[zz].view(1, -1)
                               ))
                    dice_back_.append(l_f1neg_tr[-1])

            dice_forg = torch.stack(dice_forg_).mean() if dice_forg_ else 0.
            dice_back = torch.stack(dice_back_).mean() if dice_back_ else 0.
            miou = torch.stack(miou_).mean() if miou_ else 0.
        else:
            raise NotImplementedError

        tr_stats["total_loss"].append(t_loss.item())
        tr_stats["loss_pos"].append(l_p.item())
        tr_stats["loss_neg"].append(l_n.item())

        tr_stats["acc"].append(acc * 100.)
        tr_stats["f1pos"].append(dice_forg * 100.)
        tr_stats["f1neg"].append(dice_back * 100.)
        tr_stats['miou'].append(miou * 100.)

        acc_tr += acc

    # avg
    acc_tr = acc_tr * 100. / float(cnt)
    if args.dataset == constants.GLAS:
        f1neg_tr = f1neg_tr * 100. / float(cnt)
        f1pos_tr = f1pos_tr * 100. / float(cnt)
        miou_tr = miou_tr * 100. / float(cnt)
    elif args.dataset == constants.CAMELYON16P512:
        f1neg_tr = torch.stack(l_f1neg_tr).mean().item() * 100.
        f1pos_tr = torch.stack(l_f1pos_tr).mean().item() * 100.
        miou_tr = torch.stack(l_miou_tr).mean().item() * 100.

    to_write = "Train epoch {:>2d}: f1+: {:.2f}, f1-: {:.2f}, " \
               "miou: {:.2f}, acc: {:.2f}, LR {}, t:{}".format(
                epoch, f1pos_tr, f1neg_tr, miou_tr, acc_tr,
                ['{:.2e}'.format(group["lr"]) for group in optimizer.param_groups],
                dt.datetime.now() - t0
                )
    print(to_write)
    if log_file:
        log(log_file, to_write)

    return tr_stats


def store_pred_img(i,
                   dataset,
                   pred_mask_bin,
                   pred_mask_con,
                   dice_forg,
                   dice_back,
                   prob,
                   pred_label,
                   args,
                   outd):

    color_map = plt.get_cmap("jet")

    visualiser = VisualiseMIL(alpha=args.alpha_plot,
                              floating=args.floating,
                              height_tag=args.height_tag,
                              bins=args.bins,
                              rangeh=args.rangeh,
                              color_map=color_map
                              )
    img = dataset.get_original_input_img(i)  # PIL.Image.Image uint8 RGB image.
    label = dataset.get_original_input_label_int(i)  # int.
    true_mask = np.array(dataset.get_original_input_mask(i))
    true_mask = (true_mask != 0).astype(np.float32)

    img_visu = visualiser(img,
                          prob,
                          pred_label,
                          deepcopy(pred_mask_con),
                          dice_forg,
                          dice_back,
                          args.name_classes,
                          "Final",
                          pred_mask_bin=pred_mask_bin,
                          use_tags=True,
                          label=label,
                          mask=true_mask,
                          show_hists=False,
                          bins=args.bins,
                          rangeh=args.rangeh
                          )
    name_file = str(i)
    img_visu.save(join(outd, name_file + "." + args.extension[0]),
                  args.extension[1], optimize=True)


def validate(model,
             dataset,
             dataloader,
             criterion,
             device,
             stats,
             args,
             folderout=None,
             epoch=0,
             log_file=None,
             name_set="",
             store_on_disc=False,
             store_imgs=False,
             final_mode=False,
             seg_threshold=None
             ):
    """
    Perform a validation over the validation set. Assumes a batch size of 1.
    (images do not have the same size,
    so we can't stack them in one tensor).
    Validation samples may be large to fit all in the GPU at once.

    Note: criterion is deppmil.criteria.TotalLossEval().
    """
    model.eval()
    fl_thres = seg_threshold if seg_threshold is not None else args.final_thres

    metrics = Metrics(threshold=fl_thres).to(device)
    dice_f = Dice()
    iou_f = IOU()
    metrics.eval()

    est_th_seg = EstimateThresSeg(
        metric=metrics, start=0.05, stop=1., step=0.01)
    est_th_cl_sz = EstimateThresCl()
    est_th_cl = EstimateThresCl()

    f1pos_, f1neg_, miou_, acc_ = 0., 0., 0., 0.
    l_f1pos_ = []
    l_f1neg_ = []
    l_miou_ = []

    cnt = 0.
    total_loss_ = 0.
    loss_pos_ = 0.
    loss_neg_ = 0.

    # camelyon16
    sizes_m = {
        'm_pred': [],  # metastatic.
        'm_true': [],  # metastatic
        'n_pred': [],  # normal
        'n_true': []  # normal
    }

    mask_fd = None
    name_fd_masks = "masks"  # where to store the predictions.
    name_fd_masks_bin = "masks_bin"  # where to store the bin masks and al.
    if folderout is not None:
        mask_fd = join(folderout, name_fd_masks)
        bin_masks_fd = join(folderout, name_fd_masks_bin)
        if not os.path.exists(mask_fd):
            os.makedirs(mask_fd)

        if not os.path.exists(bin_masks_fd):
            os.makedirs(bin_masks_fd)

    length = len(dataloader)
    t0 = dt.datetime.now()
    myseed = int(os.environ["MYSEED"])
    masks_sizes = []
    cancer_scores = []
    glabels = []

    avg_forward_t = dt.timedelta(0)

    with torch.no_grad():
        for i, (data, mask, label) in tqdm.tqdm(
                enumerate(dataloader), ncols=80, total=length):

            reproducibility.force_seed(myseed + epoch + 1)

            msg = "Expected a batch size of 1. Found `{}`  .... " \
                  "[NOT OK]".format(data.size()[0])
            assert data.size()[0] == 1, msg
            bsz = data.size()[0]

            data = data.to(device)
            labels = label.to(device)
            mask = mask[0].clone()
            mask_t = mask.unsqueeze(0).to(device)
            assert mask_t.ndim == 4, "ndim = {} must be 4.".format(mask_t.ndim)

            # In validation, we do not need reproducibility since everything
            # is expected to deterministic. Plus,
            # we use only one gpu since the batch size os 1.
            t0 = dt.datetime.now()
            scores_pos, scores_neg, mask_pred, sc_cl_se, cams = model(
                x=data, glabels=labels, seed=None)
            delta_t = dt.datetime.now() - t0
            avg_forward_t += delta_t

            t_loss, l_p, l_n, l_seg = criterion(
                scores_pos,
                sc_cl_se,
                labels,
                mask_pred,
                scores_neg,
                cams
            )


            mask_pred = mask_pred.squeeze()
            # check sizes of the mask:
            _, _, h, w = mask_t.shape
            hp, wp = mask_pred.shape

            assert args.padding_size in [None, 'None', 0.0], args.padding_size
            mask_pred = F.interpolate(
                input=mask_pred.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=True
            ).squeeze()

            mask_pred = mask_pred.unsqueeze(0).unsqueeze(0)

            if args.dataset == constants.GLAS:
                cl_scores = scores_pos
                acc, dice_forg, dice_back, miou = metrics(
                    scores=cl_scores,
                    labels=labels,
                    masks_pred=mask_pred.contiguous().view(bsz, -1),
                    masks_trg=mask_t.contiguous().view(bsz, -1),
                    avg=False
                )
                f1pos_ += dice_forg
                f1neg_ += dice_back
                miou_ += miou

            elif args.dataset == constants.CAMELYON16P512:
                cl_scores = sc_cl_se  # sc_cl_se, scores_pos

                plabels = metrics.predict_label(cl_scores)
                acc = ((plabels - labels) == 0.).float().sum()
                assert data.size()[0] == 1
                ppixels = metrics.get_binary_mask(
                    mask_pred.contiguous().view(bsz, -1), metrics.threshold)
                masks_trg = mask_t.contiguous().view(bsz, -1)
                dice_forg = 0.
                dice_back = 0.
                miou = 0.

                glabels.append(labels.item())
                masks_sizes.append(ppixels.mean().item())

                if labels.item() == 1:
                    sizes_m['m_pred'].append(ppixels.mean().item() * 100.)
                    sizes_m['m_true'].append(masks_trg.mean().item() * 100.)
                elif labels.item() == 0:
                    sizes_m['n_pred'].append(ppixels.mean().item() * 100.)
                    sizes_m['n_true'].append(0.0)
                else:
                    raise ValueError

                cancer_scores.append(
                    torch.softmax(cl_scores, dim=1)[0, 1].item())

                if masks_trg.sum() > 0:
                    l_f1pos_.append(dice_f(ppixels.view(bsz, -1),
                                    masks_trg.view(bsz, -1)))
                    dice_forg = l_f1pos_[-1]

                    if (1. - masks_trg).sum() > 0:
                        tmp = iou_f(ppixels.view(1, -1),
                                    masks_trg.view(1, -1))
                        tmp = tmp + iou_f(1. - ppixels.view(1, -1),
                                          1. - masks_trg.view(1, -1))
                        l_miou_.append(tmp / 2.)
                        miou = l_miou_[-1]

                if (1. - masks_trg).sum() > 0:
                    l_f1neg_.append(dice_f(1. - ppixels.view(bsz, -1),
                                           1. - masks_trg.view(bsz, -1)))
                    dice_back = l_f1neg_[-1]

                # estimate
                est_th_seg(mask_pred=mask_pred, mask_trg=masks_trg)

            else:
                raise NotImplementedError

            acc_ += acc
            cnt += bsz
            total_loss_ += t_loss.item()
            loss_pos_ += l_p.item()
            loss_neg_ += l_n.item()

            if (folderout is not None) and store_on_disc:
                # binary mask
                bin_pred_mask = metrics.get_binary_mask(mask_pred).squeeze()
                bin_pred_mask = bin_pred_mask.cpu().detach().numpy().astype(np.bool)
                if args.dataset == constants.GLAS:
                    to_save = {
                        "bin_pred_mask": bin_pred_mask,
                        "continuous_mask": mask_pred.cpu().detach().numpy(),
                        "dice_forg": dice_forg,
                        "dice_back": dice_back,
                        "i": i
                    }
                elif args.dataset == constants.CAMELYON16P512:
                    to_save = {
                        "bin_pred_mask": bin_pred_mask,
                        "dice_forg": dice_forg,
                        "dice_back": dice_back,
                        "i": i
                    }
                else:
                    raise NotImplementedError

                with open(join(bin_masks_fd, "{}.pkl".format(i)), "wb") as fbin:
                    pkl.dump(to_save, fbin, protocol=pkl.HIGHEST_PROTOCOL)

            if (folderout is not None) and store_imgs and store_on_disc:
                pred_label = int(cl_scores.argmax().item())
                probs = softmax(cl_scores.cpu().detach().numpy())
                prob = float(probs[0, pred_label])

                store_pred_img(i,
                               dataset,
                               bin_pred_mask * 1.,
                               mask_pred.squeeze().cpu().detach().numpy(),
                               dice_forg,
                               dice_back,
                               prob,
                               pred_label,
                               args,
                               mask_fd,
                               )

    if args.dataset == constants.CAMELYON16P512:
        est_th_seg.log_to_text_file(
            path_file=join(folderout,
                           'log-seg-{}-final-{}.txt'.format(epoch, final_mode)))

        est_th_cl_sz(scores=masks_sizes, glabels=glabels)
        est_th_cl_sz.log_to_text_file(
            path_file=join(folderout,
                           'log-cl-SZ-{}-final-{}.txt'.format(
                               epoch, final_mode)))
        est_th_cl(scores=cancer_scores, glabels=glabels)
        est_th_cl.log_to_text_file(
            path_file=join(folderout,
                           'log-cl-SCORE-SEG-{}-final-{}.txt'.format(
                               epoch, final_mode)))
        path_siz = join(folderout, 'size-{}-{}.pkl'.format(
            args.reg_loss, name_set))

        with open(path_siz, 'wb') as fout:
            pkl.dump(sizes_m, fout, protocol=pkl.HIGHEST_PROTOCOL)
        print('Stored sizes in {}'.format(path_siz))

    # avg
    acc_ *= (100. / float(cnt))
    if args.dataset == constants.CAMELYON16P512:
        with open(join(folderout,
                           'log-cl-{}-final-{}.txt'.format(epoch,
                                                           final_mode)),
                  'a') as fz:
            fz.write("\nClassification accuracy: {} (%)".format(acc_))

        with open(join(folderout,
                       'log-seg-{}-final-{}.txt'.format(
                           epoch, final_mode)), 'a') as fz:
            fz.write("\nClassification accuracy: {} (%)".format(acc_))

    total_loss_ /= float(cnt)
    loss_pos_ /= float(cnt)
    loss_neg_ /= float(cnt)
    avg_forward_t /= float(cnt)

    if args.dataset == constants.GLAS:
        f1pos_ *= (100. / float(cnt))
        f1neg_ *= (100. / float(cnt))
        miou_ *= (100. / float(cnt))
    elif args.dataset == constants.CAMELYON16P512:
        f1pos_ = 0.
        f1neg_ = 0.
        miou_ = 0.
        if l_f1pos_:
            f1pos_ = torch.stack(l_f1pos_).mean() * 100.
        if l_f1neg_:
            f1neg_ = torch.stack(l_f1neg_).mean() * 100.
        if l_miou_:
            miou_ = torch.stack(l_miou_).mean() * 100.
    else:
        raise NotImplementedError

    if stats is not None:
        stats["total_loss"].append(total_loss_)
        stats["loss_pos"].append(loss_pos_)
        stats["loss_neg"].append(loss_neg_)
        stats["acc"].append(acc_)
        stats["f1pos"].append(f1pos_)
        stats["f1neg"].append(f1neg_)
        stats['miou'].append(miou_)

    to_write = "EVAL ({}): TLoss: {:.2f}, L+: {:.2f}, L-: {:.2f}, " \
               "F1+: {:.2f}%, F1-: {:.2f}%, MIOU: {:.2f}%, ACC: {:.2f}%, " \
               "t:{}, epoch {:>2d}.".format(
        name_set,
        total_loss_,
        loss_pos_,
        loss_neg_,
        f1pos_,
        f1neg_,
        miou_,
        acc_,
        dt.datetime.now() - t0,
        epoch
        )

    print(to_write)
    if log_file:
        log(log_file, to_write)

    if final_mode:
        assert folderout is not None
        msg = "EVAL {}: \n".format(name_set)
        msg += "ACC {}% \n".format(acc_)
        msg += "F1+ {}% \n".format(f1pos_)
        msg += "F1- {}% \n".format(f1neg_)
        msg += "MIOU {}% \n".format(miou_)
        announce_msg(msg)
        if log_file:
            log(log_file, msg)

        with open(join(folderout, 'avg_forward_time.txt'), 'w') as fend:
            fend.write("Model: {}. \n Average forward time (eval mode): "
                       " {}.".format(args.model['model_name'], avg_forward_t
            ))

    if (folderout is not None) and store_on_disc:
        pred = {
            "total_loss": total_loss_,
            "loss_pos": loss_pos_,
            "loss_neg": loss_neg_,
            "acc": acc_,
            "f1pos": f1pos_,
            "f1neg": f1neg_,
            "miou_": miou_
        }
        with open(
                join(folderout, "pred--{}.pkl".format(name_set)), "wb") as fout:
            pkl.dump(pred, fout, protocol=pkl.HIGHEST_PROTOCOL)

        # compress. delete folder.
        cmdx = [
            "cd {} ".format(mask_fd),
            "cd .. ",
           # "tar -cf {}.tar.gz {}".format(name_fd_masks, name_fd_masks),
           # "rm -r {}".format(name_fd_masks)
        ]

        cmdx += [
            "cd {} ".format(bin_masks_fd),
            "cd .. ",
            "tar -cf {}.tar.gz {}".format(name_fd_masks_bin, name_fd_masks_bin),
            "rm -r {}".format(name_fd_masks_bin)
        ]

        cmdx = " && ".join(cmdx)
        print("Running bash-cmds: \n{}".format(cmdx.replace("&& ", "\n")))
        try:
            subprocess.run(cmdx, shell=True, check=True)
        except subprocess.SubprocessError as e:
            print("Failed to run: {}. Error: {}".format(cmdx, e))

    else:
        return stats
