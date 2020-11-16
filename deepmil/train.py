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

from tools import log
from tools import announce_msg
from tools import VisualiseMIL

from deepmil.criteria import Metrics

import reproducibility


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
    metrics.eval()

    f1pos_tr, f1neg_tr, miou_tr, acc_tr = 0., 0., 0., 0.
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
        scores_pos, scores_neg, mask_pred, sc_cl_se = model(
            x=data,
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
                                            scores_neg
                                            )
        t_loss.backward()

        # Update params.
        optimizer.step()
        # End optimization.
        acc, dice_forg, dice_back, miou = metrics(
            scores=scores_pos,
            labels=labels,
            masks_pred=mask_pred.contiguous().view(bsz, -1),
            masks_trg=masks.contiguous().view(bsz, -1),
            avg=True
            )

        # tracking
        tr_stats["total_loss"].append(t_loss.item())
        tr_stats["loss_pos"].append(l_p.item())
        tr_stats["loss_neg"].append(l_n.item())
        tr_stats["acc"].append(acc * 100.)
        tr_stats["f1pos"].append(dice_forg * 100.)
        tr_stats["f1neg"].append(dice_back * 100.)
        tr_stats['miou'].append(miou * 100.)

        f1pos_tr += dice_forg
        f1neg_tr += dice_back
        miou_tr += miou
        acc_tr += acc
        cnt += bsz

    # avg
    f1neg_tr = f1neg_tr * 100. / float(cnt)
    f1pos_tr = f1pos_tr * 100. / float(cnt)
    acc_tr = acc_tr * 100. / float(cnt)
    miou_tr = miou_tr * 100. / float(cnt)

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
    visualiser = VisualiseMIL(alpha=args.alpha_plot,
                              floating=args.floating,
                              height_tag=args.height_tag,
                              bins=args.bins,
                              rangeh=args.rangeh
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
    # name_file = dataset.absolute_paths_imgs[i].split(os.sep)[-1].split(".")[
    #     0]  # e.g. 'train_13'
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
             store_imgs=False
             ):
    """
    Perform a validation over the validation set. Assumes a batch size of 1.
    (images do not have the same size,
    so we can't stack them in one tensor).
    Validation samples may be large to fit all in the GPU at once.

    Note: criterion is deppmil.criteria.TotalLossEval().
    """
    model.eval()
    metrics = Metrics(threshold=args.final_thres).to(device)
    metrics.eval()

    f1pos_, f1neg_, miou_, acc_ = 0., 0., 0., 0.
    cnt = 0.
    total_loss_ = 0.
    loss_pos_ = 0.
    loss_neg_ = 0.

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
            mask = torch.tensor(mask[0])
            mask_t = mask.unsqueeze(0).to(device)
            assert mask_t.ndim == 4, "ndim = {} must be 4.".format(mask_t.ndim)

            # In validation, we do not need reproducibility since everything
            # is expected to deterministic. Plus,
            # we use only one gpu since the batch size os 1.
            scores_pos, scores_neg, mask_pred, sc_cl_se = model(x=data,
                                                               seed=None
                                                                )
            t_loss, l_p, l_n, l_seg = criterion(scores_pos,
                                                sc_cl_se,
                                                labels,
                                                mask_pred,
                                                scores_neg
                                               )


            mask_pred = mask_pred.squeeze()
            # check sizes of the mask:
            _, _, h, w = mask_t.shape
            hp, wp = mask_pred.shape

            if (h != hp) or (w != wp):  # This means that we have padded the
                # input image. We crop the predicted mask in the center.
                mask_pred = mask_pred[int(hp / 2) - int(h / 2): int(hp / 2) + int(h / 2) + (h % 2),
                                      int(wp / 2) - int(w / 2): int(wp / 2) + int(w / 2) + (w % 2)]

            mask_pred = mask_pred.unsqueeze(0).unsqueeze(0)

            acc, dice_forg, dice_back, miou = metrics(
                scores=scores_pos,
                labels=labels,
                masks_pred=mask_pred.contiguous().view(bsz, -1),
                masks_trg=mask_t.contiguous().view(bsz, -1),
                avg=False
            )

            # tracking
            f1pos_ += dice_forg
            f1neg_ += dice_back
            miou_ += miou
            acc_ += acc
            cnt += bsz
            total_loss_ += t_loss.item()
            loss_pos_ += l_p.item()
            loss_neg_ += l_n.item()

            if (folderout is not None) and store_on_disc:
                # binary mask
                bin_pred_mask = metrics.get_binary_mask(mask_pred).squeeze()
                bin_pred_mask = bin_pred_mask.cpu().detach().numpy().astype(np.bool)
                to_save = {
                    "bin_pred_mask": bin_pred_mask,
                    "dice_forg": dice_forg,
                    "dice_back": dice_back,
                    "i": i
                }

                with open(join(bin_masks_fd, "{}.pkl".format(i)), "wb") as fbin:
                    pkl.dump(to_save, fbin, protocol=pkl.HIGHEST_PROTOCOL)

            if (folderout is not None) and store_imgs and store_on_disc:
                pred_label = int(scores_pos.argmax().item())
                probs = softmax(scores_pos.cpu().detach().numpy())
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


    # avg
    total_loss_ /= float(cnt)
    loss_pos_ /= float(cnt)
    loss_neg_ /= float(cnt)
    acc_ *= (100. / float(cnt))
    f1pos_ *= (100. / float(cnt))
    f1neg_ *= (100. / float(cnt))
    miou_ *= (100. / float(cnt))

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


    if folderout is not None:
        msg = "EVAL {}: \n".format(name_set)
        msg += "ACC {}% \n".format(acc_)
        msg += "F1+ {}% \n".format(f1pos_)
        msg += "F1- {}% \n".format(f1neg_)
        msg += "MIOU {}% \n".format(miou_)
        announce_msg(msg)
        if log_file:
            log(log_file, msg)

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
        subprocess.run(cmdx, shell=True, check=True)
    else:
        return stats
