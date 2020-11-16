from torch.utils.data import DataLoader


from loader import PhotoDataset
from loader import default_collate
from loader import _init_fn


import reproducibility


FACTOR_MUL_WORKERS = 2  # args.num_workers * this_factor.
# Useful when setting set_for_eval to False, batch size =1,


def get_eval_dataset(args,
                     myseed,
                     valid_samples,
                     transform_tensor
                     ):
    """
    Return dataset and its dataloader.
    :return:
    """
    reproducibility.force_seed(myseed)
    pad_vld_sz = None if not args.pad_eval else args.padding_size
    pad_vl_md = None if not args.pad_eval else args.padding_mode
    validset = PhotoDataset(valid_samples,
                            args.dataset,
                            args.name_classes,
                            transform_tensor,
                            set_for_eval=False,
                            transform_img=None,
                            resize=args.resize,
                            crop_size=None,
                            padding_size=pad_vld_sz,
                            padding_mode=pad_vl_md,
                            force_div_32=False,
                            up_scale_small_dim_to=args.up_scale_small_dim_to
                            )

    reproducibility.force_seed(myseed)
    valid_loader = DataLoader(validset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.num_workers * FACTOR_MUL_WORKERS,
                              pin_memory=True,
                              collate_fn=default_collate,
                              worker_init_fn=_init_fn
                              )  # we need more workers since the batch size is
    # 1, and set_for_eval is False (need more time to prepare a sample).
    reproducibility.force_seed(myseed)
    return validset, valid_loader