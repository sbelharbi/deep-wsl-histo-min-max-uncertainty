import random
import os
import warnings


import numpy as np
import torch
from torch._C import default_generator


from tools import announce_msg

DEFAULT_SEED = 0


def get_seed():
    """
    Get the default seed from the environment variable.
    If not set, we use our default seed.
    :return: int, a seed.
    """
    try:
        print("===========================================================================")
        print("                          SEED: {}  ".format(os.environ["MYSEED"]))
        print("===========================================================================")
        return int(os.environ["MYSEED"])
    except KeyError:
        print(
            "In Bash, you need to create an environment variable of the seed named `MYSEED`, then set its value to an "
            "integer.\n"
            "For example, to create an environment named `MYSEED` and set it to the value 0, in your Bash terminal, "
            "before running this script, type: `export MYSEED=0`.")
        print(" .... [NOT OK]")

        warnings.warn("WE ARE GOING TO USE OUR DEFAULT SEED: {}  .... [NOT OK]".format(DEFAULT_SEED))
        os.environ["MYSEED"] = str(DEFAULT_SEED)
        print("===========================================================================")
        print("                          DEFAULT SEED: {}  ".format(os.environ["MYSEED"]))
        print("===========================================================================")
        return DEFAULT_SEED


def set_seed(seed=None):
    """
    Set a seed to some modules for reproducibility.

    Note:

    While this attempts to ensure reproducibility, it does not offer an absolute guarantee. The results may be
    similar to some precision. Also, they may be different due to an amplification to extremely small differences.

    See:

    https://pytorch.org/docs/stable/notes/randomness.html
    https://stackoverflow.com/questions/50744565/how-to-handle-non-determinism-when-training-on-a-gpu

    :param seed: int, a seed. Default is None: use the default seed (0).
    :return:
    """
    if seed is None:
        seed = get_seed()
    else:
        os.environ["MYSEED"] = str(seed)
        announce_msg("SEED: {} ".format(os.environ["MYSEED"]))

    force_seed(seed)


def force_seed(seed, check_cudnn=False):
    """
    For seed to some modules.
    :param seed: int. The current seend.
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Deterministic mode can have a
    # performance impact, depending on your
    # model: https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    if check_cudnn:
        # If multigpu is on, deactivate cudnn since it has many randmon things
        # that we can not control.
        if (torch.cuda.device_count() > 1) and (
                os.environ["ALLOW_MULTIGPUS"] == 'True'):
            torch.backends.cudnn.enabled = False


def manual_seed(seed):
    r"""Sets the seed for generating random numbers. Returns a
    `torch._C.Generator` object.

    NOTE: WE REMOVE MANUAL RESEEDING ALL THE GPUS. At this point, it is not
    necessary; and there is not logic/reason
    to do it since we want only to reseed the current device.

    Args:
        seed (int): The desired seed.
    """
    return default_generator.manual_seed(int(seed))


def force_seed_thread(seed):
    """
    For seed to some modules.
    :param seed:
    :return:
    """
    manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True  # Deterministic mode can have a performance impact, depending on your
    # model: https://pytorch.org/docs/stable/notes/randomness.html#cudnn
