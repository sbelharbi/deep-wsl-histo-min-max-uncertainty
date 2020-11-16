from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler


from deepmil import models, criteria
from deepmil import lr_scheduler as my_lr_scheduler
from tools import Dict2Obj, count_nb_params



def instantiate_train_loss(args):
    """
    Instantiate the evaluation (test phase) loss.

    :param args: object. Contains the configuration of the exp that has been
     read from the yaml file.
    :return: eval_loss: instance of deepmil.criteria.TotalLossEval()
    """
    return criteria.TrainLoss(use_reg=args.use_reg,
                              reg_loss=args.reg_loss,
                              use_size_const=args.use_size_const,
                              init_t=args.init_t,
                              max_t=args.max_t,
                              mulcoef=args.mulcoef,
                              normalize_sz=args.normalize_sz,
                              epsilon=args.epsilon,
                              lambda_neg=args.lambda_neg
                              )


def instantiate_models(args):
    """Instantiate the necessary models.
    Input:
        args: object. Contains the configuration of the exp that has been read
        from the yaml file.

    Output:
        segmentor: instance of module from deepmil.representation; Embeds the
         instance.
        classifier: instance of module from deepmil.decision_pooling; pools
        the score of each class.
    """
    p = Dict2Obj(args.model)

    model = models.__dict__[p.model_name](pretrained=p.pretrained,
                                          sigma=p.sigma,
                                          w=p.w,
                                          num_classes=p.num_classes,
                                          scale=p.scale_in_cl,
                                          modalities=p.modalities,
                                          kmax=p.kmax,
                                          kmin=p.kmin,
                                          alpha=p.alpha,
                                          dropout=p.dropout
                                          )

    print("Mi-max entropy model `{}` was successfully instantiated. "
          "Nbr.params: {} .... [OK]".format(
        model.__class__.__name__, count_nb_params(model)))
    return model


def instantiate_optimizer(args, model):
    """Instantiate an optimizer.
    Input:
        args: object. Contains the configuration of the exp that has been
        read from the yaml file.
        mode: a pytorch model with parameters.

    Output:
        optimizer: a pytorch optimizer.
        lrate_scheduler: a pytorch learning rate scheduler (or None).
    """
    if args.optimizer["name"] == "sgd":
        optimizer = SGD(model.parameters(), lr=args.optimizer["lr"],
                        momentum=args.optimizer["momentum"],
                        dampening=args.optimizer["dampening"],
                        weight_decay=args.optimizer["weight_decay"],
                        nesterov=args.optimizer["nesterov"])
    elif args.optimizer["name"] == "adam":
        optimizer = Adam(params=model.parameters(), lr=args.optimizer["lr"],
                         betas=args.optimizer["betas"],
                         eps=args.optimizer["eps"],
                         weight_decay=args.optimizer["weight_decay"],
                         amsgrad=args.optimizer["amsgrad"])
    else:
        raise ValueError("Unsupported optimizer `{}` .... "
                         "[NOT OK]".format(args.optimizer["name"]))

    print("Optimizer `{}` was successfully instantiated .... "
          "[OK]".format(
        [key + ":" + str(args.optimizer[key]) for key in args.optimizer.keys()])
    )

    if args.optimizer["use_lr_scheduler"]:
        if args.optimizer["lr_scheduler_name"] == "step":
            optimizer_hp = args.optimizer
            lrate_scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=optimizer_hp["step_size"],
                gamma=optimizer_hp["gamma"],
                last_epoch=optimizer_hp["last_epoch"])
            print("Learning scheduler `{}` was successfully "
                  "instantiated .... [OK]".format(
                [key + ":" + str(optimizer_hp[key]) for key in optimizer_hp.keys()]
            ))
        elif args.optimizer["lr_scheduler_name"] == "mystep":
            optimizer_hp = args.optimizer
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer,
                step_size=optimizer_hp["step_size"],
                gamma=optimizer_hp["gamma"],
                last_epoch=optimizer_hp["last_epoch"],
                min_lr=optimizer_hp["min_lr"])
            print("Learning scheduler `{}` was successfully instantiated "
                  ".... [OK]".format(
                [key + ":" + str(optimizer_hp[key]) for key in optimizer_hp.keys()]
            ))
        elif args.optimizer["lr_scheduler_name"] == "multistep":
            optimizer_hp = args.optimizer
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=optimizer_hp["milestones"],
                gamma=optimizer_hp["gamma"],
                last_epoch=optimizer_hp["last_epoch"])
            print("Learning scheduler `{}` was successfully instantiated "
                  ".... [OK]".format(
                [key + ":" + str(optimizer_hp[key]) for key in optimizer_hp.keys()]
            ))
        else:
            raise ValueError("Unsupported learning rate scheduler `{}`"
                             " .... [NOT OK]".format(
                args.optimizer["lr_scheduler_name"]))
    else:
        lrate_scheduler = None

    return optimizer, lrate_scheduler
