import os
from data import *
from src.loss import *
from src.models import *
from src.optimizer import *
from torch import nn
from .baseconfig import baseconf


class TrainConfig():
    def __init__(self):
        """
        Pre-training config
        """
        self.dir            = os.path.join(baseconf.root, "exp")                                            # root directory
        self.model          = resnet18_10cls                                                                # target model
        self.data_info      = imagenet10_data_bs16                                                          # data and meta info (data/dataloader.py)
        self.train_loss     = CrossEntropyLoss()                                                            # loss function
        self.optimizer_cls  = SGD                                                                           # optimizer
        self.lr             = 0.001                                                                         # learning rate
        self.optimizer      = self.optimizer_cls(params = self.model.parameters(), lr = self.lr)            # optimizer instance
        self.train_epochs   = 10                                                                            # number of iteration
        self.scheduler      = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)                   # lr scheduler

class FinetuneConfig():
    def __init__(self):
        """
        Finetuning config
        """
        self.checkpoint     = os.path.join(baseconf.root, "exp/resnet18/acc_0.99.pth")                     # finetune baseline (result of pretraining)
        self.dir            = os.path.join(baseconf.root, "exp")                                        
        self.model          = resnet18_10cls
        self.data_info      = imagenet10_data_bs16
        self.finetune_loss  = MyLoss(resnet18_10cls)                                                        # finetune loss (DCGLoss)
        self.optimizer_cls  = SGD
        self.lr             = 0.001
        self.optimizer      = self.optimizer_cls(params = self.model.parameters(), lr = self.lr)
        self.finetune_epochs= 5
        self.scheduler      = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

class AttackConfig():
    def __init__(self):
        """
        Attacking config
        """
        self.checkpoint     = os.path.join(baseconf.root, "exp/resnet18_finetune/epoch4_acc0.962.pth")                     # threaten model
        self.dir            = os.path.join(baseconf.root, "exp")
        self.model          = resnet18_10cls
        self.data_info      = imagenet10_data_bs16
        self.epsilon        = 16                                                                            # max perturbation
        self.step_size      = 2                                                                             # perturbation of every step
        self.num_steps      = 20                                                                            # iteration

class Evalconfig():
    def __init__(self):
        """
        Evaluating config
        """
        self.model          = densenet121_10cls                                                              # the model to evaluation performance (different from pretraining model)
        self.img_path       = os.path.join(baseconf.root, "exp", "resnet18_attack")                          # evaluation path 
        self.checkpoint     = os.path.join(baseconf.root, "exp", "densenet121/acc_0.991.pth")                # evaluation model weights




class WholeConfig():
    def __init__(self):
        self.dir                = os.path.join(baseconf.root, "exp")
        self.model              = resnet18_10cls
        self.data_info          = imagenet10_data_bs16
        self.train_epochs       = 3
        self.finetune_epochs    = 10
        self.train_loss         = CrossEntropyLoss()
        self.finetune_loss      = MyLoss(self.model)
        self.lr                 = 0.001
        self.optimizer_cls      = SGD
        self.show_perturb       = True                                                                        # if show perturb
        self.epsilon            = 16
        self.step_size          = 2
        self.num_steps          = 10
        self.optimizer          = self.optimizer_cls(params = self.model.parameters(), lr = self.lr)         
        self.scheduler          = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.eval_model         = densenet121_10cls
        self.eval_model_path    = os.path.join(baseconf.root, "exp", "densenet121", "acc_0.991.pth")






trainconfig                 = TrainConfig()
finetuneconifg              = FinetuneConfig()
attackconfig                = AttackConfig()
evalconfig                  = Evalconfig()
wholeconfig                 = WholeConfig()