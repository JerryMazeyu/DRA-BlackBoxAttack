from src.utils import *
from .train import TrainRunner
from .finetune import FinetuneRunner
from .drapgd import AttackRunner
from .eval import EvalRunner
from functools import partial


class WholeRunner():
    def __init__(self, config, exp=""):
        if exp == "":
            self.exp            = input("输入唯一实验名称: ")
        else:
            self.exp            = exp
        self.config             = config
        self.config.dir         = os.path.join(self.config.dir, self.exp)
        self.dir                = self.config.dir
        safe_mkdir(self.dir)
        self.model              = interactive_get_value(config, "model")
        self.data_info          = interactive_get_value(config, "data_info")
        self.train_epochs       = interactive_get_value(config, "train_epochs")
        self.finetune_epochs    = interactive_get_value(config, "finetune_epochs")
        self.train_loss         = interactive_get_value(config, "train_loss")
        self.finetune_loss      = interactive_get_value(config, "finetune_loss")
        self.lr                 = interactive_get_value(config, "lr")
        self.optimizer_cls      = interactive_get_value(config, "optimizer_cls")
        self.show_perturb       = interactive_get_value(config, "show_perturb")
        self.epsilon            = interactive_get_value(config, "epsilon")
        self.step_size          = interactive_get_value(config, "step_size")
        self.num_steps          = interactive_get_value(config, "num_steps")
        self.optimizer          = self.config.optimizer
        self.scheduler          = self.config.scheduler
        self.eval_model         = self.config.eval_model
        self.eval_model_path    = self.config.eval_model_path
        self.print_freq         = 10
    
    def run(self):
        # print("[1/4]\t Training... ")
        # print("="*10)
        # print("")        
        # runner = TrainRunner(self.config, "Train")
        # runner.run()
        # print("")
        # print("[2/4]\t Finetuning... ")
        # print("="*10)
        # print("")
        # setattr(self.config, "checkpoint", find_best_ckpt(os.path.join(self.dir, "Train")))
        # runner = FinetuneRunner(self.config, "Finetune")
        # runner.run()
        # print("")
        # print("[3/4]\t Attacking... ")
        # print("="*10)
        # print("")
        # print("\t DRA ")
        # setattr(self.config, "checkpoint", find_best_ckpt(os.path.join(self.dir, "Finetune")))
        # runner = AttackRunner(self.config, "Attack")
        # runner.run()
        # print("\t PGD ")
        # setattr(self.config, "checkpoint", find_best_ckpt(os.path.join(self.dir, "Train")))
        # runner = AttackRunner(self.config, "PGDAttack")
        # runner.run()

        print("")
        print("[4/4]\t Evaluating... ")
        print("="*10)
        print("")
        print("\t DRA ")
        setattr(self.config, "model", self.config.eval_model)
        setattr(self.config, "checkpoint", self.eval_model_path)
        setattr(self.config, "img_path", os.path.join(self.dir, "Attack"))
        runner = EvalRunner(self.config)
        runner.run()
        print("\t PGD ")
        setattr(self.config, "img_path", os.path.join(self.dir, "PGDAttack"))
        runner = EvalRunner(self.config)
        runner.run()





