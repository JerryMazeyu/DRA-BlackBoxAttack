from runner import *
from config import *


# trainrunner = TrainRunner(trainconfig, "resnet18")
# trainrunner.run()

# finetunerunner = FinetuneRunner(finetuneconifg, "resnet18_finetune")
# finetunerunner.run()

# attackrunner = AttackRunner(attackconfig, "resnet18_attack")
# attackrunner.run()

# evalrunner = EvalRunner(evalconfig) 
# evalrunner.run()

whole = WholeRunner(wholeconfig, exp="")
whole.run()