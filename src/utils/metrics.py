from torchmetrics import Accuracy, Recall, AUROC, F1Score

acc_top1 = Accuracy(top_k=1)
acc_top5 = Accuracy(top_k=5)

auc_cls10 = AUROC(num_classes=10)

rec_cls10 = Recall(num_classes=10)

