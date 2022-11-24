import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import FolderDataset, rec_cls10, auc_cls10, acc_top1, acc_top5, lprint
from functools import partial



class EvalRunner():
    def __init__(self, config):
        self.config         = config
        self.model          = self.config.model
        self.img_path       = self.config.img_path
        self.checkpoint     = self.config.checkpoint

    def show(self):
        print("Evaluation...")
        # print("Model:\n", self.model)
        print("Image path:\t", self.img_path)
        print("Checkpoint:\t", self.checkpoint)
    
    def run(self):

        self.show()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(self.checkpoint))
        self.model.eval()
        self.model.to(device)
        self.ori_dataset    = FolderDataset(self.img_path)
        self.att_dataset    = FolderDataset(self.img_path, type="att")
        self.ori_dataloader = DataLoader(self.ori_dataset, shuffle=False, batch_size=16)
        self.att_dataloader = DataLoader(self.att_dataset, shuffle=False, batch_size=16)
        logits = []
        target = []
        for x, y in tqdm(self.ori_dataloader, desc="original"):
            x = x.to(device)
            y = y.to(device)
            logits.append(self.model(x).detach().cpu())
            target.append(y.cpu())
        logits  = torch.vstack(logits)
        target  = torch.hstack(target)
        acc1    = acc_top1(logits, target)
        acc5    = acc_top5(logits, target)
        rec     = rec_cls10(logits, target)
        auc     = auc_cls10(logits, target)
        print(f"Ori Metrics: \t Accuracy top1: {acc1} \t Accuracy top5: {acc5} \t Recall: {rec} \t Auc: {auc}")

        logits = []
        target = []
        for x, y in tqdm(self.att_dataloader, desc="attacked"):
            x = x.to(device)
            y = y.to(device)
            logits.append(self.model(x).detach().cpu())
            target.append(y.cpu())
        logits  = torch.vstack(logits)
        target  = torch.hstack(target)
        acc1    = acc_top1(logits, target)
        acc5    = acc_top5(logits, target)
        rec     = rec_cls10(logits, target)
        auc     = auc_cls10(logits, target)
        print(f"Att Metrics: \t Accuracy top1: {acc1} \t Accuracy top5: {acc5} \t Recall: {rec} \t Auc: {auc}")

        
        




