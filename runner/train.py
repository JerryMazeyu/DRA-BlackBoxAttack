import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import copy
from src.utils import *
from functools import partial

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, data_info, criterion, optimizer, scheduler, dir, num_epochs=25, exp=""):
    print = partial(lprint, dir=os.path.join(dir, exp))
    model = model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        dataloaders = data_info["dataloader"]
        dataset_sizes = data_info["size"]

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_wts = best_model_wts
                torch.save(model.state_dict(), f"{dir}/{exp}/acc_{np.round(best_acc.cpu(), 3)}.pth")
                print(f"Save at {dir}/{exp}/acc_{np.round(best_acc.cpu(), 3)}.")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best valid Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class TrainRunner():
    def __init__(self, config, exp=""):
        self.exp            = exp
        self.config         = config
        self.model          = self.config.model
        self.data_info      = self.config.data_info
        self.criterion      = self.config.train_loss
        self.optimizer      = self.config.optimizer
        self.scheduler      = self.config.scheduler 
        self.num_epochs     = self.config.train_epochs
        self.dir            = self.config.dir
    
    def show(self):
        print = partial(lprint, dir=os.path.join(self.dir, self.exp))
        print("Train...")
        print("Model:\n", self.model)
        print("Data info:\t", self.data_info)
        print("Criterion:\t", self.criterion)
        print("Optimizer:\t", self.optimizer)
        print("Num epochs:\t", self.num_epochs)

    def run(self):
        safe_mkdir(f"{self.dir}/{self.exp}")
        self.show()
        train_model(self.model, self.data_info, self.criterion, self.optimizer, self.scheduler, self.dir, self.num_epochs, self.exp)
