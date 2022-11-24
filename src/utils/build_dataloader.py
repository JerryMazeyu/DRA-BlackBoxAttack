import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision import transforms
import re

normal_transforms = \
        torchvision.transforms.Compose([ 
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class FolderDataset(Dataset):
    def __init__(self, path, type="ori", transform=normal_transforms):
        super().__init__()
        self.p = path
        self.transform = transform
        self.imgs = list(filter(lambda x: x.endswith("png"), os.listdir(self.p)))
        self.imgs.sort()
        self.is_att = True if type != "ori" else False
        if self.is_att:  # TODO: low efficiency
            self.imgp = [osp.join(self.p, x) for x in self.imgs if x.find("ori") == -1]
        else:
            self.imgp = [osp.join(self.p, x) for x in self.imgs if x.find("ori") != -1]
        
    def __getitem__(self, index):
        x = Image.open(self.imgp[index])
        x = self.transform(x)
        y = int(re.findall(r"cls\d+", self.imgp[index])[0].split('cls')[-1])
        return x, y
    
    def __len__(self):
        return len(self.imgp)


    
    


        