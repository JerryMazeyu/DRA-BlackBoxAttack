import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from config.baseconfig import baseconf

normal_transforms = \
        torchvision.transforms.Compose([ 
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

imagenet10_dataset = torchvision.datasets.ImageFolder(baseconf.data_root, transform=normal_transforms)
validation_split = .2
dataset_size = len(imagenet10_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(2022)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_dataset_size, valid_dataset_size = len(train_indices), len(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

imagenet10_train_dataloader_bs32 = DataLoader(imagenet10_dataset, batch_size=32, num_workers=4, sampler=train_sampler)
imagenet10_valid_dataloader_bs32 = DataLoader(imagenet10_dataset, batch_size=32, num_workers=4, sampler=valid_sampler)

imagenet10_train_dataloader_bs16 = DataLoader(imagenet10_dataset, batch_size=16, num_workers=4, sampler=train_sampler)
imagenet10_valid_dataloader_bs16 = DataLoader(imagenet10_dataset, batch_size=16, num_workers=4, sampler=valid_sampler)

imagenet10_data_bs16 = {
    "dataset":          imagenet10_dataset,
    "dataloader":       {"train": imagenet10_train_dataloader_bs16, "valid": imagenet10_valid_dataloader_bs16},
    "size":             {"train": train_dataset_size, "valid": valid_dataset_size}
}

imagenet10_data_bs32 = {
    "dataset":          imagenet10_dataset,
    "dataloader":       {"train": imagenet10_train_dataloader_bs32, "valid": imagenet10_valid_dataloader_bs32},
    "size":             {"train": train_dataset_size, "valid": valid_dataset_size}
}


















# def save_images(images, img_list, output_dir='.'):
#     for i in range(images.shape[0]):
#         filename = img_list[i]
#         cur_images = (images[i, :, :, :])
#         inp = cur_images.transpose((1, 2, 0))
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         inp = std * inp + mean
#         inp = np.clip(inp, 0, 1)
#         import matplotlib.pyplot as plt
#         plt.imsave(f"{i}.png", inp)



# import cv2
# import torch
# def save_image_tensor2cv2(input_tensors: torch.Tensor, filenames):
#     assert (len(input_tensors.shape) == 4)
#     for i in range(input_tensors.shape[0]):
#         input_tensor = input_tensors[i,:,:,:]
#         filename = filenames[i]
#         # 复制一份
#         input_tensor = input_tensor.clone().detach()
#         # 到cpu
#         input_tensor = input_tensor.to(torch.device('cpu'))
#         # 反归一化
#         # unorm = UnNormalize()
#         # unorm(input_tensor)
#         # mean = torch.tensor([0.485, 0.456, 0.406])
#         # std = torch.tensor([0.229, 0.224, 0.225])
#         # input_tensor = std * input_tensor + mean
#         # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
#         input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
#         # RGB转BRG
#         input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(filename+'.png', input_tensor)


# dtld = imagenet10_data_bs16["dataloader"]["valid"]
# a = next(iter(dtld))
# # imgs = a[0][0]
# img_list = [f'{x}' for x in range(16)]

# from advertorch.attacks import PGDAttack
# import torch.nn as nn
# from torchvision.models import resnet18, resnet34

# resnet18_10cls = resnet18(pretrained=True)
# resnet18_10cls.fc = nn.Linear(resnet18_10cls.fc.in_features, 10)

# unorm = UnNormalize()
# att = PGDAttack(predict=resnet18_10cls, loss_fn=nn.CrossEntropyLoss(reduction="sum"))
# adv = att.perturb(unorm(a[0]))

# # save_image_tensor2cv2(adv, img_list)
# save_images(adv.detach().cpu().numpy(), img_list)





