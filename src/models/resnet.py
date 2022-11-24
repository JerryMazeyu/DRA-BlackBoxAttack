from torch import nn
from torchvision.models import resnet18, resnet34
from .normalization import norm

resnet18_10cls = resnet18(pretrained=True)
resnet18_10cls.fc = nn.Linear(resnet18_10cls.fc.in_features, 10)
resnet18_10cls = nn.Sequential(norm, resnet18_10cls)

resnet34_10cls = resnet34(pretrained=True)
resnet34_10cls.fc = nn.Linear(resnet34_10cls.fc.in_features, 10)
resnet34_10cls = nn.Sequential(norm, resnet34_10cls)
