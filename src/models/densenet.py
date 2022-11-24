from torch import nn
from torchvision.models import densenet121
from .normalization import norm

densenet121_10cls = densenet121(pretrained=True)
densenet121_10cls.classifier = nn.Linear(densenet121_10cls.classifier.in_features, 10)
densenet121_10cls = nn.Sequential(norm, densenet121_10cls)
