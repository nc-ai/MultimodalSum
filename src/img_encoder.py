import torchvision
import torch.nn as nn

class Resnet(nn.Module):
    def __init__ (self, embedding_dim, model='resnet101', stage=3):
        super(Resnet, self).__init__()
        self.stage = stage
        if model == 'resnet50':
            self.resnet = torchvision.models.resnet50(pretrained=True)
        elif model == 'resnet101':
            self.resnet = torchvision.models.resnet101(pretrained=True)
        elif model == 'resnet50_32':
            self.resnet = torchvision.models.resnext50_32x4d(pretrained=True)
        elif model == 'resnet101_32':
            self.resnet = torchvision.models.resnext101_32x8d(pretrained=True)
        elif model == 'resnet50_wide':
            self.resnet = torchvision.models.wide_resnet50_2(pretrained=True)
        elif model == 'resnet101_wide':
            self.resnet = torchvision.models.wide_resnet50_2(pretrained=True)

        self.stage1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool,
                                    self.resnet.layer1)
        self.stage2 = nn.Sequential(self.resnet.layer2)
        self.stage3 = nn.Sequential(self.resnet.layer3)
        if stage == 3:
            self.linear = nn.Linear(1024, embedding_dim, bias=False)
        elif stage == 4:
            self.stage4 = nn.Sequential(self.resnet.layer4)
            self.linear = nn.Linear(2048, embedding_dim, bias=False)

    def forward(self, x):
        x = self.stage1(x) # 1/4
        x = self.stage2(x).detach() # 1/8

        x = self.stage3(x) # 1/16
        if self.stage == 4:
            x = self.stage4(x) # 1/32

        x = x.flatten(start_dim=-2).transpose(1,2) # [batch_size, n_areas, out_dim]
        x = self.linear(x) # [batch_size, n_areas, embedding_dim]
        return x