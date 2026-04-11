import torch
import torch.nn as nn
import torchvision.models as models

class RSN(nn.Module):
    def __init__(self):
        super(RSN, self).__init__()
        # 使用ResNet18作为backbone，不加载预训练权重
        self.backbone = models.resnet18(pretrained=False)
        
        # 修改ResNet18的输入通道数，从3通道改为16通道
        self.backbone.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除原来的全连接层
        self.backbone.fc = nn.Identity()
        
        # 添加自定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18最后输出512维特征
            nn.ELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape

        # 通过ResNet backbone
        feats = self.backbone(x)
        # 分类
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out