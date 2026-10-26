import torch
import torch.nn as nn


class DepthFeatureExtractor(nn.Module):
    def __init__(self, out_features=2048):
        super().__init__()
        # Convs leves para extrair padrões espaciais do heatmap/depth
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # [B,64,H/2,W/2]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # [B,128,H/4,W/4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # [B,256,H/8,W/8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # [B,512,H/16,W/16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        # Global average pooling + MLP para gerar vetor 2048
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B,512,1,1] -> [B,512]
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, out_features)
        )

    def forward(self, x):
        """
        x: [B,1,H,W] heatmap/depth
        return: [B, out_features] vetor de features
        """
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
