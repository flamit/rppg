import torch.nn as nn
import torch
import torch.nn.functional as F


class RGAS(nn.Module):
    def __init__(self, inplanes, h, w, s1, affinity_out, s2):
        super(RGAS, self).__init__()
        self.outplanes1 = int(inplanes // s1)
        self.conv1 = nn.Conv2d(inplanes, self.outplanes1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.outplanes1)
        self.conv2 = nn.Conv2d(inplanes, self.outplanes1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(self.outplanes1)
        self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(inplanes, self.outplanes1, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.outplanes1)
        self.conv4 = nn.Conv2d(h * w, affinity_out, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(affinity_out)

        current_feats = 1 + affinity_out * 2
        self.outplanes2 = int(current_feats // s2)
        self.conv5 = nn.Conv2d(current_feats, self.outplanes2, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(self.outplanes2)
        self.conv6 = nn.Conv2d(self.outplanes2, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # Embed x using the two functions:
        theta = self.relu(self.bn1(self.conv1(x))).view(-1, self.outplanes1, h * w)
        phi = self.relu(self.bn2(self.conv2(x))).view(-1, self.outplanes1, h * w)
        affinity = torch.zeros(size=[b, h * w, h * w])
        # Calculate affinity as convolution over each batch for faster computation:
        for batch in range(b):
            kernel = phi[batch].permute(1, 0).view(h * w, self.outplanes1, 1, 1)
            r = F.conv2d(theta[batch].view(1, self.outplanes1, h, w), kernel).view(h * w, h * w)
            affinity[batch] = r
        # Take out affinity row wise and column wise and reshape to image dimensions.
        affinity_a = affinity.view(b, -1, h, w)
        affinity_b = affinity.permute(0, 2, 1).view(b, -1, h, w)

        # Embed x and affinity so that they are in the same feature space.
        x_embed = self.relu(self.bn3(self.conv3(x))).mean(dim=1, keepdim=True)  # Mean pool over channels
        affinitya_embed = self.relu(self.conv4(affinity_a))
        affinityb_embed = self.relu(self.conv4(affinity_b))
        # Calculate the y tensor as the concatenation of affinity and x
        y = torch.cat([x_embed, affinitya_embed, affinityb_embed], dim=1)
        # Calculate attention
        a = self.sig(self.conv6(self.relu(self.bn5(self.conv5(y)))))

        return x * a


# TODO Code cleanup and verification, add to resnet.
class RGAC(nn.Module):
    def __init__(self, inplanes, h, w, s):
        super(RGAC, self).__init__()
        self.outplanes = int(h * w // s)
        self.conv1 = nn.Conv2d(h * w, self.outplanes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.conv2 = nn.Conv2d(h * w, self.outplanes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(self.outplanes)
        self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.conv4 = nn.Conv2d(1 + 2 * inplanes, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).view(b, h * w, -1, 1)
        theta = self.relu(self.bn1(self.conv1(x_flat)))  # B x H*W x C / s
        phi = self.relu(self.bn2(self.conv2(x_flat)))  # B x H*W x C / s

        affinity = torch.zeros(size=[b, c, c])
        for batch in range(b):
            # kernel: outplanes, inplanes, 1, 1
            kernel = phi[batch].permute(1, 0, 2).contiguous().view(c, -1, 1, 1)
            r = F.conv2d(theta[batch].view(1, -1, c, 1), kernel).view(c, c)
            affinity[batch] = r
        # Take out affinity row wise and column wise and reshape to image dimensions.
        affinity_a = affinity.view(b, -1, c, c).permute(0, 2, 1, 3)
        affinity_b = affinity.permute(0, 2, 1).view(b, -1, c, c).permute(0, 2, 1, 3)
        x_embed = self.relu(self.bn3(self.conv3(x))).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).permute(0, 2, 3, 1)  # Mean pool over space
        y = torch.cat([x_embed, affinity_a, affinity_b], dim=1)
        a = self.sig(self.conv4(y)).permute(0, 3, 1, 2)

        return x * a


class RGA(nn.Module):
    def __init__(self, inplanes, h, w, s1, affinity_out, s2):
        super(RGA, self).__init__()
        self.rgas = RGAS(inplanes, h, w, s1, affinity_out, s2)
        self.rgac = RGAC(inplanes, h, w, s1)

    def forward(self, x):
        x = self.rgas(x)
        x = self.rgac(x)

        return x