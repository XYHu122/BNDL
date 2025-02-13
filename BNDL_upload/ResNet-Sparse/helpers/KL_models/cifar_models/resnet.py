'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from robustness.tools.custom_modules import SequentialWithArgs, FakeReLU


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNet, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

        self.linear_k = nn.Sequential(
            nn.Linear(feat_scale * widths[3] * block.expansion, 1),
            nn.Softplus()
        )
        self.linear_kw = nn.Sequential(
            nn.Linear(num_classes, 1),
            nn.Softplus()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def reparameterize(self, lbd, kappa, force_sample=False):
        '''
            weibull reparameterization: z = lbd * (- ln(1 - u)) ^ (1/kappa), u ~ uniform(0,1)
            z: node-community affiliation.
            lbd: scale parameter, kappa: shape parameter
        '''

        def log_max(input, SMALL=1e-10):
            device = input.device
            input_ = torch.max(input, torch.tensor([SMALL]).to(device))
            return torch.log(input_)

        # print(force_sample)
        if self.training or force_sample:
            u = torch.rand_like(lbd)
            z = lbd * (- log_max(1 - u)).pow(1 / kappa)
            # print('force sample')
        else:
            z = lbd * torch.exp(torch.lgamma(1 + kappa.pow(-1)))
            # print('fixed')
        return z

    def forward(self, x, force_sample, top_k=-1, with_latent=False, fake_relu=False, no_relu=False):
        assert (not no_relu), \
            "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        z = out.view(out.size(0), -1)

        # non-negative z
        factor_z, factor_w = 1, 0
        gelu_z = F.relu(z - factor_z)
        # print(f"z sparsity: {(gelu_z > 1e-5).sum() / gelu_z.numel()}")
        # print(f'z_relu_min: {gelu_z.min()}  z_relu_max: {gelu_z.max()}')
        z_out = gelu_z - gelu_z.data + F.relu(z - factor_z).data
        # reparameterize z
        k = self.linear_k(z) # torch.ones_like(z_out, requires_grad=False, device=z_out.device)*10
        # print(f'k_min {k.min()}\t k_max {k.max()}')
        weibull_lambda = z_out / torch.exp(torch.lgamma(1 + 1 / k))
        pre_out = self.reparameterize(weibull_lambda, k, force_sample)  # N * H

        # reparameterize w
        gelu_z_w = F.relu(self.linear.weight.transpose(1, 0) - factor_w)
        # print(f'w_relu_min: {gelu_z_w.min()} w_relu_max: {gelu_z_w.max()}')
        z_out_w = gelu_z_w - gelu_z_w.data + F.relu(self.linear.weight.transpose(1, 0) - factor_w).data
        # z_out_w = F.softmax(self.linear.weight.transpose(1, 0), dim=-1)
        k_w = self.linear_kw(self.linear.weight.transpose(1, 0)) #* 2  # torch.ones_like(z_out_w, requires_grad=False, device=z_out.device)*100
        # print(f'kw_min {k_w.min()}\t kw_max {k_w.max()}')
        weibull_lambda_w = z_out_w / torch.exp(torch.lgamma(1 + 1 / k_w))
        pre_out_w = self.reparameterize(weibull_lambda_w, k_w, force_sample)  # H * C

        if top_k > 0:
            # print(f"top_k: {top_k}")
            topk_values, topk_indices = torch.topk(pre_out_w, top_k, dim=0)
            print(f'top_k values : {topk_values}')
            results = torch.zeros_like(pre_out_w)
            # print(f'topk indices size{topk_indices.shape}')
            results.scatter_(0, topk_indices, topk_values)
            # print(f'results shape {results.shape}')
            pre_out_w = results

        # pre_out = F.normalize(pre_out, dim=-1)
        final = torch.mm(pre_out, pre_out_w) + F.relu(self.linear.bias - factor_w)

        # final = self.linear(pre_out)
        if with_latent:
            return final, pre_out, weibull_lambda, 1 / k
        return final, weibull_lambda, 1 / k


class ResNet_original(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNet_original, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale * widths[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, force_sample, top_k, with_latent=False, fake_relu=False, no_relu=False):
        assert (not no_relu), \
            "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)

        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wm=2, **kwargs)  # TODO: wm=5_original


def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], wd=.75, **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def ResNet18_original(**kwargs):
    return ResNet_original(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18Wide_original(**kwargs):
    return ResNet_original(BasicBlock, [2, 2, 2, 2], wm=2, **kwargs)  # TODO: wm=5


resnet50 = ResNet50
resnet18 = ResNet18
resnet34 = ResNet34
resnet101 = ResNet101
resnet152 = ResNet152
resnet18wide = ResNet18Wide
resnet18_original = ResNet18_original
resnet18wide_original = ResNet18Wide_original


# resnet18thin = ResNet18Thin
def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

