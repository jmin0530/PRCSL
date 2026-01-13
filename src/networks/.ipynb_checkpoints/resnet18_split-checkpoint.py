# +
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self, block, input_channel, num_layers=2):
        super(ResNet18_client_side, self).__init__()
        self.input_planes = 64
        self.conv1 = nn.Sequential (
                nn.Conv2d(input_channel, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
            )
        self.layer1 = self._layer(block, 64, num_layers, stride = 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _layer(self, block, planes, num_layers, stride = 1):
        dim_change = None
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = 1, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


# -

# Model at server side
class Basicblock_resnet18(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Basicblock_resnet18, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride = stride, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output

# Model at server side
class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_layers, classes=10):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64
        self.layer2 = self._layer(block, 128, num_layers[0], stride = 2)
        self.layer3 = self._layer(block, 256, num_layers[1], stride = 2)
        self.layer4 = self._layer(block, 512, num_layers[2], stride = 2)
        self.averagePool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, classes)
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride, bias = False),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.averagePool(x)
        x = x.view(x.size(0), -1)
        y_hat = self.fc(x)

        return y_hat
