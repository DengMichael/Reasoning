"""
Dreamnist classifier

Use like:
from dreamnist_classifier_res import WideResNet
classifier = WideResNet(3, 1, 5)
classifier.load_state_dict(torch.load('wideres-9919-(315).pth'))
get_batch_classification_accuracy(classifier, imgs)
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

import torch.nn.functional as F

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

def get_batch_classification_accuracy(classifier, imgs):
    # TODO: rewrite to be more parallel. Just test code
    num_correct = 0
    for img in imgs:
        img = img.unsqueeze(0)
        numbers = torch.cat((img[:,:, :11, :11], img[:,:, :11, -11:], img[:,:, 10:21, 10:21], img[:,:, 10:21, -11:]), dim=0)
        classes = classifier(numbers).max(1)[1].cpu().numpy().tolist()

        if img[:,:, 10:21, -11:].mean() < 0.01:
            equation_str = f"{classes[0]}+{classes[1]}={classes[2]}"
            correct = classes[0] + classes[1] == classes[2]
        else:
            equation_str = f"{classes[0]}+{classes[1]}={classes[2]}{classes[3]}"
            correct = classes[0] + classes[1] == int(str(classes[2]) + str(classes[3]))
        
        if correct:
            num_correct += 1
    
    return num_correct

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

class ModelCNN(nn.Module):

    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 256, (3, 3))
        self.conv5 = nn.Conv2d(256, 512, (3, 3))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(self.conv1(input_))
        #print(h1.shape)
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.conv2(h1))
        #print(h2.shape)
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = F.relu(self.conv3(h2))
        #print(h3.shape)
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = F.relu(self.conv4(h3))
        #print(h4.shape)
        h4 = F.dropout(h4, p=0.5, training=self.training)
        h5 = F.relu(self.conv5(h4))
        #print(h5.shape)
        h5 = F.dropout(h5, p=0.5, training=self.training)
        h5 = h5.view(-1, 512)
        h6 = F.relu(self.fc1(h5))
        h6 = F.dropout(h6, p=0.5, training=self.training)
        h7 = self.fc2(h6)
        return h7

class BasicBlock(nn.Module):
    def __init__(self, inf, outf, stride, drop):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inf)
        self.conv1 = nn.Conv2d(inf, outf, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.drop = nn.Dropout(drop, inplace=True)
        self.bn2 = nn.BatchNorm2d(outf)
        self.conv2 = nn.Conv2d(outf, outf, kernel_size=3, padding=1,
                               stride=1, bias=False)
        if inf == outf:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inf), nn.ReLU(inplace=True),
                    nn.Conv2d(inf, outf, 3, padding=1, stride=stride, bias=False))

    def forward(self, x):
        x2 = self.conv1(F.relu(self.bn1(x)))
        x2 = self.drop(x2)
        x2 = self.conv2(F.relu(self.bn2(x2)))
        r = self.shortcut(x)
        return x2.add_(r)

class WideResNet(nn.Module):
    def __init__(self, n_grps, N, k=1, drop=0.3, first_width=16):
        super().__init__()
        layers = [nn.Conv2d(1, first_width, kernel_size=3, padding=1, bias=False)]
        # Double feature depth at each group, after the first
        widths = [first_width]
        for grp in range(n_grps):
            widths.append(first_width*(2**grp)*k)
        for grp in range(n_grps):
            layers += self._make_group(N, widths[grp], widths[grp+1],
                                       (1 if grp == 0 else 2), drop)
        layers += [nn.BatchNorm2d(widths[-1]), nn.ReLU(inplace=True),
                   nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                   nn.Linear(widths[-1], 10)]
        self.features = nn.Sequential(*layers)

    def _make_group(self, N, inf, outf, stride, drop):
        group = list()
        for i in range(N):
            blk = BasicBlock(inf=(inf if i == 0 else outf), outf=outf,
                             stride=(stride if i == 0 else 1), drop=drop)
            group.append(blk)
        return group

    def forward(self, x):
        return self.features(x)




def mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

