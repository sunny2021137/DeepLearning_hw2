import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_scores = nn.Sequential(nn.Linear(c_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, nof_kernels))

    def forward(self, x, temperature=1):
        out = self.global_pooling(x)
        scores = self.to_scores(out)
        # scores: (batch_size , nof_kernels)
        return F.softmax(scores / temperature, dim=-1)


class DynamicConv2d(nn.Module):
    def __init__(self, nof_kernels, reduce, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.groups = groups
        self.conv_args = {'stride': stride, 'padding': padding, 'dilation': dilation}
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(in_channels, max(1, in_channels // reduce), nof_kernels)
        self.kernel_size = _pair(kernel_size)
        # kernels_weights: (nof_kernels, out_channels, in_channels // groups, kernel_size, kernel_size)
        # why groups? because we want to have the same number of kernels for each group. what is the group? it is the   # number of kernels that are applied to each input channel. So, if we have 2 groups, we will have 2 kernels for each input channel.
        self.kernels_weights = nn.Parameter(torch.Tensor(
            nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.kernels_bias = nn.Parameter(torch.Tensor(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter('kernels_bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x, temperature=1):
        # x: (batch_size , in_channels , H , W)
        batch_size = x.shape[0]
        # alphas: (batch_size , nof_kernels)
        alphas = self.attention(x, temperature)
        # agg_weights: (batch_size , out_channels , in_channels // groups, kernel_size, kernel_size)
        # sum over the kernels with the attention weights
        agg_weights = torch.sum(
            torch.mul(self.kernels_weights.unsqueeze(0), alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1)
        # Group the weights for each batch to conv2 all at once
        
        # agg_weights: (batch_size * out_channels , in_channels // groups, kernel_size, kernel_size)
        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])  # filters of shape ( out_channels , in_channels groups , 𝑘 𝐻 , 𝑘 𝑊 ) (out_channels, groups in_channels ​ ,kH,kW) \
        

        if self.kernels_bias is not None:
            agg_bias = torch.sum(torch.mul(self.kernels_bias.unsqueeze(0), alphas.view(batch_size, -1, 1)), dim=1)
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        
        # why view(1, -1, *x.shape[-2:])? because we want to group the input channels. So, if we have 2 groups, we will have 2 kernels for each input channel.
        x_grouped = x.view(1, -1, *x.shape[-2:])  # (1 , batch_size*out_c , H , W)
        #   out: (1 , batch_size*out_C , H' , W')
        out = F.conv2d(x_grouped, agg_weights, agg_bias, groups=self.groups * batch_size,
                       **self.conv_args)  
        # out: (batch_size , out_channels , H' , W')
        out = out.view(batch_size, -1, *out.shape[-2:]) 

        return out

class SimpleDynamicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDynamicCNN, self).__init__()
        # in_channels, out_channels, kernel_size, num_kernels=4, stride=1, padding=0, dilation=1, groups=1, bias=True
        #  nof_kernels, reduce, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        
        self.layer1 = DynamicConv2d(nof_kernels=5, reduce=4, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.layer2 = DynamicConv2d(5, 4, 64, 128, 3, 1, 1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.layer1(x))) # x shape: (batch_size, 64, 64, 64)
        x = self.pool(F.relu(self.layer2(x))) # x shape: (batch_size, 128, 32, 32)

        x = x.view(x.size(0), -1) 
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

if __name__ == '__main__':
    # Test the network with a random input
    model = SimpleDynamicCNN(num_classes=50)
    input_tensor = torch.randn(4, 3, 128, 128)
    output = model(input_tensor)
    print(output.shape)
    print(output)

