import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class RBFLinear2D(nn.Module):
    def __init__(self, in_features_x, in_features_y, out_features_x, out_features_y, grid_min=-2., grid_max=2., num_grids=(8, 8), spline_weight_init_scale=0.05):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.out_features_x = out_features_x
        self.out_features_y = out_features_y
        self.in_features_x = in_features_x
        self.in_features_y = in_features_y

        self.grid_x = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids[0]), requires_grad=False)
        self.grid_y = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids[1]), requires_grad=False)

        self.spline_weight_x = nn.Parameter(torch.randn(in_features_x * num_grids[0], out_features_x) * spline_weight_init_scale)
        self.spline_weight_y = nn.Parameter(torch.randn(in_features_y * num_grids[1], out_features_y) * spline_weight_init_scale)

    def forward(self, x):
        batch_size, height, width = x.shape
        x = x.unsqueeze(-1)

        basis_x = torch.exp(-((x - self.grid_x) / ((self.grid_max - self.grid_min) / (self.num_grids[0] - 1))) ** 2)
        basis_y = torch.exp(-((x - self.grid_y) / ((self.grid_max - self.grid_min) / (self.num_grids[1] - 1))) ** 2)
        b_x = basis_x.permute(0, 2, 1, 3).reshape(batch_size, width, height * self.num_grids[0])
        b_x = b_x.matmul(self.spline_weight_x).softmax(axis=-1)
        b_y = basis_y.permute(0, 1, 2, 3).reshape(batch_size, height, width * self.num_grids[1])
        b_y = b_y.matmul(self.spline_weight_y).softmax(axis=-1)
        return torch.einsum('bji,blk->bik', b_x, b_y)

class RBFKANLayer2D(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, output_dim_x, output_dim_y, grid_min=-2., grid_max=2., num_grids=(8, 8), use_base_update=True, base_activation=nn.SiLU(), spline_weight_init_scale=0.05):
        super().__init__()
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.output_dim_x = output_dim_x
        self.output_dim_y = output_dim_y
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.rbf_linear = RBFLinear2D(input_dim_x, input_dim_y, output_dim_x, output_dim_y, grid_min, grid_max, num_grids, spline_weight_init_scale)
        self.base_linear = nn.Linear(input_dim_x * input_dim_y, output_dim_x * output_dim_y) if use_base_update else None

    def forward(self, x):
        ret = self.rbf_linear(x)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x.view(x.shape[0], -1)))
            ret = ret + base.view(x.shape[0], self.output_dim_x, self.output_dim_y)
        return ret

class RBFKAN2D(nn.Module):
    def __init__(self, layers_hidden_x, layers_hidden_y, grid_min=-2., grid_max=2., num_grids=(8, 8), use_base_update=True, base_activation=nn.SiLU(), spline_weight_init_scale=0.05):
        super().__init__()
        self.layers = nn.ModuleList([RBFKANLayer2D(in_dim_x, in_dim_y, out_dim_x, out_dim_y, grid_min, grid_max, num_grids, use_base_update, base_activation, spline_weight_init_scale) for in_dim_x, in_dim_y, out_dim_x, out_dim_y in zip(layers_hidden_x[:-1], layers_hidden_y[:-1], layers_hidden_x[1:], layers_hidden_y[1:])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x