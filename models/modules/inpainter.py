import torch
import torch.nn as nn
import torch.nn.functional as F

class Inpainter(nn.Module):
    def __init__(self, sigma, kernel_size, reps, scale_factor=1):
        super(Inpainter, self).__init__()
        self.reps = reps
        self.padding = kernel_size//2
        
        self.downsample = nn.AvgPool2d(scale_factor) if scale_factor > 1 else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest') if scale_factor > 1 else nn.Identity()
        squared_dists = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)**2
        gaussian_kernel = torch.exp(-0.5 * squared_dists / sigma**2)
        gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).view(1, 1, kernel_size).repeat(4,1,1)
        self.register_buffer('gaussian_kernel', gaussian_kernel)

    def gaussian_filter(self, x):
        # due to separability we can apply two 1d gaussian filters to get some speedup
        v = F.conv2d(x, self.gaussian_kernel.unsqueeze(3), padding=(self.padding,0), groups=4)
        h = F.conv2d(v, self.gaussian_kernel.unsqueeze(2), padding=(0,self.padding), groups=4)
        return h

    def forward(self, x, m):
        # to perform the same convolution on each channel of x and on the mask,
        # we concatenate x and m and perform a convolution with groups=num_channels=4
        u = torch.cat((x, m), 1)
        epsilon = u.sum((2,3), keepdim=True) * 1e-8
        u = self.downsample(u)
        for _ in range(self.reps): u = self.gaussian_filter(u)
        u = self.upsample(u)
        u = u + epsilon
        filtered_x = u[:,:-1]
        filtered_m = u[:,-1:]
        return filtered_x / filtered_m