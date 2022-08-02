import torch
import torch.nn as nn

class CausalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        """
        Implementation of Causal Convolution 1d, computes 1d Convolution with mask so that values are only influenced by preceeding values

        :param int in_channels: input channels
        :param int out_channels: output channels
        :param int kernel_size: size of filter kernel
        :param int dilation: dilation of kernel, defaults to 1 <- no dilation
        """
        super().__init__()

        # calculating same padding based on kernel_size and dilation
        padding = dilation * (kernel_size-1) // 2

        # creating mask
        mask = torch.ones(kernel_size)
        mask[kernel_size//2+1:] = 0
        self.register_buffer("mask", mask[None])

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor):
        # applying mask to filter weights
        self.conv1d.weight.data *= self.mask

        return self.conv1d(x)

class GatedResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int, dilation: int):
        """
        Implementation of Gated Residual Conv Block with Causal Convolution Layers 

        :param int in_channels: input channels
        :param int hidden_channels: intermediate channels between CasusalConv layer and Conv1x1
        :param int out_channels: output channels
        :param int kernel_size: size of filter kernel
        :param int dilation: dilation of kernel
        """
        super().__init__()

        self.dilated_conv = CausalConv(in_channels, 2*hidden_channels, kernel_size, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # applying causal conv
        dilated = self.dilated_conv(x)
        # spliting channel dim into value and gate
        value, gate = dilated.chunk(2, dim=1)

        # gate
        gated_value = torch.tanh(value) * torch.sigmoid(gate)
        # output conv
        output = self.conv_1x1(gated_value)
        # residual connection
        residual_output = output + x

        # output of residual connection and output of skip connection
        return residual_output, output

class GatedConvStack(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int, num_residual_blocks: int):
        """
        Stack of Gated Residual Conv Blocks with dilation doubling at each step

        :param int in_channels: input channels
        :param int hidden_channels: intermediate channels
        :param int out_channels: output channels
        :param int kernel_size: size of filter kernels
        :param int num_residual_blocks: num of conv blocks in stack
        """
        super().__init__()

        # generating dilations -> 1, 2, 4, 8, 16, ...
        dilations = [2**i for i in range(num_residual_blocks)]

        self.conv_stack = nn.ModuleList(
            [GatedResidualConvBlock(in_channels, hidden_channels, out_channels, kernel_size, dilations[i]) for i in range(num_residual_blocks)]
        )

    def forward(self, x: torch.Tensor):
        skip_connections = []

        for layer in self.conv_stack:
            x, skip_connection = layer(x)

            skip_connections.append(skip_connection)

        # residual connection to next conv block, sum of skip connections from stack
        return x, torch.stack(skip_connections, dim=-1).sum(dim=-1)

class WaveNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int, num_stacks: int, num_residual_blocks_in_stack: int):
        """
        Implementation of WaveNet Architecture

        :param int in_channels: input channels
        :param int hidden_channels: intermediate channels
        :param int out_channels: output channels
        :param int kernel_size: size of filter kernels
        :param int num_stacks: num of stacks
        :param int num_residual_blocks_in_stack: num of gated residual conv blocks in each stack
        """
        super().__init__()

        self.causal_conv = CausalConv(in_channels, hidden_channels, kernel_size)
        self.gated_conv_stacks = nn.ModuleList(
            [GatedConvStack(hidden_channels, hidden_channels, hidden_channels, kernel_size, num_residual_blocks_in_stack) for _ in range(num_stacks)]
        )
        self.output_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x: torch.Tensor):
        skip_connections = []
        
        x = self.causal_conv(x)

        for stack in self.gated_conv_stacks:
            x, skip_connection = stack(x)

            skip_connections.append(skip_connection)

        # sum of all skip connection outputs
        output = torch.stack(skip_connections, dim=-1).sum(dim=-1)

        return self.output_block(output)
