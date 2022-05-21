# Based on this paper: https://arxiv.org/abs/2205.04095
# SmoothNets can be considered as:
# (1) Wide 
# (2) DenseNets with 
# (3) SELU activations and 
# (4) DP-compatible normalization and max pooling

# For more experiments codebase, see: https://github.com/NiWaRe/DPBenchmark
# W80D50 SmoothNet := en_scaling_residual_model, dense=True, depth=5.0, width=8.0

# torch 
import torch
from torch import nn


def getActivationFunction(
        activation_fc_str : str = "selu"
    ): 
    """
    This is a helper function to return all the different activation functions
    we want to consider. This is written in a dedicated function because
    it is called from different classes and because this is the central place
    where all possible activation_fc are listed.
    """
    if activation_fc_str == "selu": 
        activation_fc = nn.SELU()
    elif activation_fc_str == "relu": 
        activation_fc = nn.ReLU()
    elif activation_fc_str == "leaky_relu":
        activation_fc = nn.LeakyReLU()

    return activation_fc


def getPoolingFunction(
        pool_fc_str : str, 
        **kwargs
    ): 
    """
    This is a helper function to return all the different after_conv_fcs
    we want to consider. This is written in a dedicated function because
    it is called from different classes and because this is the central place
    where all possible after_conv_fct are listed.

    Args: 
        after_conv_fc_str: str to select the specific function
        num_features: number of channels, only necessary for norms 

    """
    if pool_fc_str == 'mxp': 
        # keep dimensions for CIFAR10 dimenions assuming a downsampling 
        # only through halving. 
        pool_fc = nn.MaxPool2d(
            kernel_size=3, 
            stride=1, 
            padding=1
        )
    elif pool_fc_str == 'avg': 
        # keep dimensions for CIFAR10 dimenions assuming a downsampling 
        # only through halving. 
        pool_fc = nn.AvgPool2d(
            kernel_size=3, 
            stride=1, 
            padding=1
        )
    elif pool_fc_str == 'identity': 
        pool_fc = nn.Identity()
    
    return pool_fc

def getNormFunction(
        norm_fc_str : str, 
        num_features : int,
        **kwargs
    ): 
    """
    This is a helper function to return all the different after_conv_fcs
    we want to consider. This is written in a dedicated function because
    it is called from different classes and because this is the central place
    where all possible after_conv_fct are listed.

    Args: 
        norm_fc_str: str to select the specific function
        num_features: number of channels, only necessary for norms 

    """
    if norm_fc_str == 'bn':
        norm_fc = nn.BatchNorm2d(
            num_features=num_features
        )
    elif norm_fc_str == 'gn':
        # for num_groups = num_features => LayerNorm
        # for num_groups = 1 => InstanceNorm
        min_num_groups = 8
        norm_fc = nn.GroupNorm(
            num_groups=min(min_num_groups, num_features), 
            num_channels=num_features, 
            affine=True
        )
    elif norm_fc_str == 'in':
        # could also use GN with num_groups=num_channels
        norm_fc = nn.InstanceNorm2d(
            num_features=num_features,
        )
    elif norm_fc_str == 'ln':
        # could also use nn.LayerNorm, but we would need
        # complete input dimension for that (possible but more work)
        # after_conv_fc = nn.LayerNorm(
        #     normalized_shape=input_shape[1:],
        # )
        norm_fc = nn.GroupNorm(
            num_groups=1, 
            num_channels=num_features, 
            affine=True
        )
    elif norm_fc_str == 'identity': 
        norm_fc = nn.Identity()
    
    return norm_fc

# NOTE: Specific features have been designed for CIFAR10 Input specifically
# NOTE: Difference to Stage 1 Network: Downsampling and channel adaptation
#       happens in first Conv2d.
class SmoothBlock(nn.Module):
    """
    Inspired by Dense Blocks and Residual Blocks of ResNet and DenseNet.

    Args:
        in_channels: the number of channels (feature maps) of the incoming embedding
        out_channels: the number of channels after the first convolution
        after_conv_fc_str: norm or pooling after Conv2D layer (BatchNorm2d, MaxPool, Identity)
        activation_fc_str: choose activation function
        skip_depth: how much blocks skip connection should jump,
            2 = default, 0 = no skip connections
        dsc: whether to use depthwise seperable convolutions or not
    """

    def __init__(
        self, 
        in_channels : int, 
        out_channels : int,
        pool_fc_str : str,
        norm_fc_str : str,
        activation_fc_str : str,
        dsc : bool = False,  
    ):
        super().__init__()

        if not dsc: 
            self.conv_layers = nn.Conv2d(
                        in_channels, 
                        out_channels, 
                        kernel_size = 3, 
                        stride = 1, 
                        padding = 1, 
                        bias=False
                    ) 
        else: 
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    in_channels, 
                    groups=in_channels,
                    kernel_size = 3, 
                    stride = 1, 
                    padding = 1, 
                    bias=False, 
                ), 
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    bias=False
                )
            )

        # set post conv operations
        self.pooling = getPoolingFunction(pool_fc_str=pool_fc_str)
        self.norm = getNormFunction(norm_fc_str=norm_fc_str, num_features=out_channels)
        self.activation_fcs = getActivationFunction(activation_fc_str)


    def forward(self, input):
        # go through all triples expect last one
        out = input
        out = self.conv_layers(out)
        # out = self.after_conv_fcs(out)
        out = self.pooling(out)
        out = self.norm(out)
        out = self.activation_fcs(out)

        return out

class ResidualStack(nn.Module):
    """
    Helper module to stack the different residual blocks. 
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        after_conv_fc_str: norm or pooling after Conv2D layer (BatchNorm2d, MaxPool, Identity)
        activation_fc_str: choose what activation function to use
        num_blocks: Number of residual blocks
        dsc: whether to use depthwise seperable convolutions or not
    """
    
    def __init__(
        self, 
        in_channels : int, 
        out_channels : int, 
        pool_fc_str : str,
        norm_fc_str : str,
        activation_fc_str : str,
        num_blocks : int, 
        dsc : bool
    ):
        super().__init__()

        # first block to get the right number of channels (from previous block to current)
        # and sample down if specified (specifically at the first layer in the ResidualBlock)
        self.residual_stack = nn.ModuleList(
            [
                SmoothBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    pool_fc_str=pool_fc_str,
                    norm_fc_str=norm_fc_str, 
                    activation_fc_str=activation_fc_str,
                    dsc=dsc,
                )
            ]
        )
        
        # EXTEND adds array as elements of existing array, APPEND adds array as new element of array 
        self.residual_stack.extend(
            [
                SmoothBlock(
                    in_channels=out_channels*i+in_channels, 
                    out_channels=out_channels, 
                    pool_fc_str=pool_fc_str,
                    norm_fc_str=norm_fc_str,
                    activation_fc_str=activation_fc_str,
                    dsc=dsc,
                ) 
                for i in range(1, num_blocks)
            ]
        )
        
    def forward(self, input):
        out = input
        for layer in self.residual_stack:
            temp = layer(out)
            # concatenate at channel dimension
            out = torch.cat((out, temp), 1)
        return out

# NOTE: some differences to my manually crafted CNN (not just added skip connections)
    # - downsampling is done with adaption of channels in first ConvBlock 
    # - downsampling is done through Conv2d layer and not dedicated maxpool layer

# TODO: make adaptive for MNIST (most assume dn RGB img with 3 channels)
class SmoothNet(nn.Module):
    """
    ConvNet that is based on stage 1 network of StageConvNet and where
    depth (factor multiplied number of equal size conv blocks) and width 
    (factor multiplied number of channels per conv block) can be changed seperately. 
    By default depth and width are 1 which results in the stage 1 network of StafeConvNet.
    Args: 
        pool_fc_str: set pooling operation after conv (or none)
        norm_fc_str: set a normalization layer after conv(+pooling if set)
        activation_fc_str: choose activation function
        depth: a factor multiplied with number of conv blocks per stage of base model
        width: a factor multiplied with number of channels per conv block of base model
                := num_blocks (as defined in the scaling approach)
        dsc: whether depthwise seperable convolutions are used or normal convolutions
    """
    def __init__(
        self, 
        pool_fc_str : str = 'mxp',
        norm_fc_str : str = 'gn',
        activation_fc_str : str = 'selu',
        in_channels: int = 3,
        depth: float = 1.0,
        width: float = 1.0,
        dsc: bool = False,
        ):
        super(SmoothNet, self).__init__()

        ## STAGE 0 ##
        # the stage 1 base model has 8 channels in stage 0
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_zero = int(width*8)
        depth_stage_zero = int(depth*1)

        self.stage_zero = ResidualStack(
            in_channels=in_channels,
            out_channels=width_stage_zero,
            pool_fc_str=pool_fc_str,
            norm_fc_str=norm_fc_str,
            activation_fc_str=activation_fc_str,
            num_blocks=depth_stage_zero, 
            dsc=dsc,
        )

        ## STAGE 1 ##
        # the stage 1 base model has 16 channels in stage 1
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_one = int(width*16)
        depth_stage_one = int(depth*1)
        
        # DenseTransition if using DenseBlocks #
        # recalculate the number of input features 
        # depth_stage_zero (total num of blocks) + input channels (3 for CIFAR10)
        width_stage_zero = width_stage_zero * depth_stage_zero + 3
        # same as original Tranistion Layers in DenseNet
        # features are halved through 1x1 Convs and AvgPool is used to halv the dims
        self.dense_transition = nn.Sequential(
            #getAfterConvFc(after_conv_fc_str, width_stage_zero), 
            nn.Conv2d(
                width_stage_zero, 
                width_stage_zero//2, 
                kernel_size=1, 
                stride=1, 
                bias=False
            ),
            nn.AvgPool2d(
                kernel_size=2, 
                stride=2
            ), 
        )
        width_stage_zero = width_stage_zero // 2

        self.stage_one = ResidualStack(
            in_channels=width_stage_zero,
            out_channels=width_stage_one,
            pool_fc_str=pool_fc_str,
            norm_fc_str=norm_fc_str,
            activation_fc_str=activation_fc_str,
            num_blocks=depth_stage_one, 
            dsc=dsc,
        )  

        self.pre_final = nn.AvgPool2d(kernel_size=2, stride=2)
        self.width_stage_one = width_stage_one

        ## Final FC Block ##
        # output_dim is fixed to 4 (even if 8 makes more sense for the stage 1 StageConvModel)
        output_dim = 4
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4) 

        self.fc1 = nn.Linear(width_stage_one*output_dim**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        if activation_fc_str == "selu":
            self.relu1 = nn.SELU()
            self.relu2 = nn.SELU()
        else:
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.stage_zero(x)
        # add dense transition layer if using dense connections; in this case
        # no dim or feature reduction will happen in the stages themselves
        out = self.dense_transition(out)
        out = self.stage_one(out) 
        # as input to last FC layer only the output of the last conv_block 
        # should be considered in the dense connection case
        # last pooling layer to downsampling (same as in DenseNet)
        out = self.pre_final(out)
        # only get output of last conv layer
        out = out[:, -self.width_stage_one:, :, :]
        out = self.adaptive_pool(out)
        out = out.view(batch_size, -1)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out 

# the standard SmoothNet used in the original paper is a SmoothNet W80D50
def getSmoothNets(
        width: float = 8.0, 
        depth: float = 5.0,
        **kwargs
    ): 
    model = SmoothNet(width=width, depth=depth, **kwargs)
    
    # TODO: check if using jit can speed up the training
    # example = torch.randn(1, 3, 224, 224)
    # model_jit = torch.jit.trace(model, example)
    return model