# Based on this paper: https://arxiv.org/abs/2205.04095
# SmoothNets can be considered as:
# (1) Wide 
# (2) DenseNets (w/o Bottlenecks) with 
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
    we want to consider. 
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
    This is a helper function to return all the different pooling operations.

    Args: 
        pool_fc_str: str to select the specific function

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
        norm_groups : int, 
        num_features : int,
        **kwargs
    ): 
    """
    This is a helper function to return all the different normalizations we want to consider. 

    Args: 
        norm_groups: the number of normalization groups of GN, or to select IN, Identity, BN
        num_features: number of channels

    """
    if norm_groups > 0:
        # for num_groups = num_features => InstanceNorm
        # for num_groups = 1 => LayerNorm
        norm_fc = nn.GroupNorm(
            num_groups=min(norm_groups, num_features), 
            num_channels=num_features, 
            affine=True
        )
    # extra cases: InstanceNorm, Identity (no norm), BatchNorm (not DP-compatible)
    elif norm_groups == 0:
        norm_fc = nn.InstanceNorm2d(
            num_features=num_features,
        )
    elif norm_groups == -1: 
        norm_fc = nn.Identity()
    elif norm_groups == -2:
        norm_fc = nn.BatchNorm2d(
            num_features=num_features
        )
    
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
        pool_fc_str: selected pooling operation (mxp, avg, identity)
        norm_groups: number of norm groups for group norm (or selected IN, BN)
        activation_fc_str: choose activation function
        dsc: whether to use depthwise seperable convolutions or not
    """

    def __init__(
        self, 
        in_channels : int, 
        out_channels : int,
        pool_fc_str : str,
        norm_groups : int,
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
        self.norm = getNormFunction(norm_groups=norm_groups, num_features=out_channels)
        self.activation_fcs = getActivationFunction(activation_fc_str)


    def forward(self, input):
        out = input
        out = self.conv_layers(out)
        out = self.pooling(out)
        out = self.norm(out)
        out = self.activation_fcs(out)

        return out

class SmoothStack(nn.Module):
    """
    Helper module to stack the different smooth blocks. 

    Args:
        in_channels: the number of channels (feature maps) of the incoming embedding
        out_channels: the number of channels after the first convolution
        pool_fc_str: selected pooling operation (mxp, avg, identity)
        norm_groups: number of norm groups for group norm (or selected IN, BN)
        activation_fc_str: choose activation function
        num_blocks: number of smooth blocks 
        dsc: whether to use depthwise seperable convolutions or not
    """
    
    def __init__(
        self, 
        in_channels : int, 
        out_channels : int, 
        pool_fc_str : str,
        norm_groups : str,
        activation_fc_str : str,
        num_blocks : int, 
        dsc : bool
    ):
        super().__init__()

        # first block to get the right number of channels (from previous block to current)
        self.smooth_stack = nn.ModuleList(
            [
                SmoothBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    pool_fc_str=pool_fc_str,
                    norm_groups=norm_groups, 
                    activation_fc_str=activation_fc_str,
                    dsc=dsc,
                )
            ]
        )
        
        # EXTEND adds array as elements of existing array, APPEND adds array as new element of array 
        self.smooth_stack.extend(
            [
                SmoothBlock(
                    in_channels=out_channels*i+in_channels, 
                    out_channels=out_channels, 
                    pool_fc_str=pool_fc_str,
                    norm_groups=norm_groups,
                    activation_fc_str=activation_fc_str,
                    dsc=dsc,
                ) 
                for i in range(1, num_blocks)
            ]
        )
        
    def forward(self, input):
        out = input
        for layer in self.smooth_stack:
            temp = layer(out)
            # concatenate at channel dimension
            out = torch.cat((out, temp), 1)
        return out

# TODO: make adaptive for MNIST (most assume dn RGB img with 3 channels)
class SmoothNet(nn.Module):
    """
    The SmoothNet class. The v1 SmoothNets can be considered as: 
    (1) Wide, (2) DenseNets (w/o Bottlenecks) with (3) SELU activations and (4) DP-compatible normalization and max pooling.

    Args: 
        pool_fc_str: set pooling operation after conv (or none)
        norm_groups: the number of groups to be used in the group normalization (0:=IN, -1:=ID, -2:=BN)
        activation_fc_str: choose activation function
        depth: a factor multiplied with number of conv blocks per stage of base model
        width: a factor multiplied with number of channels per conv block of base model
                := num_blocks (as defined in the scaling approach)
        dsc: whether depthwise seperable convolutions are used or normal convolutions
    """
    def __init__(
        self, 
        pool_fc_str : str = 'mxp',
        norm_groups : int = 8,
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

        self.stage_zero = SmoothStack(
            in_channels=in_channels,
            out_channels=width_stage_zero,
            pool_fc_str=pool_fc_str,
            norm_groups=norm_groups,
            activation_fc_str=activation_fc_str,
            num_blocks=depth_stage_zero, 
            dsc=dsc,
        )

        ## STAGE 1 ##
        # the stage 1 base model has 16 channels in stage 1
        # the stage 1 base model has 1 conv block per stage (in both stage 0 and 1)
        width_stage_one = int(width*16)
        depth_stage_one = int(depth*1)
        
        # DenseTransition #
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

        self.stage_one = SmoothStack(
            in_channels=width_stage_zero,
            out_channels=width_stage_one,
            pool_fc_str=pool_fc_str,
            norm_groups=norm_groups,
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
        # with dense transition layer
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
        norm_groups: int = 8,  # alternatives: 0:=IN, -1:=identity, -2:=BN
        pool_fc_str: int = 'mxp', # alternatives: avg, identity
        **kwargs
    ): 
    model = SmoothNet(
        width=width, 
        depth=depth, 
        norm_groups=norm_groups, 
        pool_fc_str=pool_fc_str,
        **kwargs
    )
    
    # NOTE: using jit can speed up the training
    # example = torch.randn(1, 3, 224, 224)
    # model_jit = torch.jit.trace(model, example)
    return model