from configs.model_config import ModelConfig
from typing import Union, List, Tuple


class HITModelConfig(ModelConfig):
    def __init__(
            self,
            is_mult_size_conv_feat_extract: bool,
            is_channel_spatial_attn: bool,
            is_fusion: bool,
            scaling_factor: int = 4,
            in_channel: int = 3,
            embed_dim: int = (6 * 3) * 4,  # 注意 embed_dim 必须是 num_heads[i] * 2 的整数倍
            base_win_size=[8, 8],
            depths=[6, 6, 6, 6],
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffledirect',
            hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
            **kwargs
    ):
        super(HITModelConfig, self).__init__(**kwargs)

        self.is_mult_size_conv_feat_extract = is_mult_size_conv_feat_extract
        self.is_channel_spatial_attn = is_channel_spatial_attn
        self.is_fusion = is_fusion
        self.scaling_factor = scaling_factor
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.base_win_size = base_win_size
        self.depths = depths
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.upsampler = upsampler
        self.hier_win_ratios = hier_win_ratios
