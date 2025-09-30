from configs.model_config import ModelConfig
from typing import Union, List, Tuple


class DenseModelConfig(ModelConfig):
    def __init__(
            self,
            is_sa_attn: bool,
            is_fusion: bool,
            is_mult_size_conv_feat_extract: bool,
            num_blocks: list[int],
            skip_blocks: list[int] = None,
            scaling_factor: int = 4,
            in_channel: int = 3,
            middle_channels: int = 64,
            **kwargs
    ):
        super(DenseModelConfig, self).__init__(**kwargs)

        self.is_sa_attn = is_sa_attn
        self.is_fusion = is_fusion
        self.is_mult_size_conv_feat_extract = is_mult_size_conv_feat_extract
        self.num_blocks = num_blocks
        self.skip_blocks = skip_blocks
        self.scaling_factor = scaling_factor
        self.in_channel = in_channel
        self.middle_channels = middle_channels
