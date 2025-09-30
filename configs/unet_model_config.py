from configs.model_config import ModelConfig
from typing import Union, List, Tuple


class UNetModelConfig(ModelConfig):
    def __init__(
            self,
            image_in_channels: int = 3,
            image_out_channels: int = 64,
            n_channels: int = 64,
            self_attention_layer_count: int = 1,
            ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 1, 1),
            is_attn: Union[Tuple[bool, ...], List[int]] = (True, True, True, True),
            n_blocks: int = 2,
            n_heads: int = 1,
            **kwargs
    ):
        super(UNetModelConfig, self).__init__(**kwargs)

        self.image_in_channels = image_in_channels
        self.image_out_channels = image_out_channels
        self.n_channels = n_channels
        self.self_attention_layer_count = self_attention_layer_count
        self.ch_mults = ch_mults
        self.is_attn = is_attn
        self.n_blocks = n_blocks
        self.n_heads = n_heads
