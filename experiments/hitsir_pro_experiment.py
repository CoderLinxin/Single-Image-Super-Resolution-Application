from models.hit_sir_pro import HiT_SIR
from experiments.experiment import Experiment
from configs.dataset_config import DatasetConfig
import copy
from configs.hit_model_config import HITModelConfig
import lpips


class HITSIRPROExperiment(Experiment):
    def __init__(self, **kwargs):
        super(HITSIRPROExperiment, self).__init__(**kwargs)

    def init_model(self):
        # 创建模型
        self.model = HiT_SIR(
            is_mult_size_conv_feat_extract=self.model_config.is_mult_size_conv_feat_extract,
            is_channel_spatial_attn=self.model_config.is_channel_spatial_attn,
            is_fusion=self.model_config.is_fusion,
            embed_dim=self.model_config.embed_dim,
            base_win_size=self.model_config.base_win_size,
            depths=self.model_config.depths,
            num_heads=self.model_config.num_heads,
            mlp_ratio=self.model_config.mlp_ratio,
            upsampler=self.model_config.upsampler,
            hier_win_ratios=self.model_config.hier_win_ratios
        ).to(self.model_config.device)

        super(HITSIRPROExperiment, self).init_model()


def hitsir_pro_experiment(
        is_test: bool,
        loss: str,
        is_mult_size_conv_feat_extract: bool,
        is_channel_spatial_attn: bool,
        is_fusion: bool,
        epochs: int,
        is_augment,
        batch_size,
        test_model_name,
        embed_dim,
        base_win_size,
        depths,
        num_heads,
        mlp_ratio,
        upsampler,
        hier_win_ratios
):
    # 数据集配置
    train_data_config = DatasetConfig(
        split='train',
        crop_size=64,
        scaling_factor=4,
        lr_img_type='[0,1]',
        hr_img_type='[0,1]',
        is_lr_amplify=False,  # rcan 的输入为 lr 图像(不需要放大到与 hr 图像相同大小)
        is_augment=is_augment
    )
    eval_data_config = copy.deepcopy(train_data_config)
    eval_data_config.split = 'eval|test'
    test_data_config = copy.deepcopy(train_data_config)
    test_data_config.split = 'eval|test'

    # 获取消融实验保存结果文件夹名称
    folder_name = f'hitsir_pro_loss({loss})_mulsizeconvextract({is_mult_size_conv_feat_extract})_casa({is_channel_spatial_attn}){"_fusion" if is_fusion else ""}_embed_dim({embed_dim})_len(depths)({len(depths)})'
    if is_augment:
        folder_name = folder_name + "_augment"

    # 模型配置
    model_config = HITModelConfig(
        batch_size=batch_size,
        learning_rate=2e-5,
        min_learning_rate=1e-7,
        optimizer='Adam',
        optimizer_params={'weight_decay': 0, 'betas': [0.9, 0.99]},
        loss_function=loss,
        epochs=epochs,
        checkpoint_folder=f'weights/{folder_name}',
        test_model_path=f'weights/{folder_name}/{test_model_name}',
        result_folder=f'results/{folder_name}',
        log_folder=f'logs/{folder_name}',
        train_data_folder='data/train',
        # train_data_name_list=['DIV2K_train_HR'],
        # train_data_name_list=['RealSR(V3)', 'DIV2K_train_HR'],
        train_data_name_list=[
            'blend', 'RealSR(V3)', 'DIV2K_train_HR', 'wuthering_wave', 'Flickr2K_HR',  # 基本训练集
            # 剩下的训练集会在每个 epoch 的时候随机选取一个来训练(暂时不用)
            # 'wed1', 'wed2', 'wed3', 'wed4',
            # 'OST_dataset/animal', 'OST_dataset/building', 'OST_dataset/grass', 'OST_dataset/mountain', 'OST_dataset/plant', 'OST_dataset/sky', 'OST_dataset/water'
        ],
        # train_data_name_list=['RealSR(V3)'],
        eval_data_folder='data/eval',
        eval_data_name_list=['DIV2K_valid_HR30'],
        test_data_folder='data/test',
        # test_data_name_list=['Canon', 'Nikon', 'BSD100', 'Urban100'],
        test_data_name_list=['Set5'],
        # test_data_name_list=['display_example1', 'display_example2', 'display_example3', 'Canon', 'Nikon', 'BSD100', 'Urban100'],
        # test_data_name_list=['Canon', 'Nikon', 'Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109'],

        is_mult_size_conv_feat_extract=is_mult_size_conv_feat_extract,
        is_channel_spatial_attn=is_channel_spatial_attn,
        is_fusion=is_fusion,
        in_channel=3,
        embed_dim=embed_dim,
        base_win_size=base_win_size,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        hier_win_ratios=hier_win_ratios
    )

    # 创建实验
    experiment = HITSIRPROExperiment(
        train_data_config=train_data_config,
        eval_data_config=eval_data_config,
        test_data_config=test_data_config,
        model_config=model_config,
        is_test=is_test
    )

    # 运行实验
    experiment.run()
