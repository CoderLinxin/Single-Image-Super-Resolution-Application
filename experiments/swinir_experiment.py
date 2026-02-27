from 参考资料.KAIR_master.models.network_swinir import SwinIR
from experiments.experiment import Experiment
from configs.dataset_config import DatasetConfig
import copy
from configs.model_config import ModelConfig
import os
import torch


class SWINIRExperiment(Experiment):
    def __init__(self, **kwargs):
        super(SWINIRExperiment, self).__init__(**kwargs)

    def init_model(self):
        # 创建模型
        self.model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
            init_type="default"
        ).to(self.model_config.device)

        super(SWINIRExperiment, self).init_model()

    def load_model_weights_scheduler(self, is_gan_start: bool = False):
        # 加载预训练模型权重(以继续训练)
        pretrain_model_path = self.model_config.test_model_path if self.is_test else self.new_model_path
        if os.path.exists(pretrain_model_path):
            print('============ 加载模型权重 start ============')

            dic = torch.load(pretrain_model_path, map_location=self.model_config.device, weights_only=True)
            self.model.load_state_dict(dic['params'])

            print(f'模型权重路径: {pretrain_model_path}')
            print('============ 加载模型权重 end ============')


def swinir_experiment(is_test: bool, is_augment):
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

    # 模型配置
    model_config = ModelConfig(
        batch_size=16,
        learning_rate=2e-4,
        min_learning_rate=2e-5,
        optimizer='Adam',
        optimizer_params={'weight_decay': 0, 'betas': [0.9, 0.99]},
        loss_function='l1',
        epochs=1000,
        checkpoint_folder='weights/swinir',
        test_model_path='weights/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
        result_folder='results/swinir',
        log_folder='logs/swinir',
        train_data_folder='data/train',
        train_data_name_list=['DIV2K_train_HR'],
        eval_data_folder='data/eval',
        eval_data_name_list=['Set5'],
        test_data_folder='data/test',
        test_data_name_list=['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109'],
    )

    # 创建实验
    experiment = SWINIRExperiment(
        train_data_config=train_data_config,
        eval_data_config=eval_data_config,
        test_data_config=test_data_config,
        model_config=model_config,
        is_test=is_test
    )

    # 运行实验
    experiment.run()
