from models.bsrgan import RRDBNet
from experiments.experiment import Experiment
from configs.dataset_config import DatasetConfig
import copy
from configs.model_config import ModelConfig


class RRDBExperiment(Experiment):
    def __init__(self, **kwargs):
        super(RRDBExperiment, self).__init__(**kwargs)

    def init_model(self):
        # 创建模型
        self.model = RRDBNet().to(self.model_config.device)

        super(RRDBExperiment, self).init_model()


def rrdb_experiment(is_test: bool, is_augment):
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
        batch_size=8,
        learning_rate=2e-4,
        min_learning_rate=2e-5,
        optimizer='Adam',
        optimizer_params={'weight_decay': 0, 'betas': [0.9, 0.99]},
        loss_function='l1',
        epochs=1000,
        checkpoint_folder='weights/rrdb',
        test_model_path='weights/rrdb/best_psnr_ssim_lpips_model.pth',
        result_folder='results/rrdb',
        log_folder='logs/rrdb',
        train_data_folder='data/train',
        train_data_name_list=['DIV2K_train_HR'],
        eval_data_folder='data/eval',
        eval_data_name_list=['Set5'],
        test_data_folder='data/test',
        test_data_name_list=['Urban100'],
    )

    # 创建实验
    experiment = RRDBExperiment(
        train_data_config=train_data_config,
        eval_data_config=eval_data_config,
        test_data_config=test_data_config,
        model_config=model_config,
        is_test=is_test
    )

    # 运行实验
    experiment.run()
