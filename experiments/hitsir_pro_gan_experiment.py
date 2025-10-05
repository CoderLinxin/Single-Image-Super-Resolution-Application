from models.hit_sir_pro import HiT_SIR
from experiments.experiment import Experiment
from configs.dataset_config import DatasetConfig
import copy
from configs.hit_model_config import HITModelConfig
from 参考资料.KAIR_master.models.network_discriminator import Discriminator_UNet as discriminator
from configs.model_config import get_optimizer, get_scheduler, get_loss_function
from 参考资料.KAIR_master.models.loss import GANLoss, PerceptualLoss
import torch
import os
from utils.utils import AverageMeter
import numpy as np


class HITSIRPROGANExperiment(Experiment):
    def __init__(self, **kwargs):
        super(HITSIRPROGANExperiment, self).__init__(**kwargs)

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

        # 创建判别器
        self.discriminator = discriminator().to(self.model_config.device)

        super(HITSIRPROGANExperiment, self).init_model()

    def init_tools(self):
        super(HITSIRPROGANExperiment, self).init_tools()
        self.epoch_discriminator_loss = AverageMeter()

    def train(self):
        self.epoch_discriminator_loss.reset()
        self.discriminator.train()
        super(HITSIRPROGANExperiment, self).train()

    def eval(self, start_epoch=None):
        self.discriminator.eval()
        super(HITSIRPROGANExperiment, self).eval(start_epoch)

    def init_optimizer_loss_function(self, params=None):
        super(HITSIRPROGANExperiment, self).init_optimizer_loss_function()

        # 创建判别器的优化器
        self.discriminator_optimizer = get_optimizer(
            optimizer_name=self.model_config.optimizer,
            model=self.discriminator,
            lr=self.model_config.learning_rate,
            kwarg=self.model_config.optimizer_params,
            params=params
        )
        # 感知损失(内部会自动加载预训练的vgg19模型)
        self.f_loss_function = PerceptualLoss(
            feature_layer=[2, 7, 16, 25, 34],
            weights=[0.1, 0.1, 1.0, 1.0, 1.0],
            lossfn_type="l1",
            use_input_norm=True,
            use_range_norm=False
        ).to(self.model_config.device)
        self.f_loss_function_weight = 1  # 感知损失的权重
        # 判别损失
        self.d_loss_function = GANLoss('gan', 1.0, 0.0).to(self.model_config.device)
        self.d_loss_function_weight = 0.1  # 判别损失的权重

    def load_model_weights_scheduler(self, is_gan_start: bool = False):
        # 加载判别器模型权重
        self.discriminator_pretrain_model_path = os.path.join(self.model_config.checkpoint_folder, 'discriminator_new_epoch_model.pth')

        if os.path.exists(self.discriminator_pretrain_model_path):
            print('============ 加载判别器模型权重 start ============')

            dic = torch.load(self.discriminator_pretrain_model_path, map_location=self.model_config.device, weights_only=True)
            self.discriminator.load_state_dict(dic['model'])
            self.discriminator_optimizer.load_state_dict(dic['optimizer'])
            self.start_epoch = dic['start_epoch'] + 1

            print(f'模型权重路径: {self.discriminator_pretrain_model_path}, 训练 epoch 数: {self.start_epoch - 1}')
            print('============ 加载判别器模型权重 end ============')

        # 同步初始学习率(这样使得修改最小学习率后学习率调整器调整学习率能够及时同步)
        for param_group in self.discriminator_optimizer.param_groups:
            if "initial_lr" in param_group:
                param_group['initial_lr'] = self.model_config.learning_rate
            print(f'同步判别器初始学习率为 {self.model_config.learning_rate}')

        # 创建判别器的优化器学习率调整器
        self.lr_discriminator_scheduler = get_scheduler(
            optimizer=self.discriminator_optimizer,
            T_max=self.model_config.epochs,
            eta_min=self.model_config.min_learning_rate,
            last_epoch=-1 if self.start_epoch == 1 else self.start_epoch - 2,
        )

        print(f'当前 epoch 的判别器学习率为: {self.discriminator_optimizer.param_groups[0]["lr"]}')
        super(HITSIRPROGANExperiment, self).load_model_weights_scheduler(is_gan_start=self.start_epoch == 1)

    def load_log(self):
        self.lr_log = [  # 初始值
            f"epoch:{self.start_epoch},lr:{format_str(self.optimizer.param_groups[0]['lr'], 25)}, discriminator_lr:{format_str(self.discriminator_optimizer.param_groups[0]['lr'], 25)}"
        ]
        super(HITSIRPROGANExperiment, self).load_log()
        # 有可能本轮更改了初始学习率,导致上一轮 epoch 记录的本轮使用的学习率旧了,需要同步下
        self.lr_log[-1] = f"epoch:{self.start_epoch},lr:{format_str(self.optimizer.param_groups[0]['lr'], 25)}, discriminator_lr:{format_str(self.discriminator_optimizer.param_groups[0]['lr'], 25)}"

    # 不使用父类的实现
    def train_batch_process(
            self,
            hr_imgs: torch.Tensor,  # (b,c,h,w)
            sr_imgs: torch.Tensor,  # (b,c,h,w)
            _: str, __: str, ___: str
    ) -> dict:
        # ------------------------------------
        # 优化生成器
        # ------------------------------------

        # 判别器不要梯度
        for p in self.discriminator.parameters():
            p.requires_grad = False
        # 清空梯度
        self.optimizer.zero_grad()
        # 计算 loss
        loss = self.loss_function(input=sr_imgs, target=hr_imgs)  # 像素损失
        loss += self.f_loss_function_weight * self.f_loss_function(sr_imgs, hr_imgs)  # vgg 感知损失
        pred_g_fake = self.discriminator(sr_imgs)  # 判别器判别假图片
        loss += self.d_loss_function_weight * self.d_loss_function(pred_g_fake, True)  # 判别损失(希望假图片为真)
        # 更新梯度
        loss.backward()
        # 更新参数
        self.optimizer.step()
        # 更新损失
        self.epoch_loss.update(loss.item() / (1 + self.f_loss_function_weight + self.d_loss_function_weight), len(hr_imgs))

        # ------------------------------------
        # 优化判别器
        # ------------------------------------
        # 判别器要梯度
        for p in self.discriminator.parameters():
            p.requires_grad = True
        # 清空梯度
        self.discriminator_optimizer.zero_grad()
        # real 损失
        pred_d_real = self.discriminator(hr_imgs)
        l_d_real = self.d_loss_function(pred_d_real, True)  # 希望真图片为真
        # 更新梯度
        l_d_real.backward()
        # fake 损失
        pred_d_fake = self.discriminator(sr_imgs.detach().clone())
        l_d_fake = self.d_loss_function(pred_d_fake, False)  # 希望假图片为假
        # 更新梯度
        l_d_fake.backward()
        # 更新参数
        self.discriminator_optimizer.step()
        # 更新损失
        self.epoch_discriminator_loss.update((l_d_real.item() + l_d_fake.item()) / 2, len(hr_imgs))

    def train_dataloader_process(
            self,
            is_end: bool,  # 遍历的是否是最后一个 data_loader
            _: str
    ):
        super(HITSIRPROGANExperiment, self).train_dataloader_process(is_end, _)

        # 每个epoch结束更新一次学习率
        self.lr_discriminator_scheduler.step()

        # 每个epoch结束保存最新模型(判别器)
        self.save_model_weights(
            model_path=self.discriminator_pretrain_model_path,
            model=self.discriminator,
            optimizer=self.discriminator_optimizer
        )

        # 每个epoch结束记录平均损失(加上判别损失)
        self.loss_log[-1].append(f'd_loss:{self.epoch_discriminator_loss.avg}')
        self.lr_log[
            -1] = f"epoch:{self.start_epoch + 1},lr:{format_str(self.optimizer.param_groups[0]['lr'], 25)}, discriminator_lr:{format_str(self.discriminator_optimizer.param_groups[0]['lr'], 25)}"
        # 保存训练指标
        np.savetxt(self.loss_log_path, self.loss_log, fmt='%s')  # 文件不存在会自动创建相应的文件
        np.savetxt(self.lr_log_path, self.lr_log, fmt='%s')


def hitsir_pro_gan_experiment(
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
    folder_name = f'hitsir_pro_gan_loss({loss})_mulsizeconvextract({is_mult_size_conv_feat_extract})_casa({is_channel_spatial_attn}){"_fusion" if is_fusion else ""}_embed_dim({embed_dim})_len(depths)({len(depths)})'
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
        train_data_name_list=['RealSR(V3)', 'DIV2K_train_HR', 'wuthering_wave', 'Flickr2K_HR', 'blend'],
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
    experiment = HITSIRPROGANExperiment(
        train_data_config=train_data_config,
        eval_data_config=eval_data_config,
        test_data_config=test_data_config,
        model_config=model_config,
        is_test=is_test
    )

    # 运行实验
    experiment.run()
