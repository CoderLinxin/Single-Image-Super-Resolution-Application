import time
from abc import ABCMeta
import numpy as np
import torch
from torch import nn
from datasets.dataset import Dataset
from configs.dataset_config import DatasetConfig
from configs.model_config import ModelConfig
from configs.unet_model_config import UNetModelConfig
from configs.dense_model_config import DenseModelConfig
from configs.hit_model_config import HITModelConfig
from utils.utils import AverageMeter, convert_image, format_str
from torch.utils.data import DataLoader
from configs.model_config import get_optimizer, get_scheduler, get_loss_function
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
import lpips
import glob
import os
import shutil


class Experiment(metaclass=ABCMeta):
    def __init__(
            self,
            train_data_config: DatasetConfig,
            eval_data_config: DatasetConfig,
            test_data_config: DatasetConfig,
            model_config: ModelConfig | UNetModelConfig | DenseModelConfig | HITModelConfig,
            is_test: bool,
    ):
        """
        :param train_data_config: 训练数据配置
        :param eval_data_config: 验证数据配置
        :param test_data_config: 测试数据配置
        :param model_config: 模型配置
        :param is_test: 是否处于测试阶段
        """
        self.train_data_config = train_data_config
        self.eval_data_config = eval_data_config
        self.test_data_config = test_data_config
        self.model_config = model_config
        self.is_test = is_test
        self.lpips_fn = lpips.LPIPS(net='vgg')

        self.eval_data_count = 0  # 统计验证集数据总数
        self.img_transform = transforms.ToPILImage()

        # 数据集相关
        self.train_loaders: list[DataLoader] = []
        self.eval_loaders: list[DataLoader] = []
        self.test_loaders: list[DataLoader] = []
        # 模型相关
        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = None
        self.loss_function = None

        # 训练开始的 epoch
        self.start_epoch = 1

        # 训练过程模型保存路径
        self.new_model_path = os.path.join(self.model_config.checkpoint_folder, 'new_epoch_model.pth')
        self.best_psnr_model_path = os.path.join(self.model_config.checkpoint_folder, 'best_psnr_model.pth')
        self.best_ssim_model_path = os.path.join(self.model_config.checkpoint_folder, 'best_ssim_model.pth')
        self.best_lpips_model_path = os.path.join(self.model_config.checkpoint_folder, 'best_lpips_model.pth')
        self.best_psnr_ssim_lpips_model_path = os.path.join(self.model_config.checkpoint_folder, 'best_psnr_ssim_lpips_model.pth')

        # 初始化数据和模型
        self.init_data_loaders()
        self.init_model()
        self.init_optimizer_loss_function()
        self.load_model_weights_scheduler()

        # 测试过程相关路径
        self.result_path = os.path.join(
            self.model_config.result_folder,  # results/srcnn
            os.path.basename(self.model_config.test_model_path).split('.')[0]  # best
        )
        # ['results/srcnn/best/Set5','results/srcnn/best/Set14',...]
        self.result_data_paths = [os.path.join(self.result_path, test_loader.name) for test_loader in self.test_loaders]

        # 训练指标
        self.loss_log = []  # 记录每个 epoch 的训练损失: [[epoch1,loss1], [epoch2,loss2], ...]
        # 验证指标
        self.best_epoch_psnr_ssim_lpips_log = [-1, -1, -1, 1]  # 记录 psnr、ssim、lpips 同时达到最优及对应的 epoch: [epoch, psnr, ssim, lpips]
        self.psnr_ssim_lpips_log = []  # 记录每个 epoch 的 psnr、ssim: [[epoch1, psnr1, ssim1, lpips1],[epoch2, psnr2, ssim2, lpips2],...]
        self.only_best_psnr = -1  # 记录单 psnr 最优: psnr
        self.only_best_ssim = -1  # 记录单 ssim 最优: ssim
        self.only_best_lpips = 1  # 记录单 lpips 最优: lpips
        # 学习率调整过程
        self.lr_log = [
            f"epoch:{self.start_epoch},lr:{format_str(self.optimizer.param_groups[0]['lr'], 25)}"
        ]  # 保存历史学习率的列表: [[epoch1,lr1], [epoch2,lr2], ...]
        # 每个 epoch 的训练、验证时长及最终的训练总时长(秒)
        # [[epoch1,训练时长:xxx,验证时长:xxx,验证数据集: xx、xx、...],[epoch2,训练时长:xxx,验证时长:xxx,验证数据集: xx、xx、...],...]
        self.train_eval_seconds_consume_log = []
        # 总的训练+验证总时长
        self.total_seconds_consume_log = [0]  # [xx]
        # 加载已保存的相关指标
        self.load_log()

        # 训练、验证、测试过程中用到的相关指标计算工具初始化
        self.init_tools()

    def init_tools(self):
        # 训练阶段
        # 记录每个 epoch 的平均损失
        self.epoch_loss = AverageMeter()
        self.train_start_time = None

        # 验证阶段
        # 记录每个 epoch 中验证集的平均 PSNR、平均 SSIM、平均 lpips 值
        self.epoch_psnr = AverageMeter()
        self.epoch_ssim = AverageMeter()
        self.epoch_lpips = AverageMeter()
        self.eval_start_time = None

        # 测试阶段
        # 统计每一个测试集的相关指标
        self.test_set_psnr = AverageMeter()
        self.test_set_ssim = AverageMeter()
        self.test_set_lpips = AverageMeter()
        self.test_start_time = None

    # 初始化 data_loader
    def init_data_loaders(self, is_shuffle=True):
        print('============ 加载数据集 start ============')

        # 准备训练集
        for i, train_data_path in enumerate(self.model_config.train_data_path_list):
            data_name = self.model_config.train_data_name_list[i]
            # 创建 train_dataset
            train_dataset = Dataset(config=self.train_data_config, data_folder=train_data_path)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.model_config.batch_size,
                shuffle=is_shuffle,
                drop_last=True,
                pin_memory=False
            )
            self.train_loaders.append(train_loader)
            # 添加数据集名称
            train_loader.name = data_name
            # 打印数据集名称,数据集大小
            if not self.is_test:
                print(
                    f'{format_str("train_data: " + train_loader.name, 20, " ")}, train_data_len: {len(train_loader.dataset)}'
                )

        # 准备验证集
        for i, eval_data_path in enumerate(self.model_config.eval_data_path_list):
            data_name = self.model_config.eval_data_name_list[i]
            eval_dataset = Dataset(config=self.eval_data_config, data_folder=eval_data_path)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=1,
                pin_memory=False
            )
            self.eval_loaders.append(eval_loader)
            # 更新验证数据总量
            self.eval_data_count += len(eval_loader)
            eval_loader.name = data_name
            if not self.is_test:
                print(
                    f'{format_str("eval_data : " + eval_loader.name, 20, " ")}, eval_data_len : {len(eval_loader.dataset)}'
                )

        if self.is_test:
            # 准备测试集
            for i, test_data_path in enumerate(self.model_config.test_data_path_list):
                data_name = self.model_config.test_data_name_list[i]
                test_dataset = Dataset(config=self.test_data_config, data_folder=test_data_path)
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=1,
                )
                self.test_loaders.append(test_loader)
                test_loader.name = data_name
                if self.is_test:
                    print(
                        f'{format_str("test_data: " + test_loader.name, 20, " ")}, test_data_len: {len(test_loader.dataset)}'
                    )

        print('============ 加载数据集 end ============')

    # 初始化模型
    def init_model(self):
        # 如果是训练，则所有(裁剪出来的)图像块 image_size 必须保持固定的分辨率且保证能够整除 scaling_factor,这个由人为传参来确保
        # 如果是验证或测试，则要求裁剪出来的图像块的宽高尽可能大且能够整除 scaling_factor
        assert self.train_data_config.image_size % self.train_data_config.scaling_factor == 0, \
            f'高分辨率图像块裁剪尺寸不能被 scaling_factor 整除!'

        self.print_total_params_num()

    # 打印模型总参数量
    def print_total_params_num(self):
        total_params = 0
        for _, param in self.model.named_parameters():
            # 只统计可训练的参数
            if param.requires_grad:
                total_params += param.numel()
        params_descript = f"Total parameters: {total_params}"
        print(params_descript)
        # 保存模型参数量
        np.savetxt(os.path.join(self.model_config.log_folder, '模型参数量.txt'), [params_descript], fmt='%s')

    # 加载预训练模型权重(如果有)以及学习率调度器
    def load_model_weights_scheduler(self, is_gan_start: bool = False):
        # 加载预训练模型权重(以继续训练)
        # 测试阶段可以选择合适的模型加载(通过指定 test_model_path)，训练阶段总是通过 new_model_path 加载
        pretrain_model_path = self.model_config.test_model_path if self.is_test else self.new_model_path
        if os.path.exists(pretrain_model_path):
            print('============ 加载模型权重 start ============')

            dic = torch.load(pretrain_model_path, map_location=self.model_config.device, weights_only=True)
            self.model.load_state_dict(dic['model'])
            if not is_gan_start:  # 如果是 gan 开始训练的第一个 epoch 则无需加载优化器
                self.optimizer.load_state_dict(dic['optimizer'])
                print('加载优化器')
            else:
                print('gan 第一个 epoch 训练, 无需加载优化器')
            if type(self).__name__ != 'HITSIRPROGANExperiment':  # 如果是 HITSIRPROGANExperiment 那么 start_epoch 以判别器模型中保存的 start_epoch 为准
                self.start_epoch = dic['start_epoch'] + 1

            print(f'模型权重路径: {pretrain_model_path}, 训练 epoch 数: {self.start_epoch - 1}')
            print('============ 加载模型权重 end ============')

        # 同步初始学习率(这样使得修改最小学习率后学习率调整器调整学习率能够及时同步)
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                param_group['initial_lr'] = self.model_config.learning_rate
            print(f'同步初始学习率为 {self.model_config.learning_rate}')

        # 创建优化器学习率调整器
        # scheduler 构造时会读取优化器的 initial_lr 并保存到内部,并同步一次优化器的学习率(计算出本轮需要使用的正确的学习率(使用了initial_lr)), 后续 step 更新学习率时使用的就是此刻拿到的 initial_lr, 而不是根据优化器上的 initial_lr
        # 假如后续优化器的 initial_lr 发生更改,只要没有重新构造 scheduler, 那么 scheduler.step 更新学习率参考的还是此处拿到的 initial_lr
        self.lr_scheduler = get_scheduler(
            optimizer=self.optimizer,
            T_max=self.model_config.epochs,
            eta_min=self.model_config.min_learning_rate,
            last_epoch=-1 if self.start_epoch == 1 else self.start_epoch - 2,
        )

        print(f'当前epoch的学习率为: {self.optimizer.param_groups[0]["lr"]}')

    # 保存模型权重(记得在调用下面函数前确保start_epoch是最新的内容)
    def save_model_weights(self, model_path: str, model=None, optimizer=None):
        torch.save({
            'start_epoch': self.start_epoch,  # 存储模型保存时的 epoch
            'model': self.model.state_dict() if model is None else model.state_dict(),
            'optimizer': self.optimizer.state_dict() if optimizer is None else optimizer.state_dict(),
        }, model_path)

    # 初始化优化器、损失函数
    def init_optimizer_loss_function(self, params=None):
        # 创建优化器
        self.optimizer = get_optimizer(
            optimizer_name=self.model_config.optimizer,
            model=self.model,
            lr=self.model_config.learning_rate,
            kwarg=self.model_config.optimizer_params,
            params=params
        )
        # 创建损失函数
        self.loss_function = get_loss_function(
            loss_function_name=self.model_config.loss_function,
            device=self.model_config.device,
        )

    # 加载已保存的相关训练指标、验证指标及训练总时长(以继续训练)
    def load_log(self):
        if self.is_test:
            return

        # 初始化相关指标保存路径
        self.loss_log_path = os.path.join(self.model_config.log_folder, 'loss_log.txt')
        self.psnr_ssim_lpips_log_path = os.path.join(self.model_config.log_folder, 'psnr_ssim_lpips_log.txt')
        self.best_epoch_psnr_ssim_lpips_log_path = os.path.join(self.model_config.log_folder, 'best_epoch_psnr_ssim_lpips_log.txt')
        self.lr_log_path = os.path.join(self.model_config.log_folder, 'lr_log.txt')
        self.train_eval_seconds_consume_log_path = os.path.join(self.model_config.log_folder,
                                                                'train_eval_seconds_consume_log.txt')
        self.total_seconds_consume_log_path = os.path.join(self.model_config.log_folder,
                                                           'total_seconds_consume_log.txt')

        print('============ 加载相关指标文件 start ============')

        # 加载已保存的相关指标
        if os.path.exists(self.loss_log_path):
            self.loss_log = np.loadtxt(self.loss_log_path, dtype=str).tolist()
            if type(self.loss_log[0]) != list:
                self.loss_log = [self.loss_log]
            print(f'{os.path.basename(self.loss_log_path).split(".")[0]}加载完毕~')
        if os.path.exists(self.psnr_ssim_lpips_log_path):
            self.psnr_ssim_lpips_log = np.loadtxt(self.psnr_ssim_lpips_log_path, dtype=str).tolist()
            # 当只存储了一行时,读取进来的时候会解包一层,这不符合我们的预期,需要进行处理
            if type(self.psnr_ssim_lpips_log[0]) != list:
                self.psnr_ssim_lpips_log = [self.psnr_ssim_lpips_log]
            psnr_ssim_log = np.array(self.psnr_ssim_lpips_log)
            self.only_best_psnr = psnr_ssim_log[:, 1].astype(float).max()
            self.only_best_ssim = psnr_ssim_log[:, 2].astype(float).max()
            self.only_best_lpips = psnr_ssim_log[:, 3].astype(float).min()
            print(f'{os.path.basename(self.psnr_ssim_lpips_log_path).split(".")[0]}加载完毕~')
        if os.path.exists(self.best_epoch_psnr_ssim_lpips_log_path):
            self.best_epoch_psnr_ssim_lpips_log = np.loadtxt(self.best_epoch_psnr_ssim_lpips_log_path, dtype=str).tolist()
            self.best_epoch_psnr_ssim_lpips_log = np.array(self.best_epoch_psnr_ssim_lpips_log).astype(float)
            print(f'{os.path.basename(self.best_epoch_psnr_ssim_lpips_log_path).split(".")[0]}加载完毕~')
        if os.path.exists(self.lr_log_path):
            self.lr_log = np.loadtxt(self.lr_log_path, dtype=str).tolist()
            print(f'{os.path.basename(self.lr_log_path).split(".")[0]}加载完毕~')
        if type(self).__name__ != 'HITSIRPROGANExperiment':
            # 有可能本轮更改了初始学习率,导致上一轮 epoch 记录的本轮使用的学习率旧了,需要同步下
            self.lr_log[-1] = f"epoch:{self.start_epoch},lr:{format_str(self.optimizer.param_groups[0]['lr'], 25)}"
        if os.path.exists(self.train_eval_seconds_consume_log_path):
            self.train_eval_seconds_consume_log = np.loadtxt(
                self.train_eval_seconds_consume_log_path, dtype=str
            ).tolist()
            if type(self.train_eval_seconds_consume_log[0]) != list:
                self.train_eval_seconds_consume_log = [self.train_eval_seconds_consume_log]
            train_eval_seconds_consume_log = np.array(self.train_eval_seconds_consume_log)
            print(f'{os.path.basename(self.train_eval_seconds_consume_log_path).split(".")[0]}加载完毕~')
            for item in train_eval_seconds_consume_log:
                self.total_seconds_consume_log[0] += float(item[1].split('训练时长:')[1])
                if item[2] != 'None':
                    self.total_seconds_consume_log[0] += float(item[2].split('验证时长:')[1])
            print(
                f'过去总共训练了{self.start_epoch - 1}个epoch,训练+验证总共花费{self.total_seconds_consume_log[0]}秒~'
            )

        print('============ 加载相关指标文件 end ============')

    # 保存相关训练指标、验证指标及训练总时长(记得在调用下面函数前确保相关的指标列表变量是最新的内容)
    def __save_log(self):
        # 保存相关指标
        np.savetxt(self.train_eval_seconds_consume_log_path, self.train_eval_seconds_consume_log, fmt='%s')
        np.savetxt(self.psnr_ssim_lpips_log_path, self.psnr_ssim_lpips_log, fmt='%s')
        np.savetxt(self.best_epoch_psnr_ssim_lpips_log_path, self.best_epoch_psnr_ssim_lpips_log, fmt='%s')
        np.savetxt(self.total_seconds_consume_log_path, self.total_seconds_consume_log)

    # 保存相关测试结果
    def __save_result(self, img: torch.Tensor, path):
        self.img_transform(img.squeeze(0)).save(path)

    # 保存相关测试指标
    def __save_test_log(self, subfolder: str):
        test_psnr_ssim_lpips_log = [f'psnr:{self.test_set_psnr.avg}', f'ssim:{self.test_set_ssim.avg}', f'lpips:{self.test_set_lpips.avg}']
        elapse = time.time() - self.test_start_time
        np.savetxt(
            os.path.join(self.result_path, subfolder, 'test_log.txt'),
            [test_psnr_ssim_lpips_log, ['test_time:', elapse, ' ']],
            fmt='%s'
        )

    # 训练过程中遍历每一个 batch 的回调
    def train_batch_process(
            self,
            hr_imgs: torch.Tensor,  # (b,c,h,w)
            sr_imgs: torch.Tensor,  # (b,c,h,w)
            _: str, __: str, ___: str
    ) -> dict:
        # 清空梯度
        self.optimizer.zero_grad()
        # 计算 loss
        loss = self.loss_function(input=sr_imgs, target=hr_imgs)
        # 更新梯度
        loss.backward()
        # 更新参数
        self.optimizer.step()
        # 更新损失
        self.epoch_loss.update(loss.item(), len(hr_imgs))

    # 训练过程中遍历完每一个 dataloaders 的回调
    def train_dataloader_process(
            self,
            is_end: bool,  # 遍历的是否是最后一个 data_loader
            _: str
    ):
        if not is_end:
            return

        # 每个epoch结束更新一次学习率
        self.lr_scheduler.step()
        # 每个epoch结束记录当前学习率(注意当前学习率是用于下一epoch的训练)
        self.lr_log.append(f"epoch:{self.start_epoch + 1},lr:{format_str(self.optimizer.param_groups[0]['lr'], 25)}")
        # 每个epoch结束记录平均损失
        self.loss_log.append([f'epoch:{self.start_epoch:05d}', f'loss:{self.epoch_loss.avg}'])
        # 每个epoch结束记录训练时长
        train_time = time.time() - self.train_start_time
        self.train_eval_seconds_consume_log.append(
            [f'epoch:{self.start_epoch:05d}', format_str(f'训练时长:{train_time}', 25), 'None', 'None']
        )
        self.total_seconds_consume_log[0] += train_time

        # 每个epoch结束保存最新模型
        self.save_model_weights(model_path=self.new_model_path)

        # 保存训练指标
        if type(self).__name__ != 'HITSIRPROGANExperiment':
            np.savetxt(self.loss_log_path, self.loss_log, fmt='%s')  # 文件不存在会自动创建相应的文件
        if type(self).__name__ != 'HITSIRPROGANExperiment':
            np.savetxt(self.lr_log_path, self.lr_log, fmt='%s')
        np.savetxt(self.train_eval_seconds_consume_log_path, self.train_eval_seconds_consume_log, fmt='%s')

    # 训练函数
    def train(self):
        """
        单 epoch 训练
        """
        # 开启训练
        self.model.train()

        # 重置训练指标
        self.epoch_loss.reset()
        self.train_start_time = time.time()

        # 遍历所有的 train_loader
        self.__dataloaders_traverse(
            dataloaders=self.train_loaders,
            stage='train',
            batch_callback=self.train_batch_process,
            data_loader_callback=self.train_dataloader_process
        )

    # 验证过程中遍历每个batch的回调
    def eval_batch_process(
            self,
            hr_img: torch.Tensor,  # (1,c,h,w)
            sr_img: torch.Tensor,  # (1,c,h,w)
            _: str, __: str, ___: str
    ) -> dict:
        # 转换为 ycbcr 中的 y 通道
        hr_img_y = convert_image(
            hr_img,
            source='[0,1]',
            target='y-channel',
            is_lr=False, is_lr_amplify=False, scaling_factor=4
        ).squeeze(0)  # (1,h,w)
        sr_img_y = convert_image(
            sr_img,
            source='[0,1]',
            target='y-channel',
            is_lr=False, is_lr_amplify=False, scaling_factor=4
        ).squeeze(0)  # (1,h,w)

        hr_img_y = hr_img_y.cpu().numpy()
        sr_img_y = sr_img_y.cpu().numpy()

        # 根据 y 通道计算 psnr、ssim、lpips
        psnr = peak_signal_noise_ratio(
            hr_img_y,
            sr_img_y,
            data_range=1
        )
        ssim = structural_similarity(
            hr_img_y,
            sr_img_y,
            data_range=1,
        )
        lpips = self.lpips_fn(torch.from_numpy(hr_img_y), torch.from_numpy(sr_img_y))

        is_psnr_nan = False
        is_ssim_nan = False
        is_lpips_nan = False

        # 统计 psnr、ssim、lpips
        if not np.isnan(psnr):
            self.epoch_psnr.update(psnr, len(hr_img))
        else:
            is_psnr_nan = True
        if not np.isnan(ssim):
            self.epoch_ssim.update(ssim, len(hr_img))
        else:
            is_ssim_nan = True
        if not np.isnan(lpips.item()):
            self.epoch_lpips.update(lpips.item(), len(hr_img))
        else:
            is_lpips_nan = True

        if is_psnr_nan or is_ssim_nan or is_lpips_nan:
            print(f'出现 {"psnr " if is_psnr_nan else ""}{"ssim " if is_ssim_nan else ""}{"lpips " if is_lpips_nan else ""}为 nan')
            raise ValueError('实验出错,实验指标为 nan')

    # 验证过程中遍历完每一个data_loader的回调
    def __eval_dataloader_process(
            self,
            is_end: bool,  # 遍历的是否是最后一个 data_loader
            dataloader_name: str,
            start_epoch=None
    ):
        if not is_end:
            return

        start_epoch = start_epoch if start_epoch is not None else self.start_epoch

        if self.epoch_lpips.avg == 0:  # 说明所有验证数据的 lpips 计算都出错了
            self.epoch_lpips.avg = 1

        # 每个epoch结束时记录相关验证指标
        self.psnr_ssim_lpips_log.append([
            f'epoch:{start_epoch:05d}',
            format_str(f'{self.epoch_psnr.avg}'),
            format_str(f'{self.epoch_ssim.avg}'),
            format_str(f'{self.epoch_lpips.avg}'),
        ])
        # 是否找到更好的权重参数
        # 单 psnr 最好
        if self.epoch_psnr.avg > self.only_best_psnr:
            self.only_best_psnr = self.epoch_psnr.avg
            self.save_model_weights(model_path=self.best_psnr_model_path)
        # 单 ssim 最好
        if self.epoch_ssim.avg > self.only_best_ssim:
            self.only_best_ssim = self.epoch_ssim.avg
            self.save_model_weights(model_path=self.best_ssim_model_path)
        # 单 lpips 最好
        if self.epoch_lpips.avg < self.only_best_lpips:
            self.only_best_lpips = self.epoch_lpips.avg
            self.save_model_weights(model_path=self.best_lpips_model_path)
        # psnr、ssim、lpips 同时达到最好
        if self.epoch_psnr.avg > self.best_epoch_psnr_ssim_lpips_log[1] and self.epoch_ssim.avg > \
                self.best_epoch_psnr_ssim_lpips_log[2] and self.epoch_lpips.avg < self.best_epoch_psnr_ssim_lpips_log[3]:
            self.best_epoch_psnr_ssim_lpips_log = [
                f'{start_epoch:05d}',
                self.epoch_psnr.avg,
                self.epoch_ssim.avg,
                self.epoch_lpips.avg
            ]
            self.save_model_weights(model_path=self.best_psnr_ssim_lpips_model_path)

        # 保存验证时间
        eval_time = time.time() - self.eval_start_time
        self.train_eval_seconds_consume_log[-1][2] = format_str(f'验证时长:{eval_time}', 25)
        if str(self.train_eval_seconds_consume_log[-1][3]) == 'None':
            self.train_eval_seconds_consume_log[-1][3] = f'验证数据集:{dataloader_name}'
        else:
            self.train_eval_seconds_consume_log[-1][3] += f'、{dataloader_name}'

        self.total_seconds_consume_log[0] += eval_time
        # 验证过程中所有 data_loader 遍历完保存相关log文件
        self.__save_log()

    # 验证函数
    def eval(self, start_epoch=None):
        """
        单 epoch 验证
        """
        # 开启验证
        self.model.eval()

        # 重置验证指标
        self.epoch_psnr.reset()
        self.epoch_ssim.reset()
        self.epoch_lpips.reset()
        self.eval_start_time = time.time()

        # 遍历所有的 eval_loader
        self.__dataloaders_traverse(
            dataloaders=self.eval_loaders,
            stage='eval',
            batch_callback=self.eval_batch_process,
            data_loader_callback=lambda is_end, dataloader_name: self.__eval_dataloader_process(is_end, dataloader_name, start_epoch),
            start_epoch=start_epoch
        )

    # 测试过程中遍历每个batch的回调
    def test_batch_process(
            self,
            hr_img: torch.Tensor,  # (1,c,h,w)
            sr_img: torch.Tensor,  # (1,c,h,w)
            filename: str,
            suffix: str,
            dataloader_name: str
    ) -> dict:
        # 转换为 ycbcr 中的 y 通道
        hr_img_y = convert_image(
            hr_img,
            source='[0,1]',
            target='y-channel',
            is_test=True,
            is_lr=False, is_lr_amplify=False, scaling_factor=4
        ).squeeze(0)  # (1,h,w)
        sr_img_y = convert_image(
            sr_img,
            source='[0,1]',
            target='y-channel',
            is_test=True,
            is_lr=False, is_lr_amplify=False, scaling_factor=4
        ).squeeze(0)  # (1,h,w)

        hr_img_y = hr_img_y.cpu().numpy()
        sr_img_y = sr_img_y.cpu().numpy()

        # 根据 y 通道计算 psnr 和 ssim
        psnr = peak_signal_noise_ratio(
            hr_img_y,
            sr_img_y,
            data_range=1.
        )
        ssim = structural_similarity(
            hr_img_y,
            sr_img_y,
            data_range=1,
            gaussian_weights=True
        )
        lpips = self.lpips_fn(torch.from_numpy(hr_img_y), torch.from_numpy(sr_img_y))

        # 更新测试指标
        self.test_set_psnr.update(psnr, len(hr_img))
        self.test_set_ssim.update(ssim, len(hr_img))
        self.test_set_lpips.update(lpips.item(), len(hr_img))

        # 检查测试结果路径是否存在
        result_path = os.path.join(
            self.result_path,  # results/srcnn/best
            dataloader_name,  # Set5
        )
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        # 获取测试结果存放路径
        hr_path = os.path.join(
            result_path,
            f'{filename}_hr.{suffix}'
        )
        sr_path = os.path.join(
            result_path,
            f'{filename}_sr.{suffix}'
        )
        # 保存测试结果到磁盘
        self.__save_result(hr_img, hr_path)
        self.__save_result(sr_img, sr_path)

    # 测试过程中遍历每个data_loader前的回调
    def __test_dataloader_prev_process(self):
        # 重置测试指标
        self.test_set_psnr.reset()
        self.test_set_ssim.reset()
        self.test_set_lpips.reset()
        self.test_start_time = time.time()

    # 测试过程中遍历完每个data_loader的回调
    def __test_dataloader_process(
            self,
            _: bool,
            dataloader_name: str
    ):
        # 遍历完每个 data_loader 保存测试指标到结果文件夹
        self.__save_test_log(dataloader_name)

    # 测试函数
    def __test(self):
        # 开启测试
        self.model.eval()

        # 检查测试结果存放路径是否存在
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        for result_data_path in self.result_data_paths:
            if not os.path.exists(result_data_path):
                os.mkdir(result_data_path)

        # 遍历所有的测试_loader
        self.__dataloaders_traverse(
            dataloaders=self.test_loaders,
            stage='test',
            batch_callback=self.test_batch_process,
            data_loader_prev_callback=self.__test_dataloader_prev_process,
            data_loader_callback=self.__test_dataloader_process
        )

    # 遍历 dataloaders
    def __dataloaders_traverse(
            self,
            dataloaders: list[DataLoader],
            stage: str,
            batch_callback,
            data_loader_prev_callback=None,
            data_loader_callback=None,
            start_epoch=None  # eval 阶段用于修改 start_epoch
    ):
        """
        Args:
            dataloaders: train_loaders or eval_loaders or test_loaders
            stage: train or eval or test
            batch_callback, 遍历完每个 batch 的回调
            data_loader_prev_callback, 遍历每个 data_loader 前的回调
            data_loader_callback, 遍历完每个 data_loader 的回调
        """
        # 遍历每一个 dataloader
        for i, dataloader in enumerate(dataloaders):
            total_size = len(dataloader.dataset) - (len(dataloader.dataset) % dataloader.batch_size)
            is_end = i == len(dataloaders) - 1  # 是否是最后一个 data_loader

            # 每个 data_loader 遍历前的回调
            if data_loader_prev_callback is not None:
                data_loader_prev_callback()

            # 进度条，不要不足 batch_size 的部分
            with tqdm(total=total_size) as t:
                # 设置描述信息
                if stage == 'train':
                    t.set_description(
                        f'train_epoch {self.start_epoch}/{self.model_config.epochs}, data: {dataloader.name}'
                    )
                elif stage == 'eval':
                    t.set_description(
                        f'eval_epoch  {start_epoch if start_epoch is not None else self.start_epoch}/{self.model_config.epochs}, data: {dataloader.name}'
                    )
                elif stage == 'test':
                    t.set_description(
                        f'start test, current test data: {dataloader.name}'
                    )

                # 遍历每一个 batch
                for lr_imgs, hr_imgs, (filename, suffix) in dataloader:
                    # 获取图片名称及后缀
                    filename = filename[0]
                    suffix = suffix[0]
                    params = (filename, suffix)

                    # 数据迁移至默认设备
                    lr_imgs = lr_imgs.to(self.model_config.device)
                    hr_imgs = hr_imgs.to(self.model_config.device)

                    # 对 lr_imgs 进一步的处理(可选)
                    lr_imgs = self.process_lr_imgs(stage, lr_imgs)
                    # 对 hr_imgs 进一步的处理(可选)
                    hr_imgs = self.process_hr_imgs(stage, hr_imgs)
                    sr_imgs = self.model(lr_imgs)

                    # 验证/测试阶段才需要将输出裁剪到 [0,1]
                    if stage == 'eval' or stage == 'test':
                        # 获取模型预测(模型输出默认为 0~1)
                        sr_imgs = sr_imgs.clip(0, 1)

                    # 对 sr_imgs 进一步的处理(可选)
                    sr_imgs = self.process_sr_imgs(stage, sr_imgs)

                    # 传递数据给回调
                    batch_callback(hr_imgs, sr_imgs, *params, dataloader.name)
                    # 更新进度条进度
                    t.update(len(sr_imgs))

                # 每个 data_loader 遍历完的回调
                if data_loader_callback is not None:
                    data_loader_callback(is_end, dataloader.name)

                # 所有 data_loader 遍历完打印训练指标
                if stage == 'train' and is_end:
                    if type(self).__name__ == 'HITSIRPROGANExperiment':
                        t.set_postfix({
                            'g_loss': f'{self.epoch_loss.avg:.6f}',
                            'd_loss': f'{self.epoch_discriminator_loss.avg:.6f}'
                        })
                    else:
                        t.set_postfix({
                            'loss': f'{self.epoch_loss.avg:.6f}'
                        })
                # 所有 data_loader 遍历完打印验证指标
                elif stage == 'eval' and is_end:
                    t.set_postfix({
                        'eval psnr': f'{self.epoch_psnr.avg:.6f}',
                        'eval ssim': f'{self.epoch_ssim.avg:.6f}',
                        'eval lpips': f'{self.epoch_lpips.avg:.6f}',
                        'best epoch': f'{self.best_epoch_psnr_ssim_lpips_log[0]}',
                        'best psnr': f'{self.best_epoch_psnr_ssim_lpips_log[1]:.6f}',
                        'best ssim': f'{self.best_epoch_psnr_ssim_lpips_log[2]:.6f}',
                        'best ssim_lpips': f'{self.best_epoch_psnr_ssim_lpips_log[3]:.6f}',
                    })
                # 遍历完每个 data_loader 打印测试指标
                elif stage == 'test':
                    t.set_postfix({
                        'psnr': f'{self.test_set_psnr.avg:.6f}',
                        'ssim': f'{self.test_set_ssim.avg:.6f}',
                        'lpips': f'{self.test_set_lpips.avg:.6f}',
                    })

    def preprocess_train(self):
        ...

    def process_lr_imgs(self, stage, lr_imgs):
        """
        在 lr_imgs 输入到模型前再作一些自定义处理(主要用于 ipg 模型中)
        stage: 当前所处阶段: train|eval|test
        lr_imgs: [b,c,h,w]
        """
        return lr_imgs

    def process_hr_imgs(self, stage, hr_imgs):
        """
        对 hr_imgs 进行一些自定义处理
        stage: 当前所处阶段: train|eval|test
        hr_imgs: [b,c,h,w]
        """
        return hr_imgs

    def process_sr_imgs(self, stage, sr_imgs):
        """
        对模型输出的 sr_imgs 再作一些自定义处理(主要用于 ipg 模型中)
        stage: 当前所处阶段: train|eval|test
        sr_imgs: [b,c,h,w]
        """
        return sr_imgs

    # 运行实验
    def run(self):
        print(f'{type(self).__name__}.run...')

        # 训练/验证阶段
        if not self.is_test:
            # 判断上次训练的验证阶段是否被中断
            if self.start_epoch - 2 == self.psnr_ssim_lpips_log.__len__():
                print(f'上次训练 epoch 为 {self.start_epoch - 1}, 缺少对应的 eval 指标,故现在补充一次 eval() 以计算验证指标')
                # 手动补充一次 eval
                with torch.no_grad():
                    self.eval(start_epoch=self.start_epoch - 1)

            # 遍历每一个 epoch
            for epoch in range(self.start_epoch, self.model_config.epochs + 1):
                # 更新 start_epoch 为当前 epoch
                self.start_epoch = epoch

                # 每次训练前的准备工作
                self.preprocess_train()
                # 训练阶段
                self.train()
                # 验证阶段
                with torch.no_grad():
                    self.eval()

                # 每隔 5 个 epoch 保存一次所有指标
                if epoch % 5 == 0:
                    print('每隔5个epoch重新保存一次_start')
                    weights_path = glob.glob(self.model_config.checkpoint_folder + '/*.pth')
                    weight_save_path = self.model_config.checkpoint_folder + f'/epoch={5 if epoch == 5 else epoch - 5}'
                    if not os.path.exists(weight_save_path):
                        os.mkdir(weight_save_path)
                    new_weight_save_path = weight_save_path if epoch == 5 else self.model_config.checkpoint_folder + f'/epoch={epoch}'
                    os.rename(weight_save_path, new_weight_save_path)  # 文件夹重命名
                    for weight_path in weights_path:
                        shutil.copy(weight_path, new_weight_save_path + f'/{os.path.basename(weight_path)}')

                    logs_path = glob.glob(self.model_config.log_folder + '/*.txt')
                    log_save_path = self.model_config.log_folder + f'/epoch={5 if epoch == 5 else epoch - 5}'
                    if not os.path.exists(log_save_path):
                        os.mkdir(log_save_path)
                    new_log_save_path = log_save_path if epoch == 5 else self.model_config.log_folder + f'/epoch={epoch}'
                    os.rename(log_save_path, new_log_save_path)  # 文件夹重命名
                    for log_path in logs_path:
                        shutil.copy(log_path, new_log_save_path + f'/{os.path.basename(log_path)}')

                    print('每隔5个epoch重新保存一次_end')

            print('已完成所有训练批次~')
        else:
            # 测试阶段
            with torch.no_grad():
                self.__test()
