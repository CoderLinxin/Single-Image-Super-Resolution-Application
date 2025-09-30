import os
import torch
from torch.nn import MSELoss, L1Loss
from utils.utils import CharbonnierLoss

optimizers = ['Adam']
loss_functions = ['mse', 'l1', 'charbonnier']


# 获取优化器
def get_optimizer(
        optimizer_name: optimizers,
        model: torch.nn.Module,
        lr: float,
        kwarg: dict = None,
        params=None
):
    optimizer = None
    params = model.parameters() if params is None else params
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr, **kwarg) if kwarg is not None else torch.optim.Adam(params, lr=lr)

    return optimizer


# 获取损失函数
def get_loss_function(
        loss_function_name: loss_functions,
        device: torch.device,
):
    print(f'loss_function_name: {loss_function_name}')

    if loss_function_name == 'mse':
        return MSELoss(reduction='mean').to(device)
    elif loss_function_name == 'l1':
        return L1Loss(reduction='mean').to(device)
    elif loss_function_name == 'charbonnier':
        return CharbonnierLoss().to(device)


# 获取学习率调整器(统一使用余弦退火学习率调整器)
def get_scheduler(
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min=0.0,
        last_epoch=-1,
):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=T_max,
        eta_min=eta_min,
        last_epoch=last_epoch
    )


class ModelConfig:
    def __init__(
            self,
            batch_size: int,
            learning_rate: float,
            min_learning_rate: float,
            optimizer: str,
            optimizer_params: dict,
            loss_function: str,
            epochs: int,
            checkpoint_folder: str,
            test_model_path: str,
            result_folder: str,
            log_folder: str,
            train_data_folder: str,
            train_data_name_list: list[str],
            eval_data_folder: str,
            eval_data_name_list: list[str],
            test_data_folder: str,
            test_data_name_list: list[str],
    ):
        """
        :param batch_size: 批大小
        :param learning_rate: 初始学习率
        :param min_learning_rate: 最小学习率
        :param optimizer: 优化器
        :param optimizer_params: 优化器参数(对应优化器构造函数的kwarg)
        :param loss_function: 损失函数
        :param epochs: 训练 epoch 总数
        :param checkpoint_folder: 训练过程中保存模型的文件夹
        :param test_model_path: 测试的模型路径
        :param result_folder: 存放测试结果的文件夹
        :param log_folder: 存放训练过程中打印指标的文件夹
        :param train_data_folder: 训练数据集文件夹
        :param train_data_name_list: 训练数据集名称列表
        :param eval_data_folder: 验证数据集文件夹
        :param eval_data_name_list: 验证数据集名称列表
        :param test_data_folder: 测试数据集文件夹
        :param test_data_name_list: 测试数据集名称列表
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss_function = loss_function
        self.epochs = epochs
        self.checkpoint_folder = checkpoint_folder
        self.test_model_path = test_model_path
        self.result_folder = result_folder
        self.log_folder = log_folder
        self.train_data_folder = train_data_folder
        self.train_data_name_list = train_data_name_list
        self.eval_data_folder = eval_data_folder
        self.eval_data_name_list = eval_data_name_list
        self.test_data_folder = test_data_folder
        self.test_data_name_list = test_data_name_list

        self.device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        assert self.optimizer in optimizers, \
            f'optimizer must be in {optimizers}'
        assert self.loss_function in loss_functions, \
            f'loss_function must be in {loss_functions}'

        # 检查相关文件夹是否存在,如果不存在则创建
        if self.checkpoint_folder is not None and not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        if self.result_folder is not None and not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        if self.log_folder is not None and not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)

        # for train_data_name in self.train_data_name_list:
        #     assert train_data_name in ['DIV2K_train_HR'], f'unsupported train_data_name {train_data_name}'
        # for eval_data_name in self.eval_data_name_list:
        #     assert eval_data_name in ['DIV2K_valid_HR'], f'unsupported eval_data_name {eval_data_name}'
        # for test_data_name in self.test_data_name_list:
        #     assert test_data_name in ['BSD100', 'display_example1', 'display_example2', 'display_example3', 'Set5',
        #                               'Set14', 'Urban100'], f'unsupported test_data_name {test_data_name}'

        assert self.train_data_name_list is not None and len(self.train_data_name_list) > 0, \
            'train_data_name_list must not be None or len(train_data_name_list) must be > 0'
        assert self.eval_data_name_list is not None and len(self.eval_data_name_list) > 0, \
            'eval_data_name_list must not be None or len(eval_data_name_list) must be > 0'
        assert self.test_data_name_list is not None and len(self.test_data_name_list) > 0, \
            'test_data_name_list must not be None or len(test_data_name_list) must be > 0'

        # 获取数据集路径
        self.train_data_path_list = []
        self.eval_data_path_list = []
        self.test_data_path_list = []
        for train_data_name in self.train_data_name_list:
            self.train_data_path_list.append(os.path.join(self.train_data_folder, train_data_name))
        for eval_data_name in self.eval_data_name_list:
            self.eval_data_path_list.append(os.path.join(self.eval_data_folder, eval_data_name))
        for test_data_name in self.test_data_name_list:
            self.test_data_path_list.append(os.path.join(self.test_data_folder, test_data_name))
