import math
import time
import tkinter
from tkinter import filedialog
import os
from load_model import *
from image_block_handle import *
from PIL import Image, ImageTk
from torchvision import transforms

to_tensor_transform = transforms.ToTensor()
to_pil_transform = transforms.ToPILImage()


# 按钮点击处理函数的装饰类(主要用于传递参数)
def button_click_handle_decorator(fun):
    def wrapper(*args, **kwargs):
        def wrapper2():
            fun(*args, **kwargs)

        return wrapper2

    return wrapper


# 选择图像按钮点击处理函数
@button_click_handle_decorator
def select_image_button_click_handle(
        select_image_text_var: tkinter.StringVar,
        input_image_size_var: tkinter.StringVar,
        output_image_size_var: tkinter.StringVar,
        save_image_filename_var: tkinter.StringVar,
        image_progress_var: tkinter.StringVar,
        image_progress_text_var: tkinter.StringVar,
        run_time_var: tkinter.StringVar,
        save_image_text_var: tkinter.StringVar,
        get_start_button,
        get_source_image_label_and_placeholder,
        get_output_image_label_and_placeholder,
):
    """
    :param select_image_text_var: 选择图像文本字符串变量
    :param input_image_size_var: 输入图像大小文本字符串变量
    :param output_image_size_var: 输出图像大小文本字符串变量
    :param save_image_filename_var: 保存文件名文本字符串变量
    :param image_progress_var: 当前进度
    :param image_progress_text_var: 当前进度文本字符串变量
    :param run_time_var: 运行时间文本字符串变量
    :param save_image_text_var: 保存图像文件夹路径文本字符串变量
    :param get_start_button: 获取开始运行按钮的函数
    :param get_source_image_label: 获取显示输入图像的 label 以及占位图
    :param get_output_image_label: 获取显示输出图像的 label 以及占位图
    """
    # 选择一个文件
    filetypes = [('PNG图片', '*.png'), ('JPEG图片', '*.jpeg'), ('JPG图片', '*.jpg')]
    file_path = filedialog.askopenfilename(title='选择图像', filetypes=filetypes)
    if file_path != '':
        img = None
        with Image.open(file_path, mode='r') as img_open:
            img = img_open.convert('RGB')

        # 更新选择的图像文件路径
        select_image_text_var.set(file_path)
        # 更新选择的图像大小
        input_image_size_var.set(f'{img.width}x{img.height}')
        # 更新输出的图像大小
        output_image_size_var.set(f'{img.width * 4}x{img.height * 4}')
        # 更新默认保存的图像文件名称
        file_name = os.path.basename(file_path)
        file_info = file_name.split('.')
        save_image_filename_var.set(f'{file_info[0]}_srx4.{file_info[1]}')
        # 重置当前进度
        image_progress_var.set(0)
        image_progress_text_var.set('0%')
        # 更新输入图像的显示
        source_image_label, _ = get_source_image_label_and_placeholder()
        global source_image  # 防止被垃圾回收
        source_image = ImageTk.PhotoImage(img.resize((550, 550), Image.BICUBIC))
        source_image_label.config(image=source_image)
        # 重置输出图像的显示
        output_image_label, output_image_placeholder = get_output_image_label_and_placeholder()
        output_image_label.config(image=output_image_placeholder)
        # 重置运行时间
        run_time_var.set('0')

        # 如果已选定结果图像保存文件夹,则启用开始运行按钮
        if save_image_text_var.get() != '无':
            get_start_button()['state'] = 'normal'

        del img


# 保存图像按钮点击处理函数
@button_click_handle_decorator
def save_image_button_click_handle(
        save_image_text_var: tkinter.StringVar,
        select_image_text_var: tkinter.StringVar,
        get_start_button
):
    """
    :param save_image_text_var: 保存图像文件夹路径文本字符串变量
    :param select_image_text_var: 选择图像文本字符串变量
    :param get_start_button: 获取开始运行按钮的函数
    """
    # 选择一个文件夹
    directory_path = filedialog.askdirectory(title='选择保存路径')
    if directory_path != '':
        # 更新保存的图像文件夹路径
        save_image_text_var.set(directory_path)

        # 如果已选定输入图像,则启用开始运行按钮
        if select_image_text_var.get() != '未选择图像':
            get_start_button()['state'] = 'normal'


# 开始运行按钮点击处理函数
@button_click_handle_decorator
def start_button_click_handle(
        select_image_text_var: tkinter.StringVar,
        save_image_text_var: tkinter.StringVar,
        input_image_size_var: tkinter.StringVar,
        image_block_var: tkinter.StringVar,
        save_image_filename_var: tkinter.StringVar,
        image_progress_var: tkinter.StringVar,
        image_progress_text_var: tkinter.StringVar,
        image_progress_bar,
        run_time_var: tkinter.StringVar,
        select_image_button: tkinter.Button,
        save_image_button: tkinter.Button,
        image_block_combobox,
        save_image_filename_entry: tkinter.Entry,
        get_start_button,
        window,
        get_output_image_label_and_placeholder
):
    """
    :param select_image_text_var: 输入图像路径
    :param save_image_text_var: 输出图像文件夹
    :param input_image_size_var: 输入图像大小
    :param image_block_var: 图像分块大小
    :param save_image_filename_var: 输出图像文件名称
    :param image_progress_var: 当前进度
    :param image_progress_text_var: 当前进度文本控件
    :param image_progress_bar: 进度条控件
    :param run_time_var: 运行时间
    :param select_image_button: 选择图像按钮
    :param save_image_button: 选择保存路径按钮
    :param image_block_combobox: 选择图像分块下拉框
    :param save_image_filename_entry: 保存图像文件名称控件
    :param get_start_button: 获取开始运行按钮的函数
    :param window: 窗口程序实例
    :param get_output_image_label_and_placeholder: 获取显示输出图像的 label 以及占位图
    """

    # 运行前的相关检查
    # 1.输入图像大小 >= 32x32
    input_image_size = input_image_size_var.get().split('x')
    if int(input_image_size[0]) < 32 or int(input_image_size[1]) < 32:
        # 弹出提示框
        tkinter.messagebox.showerror("输入图像错误", f"输入图像大小 {input_image_size_var.get()} < 32x32")
        return

    # 禁用交互
    select_image_button['state'] = 'disabled'
    save_image_button['state'] = 'disabled'
    image_block_combobox['state'] = 'disabled'
    save_image_filename_entry['state'] = 'disabled'
    get_start_button()['state'] = 'disabled'

    time_start = time.time()
    # 进度重置
    image_progress_var.set(0), image_progress_text_var.set(f'0%'), window.update()
    # 重置输出图像
    output_image_label, output_image_placeholder = get_output_image_label_and_placeholder()
    output_image_label.config(image=output_image_placeholder)

    # 加载 lr 图像并转换为 tensor 格式
    lr_img = None
    with Image.open(select_image_text_var.get(), mode='r') as img_open:
        lr_img = img_open.convert('RGB')
    lr_img = to_tensor_transform(lr_img).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # 执行图像超分
    sr_img, block_count = image_block_handle(
        lr_img, inference,
        lambda block_count: before_progress(image_progress_bar, block_count),  # 更新进度条总长度
        lambda block_count, current_block_index: update_progress(block_count, current_block_index, image_progress_var, image_progress_text_var, window),  # 更新进度
        int(image_block_var.get()), tile_overlap=16
    )
    # 进度置1
    image_progress_var.set(block_count), image_progress_text_var.set(f'100%')
    # 转换为 pil_image
    sr_img: PIL.Image = to_pil_transform(sr_img.squeeze())
    # 保存文件到磁盘
    save_path = os.path.join(save_image_text_var.get(), save_image_filename_var.get())
    sr_img.save(save_path)

    # 显示输出图像
    global output_image  # 防止被垃圾回收
    output_image = ImageTk.PhotoImage(sr_img.resize((550, 550), Image.BICUBIC))
    output_image_label.config(image=output_image)

    # 计算总执行时间
    run_time_var.set(f'{time.time() - time_start}s')

    # 重新启用交互
    select_image_button['state'] = 'normal'
    save_image_button['state'] = 'normal'
    image_block_combobox['state'] = 'normal'
    save_image_filename_entry['state'] = 'normal'
    get_start_button()['state'] = 'normal'


# 更新进度前的处理函数
def before_progress(image_progress_bar, block_count):
    """
    :param image_progress_bar: 进度条控件
    :param block_count:: 进度条长度
    """

    # 更新进度条长度
    image_progress_bar['maximum'] = block_count


# 更新进度的处理函数
def update_progress(
        block_count: int,
        current_block_index: int,
        image_progress_var: tkinter.IntVar,
        image_progress_text_var: tkinter.StringVar,
        window,
):
    """
    :param block_count: 分块总数
    :param current_block_index: 当前处理完毕的分块索引
    :param image_progress_var: 当前进度
    :param image_progress_text_var: 当前进度文本
    :param window: 窗口程序实例
    """

    # 更新进度
    image_progress_var.set(image_progress_var.get() + 1)
    # 更新进度文本
    image_progress_text_var.set(f'{math.ceil((current_block_index + 1) / block_count * 100)}%')
    # 刷新窗口
    window.update()
