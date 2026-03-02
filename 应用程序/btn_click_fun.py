import tkinter
from tkinter import filedialog


# 按钮点击处理函数的装饰类(主要用于传递参数)
def button_click_handle_decorator(fun):
    def wrapper(*args, **kwargs):
        def wrapper2():
            fun(*args, **kwargs)

        return wrapper2

    return wrapper


# 选择图像按钮点击处理函数
@button_click_handle_decorator
def select_image_button_click_handle(text_var: tkinter.StringVar):
    """
    :param text_var: 需要对应更新显示的文本组件所绑定的字符串变量
    :return:
    """
    # 选择一个文件
    filetypes = [('PNG图片', '*.png'), ('JPEG图片', '*.jpeg'), ('JPG图片', '*.jpg')]
    file_path = filedialog.askopenfilename(title='选择图像', filetypes=filetypes)
    # 将选择的文件路径设置到 text_widget 上
    if file_path != '':
        text_var.set(file_path)


# 保存图像按钮点击处理函数
@button_click_handle_decorator
def save_image_button_click_handle(text_var: tkinter.StringVar):
    """
    :param text_var: 需要对应更新显示的文本组件所绑定的字符串变量
    :return:
    """
    # 选择一个文件夹
    directory_path = filedialog.askdirectory(title='选择保存路径')
    if directory_path != '':
        text_var.set(directory_path)
