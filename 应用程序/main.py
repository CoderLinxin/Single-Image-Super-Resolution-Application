import torch.cuda
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from btn_click_fun import *

window = ttk.Window(title='图像超分辨率', size=(1440, 1150), themename='yeti')
# 禁止用户拉伸窗口
window.resizable(False, False)
# 设置窗口左上角图标
# window.iconbitmap('')
window.grid_columnconfigure(0, minsize=100)  # 设置第一列最小宽度
window.grid_columnconfigure(2, minsize=400)  # 设置第三列最小宽度
window.grid_columnconfigure(5, minsize=50)  # 设置第六列最小宽度

# 设置主标题
title = ttk.Label(window, text="图像超分辨率", font=("Arial Bold", 30))
title.grid(row=0, column=0, columnspan='9', ipady=50)

select_image_text_var = ttk.StringVar()
select_image_text_var.set('未选择图像')
select_image_text = ttk.Entry(window, width=40, state='readonly', textvariable=select_image_text_var)  # 选择图像文本
save_image_text_var = ttk.StringVar()
save_image_text_var.set('无')
save_image_text = ttk.Entry(window, state='readonly', width=40, textvariable=save_image_text_var)  # 保存图像文本
select_image_button = ttk.Button(window, text="选择图像", bootstyle='outline', command=select_image_button_click_handle(select_image_text_var))  # 选择图像按钮
save_image_button = ttk.Button(window, text="选择保存路径", bootstyle='outline', command=save_image_button_click_handle(save_image_text_var))  # 保存图像按钮
select_image_button.grid(row=1, column=1, pady=10, sticky='e')
select_image_text.grid(row=1, column=2, padx=20, columnspan='3')
save_image_button.grid(row=1, column=6)
save_image_text.grid(row=1, column=7, padx=20, columnspan='2')

input_image_size_text = ttk.Label(window, text='原图像大小 :')  # 输入图像大小
input_image_size = ttk.Label(window, text='256x256')
output_image_size_text = ttk.Label(window, text='输出图像大小 :')  # 输出图像大小
output_image_size = ttk.Label(window, text='1024x1024')  # 输出图像大小
input_image_size_text.grid(row=2, column=1, pady=20, sticky='e')
input_image_size.grid(row=2, column=2, sticky='w', padx=20)
output_image_size_text.grid(row=2, column=6, sticky='e')
output_image_size.grid(row=2, column=7, sticky='w', padx=15)

image_block_text = ttk.Label(window, text='图像分块大小 :')  # 图像分块文本
image_block_scale = ttk.Scale()  # 图像分块调节旋钮
image_block = ttk.Label(window, text='256')  # 图像分块
save_image_filename_text = ttk.Label(window, text='保存文件名称 :')
save_image_filename = ttk.Entry(window, width=40)
image_block_text.grid(row=3, column=1, pady=10, sticky='e')
image_block_scale.grid(row=3, column=2, sticky='ew', padx=20, columnspan='2')
image_block.grid(row=3, column=4, sticky='e', padx=20)
save_image_filename_text.grid(row=3, column=6, sticky='e')
save_image_filename.grid(row=3, column=7, columnspan='2')

image_progress_text = ttk.Label(window, text='当前进度 :')
image_progress_bar = ttk.Progressbar(bootstyle="striped", value=50)
image_progress = ttk.Label(window, text='50%')  # 图像分块
processor = ttk.Label(window, text='当前平台 :')
processor_select = ttk.IntVar(value=2 if torch.cuda.is_available() else 1)
processor_cpu = ttk.Radiobutton(value=1, text='cpu', state='disable', variable=processor_select)
processor_gpu = ttk.Radiobutton(value=2, text='gpu', state='disable', variable=processor_select)
image_progress_text.grid(row=4, column=1, pady=20, sticky='e')
image_progress_bar.grid(row=4, column=2, padx=20, pady=20, sticky='ew', columnspan='2')
image_progress.grid(row=4, column=4, sticky='e', padx=20)
processor.grid(row=4, column=6, sticky='e')
processor_cpu.grid(row=4, column=7, sticky='w', padx=20)
processor_gpu.grid(row=4, column=8, sticky='w')

lpips_text = ttk.Label(window, text='超分结果 lpips :')
lpips_value = ttk.Label(window, text='0')
start_button = ttk.Button(window, width=10, text="运行", bootstyle='outline')  # 运行超分模型按钮
lpips_text.grid(row=5, column=6, pady=10, sticky='e')
lpips_value.grid(row=5, column=7, sticky='w', padx=20)
start_button.grid(row=5, column=8, sticky='w')

source_image = ttk.PhotoImage(file='images/placeholder.png', width=550, height=550)
source_image_label = ttk.Label(window, image=source_image)
output_image = ttk.PhotoImage(file='images/placeholder.png', width=550, height=550)
output_image_label = ttk.Label(window, image=output_image)
source_image_label.grid(row=6, column=1, pady=20, columnspan='4')
output_image_label.grid(row=6, column=6, columnspan='3')

source_image_text = ttk.Label(window, text='原图', font=("Arial Bold", 16))
output_image_text = ttk.Label(window, text='超分后', font=("Arial Bold", 16))
source_image_text.grid(row=7, column=1, columnspan='4')
output_image_text.grid(row=7, column=6, columnspan='3')

# 调用 mainloop 函数，这个函数将让窗口等待用户与之交互，直到我们关闭它
window.mainloop()
