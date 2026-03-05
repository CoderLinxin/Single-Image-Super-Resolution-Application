import torch.cuda
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from btn_click_fun import *

window = ttk.Window(title='图像超分辨率', size=(1440, 1150), position=(200, 100), themename='yeti')
# 禁止用户拉伸窗口
window.resizable(False, False)
# 设置窗口左上角图标
# window.iconbitmap('')
window.grid_columnconfigure(0, minsize=100)  # 设置第一列最小宽度
window.grid_columnconfigure(2, minsize=400)  # 设置第三列最小宽度
window.grid_columnconfigure(5, minsize=50)  # 设置第六列最小宽度

# UI 界面相关变量
select_image_text_var = ttk.StringVar(value='未选择图像')  # 选择图像文本
save_image_text_var = ttk.StringVar(value='无')  # 保存图像文本
input_image_size_var = ttk.StringVar(value='0x0')  # 输入图像大小
output_image_size_var = ttk.StringVar(value='0x0')  # 输出图像大小
image_block_var = ttk.StringVar(value=128)  # 图像分块大小
save_image_filename_var = ttk.StringVar()  # 保存文件名
image_progress_var = ttk.IntVar(value=0)  # 当前进度
image_progress_text_var = ttk.StringVar(value='0%')  # 当前进度文本
run_time_var = ttk.StringVar(value='0')  # 运行时间

style = ttk.Style()
# 设置下拉框选择完选项时不显示背景色
style.map('TCombobox', selectbackground=[('readonly', 'transparent')], selectforeground=[('readonly', 'black')], )
# 设置单选框禁用状态下的颜色(需要用自定义颜色覆盖)
style.configure('Custom.TRadiobutton', foreground='black')
style.map('Custom.TRadiobutton', foreground=[('disabled', 'black')])

# 设置主标题
title = ttk.Label(window, text="图像超分辨率", font=("Arial Bold", 30))
title.grid(row=0, column=0, columnspan='9', ipady=50)

start_button = None
source_image_label = None
source_image_placeholder = None
output_image_label = None
output_image_placeholder = None

select_image_text = ttk.Entry(window, width=40, state='readonly', textvariable=select_image_text_var)
save_image_text = ttk.Entry(window, state='readonly', width=40, textvariable=save_image_text_var)
select_image_button = ttk.Button(
    window, text="选择图像", bootstyle='outline',
    command=select_image_button_click_handle(
        select_image_text_var, input_image_size_var, output_image_size_var,
        save_image_filename_var, image_progress_var, image_progress_text_var, run_time_var,
        save_image_text_var,
        lambda: start_button, lambda: (source_image_label, source_image_placeholder),
        lambda: (output_image_label, output_image_placeholder),
    )
)
save_image_button = ttk.Button(
    window, text="选择保存路径", bootstyle='outline',
    command=save_image_button_click_handle(save_image_text_var, select_image_text_var, lambda: start_button)
)
select_image_button.grid(row=1, column=1, pady=10, sticky='e')
select_image_text.grid(row=1, column=2, padx=20, columnspan='3')
save_image_button.grid(row=1, column=6)
save_image_text.grid(row=1, column=7, padx=20, columnspan='2')

input_image_size_text = ttk.Label(window, text='原图像大小 :')
input_image_size = ttk.Label(window, textvariable=input_image_size_var)
output_image_size_text = ttk.Label(window, text='输出图像大小 :')
output_image_size = ttk.Label(window, textvariable=output_image_size_var)
input_image_size_text.grid(row=2, column=1, pady=20, sticky='e')
input_image_size.grid(row=2, column=2, sticky='w', padx=20)
output_image_size_text.grid(row=2, column=6, sticky='e')
output_image_size.grid(row=2, column=7, sticky='w', padx=15)

image_block_text = ttk.Label(window, text='图像分块大小 :')
image_block_combobox = ttk.Combobox(state='readonly', values=[32, 64, 128, 256, 512, 1024], textvariable=image_block_var)
save_image_filename_text = ttk.Label(window, text='保存文件名称 :')
save_image_filename_entry = ttk.Entry(window, width=40, textvariable=save_image_filename_var)
image_block_text.grid(row=3, column=1, pady=10, sticky='e')
image_block_combobox.grid(row=3, column=2, sticky='ew', padx=20, columnspan='3')
save_image_filename_text.grid(row=3, column=6, sticky='e')
save_image_filename_entry.grid(row=3, column=7, columnspan='2')

image_progress_text = ttk.Label(window, text='当前进度 :')
image_progress_bar = ttk.Progressbar(bootstyle="striped", length=395, variable=image_progress_var)
image_progress_label = ttk.Label(window, textvariable=image_progress_text_var)
processor = ttk.Label(window, text='当前平台 :')
processor_select = ttk.IntVar(value=2 if torch.cuda.is_available() else 1)
processor_cpu = ttk.Radiobutton(value=1, text='cpu', state='disabled', variable=processor_select, style='Custom.TRadiobutton')
processor_gpu = ttk.Radiobutton(value=2, text='gpu', state='disabled', variable=processor_select, style='Custom.TRadiobutton')
image_progress_text.grid(row=4, column=1, pady=20, sticky='e')
image_progress_bar.grid(row=4, column=2, padx=20, pady=20, sticky='w', columnspan='3')
image_progress_label.grid(row=4, column=4, sticky='e', padx=20)
processor.grid(row=4, column=6, sticky='e')
processor_cpu.grid(row=4, column=7, sticky='w', padx=20)
processor_gpu.grid(row=4, column=8, sticky='w')

run_time_label = ttk.Label(window, text='运行时间 :')
run_time_text = ttk.Label(window, textvariable=run_time_var)
start_button = ttk.Button(
    window, width=10, text="运行", bootstyle='outline', state='disabled',
    command=start_button_click_handle(
        select_image_text_var, save_image_text_var, input_image_size_var, image_block_var,
        save_image_filename_var, image_progress_var, image_progress_text_var, image_progress_bar, run_time_var,
        select_image_button, save_image_button,
        image_block_combobox, save_image_filename_entry, lambda: start_button,
        window,
        lambda: (output_image_label, output_image_placeholder)
    )
)
run_time_label.grid(row=5, column=6, pady=10, sticky='e')
run_time_text.grid(row=5, column=7, sticky='w', padx=20)
start_button.grid(row=5, column=8, sticky='w')

source_image_placeholder = ttk.PhotoImage(file='images/placeholder.png', width=550, height=550)
source_image_label = ttk.Label(window, image=source_image_placeholder)
output_image_placeholder = ttk.PhotoImage(file='images/placeholder.png', width=550, height=550)
output_image_label = ttk.Label(window, image=output_image_placeholder)
source_image_label.grid(row=6, column=1, pady=20, columnspan='4')
output_image_label.grid(row=6, column=6, columnspan='3')

source_image_text = ttk.Label(window, text='原图', font=("Arial Bold", 16))
output_image_text = ttk.Label(window, text='超分后', font=("Arial Bold", 16))
source_image_text.grid(row=7, column=1, columnspan='4')
output_image_text.grid(row=7, column=6, columnspan='3')

# 调用 mainloop 函数，这个函数将让窗口等待用户与之交互，直到我们关闭它
window.mainloop()
