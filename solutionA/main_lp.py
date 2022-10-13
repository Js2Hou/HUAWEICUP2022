# 窗口机制
# 线性探测求解

import os
import numpy as np
import pandas as pd

from processes import item2stack, stack2stride_lp, stride2ban_lp
from utils import plot_from_csv, parse_ban_loc

## 超参数设定
constH, constW = 2440, 1220  # 板材大小
N1, N2 = 10, 10
dataset_idx = 1  # from 1 - 5

for dataset_idx in range(1, 5):
    output_dir = os.sep.join(['outputs', f'A{dataset_idx}', f'window({N1}x{N2})_lp'])

    ## 导入数据
    data_path = os.sep.join(['data', 'A', f'dataA{dataset_idx}.csv'])
    df = pd.read_csv(data_path)
    data = df.values

    # 将data中所有item_id从数字转成str
    for i in range(data.shape[0]):
        data[i, 0] = str(data[i, 0])

    # 计算所有产品项面积
    s = 0
    for i in range(data.shape[0]):
        s += data[i, 3] * data[i, 4]

    ## 排样优化
    # 产品项 >> 栈
    stacks, item_dt = item2stack(data, constH, constW)
    # 栈 >> 条带
    strides, stack_dt = stack2stride_lp(stacks, constH, constW, N1)
    # 条带>> 板
    bans, stride_dt = stride2ban_lp(strides, constH, constW, N2)

    ##计算利用率, 保存结果, 作图
    n_ban = bans.shape[0]
    ita = s / n_ban / constH / constW
    print(f'利用率为：{ita:.4f}')

    output_dir = output_dir + f'_{ita:.4f}'
    os.makedirs(output_dir, exist_ok=True)
    print(f'结果保存在：{output_dir}')

    # 结果保存值csv文件
    title = ['board_material', 'board_id', 'item_id', 'x', 'y', 'item_length', 'item_width']
    res = [title]
    for i in range(n_ban):
        ban_ = bans[i]
        cordt = parse_ban_loc(ban_, item_dt, stack_dt, stride_dt)
        for k, v in cordt.items():
            left, bottom, height, width = v
            item_ = item_dt[k]
            board_material = item_[1]

            tem_ = [board_material, i, k, left, bottom, height, width]
            res.append(tem_)

    csv_path = os.path.join(output_dir, 'cut_program.csv')
    res = np.array(res)
    df = pd.DataFrame(res)
    df.to_csv(csv_path, header=None, index=None)

    # 解析csv文件绘图
    plot_from_csv(csv_path, output_dir)