"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/10/13 10:57:37
@Description: 
"""

import os
import numpy as np
import pandas as pd

from processes import assemble_orders, item2stack, stack2stride_lp, stride2ban_lp
from utils import parse_ban_loc, plot_from_csv

## 常量限制
NUM_ITEMS, AREA_ITEMS = 1000, 250
constH, constW = 2440, 1220
N = 20  # window size for order placement
N1, N2 = 10, 10
dataset_idx = 1  # discrete value in [1, 5]

for dataset_idx in range(2, 3):
    output_dir = os.sep.join(['outputs', f'B{dataset_idx}', f'window({N1}x{N2})_lp'])

    ## 导入数据
    # item_id, item_material, item_num, item_length, item_width, item_order
    data_path = os.sep.join(['data', 'B', f'dataB{dataset_idx}.csv'])
    df = pd.read_csv(data_path)
    data = df.values

    # 将data中所有id转为str
    for i in range(data.shape[0]):
        data[i, 0] = str(data[i, 0])

    # 所有items面积
    area_items_raw = 0  
    for i in range(data.shape[0]):
        area_items_raw += data[i, 3] * data[i, 4]

    # 计算订单种类
    orders_name_set = np.unique(data[:, -1])  # 其实是numpy, 不是set类型
    n_orders = orders_name_set.shape[0]

    # 计算原材料种类
    materials_name_set = np.unique(data[:, 1])
    n_materials = materials_name_set.shape[0]

    # 建立订单信息表，np.ndarray, [order_name, num_items, area_items]
    order_info_table = []
    for _order_name in orders_name_set:
        idx_ = data[:, 5] == _order_name
        order_data = data[idx_]
        n = order_data.shape[0]
        dad = order_data[:,3: 5]
        s = np.sum(dad[:, 0] * dad[:, 1] / 1e6) 
        order_info_table.append([_order_name, n, s])

    orders = np.array(order_info_table)

    ## 将订单拼批
    # 在窗口内实行启发式搜索
    orders_placement = assemble_orders(orders, n_orders, wsize=N)
    n_batches = len(orders_placement)  # 36

    ## 排版
    # 分别对每个订单排版
    # 订单里不同材料分别排版

    batches = []
    n_bans = 0  # 所有板子的数量

    for batch_id, batch_orders_name in enumerate(orders_placement):
        # 把所有batch_orders_name数据组合在一起，按照material分类分别排版
        batch_data_ = []
        for order_name_ in batch_orders_name:
            idx_ = data[:, -1] == order_name_
            batch_data_.append(data[idx_])
        batch_data_ = np.vstack(batch_data_)  # batch内所有数据

        cur_batch_ = []
        batch_material_set_ = np.unique(batch_data_[:, 1])
        for material_name_ in batch_material_set_:
            idx_material_filter = batch_data_[:, 1] == material_name_
            batch_material_data_ = batch_data_[idx_material_filter]
            # do as problem 1
            ## 排样优化
            # 产品项 >> 栈
            stacks, item_dt = item2stack(batch_material_data_, constH, constW)
            # 栈 >> 条带
            strides, stack_dt = stack2stride_lp(stacks, constH, constW, N1)
            # 条带>> 板
            bans, stride_dt = stride2ban_lp(strides, constH, constW, N2)

            n_bans += bans.shape[0]

            # 暂存便于后面作图统计
            cur_batch_.append([material_name_, bans, item_dt, stack_dt, stride_dt])
        batches.append(cur_batch_)

    ##计算利用率, 作图, 保存结果

    # 计算利用率
    ita = area_items_raw / n_bans / constH / constW  # 利用率
    print(f'使用板材 {n_bans} 块，利用率为: {ita:.4f}')

    # 建立输出文件夹
    output_dir = output_dir + f'_{ita:.4f}'
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出文件夹: {output_dir}')

    # 结果保存值csv文件
    title = ['batch_id', 'board_material', 'board_id', 'item_id', 'x', 'y', 'item_length', 'item_width']
    res4csv = [title]
    id_rawp = -1  # 全局原片id

    # 循环保存结果
    for id_batch, batch_ in enumerate(batches):
        for id_material, material_data_ in enumerate(batch_):
            material_name_, bans, item_dt, stack_dt, stride_dt = material_data_

            for i in range(bans.shape[0]):
                id_rawp += 1  # 全局原片id
                ban_ = bans[i]
                cordt = parse_ban_loc(ban_, item_dt, stack_dt, stride_dt)

                # for csv file
                for k, v in cordt.items():
                    left, bottom, height, width = v
                    item_ = item_dt[k]
                    board_material = item_[1]

                    tem_ = [id_batch, board_material, id_rawp, k, left, bottom, height, width]
                    res4csv.append(tem_)

    csv_path = os.path.join(output_dir, 'sum_order.csv')
    res4csv = np.array(res4csv)
    df = pd.DataFrame(res4csv)
    df.to_csv(csv_path, header=None, index=None)

    # 作图
    plot_from_csv(csv_path, output_dir)
