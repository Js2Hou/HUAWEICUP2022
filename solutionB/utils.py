"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/10/12 21:53:31
@Description: 

一些工具函数。
"""

import os
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import pandas as pd


def filter_data_by_material(data, materials):
    res = []
    for e in data:
        if e[1] in materials:
            res.append(e)
    return np.array(res)


def parse_ban_loc(ban, item_dt, stack_dt, stride_dt):
    """parse items' loc in ban

    Returns:
        Dict, key is item id, value is item loc
    """
    res = {}

    strides_ids = ban[0].split('||')
    stride_loc = np.zeros(2)
    for i in range(len(strides_ids)):
        stride_id = strides_ids[i]
        # 计算当前stride左下角坐标
        if i > 0:
            stride_loc[0] += stride_dt[strides_ids[i - 1]][3]

        stacks_ids = stride_id.split(',')
        stack_loc = stride_loc.copy()  # stack从stride左下角开始排布
        for j in range(len(stacks_ids)):
            stack_id = stacks_ids[j]
            # 计算当前stack左下角坐标
            if j > 0:
                stack_loc[1] += stack_dt[stacks_ids[j - 1]][4]

            items_ids = stack_id.split('-')
            item_loc = stack_loc.copy()
            for k in range(len(items_ids)):
                item_id = items_ids[k]
                if k > 0:
                    item_loc[0] += item_dt[items_ids[k - 1]][3]

                tem = np.hstack(
                    (item_loc.copy(), item_dt[item_id][3:5].copy()))
                res[item_id] = tem
    return res


def plot_from_csv(csv_path, output_dir):
    constH, constW = 2440, 1220

    df = pd.read_csv(csv_path)
    data = df.values
    batch_ids_set = np.unique(data[:, 0])

    for batch_id in batch_ids_set:
        batch_data_ = data[data[:, 0] == batch_id]
        material_ids_set_ = np.unique(batch_data_[:, 1])

        for material_id in material_ids_set_:
            batch_material_data_ = batch_data_[batch_data_[:, 1] == material_id]
            board_ids_set_ = np.unique(batch_material_data_[:, 2])
            
            cur_output_dir = os.path.join(
                output_dir, f'batch_{batch_id}', str(material_id))
            os.makedirs(cur_output_dir, exist_ok=True)

            for board_id in board_ids_set_:
                plt.figure(figsize=(16, 8))
                plt.xlim(xmin=0, xmax=constH)
                plt.ylim(ymin=0, ymax=constW)
                plt.xticks([])
                plt.yticks([])

                bwith = 2  # 边框宽度设置为2
                ax = plt.gca()  # 获取边框
                ax.spines['bottom'].set_linewidth(bwith)
                ax.spines['left'].set_linewidth(bwith)
                ax.spines['top'].set_linewidth(bwith)
                ax.spines['right'].set_linewidth(bwith)

                s_ = 0  # 板材内item项总面积
                cur_data = batch_material_data_[batch_material_data_[:, 2] == board_id]
                for _item in cur_data:
                    item_id, left, bottom, width, height = _item[3:]
                    assert left + width <= constH and bottom + \
                        height <= constW, print('size error')

                    s_ += width * height
                    rect = patches.Rectangle((left, bottom), width, height, alpha=1,
                                             edgecolor='#000000', facecolor='#ffffff', fill=True)
                    ax.add_patch(rect)
                    plt.text(left + width / 2 - 12, bottom + height /
                             2 - 12, str(item_id), fontsize=12, color="red")
                # 计算利用率，保存图片
                ita_ = s_ / constH / constW
                save_path_ = os.path.join(cur_output_dir, f'{board_id}_{ita_:.4f}.jpg')
                plt.savefig(save_path_,  bbox_inches='tight', dpi=100)
                plt.close()
