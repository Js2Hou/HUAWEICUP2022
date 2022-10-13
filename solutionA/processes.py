"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/10/12 21:47:01
@Description: 

我们将排版过程拆分为产品项拼成栈、栈拼成条、条拼成板三个过程，本文件提供这三个过程的代码实现，
其中后两个过程有两个实现方式：线性探测法linear probe和遗传算法genetic algorithm。

- [x] item2stack
- [x] stack2stride (lp / ga)
- [x] stride2ban (lp / ga)
"""

import os
import numpy as np

from optimization import stack2stripe_solver, stripe2ban_solver


def item2stack(data, constH, constW):
    """assemble items into stacks along w

    Returns:
        stacks: (N, 6)

    """
    data = data.copy()

    # 保存item字典
    item_dt = {}
    for e in data:
        item_dt[str(e[0])] = e

    # 沿着w降序排列
    index = np.argsort(data[:, 4])
    data = data[index[::-1]]

    # 开始拼装item, ','分隔
    lst_ = []
    p = data[0].copy()
    for i in range(1, data.shape[0]):
        if p[4] == data[i, 4] and p[3] + data[i, 3] <= constH:  # 宽度相等，长度和小于constH
            p[0] = str(p[0]) + '-' + str(data[i, 0])
            p[3] = p[3] + data[i, 3]
        else:
            lst_.append(p)
            p = data[i].copy()

    lst_.append(p)
    stacks = np.array(lst_)
    return stacks, item_dt


def stack2stride_lp(stacks, constH, constW, wsize):
    # linear probe
    stacks = stacks.copy()

    # 保存stack字典
    stack_dt = {}
    for e in stacks:
        stack_dt[str(e[0])] = e

    # 沿着h降序排列
    index = np.argsort(stacks[:, 3])
    sorted_stacks = stacks[index[::-1]]

    # 开始拼装stack, ','分隔
    lst_ = []
    slide_space_ = []
    n_stacks = sorted_stacks.shape[0]
    p = sorted_stacks[0].copy()
    for i in range(1, n_stacks):
        if i in slide_space_:
            slide_space_.remove(i)
            continue
        if p[4] + sorted_stacks[i, 4] <= constW:  # 栈宽度和小于等于constW
            p[0] = str(p[0]) + ',' + str(sorted_stacks[i, 0])
            p[4] = p[4] + sorted_stacks[i, 4]
        else:  # 向后探测 N - 1步
            cur_window_size = wsize
            for j in range(i + 1, n_stacks):
                if not cur_window_size > 0:
                    break
                if j in slide_space_:
                    continue
                cur_window_size = cur_window_size - 1
                if p[4] + sorted_stacks[j, 4] <= constW:
                    slide_space_.append(j)
                    p[0] = str(p[0]) + ',' + str(sorted_stacks[j, 0])
                    p[4] = p[4] + sorted_stacks[j, 4]
            lst_.append(p)
            p = sorted_stacks[i].copy()

    lst_.append(p)
    strides = np.array(lst_)

    return strides, stack_dt


def stride2ban_lp(strides, constH, constW, wsize):
    # linear probe
    strides = strides.copy()

    # 保存stride字典
    stride_dt = {}
    for e in strides:
        stride_dt[str(e[0])] = e

    # 沿着h随机排序
    sorted_strides = strides.copy()
    np.random.shuffle(sorted_strides)

    # 开始拼接ban, '||'分隔
    lst_ = []
    slide_space_ = []  # 存放探测时已经用过的index
    n_strides = sorted_strides.shape[0]
    p = sorted_strides[0].copy()
    for i in range(1, n_strides):
        if i in slide_space_:
            slide_space_.remove(i)
            continue
        if p[3] + sorted_strides[i, 3] <= constH:  # 条长度和小于等于constH
            p[0] = str(p[0]) + '||' + str(sorted_strides[i, 0])
            p[3] += sorted_strides[i, 3]
        else:  # 向后探测 wsize - 1步
            cur_window_size = wsize
            for j in range(i + 1, n_strides):
                if not cur_window_size > 0:
                    break
                if j in slide_space_:
                    continue
                cur_window_size = cur_window_size - 1
                if p[3] + sorted_strides[j, 3] <= constH:  # 条长度和小于等于constH
                    slide_space_.append(j)
                    p[0] = str(p[0]) + '||' + str(sorted_strides[j, 0])
                    p[3] += sorted_strides[j, 3]

            lst_.append(p)
            p = sorted_strides[i].copy()

    lst_.append(p)  # 最后一个条带单独装箱
    bans = np.array(lst_)

    return bans, stride_dt


def stack2stride_ga(stacks, constH, constW, wsize):
    # 局部窗口机制的遗传算法
    stacks = stacks.copy()

    # 保存stack字典
    stack_dt = {}
    for e in stacks:
        stack_dt[str(e[0])] = e

    # 沿着h降序排列
    index = np.argsort(stacks[:, 3])
    sorted_stacks = stacks[index[::-1]]

    lst_ = []

    n_stacks = sorted_stacks.shape[0]
    rest_idx_stacks = list(range(n_stacks))  # unselected idx of stacks
    # set_stack_selected = set()

    while rest_idx_stacks:
        # idx_stacks_window_同rest_idx_stacks，为后者的子集
        idx_stacks_window_ = rest_idx_stacks[:wsize]
        x_ = sorted_stacks[idx_stacks_window_, 3:5].astype(np.float32)

        # res: (1, N1), 仅包括0或者1, 1表示被用掉
        res = stack2stripe_solver(x_).flatten()
        res = np.where(res.flatten() != 0)[0]  # 窗口内相对索引

        # 更新rest_idx_stacks
        p = None
        for widx_ in res:
            # widx_为窗口内被挑中的栈的相对索引
            # idx_stacks_window_[widx_]获取真实索引
            cur_stack_ = sorted_stacks[idx_stacks_window_[widx_]]

            if p is None:
                p = cur_stack_.copy()
            else:
                p[0] = str(p[0]) + ',' + str(cur_stack_[0])
                p[4] = p[4] + cur_stack_[4]

        for widx_ in res[::-1]:
            # or rest_idx_stacks.remove(idx_stacks_window_[widx_])
            rest_idx_stacks.pop(widx_)
        lst_.append(p)

    strides = np.array(lst_)

    return strides, stack_dt


def stride2ban_ga(strides, constH, constW, wsize):
    strides = strides.copy()

    # 保存stride字典
    stride_dt = {}
    for e in strides:
        stride_dt[str(e[0])] = e

    # 沿着h随机排序
    sorted_strides = strides.copy()
    np.random.shuffle(sorted_strides)

    # 开始拼接ban, '||'分隔
    lst_ = []
    n_strides = sorted_strides.shape[0]
    rest_idx_strides = list(range(n_strides))  # unselected idx of stacks

    while rest_idx_strides:
        # idx_stacks_window_同rest_idx_stacks，为后者的子集
        idx_stacks_window_ = rest_idx_strides[:wsize]
        x_ = sorted_strides[idx_stacks_window_, 3:5].astype(np.float32)

        # res: (1, N2), 仅包括0或者1, 1表示被用掉
        res = stripe2ban_solver(x_).flatten()
        res = np.where(res.flatten() != 0)[0]  # 窗口内相对索引

        # 更新rest_idx_stacks
        p = None
        for widx_ in res:
            # widx_为窗口内被挑中的栈的相对索引
            # idx_stacks_window_[widx_]获取真实索引
            cur_stride_ = sorted_strides[idx_stacks_window_[widx_]]
            if p is None:
                p = cur_stride_.copy()
            else:
                p[0] = str(p[0]) + '||' + str(cur_stride_[0])
                p[3] = p[3] + cur_stride_[3]

        for widx_ in res[::-1]:
            rest_idx_strides.pop(widx_)
        lst_.append(p)

    bans = np.array(lst_)

    return bans, stride_dt
