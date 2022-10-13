"""
@Author: Js2Hou 
@github: https://github.com/Js2Hou 
@Time: 2022/10/12 21:45:38
@Description: 

采用遗传算法解决优化问题，包括将栈拼成条，和将条拼成板两个优化过程。
"""

import os
import numpy as np
import geatpy as ea


class PStack2Stripe(ea.Problem):
    """opmiziation probrom of stack to stripe
    """

    def __init__(self, x):
        # x: (N, 2), [[h1,w1],[h2,w2],...,[hn,wn]]
        # 存储所有栈的长和宽
        assert isinstance(x, np.ndarray)
        if len(x.shape) == 1:
            x = x[np.newaxis]

        name = 'PStack2Stripe'
        dim_obj = 1
        maxormins = [-1]  # -1最大化该目标
        dim_vars = x.shape[0]
        varTypes = [1] * dim_vars
        lb = [0] * dim_vars
        ub = [1] * dim_vars
        lbin = [1] * dim_vars
        ubin = [1] * dim_vars
        self.x = x.astype(np.float32)
        self.constH, self.constW = 2440, 1220

        self.flag = 0
        ea.Problem.__init__(self, name, dim_obj, maxormins,
                            dim_vars, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        # 初始化种群
        if self.flag == 0:
            pop.Phen = np.zeros_like(pop.Phen)
            pop.Phen[:, :3] = 1
            self.flag += 1
        Vars = pop.Phen  # n_popu, n_stack

        s_ = np.sum(Vars * self.x[:, 0] * self.x[:, 1], axis=1)  # 所有栈面积
        ss_ = np.max(Vars * self.x[:, 0], axis=1) * \
            self.constW + 1e-3  # 最长的栈的长度*板材宽度

        pop.ObjV = s_[:, np.newaxis] / ss_[:, np.newaxis]  # 条带面积利用率

        # 采用可行性法则处理约束
        constraint1 = np.sum(
            Vars * self.x[:, 1], axis=1) - self.constW  # n_popu,
        constraint2 = np.sum(Vars, axis=1) - 10  # n_popu,
        pop.CV = np.vstack([constraint1, constraint2]
                           ).T  # (n_popu, n_constraint)


def stack2stripe_solver(x, output_dir=None):
    problem = PStack2Stripe(x)

    encoding = 'RI'
    NIND = 50  # 种群规模
    field = ea.crtfld(encoding, problem.varTypes,
                      problem.ranges, problem.borders)
    population = ea.Population(encoding, field, NIND)

    # 算法参数设置
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)
    myAlgorithm.MAXGEN = 500  # 最大进化代数
    myAlgorithm.mutOper.F = 0.7  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = 0.7  # 重组概率
    myAlgorithm.logTras = 1  # 每隔多少代记录日志，设置成0表示不记录日志
    myAlgorithm.verbose = False
    myAlgorithm.drawing = 0  # 0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画
    myAlgorithm.dirName = output_dir if output_dir else 'tmp'

    res = ea.optimize(myAlgorithm, verbose=False,
                      drawing=0,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    return res['Vars']


class PStripe2Ban(ea.Problem):
    """optimization of stripe to ban
    """

    def __init__(self, x):
        assert isinstance(x, np.ndarray)
        if len(x.shape) == 1:
            x = x[np.newaxis]

        name = 'PStripe2Ban'
        dim_obj = 1
        maxormins = [-1]  # -1最大化该目标
        dim_vars = x.shape[0]
        varTypes = [1] * dim_vars
        lb = [0] * dim_vars
        ub = [1] * dim_vars
        lbin = [1] * dim_vars
        ubin = [1] * dim_vars
        self.x = x.astype(np.float32)
        self.constH, self.constW = 2440, 1220

        self.flag = 0
        ea.Problem.__init__(self, name, dim_obj, maxormins,
                            dim_vars, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        if self.flag == 0:  # 只有第一次运行时执行这个：
            pop.Phen = np.zeros_like(pop.Phen)
            pop.Phen[:, :3] = 1
            self.flag += 1
        Vars = pop.Phen  # n_popu, n_stack

        s_ = np.sum(Vars * self.x[:, 0], axis=1)  # 所有条的长度和
        pop.ObjV = s_[:, np.newaxis] / self.constH

        # 采用可行性法则处理约束
        constraint1 = np.sum(
            Vars * self.x[:, 0], axis=1) - self.constH  # n_popu,
        constraint2 = np.sum(Vars, axis=1) - 10  # n_popu,
        pop.CV = np.vstack([constraint1, constraint2]
                           ).T  # (n_popu, n_constraint)


def stripe2ban_solver(x, output_dir=None):
    problem = PStripe2Ban(x)

    encoding = 'RI'
    NIND = 50  # 种群规模
    field = ea.crtfld(encoding, problem.varTypes,
                      problem.ranges, problem.borders)
    population = ea.Population(encoding, field, NIND)

    # 算法参数设置
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)
    myAlgorithm.MAXGEN = 500  # 最大进化代数
    myAlgorithm.mutOper.F = 0.7  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = 0.7  # 重组概率
    myAlgorithm.logTras = 1  # 每隔多少代记录日志，设置成0表示不记录日志
    myAlgorithm.verbose = False
    myAlgorithm.drawing = 0  # 0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画
    myAlgorithm.dirName = output_dir if output_dir else 'tmp'

    res = ea.optimize(myAlgorithm, verbose=False,
                      drawing=0,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    return res['Vars']
