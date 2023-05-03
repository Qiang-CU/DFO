# -*- coding: utf-8 -*-
# @author: LI Qiang
# @email: qiangli@link.cuhk.edu.hk 
# @date: 2023/01/26

import math
import sys
import os.path
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
from collections import OrderedDict

from config import config


class AlgoBase(object):
    def __init__(self):
        self.iter = 0
        self.config = config
        self.metric = {'iter': [], 'gap': []}
        self.output_path = './res-new/'
        self.sample_time = self.create_sampling_time()
        self.max_record_time = self.sample_time[-1]

        
        # mpi 相关的设定
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()


    def create_sampling_time(self):
        """生成对数刻度或者正常刻度，sample_num记录metric运行的时间点"""
        if self.config.log_scale:
            L = np.logspace(0, self.config.max_iter_log, self.config.num_points, endpoint=False,
                            dtype=int).tolist()  # L stores the time point when we sample
            sample_num = list(OrderedDict.fromkeys(L))  # 去掉L中重复的元素(note: L中的元素都是non-decreasing的)
        else:
            sample_num = list(range(0, self.config.max_iter_num, self.config.step))  # 选取测算measurement的时间点
            # 这里原本时用range(0, max_iter)的，但是由于DFO执行时while-loop, self.iter=0的时候会出错
        return sample_num

    def save_res(self, filename):
        save_path = self.output_path
        np.save(save_path+"{}.npy".format(filename), self.metric)
        # np.save(os.path.join(save_path, "xval.npy"), self.sample_time)


class DFO(AlgoBase):

    def __init__(self, problem, forgetting_factor=0):
        super().__init__()
        self.problem = problem
        self.theta, self.sample_z = self.initialization()

        self.rho = 0.5
        self.forgetting_factor = forgetting_factor # corresponding to lambda in paper

        if self.forgetting_factor == 1:
            self.tau0 = 2/np.log(1/max(self.rho, self.forgetting_factor + 1e-8)) # 这里的1e-6可以防止出现log(0)的情况
        else:
            self.tau0 = 2/np.log(1/max(self.rho, self.forgetting_factor)) 
        self.outer_iter, self.inner_iter, self.sample_count = 0, 0, 0

        # step size 相关的参数
        self.delta0 = self.problem.dim ** (1/6) * 100 #100 # 这里的50非常重要，决定了收敛的CPU time
        self.beta = 1/6

        self.eta0 = self.problem.dim ** (-2/3) * 0.3 #0.3
        self.alpha = -2/3

        self.flag = False
        self.flag2 = True


    def initialization(self):
        init_theta = np.array([2, -2, 2, -2, 2])
        sample_z = np.zeros((config.batch, self.problem.dim))
        return init_theta, sample_z

    def info_bar(self, err):
        head = ['Outer Iter', 'Inner Iter', 'Sample Num', 'Squa. Dist.']
        row_format = "{:^18}|{:^18}|{:^18}|{:^18}"
        if self.outer_iter == 0:
            print(row_format.format(*head))
        row = [f'Outer Iter: {str(self.outer_iter)}', f'Inner Iter: {str(self.inner_iter)}',
               f'Sample Num: {str(self.sample_count)}', 'Error: %.2E' % err]
        print(row_format.format(*row), flush=True)

    def sample_unit_sphere(self):
        s = np.random.normal(0, 1, self.problem.dim)
        norm = math.sqrt(sum(s * s))

        return s / norm

    def record(self):
        res = np.linalg.norm(self.problem.grd_true_loss(self.theta), ord=2)

        self.metric['iter'].append(self.sample_count)
        self.metric['gap'].append(res)

    def step_size(self, name='delta'):
        if name == 'delta':
            stepsize = self.delta0 / ((self.outer_iter+1) ** self.beta)
        elif name == 'eta':
            stepsize = self.eta0 * ((1+self.outer_iter) ** self.alpha)
        else:
            print('输入参数错误')
            sys.exit(1)
        return stepsize

    def fit_innner_loop(self, tau_k, uk, delta_k, bar):

        pert_theta = self.theta + delta_k * uk
        new_sample = self.problem.sample_from_AR(pert_theta)
        # new_sample = self.problem.sample_from_stationary_dist(pert_theta)
        self.sample_count += 1


        grd = self.problem.dim / delta_k * self.problem.ell_loss(pert_theta, new_sample) * uk
        # grd = self.problem.dim / delta_k * self.problem.expect_loss(pert_theta) * uk  #如果用真正的grd是可以收敛的

        # update theta
        self.theta = self.theta - self.step_size('eta') * (self.forgetting_factor ** (tau_k - self.inner_iter)) * grd

        if np.linalg.norm(self.theta) >= 1e8: # for debug use
            print('Error! There are some issue in programming')
            exit(0)

        if self.sample_time != [] and self.sample_count == self.sample_time[0]:
            self.sample_time.pop(0)
            self.record()
            # if self.rank == 0:
            #     self.info_bar(err=self.metric['gap'][-1])
            if self.sample_count == self.max_record_time:
                self.flag = True

        bar.update(1)
        if self.rank == 0:
            bar.set_description(f'rho = {self.forgetting_factor}, '
                                f'Current Sample Num {self.sample_count}, gap {self.metric["gap"][-1]:.5f}')


    def run(self, rep):
        tqdm_bar = tqdm(total = config.max_iter_num, file=sys.stdout)

        while self.flag2:
            self.outer_iter += 1

            temp = self.tau0 * np.log(self.outer_iter + 1)
            tau_k = max(1, int(temp))

            new_uk = self.sample_unit_sphere()  # direction
            delta_k = self.step_size('delta')

            for it in range(1, tau_k + 1):
                self.inner_iter = it
                self.fit_innner_loop(tau_k, uk=new_uk, delta_k=delta_k, bar=tqdm_bar)
                if self.flag:
                    break

            if self.flag:
                self.flag2 = False

        self.save_res(f'DFO-lambda{self.forgetting_factor}-rep{rep}')
        tqdm_bar.close()


class RGD(AlgoBase):

    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.theta = self.initialization()

    def initialization(self):
        init_theta = np.array([2, -2, 2, -2, 2])
        return init_theta


    def fit_one_step(self):
        sample_z = self.problem.sample_from_AR(self.theta)
        grd = self.problem.grd_ell(sample_z)
        self.theta = self.theta - self.step_size('diminishing') * grd

        # sample_z = self.problem.sample_from_stationary_dist()
        # grd = self.problem.grd_true_loss(self.theta) # ture gradient for test only

    def step_size(self, type='constant'):
        if type == 'constant':
            lr = 0.01
        elif type == 'diminishing':
            a0 = 50
            a1 = 1000
            lr = a0 / (a1 + self.iter)
        else:
            lr = None
            print('Error! No matching step size')
        return lr

    def record(self):
        res = np.linalg.norm(self.problem.grd_true_loss(self.theta), ord=2)
        self.metric['gap'].append(res)
        self.metric['iter'].append(self.iter)

    def run(self, rep):
        if self.rank == 0:
            loop = tqdm(range(self.config.max_iter_num))
        else:
            loop = range(self.config.max_iter_num)

        for i in loop:
            self.iter = i
            self.fit_one_step()

            if self.iter in self.sample_time:
                self.record()
                if self.rank == 0:
                    loop.set_description('RGD, Squa. Error %.3e' % (self.metric['gap'][-1]))

        self.save_res(f'RGD-rep{rep}')







