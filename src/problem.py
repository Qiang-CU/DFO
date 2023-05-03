# -*- coding: utf-8 -*-
# @author: LI Qiang
# @email: qiangli@link.cuhk.edu.hk 
# @date: 2023/01/26

import numpy as np
from config import config


class LinearMinimization(object):

    def __init__(self):
        self.dim = 5
        self.gamma = 0.5 # AR model parameter, 1->sample from gaussian distribution
        self.mu0 = np.array([-2, 2, -2, 2, -2]) # mu0 - kappa * theta is mean
        self.kappa = 0.1 # 代表shift的强度
        self.sigma = 1

        self.cov = self.sigma**2 * np.eye(self.dim)
        self.last_sample = np.zeros(self.dim)
        self.opt_sol = self.mu0 / (2*self.kappa)
        self.opt_ps = self.mu0 / self.kappa

    # 两种loss
    def ell_loss(self, theta, sample_z):
        return - theta @ sample_z

    def expect_loss(self, theta):
        rv_mean = self.mu0 - self.kappa * theta
        loss = - np.dot(theta, rv_mean)
        return loss

    # 两种gradient
    def grd_ell(self, sample_z):
        return -sample_z

    def grd_true_loss(self, theta):
        res = - (self.mu0 - 2 * self.kappa * theta)
        return res

    # 两种sampling的方式
    def sample_from_stationary_dist(self, theta):
        
        res = np.random.multivariate_normal(self.mu0 - self.kappa * theta, self.cov, size=(config.batch,))[0]
        return res

    def sample_from_AR(self, theta):

        cov = (2-self.gamma) / self.gamma * self.sigma**2 * np.eye(self.dim)
        current_mean = (self.mu0 - self.kappa * theta).transpose()
        new_sample = (1 - self.gamma) * self.last_sample + self.gamma * \
                     np.random.multivariate_normal(mean = current_mean, cov = cov, size=(config.batch,))[0]

        # 注意这里最后的[0]
        self.last_sample = new_sample

        return new_sample
