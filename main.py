# -*- coding: utf-8 -*-
# @author: LI Qiang
# @email: qiangli@link.cuhk.edu.hk 
# @date: 2023/01/26


from src.problem import LinearMinimization as LM
from src.Algo import DFO, RGD

from mpi4py import MPI

if __name__ == "__main__":
    """
        多进程加速任务执行
        运行方法：  mpirun -n 10 python main.py
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    model = LM()

    ## run gradient descent algorithm
    # algo_rgd = RGD(model)
    # algo_rgd.run(rep=rank)
    
    ## run DFO(lambda) algorithm

    if rank < 5:
        lambda_list = [0, 0.25, 0.5]

        for factor in lambda_list:
            algo_dfo = DFO(model, forgetting_factor=factor)
            algo_dfo.run(rep=rank)
    elif rank>=5 and rank <= 10:
        algo_rgd = RGD(model)
        algo_rgd.run(rep=rank)

        lambda_list = [0.75, 1]

        for factor in lambda_list:
            algo_dfo = DFO(model, forgetting_factor=factor)
            algo_dfo.run(rep=rank)
    else:
        print('错误')
