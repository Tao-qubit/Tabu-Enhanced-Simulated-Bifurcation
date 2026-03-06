import sys
import os
import torch
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
np.random.seed(666)

class SB():
    def __init__(self, A, h=None, K=1, delta=1, dt=1., sigma=1., M=2, n_iter=1000, xi=None, sk=False, batch_size=1, num_tabu=1, device='cpu'):
        self.N = A.shape[0]
        self.A = A
        self.h = h
        # self.A_sparse = csr_matrix(A)
        # The number of node
        self.batch_size = batch_size
        self.K = K
        self.delta = delta
        self.dt = dt
        self.M = M
        self.n_iter = n_iter
        self.device = device
        self.sigma = sigma
        self.p = np.linspace(0, 1,self.n_iter)
        self.dm = self.dt / self.M
        self.num_tabu = num_tabu
        self.sk = sk
        self.xi = xi
        if xi is None:
            if sk:
                self.xi = 0.7 * np.sqrt(self.N-1) / np.sqrt((self.A ** 2).sum())
            else:
                self.xi = 1 / np.abs(self.A.sum(axis=1)).max()
        
        
        self.initialize()

    def initialize(self):
        # self.x = 0.01 * (np.random.rand(self.N, self.batch_size)-0.5)
        self.x = 0.01 * (torch.rand(self.N, self.batch_size, device=self.device)-0.5)
        self.y = 0.01 * (torch.rand(self.N, self.batch_size, device=self.device)-0.5)
        #self.y = np.zeros((self.N, self.batch_size))
        # self.y = 0.01 * (np.random.rand(self.N, self.batch_size)-0.5)




    '''
    def calc_cut(self, x):
        sign = torch.sign(x)
        sign[sign==0] = 1.0
        cut = 0.25*(torch.sum((torch.mm(sign,self.A))*sign, dim=1))  - 0.25*self.asum
        return cut
    
    def energy_from_cut(self, cuts, offset=0):
        eng = -2*(cuts + 0.25*self.asum)+offset
        return eng
    
    def calc_energy(self, x):
        sign = torch.sign(x)
        sign[sign == 0] = 1.0
        energy = -0.5 * torch.mm(torch.mm(sign, self.A), sign.transpose(1,0))
        return energy
    '''
    def update(self):
        # iterate on the number of MVMs
        for i in range(self.n_iter):
            for j in range(self.M):
                #self.x += self.dm * self.y * self.delta
                self.y -= (self.K * self.x**3 + (self.delta - self.p[i])*self.x)*self.dm
                self.x += self.dm * self.y * self.delta

            self.y += self.xi * self.dt * self.A@self.x

    

    def update_b(self,lam=np.ones(10000)):
        # h=0
        for i in range(self.n_iter):
            if self.h is None:
                self.y += (-(self.delta - self.p[i])*self.x + self.xi * torch.sparse.mm(self.A, self.x)) * self.dt
            else:
                
                num_tabu_sample = self.h.shape[1]
                if num_tabu_sample > 1:
#                     if i % 10 == 0:
                    num_tabu = self.num_tabu
                    tabu_index = np.random.randint(0, num_tabu_sample, num_tabu)
                    # tabu_index = np.random.choice(num_tabu_sample,num_tabu,replace=False)
                    h = self.h[:, tabu_index].sum(dim=1, keepdim=True)/num_tabu
                    
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (torch.sparse.mm(self.A,self.x)-lam[i]*(h))) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (torch.sparse.mm(self.A,self.x)+self.h)) * self.dt
            self.x += self.dt * self.y * self.delta

            
            
            cond = torch.abs(self.x) > 1
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
            # self.y +=  - lam[i]*h* self.dt

        
    def update_d(self,lam=np.ones(10000)):
        for i in range(self.n_iter):
            if self.h is None:
                self.y += (-(self.delta - self.p[i])*self.x + self.xi * (torch.sparse.mm(self.A, torch.sign(self.x)))) * self.dt
            else:
                
                num_tabu_sample = self.h.shape[1]
                if num_tabu_sample > 1:
#                     if i % 10 == 0:
                    num_tabu = self.num_tabu
                    tabu_index = np.random.randint(0, num_tabu_sample, num_tabu)
                    # tabu_index = np.random.choice(num_tabu_sample,num_tabu,replace=False)
                    h = self.h[:, tabu_index].sum(dim=1, keepdim=True)/num_tabu
                    
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (torch.sparse.mm(self.A, torch.sign(self.x))-lam[i]*h)) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (torch.sparse.mm(self.A, torch.sign(self.x)))+self.h) * self.dt
            self.x += self.dt * self.y * self.delta

            
            cond = torch.abs(self.x) > 1
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
            
            
#         for i in range(self.n_iter):
#             if self.h is None:
#                 self.y += (-(self.delta - self.p[i])*self.x + self.xi * self.A.dot(np.sign(self.x))) * self.dt
#             else:
                
#                 num_tabu_sample = self.h.shape[1]
#                 if num_tabu_sample > 1:
# #                     if i % 10 == 0:
#                     num_tabu = num_tabu_sample//30
#                     tabu_index = np.random.randint(0, num_tabu_sample, num_tabu)
#                     h = self.h[:, tabu_index].sum(axis=1, keepdims=True)/num_tabu
                    
#                     self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(np.sign(self.x))+h)) * self.dt
#                 else:
#                     self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(np.sign(self.x))+self.h)) * self.dt
#             self.x += self.dt * self.y * self.delta

            
            
#             cond = np.abs(self.x) > 1
#             self.x = np.where(cond, np.sign(self.x), self.x)
#             self.y = np.where(cond, np.zeros_like(self.y), self.y)


def read_gset(filename, negate=True):
    # read graph
    graph = pd.read_csv(filename, sep=' ')
    # the number of vertices
    n_v = int(graph.columns[0])
    # the number of edges
    n_e = int(graph.columns[1])

    assert n_e == graph.shape[0], 'The number of edges is not matched'

    G = csr_matrix((graph.iloc[:,-1], (graph.iloc[:, 0]-1, graph.iloc[:, 1]-1)), shape=(n_v, n_v))
    G = G+G.T       
    if negate:
        return -G
    else:
        return G

if __name__=='__main__':
    J = read_gset('mcdata/G1.txt', negate=True)

    # J = read_metis('../metis-5.1.0/graphs/mdual.graph', negate=True)
    # n_iter_list = np.concatenate([np.arange(10, 100, 10), np.arange(100, 1100, 100)])
    #n_iter_list = [100, 250, 500, 750,1000,2500,5000,7500,10000]
    # n_iter_list = [25000, 50000, 75000,100000]
    # time_list = []
    # energy_list = []
    # cut_list = []
    # for n_iter in n_iter_list:
    #    energy_list = []
    #    cut_list = []
        # for i in range(100):
    device='cuda'
    # n_iter = 1000
    #K = sys.argv[2]
    J = torch.from_numpy(J.todense())
    xi = 1 / torch.abs(J.sum(axis=1)).max()
    J = J.cuda().float()
    J = J.to_sparse_csr()
    import time
    start = time.time()
    s = SB(J, n_iter=1000, xi=xi, dt=1., K=1, sk=False, batch_size=100, device=device)
    # start = time.time()
    s.update_d()
    # end =  time.time()
    # print(end-start)
    best_sample = torch.sign(s.x).clone()

    energy = -0.5 * torch.sum(J@best_sample * best_sample, dim=0)
    cut = -0.5 * energy-0.25 * J.sum()
    # cut.max()
    # print(cut)
    tabu_index = torch.nonzero(cut != cut.max())[:,0]
    tabu_sample = best_sample[:, tabu_index].clone()
    num_tabu = len(tabu_index)
    # tabu_mat = tabu_sample.dot(tabu_sample.T) - num_tabu * np.eye(s.N)
    tabu_field = tabu_sample.sum(dim=1, keepdim=True)
    start1 = time.time()
    s = SB(J, tabu_sample, xi=0.001, n_iter=10000, dt=1, K=1, sk=True, batch_size=1, num_tabu=1, device=device)
    s.update_d()
    end = time.time()
    print(end-start)
    print(end-start1)

    best_sample2 = torch.sign(s.x).clone()
    energy = -0.5 * torch.sum(J @ best_sample2 * best_sample2, dim=0)
    cut = -0.5 * energy-0.25 * J.sum()

    print(torch.max(cut))
    print(torch.mean((torch.max(cut)==cut).float()))
    print(torch.mean(cut))
    print(torch.std(cut))
