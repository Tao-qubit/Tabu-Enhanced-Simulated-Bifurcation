import sys
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
np.random.seed(189312)

class SB():
    def __init__(self, A, h=0 ,tabu=None, K=1, delta=1, dt=1., sigma=1., M=2, n_iter=1000, xi=None, sk=False, batch_size=1, num_tabu=1, device='cpu'):
        self.N = A.shape[0]
        self.A = A
        self.h = h
        self.tabu = tabu
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
        self.x = 0.01 * (np.random.rand(self.N, self.batch_size)-0.5)
        self.y = 0.01 * (np.random.rand(self.N, self.batch_size)-0.5)


    def update(self):
        # iterate on the number of MVMs
        for i in range(self.n_iter):
            for j in range(self.M):
                #self.x += self.dm * self.y * self.delta
                self.y -= (self.K * self.x**3 + (self.delta - self.p[i])*self.x)*self.dm
                self.x += self.dm * self.y * self.delta

            self.y += self.xi * self.dt * self.A.dot(self.x)

    

    def update_b(self):
        # self.evo = []
        for i in range(self.n_iter):
            if self.tabu is None:
                self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(self.x) + self.h )) * self.dt
            else:
                
                num_tabu_sample = self.tabu.shape[1]
                if num_tabu_sample > 1:
#                     if i % 10 == 0:
                    num_tabu = self.num_tabu
                    tabu_index = np.random.randint(0, num_tabu_sample, num_tabu)
                    tabu = self.tabu[:, tabu_index].sum(axis=1, keepdims=True)/num_tabu
                    
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(self.x)+self.h-tabu)) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(self.x)+self.h-self.tabu)) * self.dt
            self.x += self.dt * self.y * self.delta
            
            
            cond = np.abs(self.x) > 1
            self.x = np.where(cond, np.sign(self.x), self.x)
            self.y = np.where(cond, np.zeros_like(self.y), self.y)

            # energy_evo = -0.5 * np.sum(self.A.dot(np.sign(self.x)) * np.sign(self.x), axis=0)
            # self.evo.append(energy_evo.min())
            
        
    def update_d(self):
        # self.evo = []
        for i in range(self.n_iter):
            if self.tabu is None:
                self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(np.sign(self.x))+self.h)) * self.dt
            else:
                
                num_tabu_sample = self.tabu.shape[1]
                if num_tabu_sample > 1:
#                     if i % 10 == 0:
                    num_tabu = self.num_tabu
                    tabu_index = np.random.randint(0, num_tabu_sample, num_tabu)
                    tabu = self.tabu[:, tabu_index].sum(axis=1, keepdims=True)/num_tabu
                    
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(np.sign(self.x))+self.h-tabu)) * self.dt
                else:
                    self.y += (-(self.delta - self.p[i])*self.x + self.xi * (self.A.dot(np.sign(self.x))+self.h-self.tabu)) * self.dt
            self.x += self.dt * self.y * self.delta
            
            cond = np.abs(self.x) > 1
            self.x = np.where(cond, np.sign(self.x), self.x)
            self.y = np.where(cond, np.zeros_like(self.y), self.y)
            # energy_evo = -0.5 * np.sum(self.A.dot(np.sign(self.x)) * np.sign(self.x), axis=0)
            # self.evo.append(energy_evo.min())
            

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

if __name__ == "__main__":

    J = read_gset('gset/G1.txt', negate=True)

    s = SB(J, n_iter=1000, xi=None, dt=1., K=1, sk=False, batch_size=100, device='cpu')
    s.update_b()
    best_sample = np.sign(s.x).copy()
    energy = -0.5 * np.sum(J.dot(best_sample) * best_sample, axis=0)
    # cut = -0.5 * energy-0.25 * J.sum()
    # tabu_index = np.nonzero(cut != cut.max())[0]
    # tabu_sample = best_sample[:, tabu_index].copy()
    # num_tabu = len(tabu_index)

    s = SB(J, tabu=best_sample, xi=0.05, n_iter=9000, dt=1, K=1, sk=False, batch_size=10, num_tabu=2, device='cpu')
    s.update_b()

    best_sample2 = np.sign(s.x).copy()
    energy = -0.5 * np.sum(J.dot(best_sample2) * best_sample2, axis=0)
    cut = -0.5 * energy-0.25 * J.sum()

    print(np.max(cut))
    print(np.mean(np.max(cut)==cut))
    print(np.mean(cut))
    print(np.std(cut))