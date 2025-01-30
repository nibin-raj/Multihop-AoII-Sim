import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os
from itertools import combinations

from scipy.io import loadmat
from scipy import linalg as la

import logging
import itertools

from .constants import *

class Source_Pre:
    def __init__(self, p_c, hs):
        self.p_c = p_c  # State change probability
        self.hs = hs    # Hop distance
        self.h = self.hs     
        self.xvalue = 0 
        self.estimate = np.ones(hs + 1) * self.xvalue
        self.s = 0 
        self.aoii_track = []
        self.aoii = 0
        self.scheduled_track = []
        
    def step_initial(self):
        if np.random.rand() < self.p_c:
            self.xvalue = np.random.uniform()
            self.h = self.hs
            self.s = 0
            
        if self.estimate[0] == self.xvalue:
            self.aoii = 0
        else:
            self.aoii += 1
        self.aoii_track.append(self.aoii)

    def step(self, scheduled, p_s):
        if scheduled:
            if np.random.rand() < p_s:
                if self.h > 1:
                    if self.h == self.hs:
                        self.estimate[self.h] = self.xvalue
                    self.estimate[self.h - 1] = self.estimate[self.h]
                    self.h = self.h - 1
                    self.s = self.s + 1
                elif self.h == 1:
                    if self.h == self.hs:
                        self.estimate[self.h] = self.xvalue
                    self.estimate[self.h - 1] = self.estimate[self.h]
                    self.h = self.hs
                    self.s = 0
            else:
                if not self.h == self.hs:
                    self.s = self.s + 1
        else:
            if not self.h == self.hs:
                self.s = self.s + 1
                       
#     def tau_find(self):
#         return self.aoii

    def transition_matrix(self, ps_val):
        num_states = self.hs + 1
        T = np.zeros((num_states, num_states))
        for i in range(1,num_states):
            if i == self.hs:
                T[i, i] = self.p_c + (1 - self.p_c) * (1 - ps_val)  # Stay in h_s
                T[i, i - 1] = (1 - self.p_c) * ps_val  # Transition to h_s-1
            else:
                T[i, i] = (1 - self.p_c) * (1 - ps_val)  # Remain in the current state
                T[i, i - 1] = (1 - self.p_c) * ps_val  # Transition to the next state
                T[i, self.hs] = self.p_c  # Reset to h_s
        return T[1:,1:]

    def compute_expected_times(self, ps):
        num_states = self.hs + 1 # Number of states (h_s, h_s-1, ..., 1)
        T = self.transition_matrix(ps) # transition probability matrix (T)
        I = np.eye(num_states-1)
        I_minus_T = I - T
        ones = np.ones((num_states-1, 1))
        expected_times = np.linalg.solve(I_minus_T, ones)
        return expected_times

    def compute_second_moment_times(self,ps):
        num_states = self.hs + 1
        T = self.transition_matrix(ps)
        I = np.eye(num_states-1)
        X = self.compute_expected_times(ps)
        I_minus_T = I - T
        # Compute 1 + 2 * T * X
        term = np.ones((num_states-1, 1)) + 2 * T @ X.reshape(-1, 1)
        # Solve for E[T(i)^2] 
        second_moments = np.linalg.solve(I_minus_T, term)
        return second_moments
    
    def compute_D_tau(self, tau, E_T):
        if tau>0:
            g1 = 1 / self.p_c
            return E_T / (g1 + tau - 1 + E_T)
        else:
            return 1

    def compute_A_tau(self, tau, E_T, E_T_squared):
        if tau>0:
            g1 = 1 / self.p_c
            numerator = (tau * (tau - 1)/2 + tau * E_T+ (E_T_squared - E_T) / 2 )
            denominator = g1 + tau - 1 + E_T
            return numerator / denominator
        else:
            g1 = 1 / self.p_c
            numerator =  E_T/2 + (E_T_squared /2 )
            denominator = g1 + E_T 
            return numerator / denominator

    
    def compute_index(self, psval):
        tau = self.aoii
        E_T = self.compute_expected_times(psval) #self.simulate_markov_chain(num_simulations=1000)
        E_T = E_T[self.hs-1]
        E_T_squared = self.compute_second_moment_times(psval)
        E_T_squared =E_T_squared[self.hs-1]
        D_tau = self.compute_D_tau(tau, E_T)
        A_tau = self.compute_A_tau(tau, E_T, E_T_squared)
        D_tau_plus1 = self.compute_D_tau(tau+1, E_T)
        A_tau_plus1 = self.compute_A_tau(tau+1, E_T, E_T_squared)
        return (A_tau_plus1 - A_tau)  /(D_tau - D_tau_plus1)
