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



class Source:
    def __init__(self, p, hs):
        self.p = p
        self.hs = hs
        
        self.xvalue = 0
        self.estimate = np.ones((hs + 1, 1)) * self.xvalue

        self.aoii_track = []
        self.aoii = 0
        self.h = self.hs
        self.s = 0
        
    def step_initial(self):
        if np.random.rand() <= self.p:
            self.xvalue = np.random.uniform()

        if self.estimate[0] == self.xvalue:
            self.aoii = 0
        else:
            self.aoii += 1
        self.aoii_track.append(self.aoii)

    def step(self, scheduled):
        if scheduled:
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
    
    def _frac_scheduled(self, tau):
        if tau > 0:
            g1 = 1/self.p
            g2 = 1/(1 - self.p)**self.hs
            frac_notscheduled = (g1 + tau - 1)/(g1 + tau - 1 + g2 * self.hs)
            return 1 - frac_notscheduled
        else:
            return 1
    
    def _aaoii(self, tau):
        if tau > 0:
            pg = (1 - self.p)**self.hs
            g1 = 1/self.p
            g2 = 1/(1 - self.p)**self.hs
            num = tau*(tau - 1)/2 + tau * g2 * self.hs + self.hs*self.hs/2 * (2 - pg)/pg/pg - self.hs/2*g2
            denom = g1 + tau - 1 + self.hs * g2
            aaoii = num/denom
            return aaoii
        else:
            pi0 = (1 - self.p)**self.hs # analytical expression for pi0
            ea = (self.hs + 1) - (1 - self.p)**self.hs - 1/self.p * (1 - (1 - self.p)**self.hs) + (1 - pi0)*self.hs*(1 - (1 - self.p)**self.hs)/(1 - self.p)**self.hs
            y = self.hs + 1
            r = 1/(1 - self.p)
            f = (y*(y - 1)*(r ** (y - 2)) - 2)/(r - 1) - (y*r**(y - 1) - 2*r)/(r - 1)**2 - (y*r**(y - 1) - 2*r)/(r - 1)**2 + 2*(r**y - r*r)/(r - 1)**3
            aaoii = ea + (self.hs - 1)/2 + pi0/self.hs * (self.p*(1 - self.p)**self.hs/2/(1 - self.p)**2 * f - self.hs*(self.hs - 1)/2)
            return aaoii
        
    def compute_index(self):
        tau = self.aoii
        atau1 = self._aaoii(tau + 1)
        atau  = self._aaoii(tau)
        dtau1 = self._frac_scheduled(tau + 1)
        dtau  = self._frac_scheduled(tau)
        epsilon1 = 1e-6  
        if abs(dtau - dtau1) < epsilon1:
            INDX = np.Inf
        else:
            INDX = (atau1 - atau) / (dtau - dtau1)

        return INDX
