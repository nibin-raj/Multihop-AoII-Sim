import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

from scipy.io import loadmat
from scipy import linalg as la

import logging
import itertools

from .constants import *

class AgeDifferenceScheduler_Node:
    def __init__(self, totalnum_nodes):
        self.age = [0] * totalnum_nodes
        self.updateflag = [False] * totalnum_nodes
        self.aoii = [0] * totalnum_nodes
        self.state = [0] * totalnum_nodes 
        self.timedata = [0] * totalnum_nodes


class AgeDifferenceScheduler:
    def __init__(self):
        self.network  = None
        # the age-diff scheduler has to maintain some data for every node in order to schedule
    def setup_nodedata(self):
        
        self.agedata = []
        for i in range(self.network.totalnum_nodes):
            self.agedata.append([])
        self.nodedata = [] 
        for n in self.network.G_up.nodes:
            d = AgeDifferenceScheduler_Node(self.network.totalnum_nodes)
            self.nodedata.append(d)
        
        self.aoiidata = []
        for i in range(self.network.totalnum_nodes):
            self.aoiidata.append([])
            
    def aoii_update(self,timeis):
        for d in range(self.network.totalnum_nodes):
            for s in self.network.source_list:
                if self.nodedata[d].state[s] == self.nodedata[s].state[s]:
                    self.nodedata[d].aoii[s] = 0
                    self.nodedata[d].timedata[s] = timeis
                else:
                    self.nodedata[d].aoii[s] +=1 #timeis - self.nodedata[d].timedata[s]
                if d == 0:
                    self.aoiidata[s].append(self.nodedata[d].aoii[s])
   
    def age_function_update(self, activation_vector):
        sht = activation_vector
        for n in self.network.G_up.nodes:
            for k in self.network.source_list:
                self.nodedata[n].updateflag[k] = 0
                
        for n in self.network.G_up.nodes:
            self.nodedata[n].age[n] = 0
            self.nodedata[n].updateflag[n] = 1
            
        for (s, j, i) in sht: # if there is a transmission, age update 
            if self.nodedata[j].updateflag[s] == 0:
                self.nodedata[j].age[s] = np.min([self.nodedata[j].age[s], self.nodedata[i].age[s]]) + 1
                self.nodedata[j].updateflag[s] = 1
                
        for n in self.network.G_up.nodes:
            for k in self.network.source_list:
                if self.nodedata[n].updateflag[k] == 0:
                    self.nodedata[n].age[k] = self.nodedata[n].age[k] + 1
                    
        for k in self.network.source_list:
            d1 = self.nodedata[ROOTNODE_ID].age[k]
            self.agedata[k].append(d1)
#         print('self.agedata',self.agedata)
                           
    def get_packet_generated_slot(self, activation_vector): 
        sources_packet_generated = []
        # sht = self.network.get_link_s_ht(activation_vector)
        sht = activation_vector
        for (s,h,t) in sht:
            if s == t:
                sources_packet_generated.append(t)
        return sources_packet_generated
                    
    def get_activation_vector_slot(self):
        max_age_diff = -np.inf
        max_age_diff_indx = 0
        for ai, a in enumerate(self.network.A):
            sht = a
            age_diff = 0
            for (s, h, t) in sht:
                age_diff +=  np.max([self.nodedata[h].age[s] - self.nodedata[t].age[s], 0])
            if age_diff> max_age_diff:
                max_age_diff = age_diff
                max_age_diff_indx = ai
        best_activation_vector = self.network.A[max_age_diff_indx]
#         print(best_activation_vector)
        sources_packet_generated = self.get_packet_generated_slot(best_activation_vector)
        self.age_function_update(best_activation_vector)
        return  max_age_diff_indx, sources_packet_generated
