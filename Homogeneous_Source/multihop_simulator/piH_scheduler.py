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



class Thresholdscheduler_Node:
    def __init__(self, totalnum_nodes):
        self.age = [0] * totalnum_nodes
        self.updateflag = [False] * totalnum_nodes
        self.aoii = [0] * totalnum_nodes
        self.state = [0] * totalnum_nodes 
        self.timedata = [0] * totalnum_nodes
        self.pival = [0] * totalnum_nodes
        self.piflag = [False] * totalnum_nodes
        self.Abar = [0] * totalnum_nodes
        self.Aflag = [False] * totalnum_nodes
        self.packgentimeBelief = [0] * totalnum_nodes

class Thresholdscheduler:
    def __init__(self):
        self.network  = None
    def setup_nodedata(self):
        
        self.agedata = []
        for i in range(self.network.totalnum_nodes):
            self.agedata.append([])
            
        self.nodedata = [] 
        for n in self.network.G_up.nodes:
            d = Thresholdscheduler_Node(self.network.totalnum_nodes)
            self.nodedata.append(d)
            
        self.aoiidata = []
        for i in range(self.network.totalnum_nodes):
            self.aoiidata.append([])
            
        self.abardata = {n: {k: [] for k in self.network.source_list} for n in self.network.G_up.nodes}
          
    def aoii_update(self,timeis):
        for d in range(self.network.totalnum_nodes):
            for s in self.network.source_list:
                if self.nodedata[d].state[s] == self.nodedata[s].state[s]:
                    self.nodedata[d].aoii[s] = 0
                    self.nodedata[d].timedata[s] = timeis
                else:
                    self.nodedata[d].aoii[s] +=1 #= timeis - self.nodedata[d].timedata[s]
                if d == 0:
                    self.aoiidata[s].append(self.nodedata[d].aoii[s])
                    
    def belief_and_A_bar_update(self, activation_vector, prb, timeis):
        shts = activation_vector
        for n in self.network.G_up.nodes:
            for k in self.network.source_list:
                self.nodedata[n].piflag[k] = 0   
                    
        for n in self.network.G_up.nodes:
            self.nodedata[n].pival[n] = 1
            self.nodedata[n].piflag[n] = 1
                
        for (s, j, i) in shts: # if there is a transmission, update pi 
            if self.nodedata[j].piflag[s] == 0:
                last_rxd_pack = self.network.G_up.nodes[j]["Node"].get_latest_received_packet(s)
                if not last_rxd_pack == NOPACKET:
                    self.nodedata[j].packgentimeBelief[s] = last_rxd_pack.generated_time
                    del_of_t = timeis - self.nodedata[j].packgentimeBelief[s] +1
                    Belief_est = (1-prb)**del_of_t
                    self.nodedata[j].pival[s] = Belief_est
                    self.nodedata[j].piflag[s] = 1
                
        for n in self.network.G_up.nodes:
            for k in self.network.source_list:
                if self.nodedata[n].piflag[k] == 0:
                    self.nodedata[n].pival[k] = self.nodedata[n].pival[k]*(1-prb)
 
 #           For Age(or AoII) update
        for n in self.network.G_up.nodes:
            for k in self.network.source_list:
                self.nodedata[n].Aflag[k] = 0 
        for n in self.network.G_up.nodes:
            self.nodedata[n].Abar[n] = 0
            self.nodedata[n].Aflag[n] = 1
                
        for (s, j, i) in shts: # if there is a transmission, update A 
            if self.nodedata[j].Aflag[s] == 0:
                self.nodedata[j].Abar[s] = (1- self.nodedata[j].pival[s])*(self.nodedata[i].Abar[s] +1) 
                self.nodedata[j].Aflag[s] = 1
                    
        for n in self.network.G_up.nodes:
            for k in self.network.source_list:
                if self.nodedata[n].Aflag[k] == 0:
                    self.nodedata[n].Abar[k] = (1- self.nodedata[n].pival[k])* (self.nodedata[n].Abar[k] +1) 

               
        for n in self.network.G_up.nodes:
            for k in self.network.source_list:
                d1 = self.nodedata[n].Abar[k]
#                 print('n',n,'s',k, d1)
                self.abardata[n][k].append(d1)
                
   
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
    
    def get_index_slot(self,src,thres_tau):
        IndxComputed = self.network.G_up.nodes[src]["Node"].index_finder.compute_index(thres_tau)
        return IndxComputed 
            
            
    def get_activation_vector_slot(self):
        index_computed = []
        for s in self.network.source_list:
            thres = self.nodedata[0].Abar[s] #Implementable IndexPOlicy #Using AoiiEstimate
#             thres = self.nodedata[0].aoii[s] #Ideal IndexPolicy
            indexobtained = self.get_index_slot(s, thres)
            index_computed.append(indexobtained)
        source_index = np.argmax(index_computed)
        sourceforpacketgen_is = self.network.source_list[source_index]
        source_generated_packet = [sourceforpacketgen_is]
#         print('srclist', self.network.source_list)
#         print('index_computed',index_computed,'maxindex', np.argmax(index_computed), 'sourcegen packet', source_generated_packet)    
         
        for s in source_generated_packet:
            found_flag = False
            for d in self.network.commissioned_nodes[s]:
                last_rxd_pack = self.network.G_up.nodes[d]["Node"].get_latest_received_packet(s)
                if not last_rxd_pack == NOPACKET:
                    found_flag =True
                    head = list(self.network.G_up.neighbors(d))[0]
                    actvec = [[s, head, d]]
                    for i, item in enumerate(self.network.A):
                        if item == actvec:
                            activation_vector_index = i
                    break
            if not found_flag:
                head = list(self.network.G_up.neighbors(s))[0]
                actvec = [[s,head,s]]
                for i, item in enumerate(self.network.A):
                    if item == actvec:
                        activation_vector_index = i 
        best_activation_vector = self.network.A[activation_vector_index]
        sources_packet_generated = self.get_packet_generated_slot(best_activation_vector)
        self.age_function_update(best_activation_vector)
#         print('best_activation_vector',best_activation_vector)
        return activation_vector_index, sources_packet_generated
        
