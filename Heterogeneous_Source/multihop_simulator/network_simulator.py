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
from .age_difference_scheduler import *
from .piH_scheduler import *
from .Index_scheduler import *
from .piH_LP_scheduler import *
from .pi_Index_LP_scheduler import *


class MarkovChain: #Sample from a uniform distribution
    def __init__(self, in_state, num_states, p , u):
        self.p = p  # probability of remaining in the same state
        self.current_state = in_state
        self.num_states = num_states 
        self.u = u
        
    def transition(self,src):
        if random.random() < self.p[src]:
            self.current_state[src] = random.uniform(-self.u, self.u)
        return self.current_state[src]
    
# class StateTransitionMatrix:
#     def __init__(self, p, num_states):
#         self.p = p
#         self.num_states = num_states
#         self.r = (1 - self.p) / (self.num_states - 1)  # probability of transitioning to another state

#         #transition matrix
#         self.S = np.full((num_states, num_states), self.r)
#         np.fill_diagonal(self.S, self.p)

#     def transition_n_times(self, n):
#         S_power_n = np.linalg.matrix_power(self.S, n)
#         return S_power_n
    


# In[6]:


class MPacket:
    def __init__(self, sourceid, destinationid, seqnum, n_state, generated_time, pvalue):
        self.sourceid = sourceid
        self.destinationid = destinationid
        self.seqnum = seqnum
        self.n_state = n_state
        self.generated_time = generated_time
        self.pvalue = pvalue
        logging.info("Packet:from %u: to %u: seq %u: slot %u" % (sourceid, destinationid, seqnum, generated_time))
        
    def get_packetage(self, current_time):
        return current_time - self.generated_time + 1 # add +1; pkt generated in slot t can be served in t itself, but takes one slot time
    

class Packet:
    def __init__(self, sourceid, destinationid, seqnum, n_state, generated_time):
        self.sourceid = sourceid
        self.destinationid = destinationid
        self.seqnum = seqnum
        self.n_state = n_state
        self.generated_time = generated_time
        logging.info("Packet:from %u: to %u: seq %u: slot %u" % (sourceid, destinationid, seqnum, generated_time))
        
    def get_packetage(self, current_time):
        return current_time - self.generated_time + 1 # add +1; pkt generated in slot t can be served in t itself, but takes one slot time
    


# In[7]:


class Node:
    def __init__(self, nodeid, totalnum_nodes, issource, destinationid, hss, prob, no_ofstates, u_limit):
        self.buffer = []
        for i in range(totalnum_nodes):
            self.buffer.append([])
            
        self.count_uppackets = [] 
        for i in range(totalnum_nodes):
            self.count_uppackets.append([0])
    
        self.nodeid = nodeid
        self.prob = prob
        self.packet_seqnum = 0
        self.issource = issource
        self.destinationid = destinationid
        self.current_states = {self.nodeid: 0}
        
#         self.current_states = {i: 0 for i in range(totalnum_nodes)}
#         print('nodeid = ',self.nodeid,'current_states',self.current_states)
        logging.info("Node:Created:ID %u: IsSrc %u: DestId %u" % (nodeid, issource, destinationid))
        self.no_ofstates =no_ofstates
        self.last_removed_packet = [NOPACKET] * totalnum_nodes
        self.markov_chain = MarkovChain(self.current_states, self.no_ofstates, self.prob, u_limit) 
        
        if self.issource:
            self.hss = hss[self.nodeid]
            self.index_finder = Index_Finder(self.prob[self.nodeid], self.hss)
            self.index_finder_pre = Index_Finder_Pre(self.prob[self.nodeid], self.hss)
#         self.StateTransitionMatrix = StateTransitionMatrix(prob, no_ofstates)
            
    def generate_packet(self, current_time, n_state):
        if self.issource:
            packet = Packet(self.nodeid, self.destinationid, self.packet_seqnum, n_state, current_time)
            self.buffer[self.nodeid].append(packet)
            self.packet_seqnum += 1
            logging.info("Node:Packet g+:ID %u: PacketSeq %u: Bufferlength %u" % (self.nodeid, packet.seqnum, len(self.buffer[self.nodeid])))
    
    def generate_packet_mbelief(self, current_time, n_state, pvalue):
        if self.issource:
            packet = MPacket(self.nodeid, self.destinationid, self.packet_seqnum, n_state, current_time, pvalue)
            self.buffer[self.nodeid].append(packet)
            self.packet_seqnum += 1
            logging.info("Node:Packet g+:ID %u: PacketSeq %u: Bufferlength %u" % (self.nodeid, packet.seqnum, len(self.buffer[self.nodeid])))
            
    def add_packet(self, packet):
        # assuming tree like topologies
        # self.buffer[packet.sourceid].append(packet)
        self.buffer[packet.sourceid] = [packet]
        logging.info("Node:Packet +:ID %u: PacketSeq %u: Bufferlength %u" % (self.nodeid, packet.seqnum, len(self.buffer[packet.sourceid])))
        
    def add_packet_to_root(self, packet):
        self.buffer[packet.sourceid] = [packet]
        self.count_uppackets[packet.sourceid][0] += 1
        logging.info("Node:Packet +:ID %u: PacketSeq %u: Bufferlength %u" % (self.nodeid, packet.seqnum, len(self.buffer[packet.sourceid])))
        
        
    def remove_packet_from_hol(self, sourceid):
        if len(self.buffer[sourceid]) > 0:
            packet = self.buffer[sourceid].pop(0)
            logging.info("Node:Packet -:ID %u: PacketSeq %u: Bufferlength %u" % (self.nodeid, packet.seqnum, len(self.buffer[packet.sourceid])))
            self.last_removed_packet[sourceid] = packet
            return packet
        else:
            return NOPACKET
        
    def get_latest_received_packet(self, sourceid):
        if len(self.buffer[sourceid]) > 0:
            return self.buffer[sourceid][-1]
        else:
            return NOPACKET
        
    def logmeasurements_oneslot(self, t):
        ages = []
        for si, b in enumerate(self.buffer):
            pkt = self.get_latest_received_packet(si)
            if not pkt == NOPACKET:
                age = pkt.get_packetage(t)
            else:
                if not self.last_removed_packet[si] == NOPACKET:
                    age = self.last_removed_packet[si].get_packetage(t)
                else:
                    age = 0
            ages.append(age)
        ages = ",".join([str(i) for i in ages])
        logging.info("AgeMeasurement,%u,%u,%s" % (t, self.nodeid, ages))
        
#     def node_one_step_process(self):
#         new_state = self.markov_chain.transition()
#         return new_state
    def node_one_step_process(self, node): 
#         print('self.current_states[node]',self.current_states[node])
        nwstate = self.markov_chain.transition(node)
        return nwstate
    
    def state_transition_nstep(self, n):
        S_power = self.StateTransitionMatrix.transition_n_times(n)
        return S_power


# In[12]:


class Network:
    def __init__(self, totalnum_nodes, link_list, source_list, commissioned_nodes, network_type, hss, prob, no_of_states, u_limit, interference_model, interference_k_hop):
        self.totalnum_nodes = totalnum_nodes
        self.link_list = link_list
        self.source_list = source_list
        self.commissioned_nodes = commissioned_nodes
        
        self.network_type = network_type
        self.interference_model = interference_model
        self.interference_k_hop = interference_k_hop
        
        self.G, self.G_up, self.G_down, self.leaf_nodes, self.line_graphs, self.subtree_roots = self.make_graph()
        self.A = self.get_activation_vectors()
        
        for n in self.G_up.nodes:
            issource = False
            if n in self.source_list:
                issource = True
            node = Node(n, self.totalnum_nodes, issource, ROOTNODE_ID, hss, prob, no_of_states, u_limit)
            self.G_up.nodes[n]["Node"] = node
        
    def make_graph(self):
        G_up = nx.DiGraph()
        G_down = nx.DiGraph()
        G = nx.Graph()
        root = 0
        for li, l in enumerate(self.link_list):
            G_up.add_edge(l[1], l[0])
            G_down.add_edge(l[0], l[1])
            G.add_edge(l[1],l[0])
            
        leaf_nodes = []
        for n, d in G_up.in_degree:
            if d == 0:
                leaf_nodes.append(n)
        
        line_graphs = []
        for n in sorted(leaf_nodes):
            line_graphs.append(nx.shortest_path(G_up, n, 0))
        
        subtree_roots = [n for n in nx.neighbors(G_down, 0)]

        return G, G_up, G_down, leaf_nodes, line_graphs, subtree_roots
    
    def logmeasurements_oneslot(self, t):
        for n in self.G_up.nodes:
            self.G_up.nodes[n]["Node"].logmeasurements_oneslot(t)

    def update_node_state(self, state):
        state_dict = {"R":0, "T":1, "I":2}
        next_state = ["T","I","R"]
        return next_state[state_dict[state]]
        
    def get_activation_vectors_only_edge(self):
        radio_state = ["R","T","I"]
        Mat = []
        for b in sorted(self.line_graphs):
            for n in sorted(b):
                if n == 0:
                    continue
                self.G_up.nodes[n]["State"] = radio_state[nx.shortest_path_length(self.G_up, n, 0) % 3]

        for aa in range(3):
            for b in sorted(self.line_graphs):
                for n in sorted(b):
                    if n == 0:
                        continue
                    if self.G_up.nodes[n]["State"] == "T":
                        Mat.append(1)
                    else:
                        Mat.append(0)
                    self.G_up.nodes[n]["State"] = self.update_node_state(self.G_up.nodes[n]["State"])

        for n in self.G_up.nodes():
            if n == 0:
                continue
            del self.G_up.nodes[n]["State"]

        Mat = np.array(Mat)
        M_mat0 = Mat.reshape(3, len(self.G_up.nodes) - 1)
        return M_mat0
    
    def get_activation_vectors_singleline(self):
        radio_state = ["R","T","I"]
        Mat = []
        for b in sorted(self.line_graphs):
            for n in sorted(b):
                if n == 0:
                    continue
                self.G_up.nodes[n]["State"] = radio_state[nx.shortest_path_length(self.G_up, n, 0) % 3]

        for aa in range(3):
            for b in sorted(self.line_graphs):
                for n in sorted(b):
                    if n == 0:
                        continue
                    if self.G_up.nodes[n]["State"] == "T":
                        Mat.append(1)
                    else:
                        Mat.append(0)
                    self.G_up.nodes[n]["State"] = self.update_node_state(self.G_up.nodes[n]["State"])

        for n in self.G_up.nodes():
            if n == 0:
                continue
            del self.G_up.nodes[n]["State"]

        Mat = np.array(Mat)
        M_mat0 = Mat.reshape(3, len(self.G_up.nodes) - 1)

        awithsrc = []
        
        for i in range(M_mat0.shape[0]):
            a = M_mat0[i]
            l = len(np.where(a)[0])
            if l == 0:
                continue
            for j in itertools.product(self.source_list, repeat = l):
                ta = a.copy()
                ta[np.where(a)[0]] = j
                awithsrc.append(list(ta))
        return awithsrc
    
    def get_edgenode_graph(self):
        H = nx.Graph()
        for u,v in self.G_up.edges:
            H.add_node((u,v))
            
        hnodes = list(H.nodes)
        l_hnodes = len(hnodes)
        for i in range(l_hnodes):
            for j in range(i + 1, l_hnodes):
                s1 = set(hnodes[i])
                s2 = set(hnodes[j])
                if len(s1.intersection(s2)) > 0:
                    H.add_edge(hnodes[i], hnodes[j])
        return H
    
    def get_edgenode_graph_ind(self):
        H = nx.Graph()
        for u,v in self.G_up.edges:
            H.add_node((u,v))
            
        hnodes = list(H.nodes)
        l_hnodes = len(hnodes)
        for i in range(l_hnodes):
            for j in range(i + 1, l_hnodes):
                s1 = set(hnodes[i])
                s2 = set(hnodes[j])
                if 0 in s1 and 0 in s2:
                    continue
                if len(s1.intersection(s2)) > 0:
                    H.add_edge(hnodes[i], hnodes[j])
        return H
    
        
    def get_interference_matrix(self):
        num_links = len(self.link_list)
        links = list(self.G_up.edges)
        
        if self.interference_model == COMPLETE_INT:
            interference_matrix = np.ones((num_links, num_links))
            interference_matrix = interference_matrix - np.eye(num_links)
            return interference_matrix
        
        if self.interference_model == K_HOP_INT:
            H = self.get_edgenode_graph()
            interference_matrix = np.zeros((num_links, num_links))
            link_to_index_map = {}
            for li, l in enumerate(links):
                link_to_index_map[l] = li

            for li, l in enumerate(links):
                khop_graph = nx.ego_graph(H, l, undirected = True, radius = self.interference_k_hop)
                khop_neighbors = list(khop_graph.nodes)
                for n in khop_neighbors:
                    interference_matrix[li, link_to_index_map[n]] = 1
            interference_matrix = interference_matrix - np.eye(num_links)
            return interference_matrix
        
        if self.interference_model == K_HOP_INT_IND:
#             print('K_HOP_INT_IND')
            H = self.get_edgenode_graph_ind()
            interference_matrix = np.zeros((num_links, num_links))
            link_to_index_map = {}
            for li, l in enumerate(links):
                link_to_index_map[l] = li

            for li, l in enumerate(links):
                khop_graph = nx.ego_graph(H, l, undirected = True, radius = self.interference_k_hop)
                khop_neighbors = list(khop_graph.nodes)
                for n in khop_neighbors:
                    interference_matrix[li, link_to_index_map[n]] = 1
            interference_matrix = interference_matrix - np.eye(num_links)
            for b in self.subtree_roots:
                l = (b,0)
                for b_dash in self.subtree_roots:
                    l_dash = (b_dash,0)
                    interference_matrix[ link_to_index_map[l], link_to_index_map[l_dash]] = 1
            return interference_matrix
                
        if self.interference_model == GENERAL_INT:
            interference_matrix = np.zeros((num_links, num_links))
            return interference_matrix

            # TODO: implementation of a general interference model
            # (a) Load interference matrix from a file
            # (b) Distance based model - need additional attributes for the nodes
            # (c) Cooja model
            
    def is_independent_set(self, graph, nodes):
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2 and graph.has_edge(node1, node2):
                    return False
        return True

    def find_all_independent_sets(self, graph):
        independent_sets = []
        for nodes in graph.nodes():
            if self.is_independent_set(graph, [nodes]):
                independent_sets.append([nodes])

        for size in range(2, len(graph.nodes()) + 1):
            for nodes in combinations(graph.nodes(), size):
                if self.is_independent_set(graph, nodes):
                    independent_sets.append(list(nodes))
        return independent_sets


    def get_activation_vectors(self):
        interference_matrix = self.get_interference_matrix()
        # print("IMatrix", interference_matrix)
        r, c = np.where(interference_matrix)
        interference_graph = nx.Graph(zip(r,c))
        
        if self.network_type == multisource:
#             print('Multisource')
            interference_graph_complement = nx.complement(interference_graph)
            cliques = nx.find_cliques(interference_graph_complement)
    #         print("Cliques", cliques)
            links = list(self.G_up.edges)
            activation_vector_list = []

            for c in cliques:
                activation_vector = []
                for li in c:
                    link = links[li]
                    activation_vector.append(link)
                activation_vector_list.append(activation_vector)

#             print("Act.List", activation_vector_list)
#             # this needs to be optimized - some source link combinations are not possible
#             activation_vector_list_with_src = []
#             for av in activation_vector_list:
#                 lav = len(av)
#                 # print(av, lav)
#                 for j in itertools.product(self.source_list, repeat = lav):
#                     avs = []
#                     for l in range(lav):
#                         avs.append([j[l], av[l][1], av[l][0]])
#                     activation_vector_list_with_src.append(avs)
#             print(activation_vector_list_with_src)
            
            activation_vector_list_with_src = []
            for av in activation_vector_list:
                lav = len(av)
                for j in itertools.product(self.source_list, repeat = lav):
                    avs = []
                    for l in range(lav):
                        if  j[l] in list(nx.ancestors(self.G_up, av[l][0])) or (j[l] == av[l][0]):
                            avs.append([j[l], av[l][1], av[l][0]]) #upstream flow
#                             activation_vector_list_with_src.append(avs)
                        else:
                            continue
                        if len(avs)==lav:
                            activation_vector_list_with_src.append(avs)
                            
                    
        if self.network_type == singlesource:
            print('Singlesource')
            independent_sets = self.find_all_independent_sets(interference_graph)
            links = list(self.G_up.edges)
            
            activation_vector_list = []
            for c in independent_sets:
                activation_vector = []
                for li in c:
                    link = links[li]
                    activation_vector.append(link)
                activation_vector_list.append(activation_vector)

            activation_vector_list_with_src = []
            for av in activation_vector_list:
                lav = len(av)
                # print(av, lav)
                for j in itertools.product(self.source_list, repeat = lav):
                    avs = []
                    for l in range(lav):
                        avs.append([j[l], av[l][1], av[l][0]])
                    activation_vector_list_with_src.append(avs)
                
        return activation_vector_list_with_src
    
        
    def get_link_s_ht(self, activation_vector):
        ht = []
        for li, l in enumerate(activation_vector):
            if l > 0:
                ht.append([l, li, li + 1])
        return ht
    
    def isfeasible(self, activation_vector):
        # sht = self.get_link_s_ht(activation_vector)
        sht = activation_vector
        for (s, h, t) in sht:
            if (s != t) and len(self.G_up.nodes[t]["Node"].buffer[s]) == 0:
                return False
        return True
    
    def pack_age(self, source, current_time):
        last_received_packet = self.G_up.nodes[ROOTNODE_ID]["Node"].get_latest_received_packet(source)
        if not last_received_packet == NOPACKET:
            packetageis = last_received_packet.get_packetage(current_time)
            return packetageis
        else:
            return current_time + 1


# In[8]:


class NetworkSimulator:
    def __init__(self, totalnode_num, link_list, source_list, commissioned_nodes, network_type, hss, prob, ps_val, no_of_states, u_limit, scheduler_id, scheduler, 
                 interference_model, interference_k_hop):
        self.totalnode_num = totalnode_num
        self.link_list = link_list
        self.source_list = source_list
        self.commissioned_nodes = commissioned_nodes
        self.network_type = network_type
        self.prob =prob
        self.ps_val = ps_val
        self.scheduler = scheduler
        self.scheduler_id = scheduler_id
        self.network = Network(self.totalnode_num, self.link_list, self.source_list, self.commissioned_nodes, 
                               self.network_type,hss,  prob, no_of_states, u_limit, interference_model, interference_k_hop)
        self.initialize_scheduler()
        
            
        self.packetagedata = [] 
        for i in range(self.network.totalnum_nodes):
            self.packetagedata.append([])
            
        self.count_generatedpackets = [] 
        for i in range(self.network.totalnum_nodes):
            self.count_generatedpackets.append([0])
            
        self.aoiiplusdata = []
        for i in range(self.network.totalnum_nodes):
            self.aoiiplusdata.append([])

    def initialize_scheduler(self):
        self.scheduler.network = self.network
        if self.scheduler_id == SCHEDULER_AGEDIFF:
            self.scheduler.setup_nodedata()
        if self.scheduler_id == SCHEDULER_Genie:
            self.scheduler.setup_nodedata()
        if self.scheduler_id ==  SCHEDULER_Belief:
            self.scheduler.setup_nodedata()
        if self.scheduler_id == SCHEDULER_MGenie:
            self.scheduler.setup_nodedata()
        if self.scheduler_id ==  SCHEDULER_MBelief:
            self.scheduler.setup_nodedata()
            
        if self.scheduler_id == SCHEDULER_Threshold:
            self.scheduler.setup_nodedata()
            
        if self.scheduler_id == SCHEDULER_PiHLP_id:
            self.scheduler.setup_nodedata()
        
    def simulate_oneslot(self, ts):
        logging.info("Simulator:Slot %u" % (ts))
        # print('%%%%',ts,'%%%%%%')
#         print('################',ts,'##################')

        # Computing Next state... 
        for s in self.source_list:
            prev_state = self.scheduler.nodedata[s].state[s]
            newstateis = self.network.G_up.nodes[s]["Node"].node_one_step_process(s)
            self.scheduler.nodedata[s].state[s] = newstateis
#             print('s =', s, 'prev_state =',prev_state,'newstateis =',newstateis)
            if self.scheduler_id == SCHEDULER_MBelief:
                if prev_state == newstateis:
                    self.scheduler.nodedata[s].pval[s] = ((self.scheduler.nodedata[s].pval[s] *self.scheduler.nodedata[s].Npval[s])) / (self.scheduler.nodedata[s].Npval[s] +1)
                else:
                    self.scheduler.nodedata[s].pval[s] = (self.scheduler.nodedata[s].pval[s] *self.scheduler.nodedata[s].Npval[s])+1 / (self.scheduler.nodedata[s].Npval[s] +1)
                if self.scheduler.nodedata[s].pval[s] < 0.05:
                    self.scheduler.nodedata[s].pval[s] = 0.05
                if self.scheduler.nodedata[s].pval[s] > 0.95:
                    self.scheduler.nodedata[s].pval[s] = 0.95
                self.scheduler.nodedata[s].Npval[s] = self.scheduler.nodedata[s].Npval[s] +1
            if self.scheduler_id == SCHEDULER_AoII_plus:
                self.scheduler.nodedata[s].prevstate[s] = prev_state                 

#         # Compute the metrics
#         for s in self.source_list:
#             find_packetage = self.network.pack_age(s, delivery_t)
#             self.packetagedata[s].append(find_packetage)

                        
        if self.scheduler_id == SCHEDULER_Belief:
#             self.scheduler.belief_and_A_bar_update(activation_vector,ts)
            self.scheduler.aoii_update(ts)
            
        if self.scheduler_id == SCHEDULER_MBelief:
#             self.scheduler.belief_and_A_bar_update(activation_vector,ts)
            self.scheduler.aoii_update(ts)
            
        if self.scheduler_id == SCHEDULER_MGenie:
            self.scheduler.aoii_update(ts)

        if self.scheduler_id == SCHEDULER_Genie:
            self.scheduler.aoii_update(ts)
        if self.scheduler_id == SCHEDULER_AGEDIFF:
            self.scheduler.aoii_update(ts)
            
        if self.scheduler_id == SCHEDULER_Threshold:
            self.scheduler.aoii_update(ts)
            
        if self.scheduler_id == SCHEDULER_PiHLP_id:
            self.scheduler.aoii_update(ts)

        # Make decision on what to transmit
        source_generatedpacket = []
        activation_vector_index = NOT_SELECTED
        if self.scheduler_id == SCHEDULER_AGEDIFF:
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot()
        if self.scheduler_id == SCHEDULER_Genie:
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot()
        if self.scheduler_id == SCHEDULER_ML:
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot()
        if self.scheduler_id == SCHEDULER_Belief:
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot()
        if self.scheduler_id == SCHEDULER_MGenie:
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot(self.prob)
            
        if self.scheduler_id == SCHEDULER_MBelief:
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot()
            
        if self.scheduler_id == SCHEDULER_Threshold:
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot()
            
        if self.scheduler_id == SCHEDULER_PiHLP_id:
            self.scheduler.pre_emption_decision(ts, self.prob)
            activation_vector_index, source_generatedpacket = self.scheduler.get_activation_vector_slot(self.ps_val)
    
        
        
        logging.info("Simulator:Slot %u: Source Gen Packets %s" % (ts, source_generatedpacket))
        
        for s in source_generatedpacket:
            nstate = self.scheduler.nodedata[s].state[s]
            if self.scheduler_id == SCHEDULER_MBelief:
                pvalue = self.scheduler.nodedata[s].pval[s]
                self.network.G_up.nodes[s]["Node"].generate_packet_mbelief(ts, nstate, pvalue)
                self.count_generatedpackets[s][0] += 1
            else:
                self.network.G_up.nodes[s]["Node"].generate_packet(ts, nstate)
                self.count_generatedpackets[s][0] += 1
                
        if not activation_vector_index == NOT_SELECTED:
            activation_vector = self.network.A[activation_vector_index]
            logging.info("Simulator:Slot %u: Act. vector %s" % (ts, activation_vector))
            delivery_t = ts 
            shts = activation_vector
            for (s, h, t) in shts:
                packet = self.network.G_up.nodes[t]["Node"].remove_packet_from_hol(s)
                if not packet == NOPACKET:
                    if not h == ROOTNODE_ID:
                        self.network.G_up.nodes[h]["Node"].add_packet(packet)
                        self.scheduler.nodedata[h].state[packet.sourceid] = packet.n_state
                        if self.scheduler_id == SCHEDULER_MBelief:
                            self.scheduler.nodedata[h].pval[packet.sourceid] = packet.pvalue
                    else:
                        self.network.G_up.nodes[h]["Node"].add_packet_to_root(packet)
                        self.scheduler.nodedata[h].state[packet.sourceid] = packet.n_state
                        if self.scheduler_id == SCHEDULER_MBelief:
                            self.scheduler.nodedata[h].pval[packet.sourceid] = packet.pvalue
                            
        if self.scheduler_id == SCHEDULER_Belief:
            self.scheduler.belief_and_A_bar_update(activation_vector, self.prob, ts)
            
        if self.scheduler_id == SCHEDULER_MBelief:
            self.scheduler.belief_and_A_bar_update(activation_vector, ts)
            
        if self.scheduler_id == SCHEDULER_Threshold: 
            self.scheduler.belief_and_A_bar_update(activation_vector, self.prob, ts)
            
        if self.scheduler_id == SCHEDULER_PiHLP_id: 
            self.scheduler.belief_and_A_bar_update(activation_vector, self.prob, ts)
            
        
    def logmeasurements_oneslot(self, t):
        self.network.logmeasurements_oneslot(t)
        
    def simulate(self, max_slots):
        for t in range(max_slots):
            self.simulate_oneslot(t)
            self.logmeasurements_oneslot(t)
            

class Index_Finder:
    def __init__(self, p, hs):
        self.p = p
        self.hs = hs

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
        
    def compute_index(self,tau):
#         tau = self.aoii
        #tau = self.aoii
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
class Index_Finder_Pre:
    def __init__(self, p_c, hs):
        self.p_c = p_c  # State change probability
        self.hs = hs    # Hop distance
        
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

    
    def compute_index(self, tau, psval):
#         tau = self.aoii
        E_T = self.compute_expected_times(psval) #self.simulate_markov_chain(num_simulations=1000)
        E_T = E_T[self.hs-1]
        E_T_squared = self.compute_second_moment_times(psval)
        E_T_squared =E_T_squared[self.hs-1]
        D_tau = self.compute_D_tau(tau, E_T)
        A_tau = self.compute_A_tau(tau, E_T, E_T_squared)
        D_tau_plus1 = self.compute_D_tau(tau+1, E_T)
        A_tau_plus1 = self.compute_A_tau(tau+1, E_T, E_T_squared)
        return (A_tau_plus1 - A_tau)  /(D_tau - D_tau_plus1)

