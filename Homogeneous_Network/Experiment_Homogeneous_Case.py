from multihop_simulator import *

link_list = [(0, 1), (0, 9), (1, 4), (1, 8), (5, 6), (7, 5), (9, 7), (9, 10), (10, 2), (10, 3)]  
interference_k_hop = 2
# Network_Type = singlesource 
Network_Type = multisource
iterno = 1000 
prob_i = 0.2
state_i = 0
no_of_states = 15
u_limit = 1
ps_value = 1

G_up = nx.DiGraph()
G_down = nx.DiGraph()
root = 0
for l in link_list:
    G_up.add_edge(l[1], l[0])
    G_down.add_edge(l[0], l[1])
            
# Create a dictionary to hold the upstream nodes of each node
commissioned_nodes = {}
for node in G_down.nodes():
    if node ==0: continue
    ancestors = nx.ancestors(G_down, node)
    ancestors_list = sorted(list(ancestors))
    ancestors_list = [x for x in ancestors_list if x != 0]
    commissioned_nodes[node] = ancestors_list
#print('commissioned_nodes',commissioned_nodes)
            
source_list = []
for node in G_up.nodes():
    if node ==0: continue 
    source_list.append(node)
    
source_hss = {node: nx.shortest_path_length(G_up, source=node, target=0) for node in source_list}
totalnum_nodes = len(link_list) + 1 
source_hs = [nx.shortest_path_length(G_up, source=node, target=0) for node in source_list]


source_pc = [prob_i]* len(source_list)
num_sources_indxpolicy = len(source_pc)
source_ps = [ps_value]* len(source_list)

######## Pi_Index ###########
sources = []
for i in range(len(source_pc)):
    sources.append(Source(source_pc[i], source_hs[i]))
max_steps = iterno
for t in range(max_steps):
    indices = np.zeros((num_sources_indxpolicy, 1))
    for s in range(num_sources_indxpolicy):
        sources[s].step_initial()
        indices[s] = sources[s].compute_index()

    scheduled_source = np.argmax(indices)
    for s in range(num_sources_indxpolicy):
        if s == scheduled_source:
            sources[s].step(True)
        else:
            sources[s].step(False)
                
scheduler_aoii = []
for i in range(len(source_pc)):
    scheduler_aoii.append(np.mean(sources[i].aoii_track))
AoII_Indexscheduler =  np.mean(scheduler_aoii)
# print(AoII_Indexscheduler)

######## Age-Difference ###########
agediffscheduler = AgeDifferenceScheduler() 
agediffsim = agediffsim = NetworkSimulator(totalnum_nodes, link_list, source_list, commissioned_nodes, Network_Type,source_hss,state_i, prob_i,  ps_value,no_of_states, u_limit,SCHEDULER_AGEDIFF, agediffscheduler, COMPLETE_INT, interference_k_hop)
agediffsim.simulate(iterno)

AGEDIFFscheduler_aoii =[]
for n in source_list:
    AGEDIFFscheduler_aoii.append(np.sum(agediffscheduler.aoiidata[n])/len(agediffscheduler.aoiidata[n]))
AoII_agediffscheduler  =  np.mean(AGEDIFFscheduler_aoii)
# print(AoII_agediffscheduler)

######## Pi-H ###########
Thresholderscheduler = Thresholdscheduler() 
Thresholdersim = NetworkSimulator(totalnum_nodes, link_list, source_list, commissioned_nodes, Network_Type, source_hss, state_i,  prob_i, ps_value, no_of_states, u_limit,SCHEDULER_Threshold, Thresholderscheduler, COMPLETE_INT, interference_k_hop)

Thresholdersim.simulate(iterno)
        
aoii_thresholder =[]
for n in source_list:
    aoii_thresholder.append(np.sum(Thresholderscheduler.aoiidata[n])/len(Thresholderscheduler.aoiidata[n]))
Aoii_Thresholder =  np.mean(aoii_thresholder)
# print(Aoii_Thresholder)

######## Pi-Index-LP ###########
sources_pre = []
for i in range(len(source_pc)):
    sources_pre.append(Source_Pre(source_pc[i], source_hs[i]))

max_steps = iterno 
for t in range(max_steps):
    indices = np.zeros((num_sources_indxpolicy, 1))
    for s in range(num_sources_indxpolicy):
        sources_pre[s].step_initial()
        indices[s] = sources_pre[s].compute_index(source_ps[s])

    scheduled_source = np.argmax(indices)
    for s in range(num_sources_indxpolicy):
        if s == scheduled_source:
            sources_pre[s].step(True, source_ps[s])
        else:
            sources_pre[s].step(False, source_ps[s])
scheduler_aoii_pre = []
for i in range(len(source_pc)):
    scheduler_aoii_pre.append(np.mean(sources_pre[i].aoii_track))
AoII_scheduler_pre =  np.mean(scheduler_aoii_pre)
# print(AoII_scheduler_pre)    
    
######## Pi-H-LP ###########
PiHLP_scheduler = PiHLP()
PiHLP_sim = NetworkSimulator(totalnum_nodes, link_list, source_list, commissioned_nodes, Network_Type, source_hss, state_i, prob_i, ps_value, no_of_states, u_limit, SCHEDULER_PiHLP_id, PiHLP_scheduler, COMPLETE_INT, interference_k_hop)
PiHLP_sim.simulate(iterno)

aoii_PiHLP =[]
for n in source_list:
    aoii_PiHLP.append(np.sum(PiHLP_scheduler.aoiidata[n])/len(PiHLP_scheduler.aoiidata[n]))
Aoii__PiHLP =  np.mean(aoii_PiHLP)
# print(Aoii__PiHLP)

print(AoII_agediffscheduler, AoII_Indexscheduler, Aoii_Thresholder,  AoII_scheduler_pre, Aoii__PiHLP)
    
    
    
    
    