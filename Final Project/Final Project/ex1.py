#!/usr/bin/env python
# coding: utf-8

# In[38]:


import scipy.io
import pandas as pd
import numpy as np
import itertools
from ci_test import ci_test
import pickle
import cdt
import networkx as nx
import matplotlib.pyplot as plt


# In[39]:


datasets = []

for dataset_i in range(1,5):
    d_mat = scipy.io.loadmat(f"D{dataset_i}.mat")
    datasets.append(np.array(pd.DataFrame(d_mat['D'])))


# ## 1.1 SGS Algorithm

# In[40]:


def sgs(dataset: np.ndarray) -> (np.ndarray, [set]):
    n_columns = dataset.shape[1]
    all_nodes = range(n_columns)

    adjacency_matrix =  np.zeros((n_columns, n_columns))

    test_counter = []
    counter = 0

    # Initialize a list of sets to keep for the edge orientation phase
    #Z_set = np.full((n_columns, n_columns), set())
    Z_set = [[set() for i in range(n_columns)] for j in range(n_columns)]

    for node_i in all_nodes:
        nodes = list(all_nodes)
        nodes.remove(node_i)
        # Pick a different 2nd variable
        for node_j in nodes:
            possible_controls = list(all_nodes).copy()
            possible_controls.remove(node_i)
            possible_controls.remove(node_j)

            conditionnal_indep_does_not_hold = True
            for n_controls in range(len(possible_controls)):
                if not conditionnal_indep_does_not_hold:
                    break
                condition_set = itertools.combinations(possible_controls, n_controls)


                for condition in condition_set:
                    counter += 1
                    if ci_test(dataset, node_i, node_j, condition):
                        conditionnal_indep_does_not_hold = False

                        Z_set[node_i][node_j] |= set(condition)
                        Z_set[node_i][node_j] |= set(condition)

                        break
            if conditionnal_indep_does_not_hold:
                adjacency_matrix[node_i][node_j] = adjacency_matrix[node_j][node_i] = 1

    print(counter)
    return adjacency_matrix, Z_set


# ## 1.1 PC Algorithm

# In[41]:


def pc1(dataset: np.ndarray) -> (np.ndarray, [set]):
    n_columns = dataset.shape[1]
    all_nodes = range(n_columns)

    # Initialize a fully connected adjency matrix (no self connection since its a DAG)
    adjacency_matrix = np.ones((n_columns, n_columns)) - np.identity(n_columns)

    Z_set = [[set() for i in range(n_columns)] for j in range(n_columns)]
    # Pick the 1st variable
    counter = 0
    for node_i in all_nodes:
        nodes = list(all_nodes)
        nodes.remove(node_i)
        # Pick a different 2nd variable
        for node_j in nodes:
            # Pick a condition set
            for n_controls in all_nodes:
                possible_controls = list(all_nodes)
                possible_controls.remove(node_i)
                possible_controls.remove(node_j)


                for condition in list(itertools.combinations(possible_controls, n_controls)):
                    counter += 1
                    if ci_test(dataset, node_i, node_j, condition):
                        adjacency_matrix[node_i][node_j] =  adjacency_matrix[node_j][node_i] = 0

                        Z_set[node_i][node_j] |= set(condition)
                        Z_set[node_j][node_i] |= set(condition)
    print(counter)
    return adjacency_matrix, Z_set


# 1.2 Modified PC algorithm

# In[42]:


def pc2(dataset: np.ndarray) -> (np.ndarray, [set]):
    n_columns = dataset.shape[1]
    all_nodes = range(n_columns)

    adjacency_matrix =  np.zeros((n_columns, n_columns))
    Z_set = [[set() for i in range(n_columns)] for j in range(n_columns)]
    counter = 0
    for node_i in all_nodes:
        nodes = list(all_nodes)
        nodes.remove(node_i)
        # Pick a different 2nd variable
        for node_j in nodes:
            # Pick a condition set
            possible_controls = list(all_nodes)
            possible_controls.remove(node_i)
            possible_controls.remove(node_j)

            counter += 1
            if not ci_test(dataset, node_i, node_j, possible_controls):
                adjacency_matrix[node_i][node_j] = adjacency_matrix[node_j][node_i] =  1

            for n_controls in all_nodes:
                for condition in itertools.combinations(possible_controls, n_controls):
                    counter += 1
                    if ci_test(dataset, node_i, node_j, condition):
                        adjacency_matrix[node_i][node_j] = adjacency_matrix[node_j][node_i] = 0

                        Z_set[node_i][node_j] |= set(condition)
                        Z_set[node_j][node_i] |= set(condition)
                        break


    print(counter)
    return adjacency_matrix, Z_set


# ## 2 Orienting the edges
# ### 2.1 Starting with V-structures

# In[43]:


def add_v_structures(graph: nx.Graph, Z_sets: [set]) -> nx.DiGraph:
    directed_graph = nx.DiGraph(graph)
    nodes = range(len(graph.nodes))

    for i in nodes :
        nodes_j = list(nodes)
        nodes_j.remove(i)
        for j in nodes_j:
            nodes_k = list(nodes_j)
            nodes_k.remove(j)
            for k in nodes_k:
                # if i -> j -> k and not (i -> k && j not in Z), keep i->j<-k
                if directed_graph.has_edge(i,j) and directed_graph.has_edge(j,k) and (not directed_graph.has_edge(i,k)):
                    if j not in Z_sets[i][k]:
                        if directed_graph.has_edge(j,i):
                            directed_graph.remove_edge(j,i)
                        if directed_graph.has_edge(k,j):
                            directed_graph.remove_edge(k,j)
    return directed_graph


# ## 2.2 Adding the two meek rules

# In[44]:


def add_meek_rules(directed_graph: nx.DiGraph) -> nx.DiGraph:
    nodes = range(len(graph.nodes))

    for i in nodes :
        nodes_j = list(nodes)
        nodes_j.remove(i)
        for j in nodes_j:
            nodes_k = list(nodes_j)
            nodes_k.remove(j)

            # meek 1
            # if  parent->i-j and not parent-j, do i->j
            if directed_graph.has_edge(i, j) & directed_graph.has_edge(j, i):
                to_remove = set()
                for parent in directed_graph.predecessors(i):
                    if directed_graph.has_edge(i, parent) or (directed_graph.has_edge(parent, j) or directed_graph.has_edge(j, parent)):
                        continue
                    else:
                       to_remove.add((j,i))

                for pair in to_remove:
                    directed_graph.remove_edge(*pair)

    for i in nodes :
        nodes_j = list(nodes)
        nodes_j.remove(i)
        for j in nodes_j:
            nodes_k = list(nodes_j)
            nodes_k.remove(j)
            # meek 2
            # if i->k->j and i-j, do i->j
            if directed_graph.has_edge(i, j) & directed_graph.has_edge(j, i):
                for child in directed_graph.successors(i):
                    if directed_graph.has_edge(child, j):
                        directed_graph.remove_edge(j, i)

    return  directed_graph


# 

# In[45]:


sgs_directed_graphs = []

for dataset in datasets[:3]:
    adjacency_mat, condition_set = sgs(dataset)
    graph = nx.from_numpy_matrix(adjacency_mat)

    directed_graph = add_v_structures(graph, condition_set)
    sgs_directed_graphs.append(add_meek_rules(directed_graph))


# In[46]:


pc1_directed_graphs = []

for dataset in datasets[:3]:
    adjacency_mat, condition_set = pc1(dataset)
    graph = nx.from_numpy_matrix(adjacency_mat)

    directed_graph = add_v_structures(graph, condition_set)
    pc1_directed_graphs.append(add_meek_rules(directed_graph))


# In[47]:


pc2_directed_graphs = []

for dataset in datasets[:3]:
    adjacency_mat, condition_set = pc2(dataset)
    graph = nx.from_numpy_matrix(adjacency_mat)

    directed_graph = add_v_structures(graph, condition_set)
    pc2_directed_graphs.append(add_meek_rules(directed_graph))


# ## D4 Matrix results

# In[ ]:


# For heavier matrix
D4_adjacency_mat, D4_condition_set = pc2(datasets[3])
D4_graph = nx.from_numpy_matrix(adjacency_mat)

D4_directed_graph = add_v_structures(graph, condition_set)
pc2_directed_graphs.append(add_meek_rules(directed_graph))


# ## 3. Results
# ## 3.1 SGS Graphs

# In[49]:


plt.figure(figsize=(10,6))
nx.draw_networkx(sgs_directed_graphs[0], with_labels=True)


# In[50]:


plt.figure(figsize=(10,5))
nx.draw_networkx(sgs_directed_graphs[1], with_labels=True)


# In[51]:


plt.figure(figsize=(10,6))
nx.draw_networkx(sgs_directed_graphs[2], with_labels=True)


# ## 3.2 PC1 Graphs

# 

# In[52]:


plt.figure(figsize=(6,4))
nx.draw(pc1_directed_graphs[0], with_labels=True)


# 

# In[53]:


plt.figure(figsize=(8,5))
nx.draw(pc1_directed_graphs[1], with_labels=True)


# 

# In[54]:


nx.draw(pc1_directed_graphs[2], with_labels=True)


# ## 3.2 PC2 Graphs

# In[55]:


nx.draw(pc2_directed_graphs[0], with_labels=True)


# In[56]:


plt.figure(figsize=(8,5))
nx.draw(pc2_directed_graphs[1], with_labels=True)


# In[57]:


plt.figure(figsize=(8,6))
nx.draw(pc2_directed_graphs[2], with_labels=True)

