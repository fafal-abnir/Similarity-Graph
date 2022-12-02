import networkx as nx
import os
import numpy as np

G_L2_100 = nx.read_gpickle("../../data/final/graphs/100_IP/(147599.0, 151199.0]")

edge_distance_list = [d["weight"] for *_,d in G_L2_100.edges(data=True)]

edge_distance_list_np = np.array(edge_distance_list)

print(np.min(edge_distance_list_np))
print(np.percentile(edge_distance_list_np,[25,50,75,90,99]))
print(np.max(edge_distance_list_np))
