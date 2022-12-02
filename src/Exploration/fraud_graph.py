import os

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def draw_frauds_graph(graph_dir_path):
    if not os.path.exists(f"{graph_dir_path}/fraud_graph/"):
        os.makedirs(f"{graph_dir_path}/fraud_graph/")
    trans_df = pd.read_csv("../../data/final/creditcard.csv")
    fraud_nodes = list(trans_df[trans_df["Class"] == 1].index)
    max_time = trans_df["Time"].max()
    group_df = trans_df.groupby(pd.cut(trans_df["Time"], np.arange(-1, max_time + 3600, 3600)))
    past_fraud_nodes = []
    for group_time, group in group_df:
        group_number = int((group_time.left + 1) / 3600 + 1)
        G = nx.read_gpickle(f"{graph_dir_path}/{group_time}")
        current_fraud_nodes = list(group[group["Class"] == 1].index)
        H = G.subgraph(fraud_nodes)
        color_map = []
        for node in H:
            if node in current_fraud_nodes:
                color_map.append("red")
            else:
                color_map.append("blue")
        past_fraud_nodes = past_fraud_nodes + current_fraud_nodes
        nx.draw(H, node_color=color_map)
        connected_components = nx.connected_components(H)
        column1 = []
        column2 = []
        for c in connected_components:
            column1.append(c.nodes)
        plt.savefig(f"{graph_dir_path}/fraud_graph/{group_number}.png")


path = "../../data/final/graphs/50_L2"
draw_frauds_graph(path)