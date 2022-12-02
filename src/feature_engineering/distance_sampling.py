import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

trans_df = pd.read_csv("../../data/final/creditcard.csv").drop(['Time', 'Amount', 'Class'], axis=1)
l = []
group_size = 3000
distance_list_np = np.array([])
for i in tqdm(range(len(trans_df) // group_size)):
    g = trans_df.iloc[i * group_size:(i + 1) * group_size]
    sample_df = trans_df.sample(100)

    distance_list_np = np.concatenate((distance_list_np, cdist(g, sample_df, metric='euclidean').flatten()), axis=0)
print(np.percentile(distance_list_np, [0.01, 0.05, 0.1, 0.2, 0.5,10,50,90,99]))
datamin = 0
datamax = 100
numbins = 2000
mybins = np.linspace(datamin, datamax, numbins)
fig, axs = plt.subplots(2)
axs[0].hist(distance_list_np, bins=mybins, color="b")
axs[0].set(xlabel="Distance", ylabel="Count")
axs[1].hist(distance_list_np, bins=mybins, color="b", cumulative=True, density=True)
axs[1].set(xlabel="Distance", ylabel="Percentage")
fig.suptitle("Sampling Distribution")

plt.savefig(f"../../data/final/distributions/distance_sampling_distribution.png")
