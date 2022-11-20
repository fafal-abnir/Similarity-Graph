import csv

import pandas as pd
import networkx as nx
from networkx.algorithms import community
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import datetime
import numpy as np
import time
import csv
from sklearn.preprocessing import normalize

start_time = time.time()


def inverse_distance(x):
    if x == -1:
        return 0
    else:
        return 1 / x

def jacc_sim(set1,set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def create_collection(collection_name: str, index_t, index_p, embedding_size: int,
                      drop_exist: bool = True, description: str = "No description", metric_type: str = "L2"):
    # TODO: add connection properties
    connections.connect("default", host="localhost", port="19530")
    # if a collection exists, drop the collection
    has = utility.has_collection(collection_name)
    print(f"Does collection credit_card_similarity exist in Milvus: {has}")
    if utility.has_collection(collection_name) and drop_exist:
        utility.drop_collection(collection_name)
        print(f"Collection {collection_name} dropped")
    ## Definition of schema
    field_name = "collection_fields"
    pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)
    field = FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=embedding_size)
    schema = CollectionSchema(fields=[pk, field], description=description)

    ## Definition of index
    index_param = {
        "index_type": index_t,
        "metric_type": metric_type,
        "params": index_p

    }

    milvus_collection = Collection(name=collection_name, schema=schema)
    milvus_collection.create_index(field_name=field_name, index_params=index_param)
    return milvus_collection


def vector_search(data, top_k: int, collection, output="results"):
    # calling the collection refresh function
    # collection = CollectionRefresh(drop=True, collection_name="transactions", sn=sn, cl="Strong")

    # registration of the start time for benchmarking
    start = time.time()

    # insertion of the data into the database
    # data = [df.index.tolist(), df.iloc[:, 1:29].values.tolist()]
    # collection.insert(data)

    # definition of the parameters
    # index = {"metric_type":"L2", "index_type":"HNSW", "params":{"M":M, "efConstruction":efC}}
    search_params = {"metric_type": "L2", "params": {"ef": 32}}

    # extraction of the results
    results = collection.search(data=data, anns_field="collection_fields",
                                param=search_params, limit=top_k, expr=None, consistency_level="Strong")

    # registration of the end time for benchmarking
    end = time.time()
    duration = end - start
    # print(f"Search latency: {duration}")

    # specification of the output
    if output == "duration":
        return duration
    elif output == "results":
        return results


index_type = "HNSW"
index_params = {"M": 16, "efConstruction": 32}
credit_card_collection = create_collection("credit_card_graph_feature", index_t=index_type, index_p=index_params,
                                           embedding_size=28)

# df_np = df.to_numpy()
# df_np = normalize(df_np)
# mr = credit_card_collection.insert([df_np])
# milvus_ids = mr.primary_keys
#
# index_mapping_df_to_milvus = {key:milvus_ids[key] for key in range(len(df))}
# index_mapping_milvus_to_df = inv_map = {v: k for k, v in index_mapping_df_to_milvus.items()}


trans_df = pd.read_csv("../../data/raw/creditcard.csv")
max_time = trans_df["Time"].max()
group_df = trans_df.groupby(pd.cut(trans_df["Time"], np.arange(-1, max_time + 3600, 3600)))
fraud_trans = []
index_mapping_df_to_milvus = {}
index_mapping_milvus_to_df = {}
avg_distance_from_frauds = {}
personalized_page_rank = {}
community_risk = {}
fraud_neighbor_count = {}
G = nx.DiGraph()
TOP_K = 100
# source_id = 0
for group_time, group in group_df:
    group_number = int((group_time.left + 1) / 3600 + 1)
    print("")
    print(f"===== Processing group:{group_time} , group number:{group_number}=====")

    print(f"Inserting group to milvus")
    t = time.time()
    group_feature = group.drop(['Time', 'Amount', 'Class'], axis=1).to_numpy()
    mr = credit_card_collection.insert([group_feature])
    milvus_ids = mr.primary_keys
    print(f"Time of inserting group in milvus: {time.time() - t:.3f} s")

    print("Searching similarity from Milvus")
    t = time.time()
    first_group_index = min(group.index)
    index_mapping_df_to_milvus = index_mapping_df_to_milvus | {key: milvus_ids[key - first_group_index] for key in
                                                               group.index}
    index_mapping_milvus_to_df = index_mapping_milvus_to_df | {v: k for k, v in index_mapping_df_to_milvus.items()}
    credit_card_collection.load()
    x = vector_search(list(group_feature), TOP_K, credit_card_collection)
    print(f"Searching time: {time.time() - t:.3f} s")

    # Similarity to past fraud transaction
    print("Finding similarity with past frauds")
    t = time.time()
    source_id = first_group_index
    for transaction in x:
        distances_fraud = []
        for result in transaction:
            if index_mapping_milvus_to_df[result.id] in fraud_trans:
                distances_fraud.append(result.distance)
        if len(distances_fraud) > 0:
            avg_distance_from_frauds[source_id] = sum(distances_fraud) / TOP_K
        else:
            avg_distance_from_frauds[source_id] = -1
        source_id += 1
    print(f"Similarity to past fraud time: {time.time() - t:.3f} s")
    # Counting number fraud neighbor
    print("Counting number of fraud neighbor at Top_K")
    t = time.time()
    source_id = first_group_index
    for transaction in x:
        distances_fraud = []
        fraud_count = 0
        for result in transaction:
            if index_mapping_milvus_to_df[result.id] in fraud_trans:
                fraud_count+=1
        fraud_neighbor_count[source_id] = fraud_count
        source_id += 1

    # Generating graph
    print("Generating dynamic graph")
    t = time.time()
    l = []
    source_id = first_group_index
    for transaction in x:
        for result in transaction:
            if result.distance != 0:
                l.append((source_id, index_mapping_milvus_to_df[result.id], 1 / result.distance))
        source_id += 1
    # Drop self loop
    G.add_weighted_edges_from(l)
    del l
    print(f"Add edge to the graph time: {time.time() - t:.3f} s")

    # Personalized page rank
    print("Computing Personalized page rank")
    t = time.time()
    personalization = {key: 1 for key in fraud_trans}
    if len(fraud_trans) == 0:
        personalization = None
    page_ranks = nx.pagerank(G, personalization=personalization)
    for current_index in group.index:
        personalized_page_rank[current_index] = page_ranks[current_index]
    del personalization
    print(f"Personalized page rank with respect to known fraud: {time.time() - t:.3f} s")

    # Community detection in dynamic graph
    print("Computing Communities")
    t = time.time()
    # Initiating communities
    source_id = first_group_index
    if group_number == 2:
        communities = nx.community.louvain_communities(G, seed=1234)
        for community in communities:
            for node_transaction in community:
                community_risk[node_transaction] = len(community.intersection(fraud_trans)) / len(community)
    if group_number>2:
        for transaction in x:
            best_community = None
            neighbors = [index_mapping_milvus_to_df[result.id] for result in transaction]
            community_sim = [ jacc_sim(community,neighbors) for community in communities]
            comm_index = community_sim.index(max(community_sim))
            community_risk[source_id] = len(communities[comm_index].intersection(fraud_trans)) / len(communities[comm_index])
            communities[comm_index].add(source_id)
            source_id+=1

    print(f"Calculating communities risk time: {(time.time() - t):.3f} s")

    # Updating seen fraud transactions
    current_window_fraud_trans = list(group[group["Class"] == 1].index)
    fraud_trans = fraud_trans + current_window_fraud_trans


dt = datetime.datetime.now()

# Format datetime string
x = dt.strftime("%Y-%m-%d_%H:%M:%S")

trans_df["inversed_dist"] = {k: inverse_distance(v) for (k, v) in avg_distance_from_frauds.items()}
trans_df["fraud_neighbor_count"] = fraud_neighbor_count
trans_df["community_risk"] = community_risk
trans_df["personalized_page_rank"] = personalized_page_rank

trans_df.to_csv(f"trans_fraud_extra_{x}.csv")

print(f"total time:{(time.time() - start_time):.3f} s")
