import csv

import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import time
import csv
from sklearn.preprocessing import normalize


def create_collection(collection_name:str,index_t,index_p,embedding_size:int,
                      drop_exist:bool=True,description:str="No description",metric_type:str="L2"):
    # TODO: add connection properties
    connections.connect("default", host="localhost", port="19530")
    #if a collection exists, drop the collection
    has = utility.has_collection(collection_name)
    print(f"Does collection credit_card_similarity exist in Milvus: {has}")
    if utility.has_collection(collection_name) and drop_exist:
        utility.drop_collection(collection_name)
    ## Definition of schema
    field_name = "collection_fields"
    pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)
    field = FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=embedding_size)
    schema = CollectionSchema(fields=[pk,field],description=description)

    ## Definition of index
    index_param = {
    "index_type": index_t,
    "metric_type": metric_type,
    "params":index_p

    }

    milvus_collection = Collection(name=collection_name, schema=schema)
    milvus_collection.create_index(field_name=field_name, index_params=index_param)
    return milvus_collection

def vector_search(data,top_k:int,collection, output="results"):
    #calling the collection refresh function
    # collection = CollectionRefresh(drop=True, collection_name="transactions", sn=sn, cl="Strong")

    #registration of the start time for benchmarking
    start = time.time()

    #insertion of the data into the database
    # data = [df.index.tolist(), df.iloc[:, 1:29].values.tolist()]
    # collection.insert(data)

    #definition of the parameters
    # index = {"metric_type":"L2", "index_type":"HNSW", "params":{"M":M, "efConstruction":efC}}
    search_params = {"metric_type": "L2", "params": {"ef": 32}}

    #extraction of the results
    results = collection.search(data=data, anns_field="collection_fields",
                                param=search_params, limit=top_k, expr=None, consistency_level="Strong")

    #registration of the end time for benchmarking
    end = time.time()
    duration = end - start
    print(f"Search latency: {duration}")

    #specification of the output
    if output == "duration":
        return duration
    elif output == "results":
        return results

index_type = "HNSW"
index_params = {"M":16, "efConstruction":32}
credit_card_collection = create_collection("credit_card_kaggle",index_t=index_type,index_p=index_params,embedding_size=28)

df = pd.read_csv("../../data/raw/creditcard.csv").drop(['Time', 'Amount', 'Class'], axis=1)
df_np = df.to_numpy()
df_np = normalize(df_np)
mr = credit_card_collection.insert([df_np])
milvus_ids = mr.primary_keys

index_mapping_df_to_milvus = {key:milvus_ids[key] for key in range(len(df))}
index_mapping_milvus_to_df = inv_map = {v: k for k, v in index_mapping_df_to_milvus.items()}

header = ["source","destination","distance"]
with open("../../graph.csv", 'w') as f:
    w = csv.writer(f)
    w.writerow(header)


credit_card_collection.load()
source_id=0
with open("../../graph.csv", "a") as f:
    w = csv.writer(f)
    for i in range(0,len(df),1000):
        print(i)
        x = vector_search(list(df_np[i:min(i+1_000,len(df))]),128,credit_card_collection)
        for transaction in x:
            l = []
            for result in transaction:
                l.append([source_id,index_mapping_milvus_to_df[result.id],result.distance])
            w.writerows(l)
            source_id+=1