import logging
import os
import matplotlib.pyplot as plt
import hydra
from hydra.core.config_store import ConfigStore
import pandas as pd
import numpy as np
import time
from omegaconf import DictConfig
from src.feature_engineering.feature_extraction import vector_search, create_collection

log = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path="../../config/feature_extractor", config_name="config")
def distance_distribution(cfg: DictConfig):
    log.info(cfg)
    if not os.path.exists(f"{cfg.paths.output_dir}/distance_dist/{cfg.milvus.top_k}_{cfg.milvus.metric_type}"):
        os.makedirs(f"{cfg.paths.output_dir}/distance_dist/{cfg.milvus.top_k}_{cfg.milvus.metric_type}")

    start_time = time.time()
    credit_card_collection = create_collection(cfg.milvus, embedding_size=28)
    top_k = cfg.milvus.top_k
    trans_df = pd.read_csv(f"{cfg.paths.data}")
    max_time = trans_df["Time"].max()
    group_df = trans_df.groupby(pd.cut(trans_df["Time"], np.arange(-1, max_time + 3600, 3600)))
    index_mapping_df_to_milvus = {}
    index_mapping_milvus_to_df = {}
    l = []
    for group_time, group in group_df:
        group_number = int((group_time.left + 1) / 3600 + 1)
        log.info("")
        log.info(f"===== Processing group:{group_time} , group number:{group_number}=====")

        log.info(f"Inserting group to feature_extractor")
        t = time.time()
        group_feature = group.drop(['Time', 'Amount', 'Class'], axis=1).to_numpy()
        mr = credit_card_collection.insert([group_feature])
        milvus_ids = mr.primary_keys
        log.info(f"Time of inserting group in feature_extractor: {time.time() - t:.3f} s")

        log.info("Searching similarity from Milvus")
        t = time.time()
        first_group_index = min(group.index)
        index_mapping_df_to_milvus = index_mapping_df_to_milvus | {key: milvus_ids[key - first_group_index] for key in
                                                                   group.index}
        index_mapping_milvus_to_df = index_mapping_milvus_to_df | {v: k for k, v in index_mapping_df_to_milvus.items()}
        credit_card_collection.load()
        x = vector_search(list(group_feature), top_k, credit_card_collection, cfg.milvus.metric_type)
        log.info(f"Searching time: {time.time() - t:.3f} s")

        # Similarity to past fraud transaction
        source_id = first_group_index
        for transaction in x:
            for result in transaction:
                if result.distance != 0:
                    l.append((source_id, index_mapping_milvus_to_df[result.id], result.distance))
            source_id += 1
        # Drop self loop
        distance_list_np = np.array([t[2]for t in l])
        datamin = 0
        datamax = 100
        numbins = 100
        mybins = np.linspace(datamin, datamax, numbins)
        plt.hist(distance_list_np,bins=mybins,color="b")
        plt.ylabel("Count")
        plt.xlabel("Distance")
        plt.title(f"Distribution in group {group_number}")
        plt.savefig(f"{cfg.paths.output_dir}/distance_dist/{cfg.milvus.top_k}_{cfg.milvus.metric_type}/{group_number}.png")
    log.info(f"Calculate distribution: {time.time() - t:.3f} s")
    # Format datetime string
    print(f"total time:{(time.time() - start_time):.3f} s")

