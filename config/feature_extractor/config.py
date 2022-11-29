from dataclasses import dataclass


@dataclass
class Milvus:
    index_params: dict
    user: str = "default"
    password: str = ""
    url: str = "localhost"
    port: str = "19530"
    collection_name: str = "credit_card_graph_feature"
    top_k: int = 128
    index_type: str = "HNSW"
    metric_type: str = "L2"


@dataclass
class Paths:
    data: str
    output_dir: str
    save_graphs: bool


@dataclass
class FeatureExtractionConfig:
    paths: Paths
    milvus: Milvus
