from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar

import numpy as np

from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")

# # 问题参数类
# 自动为类生成构造函数 (__init__)、字符串表示方法 (__repr__)、比较方法 (__eq__) 等。
# 用于数据存储： 数据类通常用于封装数据，类似于数据库记录或 API 请求/响应中的数据对象
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "global" # Literal 是一种静态类型提示，用来限制变量的值只能是指定的某些常量值
    only_need_context: bool = False # 是否需要上下文
    response_type: str = "Multiple Paragraphs"
    # Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode.
    # 要检索的前k个元素的数量；对应于“局部”模式中的实体和“全局”模式中的关系。
    top_k: int = 60
    # Number of tokens for the original chunks. 原始块的标记数。
    max_token_for_text_unit: int = 4000
    # Number of tokens for the relationship descriptions 关系描述的标记数
    max_token_for_global_context: int = 4000
    # Number of tokens for the entity descriptions  实体描述的标记数
    max_token_for_local_context: int = 4000

#  存储基类？
@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing""" # 建立索引后提交存储操作
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying""" # 查询完成后提交存储操作
        pass

# 向量存储嘞
@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc # 需要一个embedding 模型
    meta_fields: set = field(default_factory=set) # 存储元数据字段

    async def query(self, query: str, top_k: int) -> list[dict]: # 需要实现一个异步的 query 方法
        raise NotImplementedError
    # 插入数据
    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.    使用value中的` content `字段进行嵌入，使用key作为id。
        If embedding_func is None, use 'embedding' field from value        如果embedding_func为None，则从value中使用` embedding `字段
        """
        raise NotImplementedError

# 使用泛型
@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc
    # 获取所有的key
    async def all_keys(self) -> list[str]:
        raise NotImplementedError
    # 根据id获取数据
    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError
    # 批量获取数据
    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError
    # 过滤数据
    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError
    # 插入数据
    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError
    # 删除数据
    async def drop(self):
        raise NotImplementedError

# 图数据库的基类
@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None
    # 节点是否存在
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError
    # 边是否存在
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError
    # 节点的度
    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError
    # 边的度
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError
    # 获取节点
    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError
    # 获取边
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError
    # 获取节点的边
    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError
    # 插入节点
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError
    # 插入边
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError
    # 删除节点
    async def delete_node(self, node_id: str):
        raise NotImplementedError
    # 节点嵌入
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in lightrag.")
