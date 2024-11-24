import asyncio
import html
import os
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from .utils import (
                        logger,
                        load_json,
                        write_json,
                        compute_mdhash_id,
                    )

from .base import (
                            BaseGraphStorage,
                            BaseKVStorage,
                            BaseVectorStorage,
                        )

# Json 缓存
@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        # 构建文件名
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        # 加载已有的数据
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")
    # 获取所有的字典文件
    async def all_keys(self) -> list[str]:
        return list(self._data.keys())
    # 完成回调，写入数据
    async def index_done_callback(self):
        write_json(self._data, self._file_name)
    # 根据id获取数据
    async def get_by_id(self, id):

        return self._data.get(id, None)

    # 根据ids 获取数据，如果有fields，则只返回对应的字段 
    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]
    # 过滤数据
    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])    
    # 更新数据
    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data
    # 清空
    async def drop(self):
        self._data = {}

# Nano向量数据库
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    # cos 相似阈值
    cosine_better_than_threshold: float = 0.2
    # self.global_config在StorageNameSpace基类（StorageNameSpace是BaseVectorStorage的父类）中实现
    def __post_init__(self):
        # 客户端的文件名
        self._client_file_name = os.path.join(self.global_config["working_dir"], f"vdb_{self.namespace}.json")
        # 最大批次
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # 初始化客户端，客户端要传入embedding_dim，这里传入的是self.embedding_func.embedding_dim
        self._client = NanoVectorDB(self.embedding_func.embedding_dim, storage_file=self._client_file_name)
        # 初始化阈值
        self.cosine_better_than_threshold = self.global_config.get("cosine_better_than_threshold", self.cosine_better_than_threshold)

    async def upsert(self, data: dict[str, dict]):
        # 插入数据
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")

        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        # entities_vdb的meta_fields： {"entity_name"}， relationships_vdb： {"src_id", "tgt_id"}  | chunks_vdb 好像没发现 meta_fields
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}, # 字典解包 # 如果数据中有元数据的相关字段，就保留
            }
            for k, v in data.items()
        ]
        # 拿到内容？
        contents = [v["content"] for v in data.values()]
        # 拿到batch ，按照batch 划分
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        # 文档内容的嵌入 
        embeddings_list = await asyncio.gather(*[self.embedding_func(batch) for batch in batches])
        
        embeddings = np.concatenate(embeddings_list)
        
        for i, d in enumerate(list_data):
            # 拿到每个文本的嵌入
            d["__vector__"] = embeddings[i]
        # 插入数据
        results = self._client.upsert(datas=list_data)
        
        return results

    async def query(self, query: str, top_k=5):
        """Query the vector DB with a query string and return the top k results."""
        # 问题嵌入
        embedding = await self.embedding_func([query])
        # 去掉batch维度
        embedding = embedding[0]
        # 调用nano_vectordb的query方法
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        # 拿到id，distance
        # 对于实体数据"entity_name"， __id__, 文本对应的向量，相似度，
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")
    # 删除实体
    async def delete_entity(self, entity_name: str):
        try:
            # 计算实体的hash id
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]
            # 获取数据
            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")
    # 删除关系
    async def delete_relation(self, entity_name: str):
        try:
            # 遍历self.client_storage["data"] 如果src_id或者tgt_id等于entity_name，放入relations列表
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            # 拿到带删除的ids
            ids_to_delete = [relation["__id__"] for relation in relations]
            # 删除
            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(f"All relations related to entity {entity_name} have been deleted.")
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting relations for entity {entity_name}: {e}")
    # 保存客户端数据
    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    # 加载图，根据xml文件名加载
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None
    # 写入图数据
    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        # 返回图的最大连通分支，节点和边以稳定的方式排序。
        """
        from graspologic.utils import largest_connected_component
        # 拷贝图
        graph = graph.copy()
        # 静态类型检查。
        graph = cast(nx.Graph, largest_connected_component(graph))
        # html.unescape 会将这些编码的实体转换回它们的原始字符。用于将HTML中的实体（如 &lt;, &gt;, &amp; 等）解码为它们对应的字符。
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        # 用于根据指定的映射关系重新标记图中的节点。
        # 创建一个包含节点 0, 1, 2 的路径图
        # Example:
                # G = nx.path_graph(3)
                # print(sorted(G.nodes()))  # 输出: [0, 1, 2]

                # # 定义映射关系
                # mapping = {0: 'a', 1: 'b', 2: 'c'}

                # # 根据映射关系重新标记节点
                # H = nx.relabel_nodes(G, mapping)
                # print(sorted(H.nodes()))  # 输出: ['a', 'b', 'c']
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        确保具有相同关系的无向图始终以相同的方式读取。
        """
        # 如果 graph 是有向图（即 graph.is_directed() 返回 True），则 fixed_graph 被初始化为 nx.DiGraph()，表示一个新的有向图。
        # 如果 graph 是无向图（即 graph.is_directed() 返回 False），则 fixed_graph 被初始化为 nx.Graph()，表示一个新的无向图
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        # ，graph.nodes(data=True) 方法用于获取图中所有节点及其关联的数据。
        # 当 data=True 时，该方法返回一个包含所有节点及其属性的迭代器，每个元素是一个二元组 (node, attributes)
        sorted_nodes = graph.nodes(data=True)
        # 根据第0个数据进行排序
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
        # 从排序后的节点构造一个新的图
        fixed_graph.add_nodes_from(sorted_nodes)
        # 返回边的数据
        # graph.edges(data=True) 方法用于获取图中所有边及其关联的数据。
        # 当 data=True 时，该方法返回一个包含所有边及其属性的迭代器，每个元素是一个三元组 (u, v, attributes)，其中 u 和 v 是边的两个端点，attributes 是该边的属性字典。
        edges = list(graph.edges(data=True))
        # 如果 graph 是无向图，则对边进行排序
        if not graph.is_directed():
            # 调整边的结构，使得源节点小于目标节点
            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]
        # 最后，对 edges 列表进行排序。排序的依据是每条边的键值，即 source -> target 的字符串形式。
        # 通过这种排序，确保边的顺序在每次读取时都是一致的。
        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"
        # 依照字符串规则进行排序
        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))
        # 添加边
        fixed_graph.add_edges_from(edges)
        return fixed_graph
    # 初始化后方法
    def __post_init__(self):
        # 获取图文件名
        self._graphml_xml_file = os.path.join(self.global_config["working_dir"], f"graph_{self.namespace}.graphml")
        # 预加载？
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        # log输出
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        # 初始化图，如果预加载成功，直接调用
        self._graph = preloaded_graph or nx.Graph()
        # 嵌入算法
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
    # 回调，写入
    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)
    # 是否存在节点
    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)
    # 是否存在边
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)
    # 获取节点
    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)
    # 节点度
    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)
    # 边度
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)
    # 获取边
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))
    # 获取节点的边
    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None
    # 插入节点和对应的数据
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)
    # 插入边
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)
    # 删除节点
    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")
    # 节点嵌入
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
