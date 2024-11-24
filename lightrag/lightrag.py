import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .operate import (
    chunking_by_token_size,
    extract_entities,
    local_query,
    global_query,
    hybrid_query,
    naive_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

from .kg.neo4j_impl import Neo4JStorage

from .kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )

# 确保返回一个有效的 asyncio 事件循环对象 (AbstractEventLoop)，即使在当前上下文中没有现成的事件循环。
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # 检查当前线程是否已经存在一个活动的事件循环。如果有，直接返回
        return asyncio.get_event_loop()

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop


@dataclass
class LightRAG:
    # 这个代码的核心功能是为 working_dir 提供一个动态的默认值，这个值会包含当前的日期和时间
    working_dir: str = field(default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" )
    # cache 
    kv_storage: str = field(default="JsonKVStorage")
    # 向量储存
    vector_storage: str = field(default="NanoVectorDBStorage")
    # 图储存
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    # 分词器
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction 
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500 # 实体摘要的最大长度

    # node embedding
    node_embedding_algorithm: str = "node2vec" # 节点嵌入的算法
    # node2vec的cahsy7
    node2vec_params: dict = field(
                                    default_factory=lambda: {
                                        "dimensions": 1536,
                                        "num_walks": 10,
                                        "walk_length": 40,
                                        "window_size": 2,
                                        "iterations": 3,
                                        "random_seed": 3,
                                    }
                                )

    # embedding_func: EmbeddingFunc = field(default_factory=lambda:hf_embedding)
    # 实例化嵌入
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32 # batch 大小
    embedding_func_max_async: int = 16 #最大的异步数量

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete  # hf_model_complete# 调用大模型
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16 #最大的异步数量
    llm_model_kwargs: dict = field(default_factory=dict) # 其他参数

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    # 支持llm cache ？
    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    # 抽取回答 到json的函数
    convert_response_to_json_func: callable = convert_response_to_json
    # ，用于在 dataclass 初始化完成后自动调用。它的主要作用是对已经初始化的字段进行进一步的验证、计算或处理。
    # __post_init__ 方法会在所有字段被 __init__ 方法赋值后自动调用。
    def __post_init__(self):
        log_file = os.path.join(self.working_dir, "lightrag.log")
        set_logger(log_file) # 创建log
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.
        # 应该将所有存储设置移到这里，以利用附加到self的初始启动参数。
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
                                                                            self._get_storage_class()[self.kv_storage]
                                                                        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[self.vector_storage]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[self.graph_storage]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        # 穿件缓存类
        self.llm_response_cache = (
                                    self.key_string_value_json_storage_cls(
                                                                                namespace="llm_response_cache",
                                                                                global_config=asdict(self),
                                                                                embedding_func=None,
                                                                            )
                                    if self.enable_llm_cache
                                    else None
                                )
        # 封装embedding函数，限制并行数量
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)

        ####
        # add embedding func by walter
        ####
        # 所有文档？
        self.full_docs = self.key_string_value_json_storage_cls(
                                                                    namespace="full_docs",
                                                                    global_config=asdict(self),
                                                                    embedding_func=self.embedding_func,
                                                                )
        # 分割后的文档 JsonKVStorage
        self.text_chunks = self.key_string_value_json_storage_cls(
                                                                        namespace="text_chunks",
                                                                        global_config=asdict(self),
                                                                        embedding_func=self.embedding_func,
                                                                    )
        # 带有chunk的实体图谱数据库
        self.chunk_entity_relation_graph = self.graph_storage_cls(namespace="chunk_entity_relation", global_config=asdict(self))
        ####
        # add embedding func by walter over
        ####
        # 实体向量库 NanoVectorDBStorage
        self.entities_vdb = self.vector_db_storage_cls(
                                                            namespace="entities",
                                                            global_config=asdict(self),
                                                            embedding_func=self.embedding_func,
                                                            meta_fields={"entity_name"},
                                                        )
        # relationships 向量库
        self.relationships_vdb = self.vector_db_storage_cls(
                                                                namespace="relationships",
                                                                global_config=asdict(self),
                                                                embedding_func=self.embedding_func,
                                                                meta_fields={"src_id", "tgt_id"},
                                                            )
        # chunks 的存储 向量库 NanoVectorDBStorage
        self.chunks_vdb = self.vector_db_storage_cls(
                                                        namespace="chunks",
                                                        global_config=asdict(self),
                                                        embedding_func=self.embedding_func,
                                                    )
        # 封装LLM的异步调用模式
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
                                                                                partial(
                                                                                    self.llm_model_func,
                                                                                    hashing_kv=self.llm_response_cache,
                                                                                    **self.llm_model_kwargs,
                                                                                )
                                                                            )
    # 返回存储的类 cls
    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }
    # 插入数据，可以是字符串，也可以是字符串列表
    def insert(self, string_or_strings):
        # 获取事件循环
        loop = always_get_an_event_loop()
        # 异步插入
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            # 如果是字符串，变成列表
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # 计算整个文档的hash值
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            # 过滤重复文档， 文档级别的
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            # 生成不重复的文档
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            
            update_storage = True
            # 开始更新
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")
            # 初始化 待插入的chunk
            inserting_chunks = {}
            
            for doc_key, doc in new_docs.items():
                # 构建 chunks 字典，每个字典维护内容的
                #           md5hash（key）: { "tokens": 长度, "content": 内容, "chunk_order_index", chunk的idx "full_doc_id": doc_key}
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                                                            **dp, # 把字典数据解包 
                                                                            "full_doc_id": doc_key,
                                                                        }
                    # 把文档分块 返回的是一个 [{chunk 1}, ...., {chunk 2}]
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                # 更新chunk字典
                inserting_chunks.update(chunks)
            # 获得不重复的chunk 数据
            _add_chunk_keys = await self.text_chunks.filter_keys(
                                                                    list(inserting_chunks.keys())
                                                                )
            # 获得需要插入的chunk
            inserting_chunks = {
                                    k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
                                }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            # 开始插入chunk
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            # 插入chunk到向量数据库
            await self.chunks_vdb.upsert(inserting_chunks)
            # 开始实体抽取
            logger.info("[Entity Extraction]...")
            # 抽取实体
            maybe_new_kg = await extract_entities(
                                                    inserting_chunks,
                                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                                    entity_vdb=self.entities_vdb,
                                                    relationships_vdb=self.relationships_vdb,
                                                    global_config=asdict(self),
                                                )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            # 复制给 graph_storage
            self.chunk_entity_relation_graph = maybe_new_kg
            # 插入文档
            await self.full_docs.upsert(new_docs)
            # 插入文本chunk
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            # 如果更新
            if update_storage:
                # 完成更新
                await self._insert_done()
    # 通知各个存储实例完成索引更新的回调操作。
    # 这通常是在数据插入或更新操作完成后，进行的清理或后续处理步骤，用于确保存储系统的状态一致性或触发必要的后续行为。
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            # index_done_callback() 是实际的回调函数，可能用于刷新索引、清理缓存或触发其他必要的操作。
            # cast(StorageNameSpace, storage_inst) 明确声明存储实例属于 StorageNameSpace 类型，可能是为了静态类型检查。
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        # 使用 asyncio.gather 并发执行所有存储实例的回调任务。asyncio.gather 会等待所有任务完成。
        await asyncio.gather(*tasks)
    # 查询 异步查询
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "local":
            response = await local_query(
                                        query,
                                        self.chunk_entity_relation_graph,
                                        self.entities_vdb,
                                        self.relationships_vdb,
                                        self.text_chunks,
                                        param,
                                        asdict(self), # 把自己的属性包裹成字典？
                                    )
        elif param.mode == "global":
            response = await global_query(
                                            query,
                                            self.chunk_entity_relation_graph,
                                            self.entities_vdb,
                                            self.relationships_vdb,
                                            self.text_chunks,
                                            param,
                                            asdict(self),
                                        )
        elif param.mode == "hybrid":
            response = await hybrid_query(
                                            query,
                                            self.chunk_entity_relation_graph,
                                            self.entities_vdb,
                                            self.relationships_vdb,
                                            self.text_chunks,
                                            param,
                                            asdict(self),
                                        )
        elif param.mode == "naive":
            response = await naive_query(
                                            query,
                                            self.chunks_vdb,
                                            self.text_chunks,
                                            param,
                                            asdict(self),
                                        )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response
    # 异步回调用，需要异步等待
    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    # 删除实体
    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))
    # 实际异步删除实体
    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            # 实体删除
            await self.entities_vdb.delete_entity(entity_name)
            # 关系删除
            await self.relationships_vdb.delete_relation(entity_name)
            # 删除节点
            await self.chunk_entity_relation_graph.delete_node(entity_name)
            # 打印日志
            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")
    # 异步回调
    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
