import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    locate_json_string_body_from_string,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS

# 按token后的数据进行划分
def chunking_by_token_size(content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"):
    # 拿到token
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        # 分块
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

# 处理实体和关系的摘要
async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    # 对描述进行分析
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    # 判断是不是超出
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    # 超出后进行描述重新总结
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    # 获得描述文本
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )

    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # 如果，长度小于4 且 第一个不是entity，则不处理
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper()) # 处理第一个，转化为小写
    # 判断实体名称
    if not entity_name.strip():
        return None
    # 清理类型
    entity_type = clean_str(record_attributes[2].upper())
    # 获得实体描述
    entity_description = clean_str(record_attributes[3])
    # 返回实体的chunk_key
    entity_source_id = chunk_key

    return dict(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    description=entity_description,
                    source_id=entity_source_id,
                )

# 关系抽取
async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id, # source_id！！！！！！！！
    )

# 合并节点和插入
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    # 查询
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])

        already_source_ids.extend(split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP]))

        already_description.append(already_node["description"])

    entity_type = sorted(
                            # 统计列表中每个元素的出现次数。结果是一个字典，键是实体类型，值是其出现次数
                            Counter([dp["entity_type"] for dp in nodes_data] + already_entitiy_types).items(),
                            key=lambda x: x[1], # 根据结果排序
                            reverse=True,
                        )[0][0] # 找到频率最高的实体类型
    # 合并描述
    description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in nodes_data] + already_description)))
    # 合并source id
    source_id = GRAPH_FIELD_SEP.join(set([dp["source_id"] for dp in nodes_data] + already_source_ids))
    # 处理实体描述--超出描述阈值才处理
    description = await _handle_entity_relation_summary(entity_name, description, global_config)

    node_data = dict(
                        entity_type=entity_type,
                        description=description,
                        source_id=source_id,
                    )
    # 插入节点
    await knowledge_graph_inst.upsert_node(entity_name, node_data=node_data,)
    # 节点数据实体名称复制
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    # 获取已经存在的数据
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    # 拿到，权重，source_id，描述，关键词
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        # 查询
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )
    # 曲终求和
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    # 描述融合
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    # 关键词融合
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    # source_id融合
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    # 判断是否存在节点
    for need_insert_id in [src_id, tgt_id]:
        # 如果不存在节点，需要更新节点数据
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    # 对描述数据进行插入
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    # 插入
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )
    # 返回节点数据
    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data

# 实体抽取函数
async def extract_entities(
    chunks: dict[str, TextChunkSchema], # 文本
    knowledge_graph_inst: BaseGraphStorage, # 图数据库
    entity_vdb: BaseVectorStorage, # 实体数据库
    relationships_vdb: BaseVectorStorage, #关系数据库
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"] # 大模型
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"] 

    ordered_chunks = list(chunks.items())
    # 实体抽取提示词
    entity_extract_prompt = PROMPTS["entity_extraction"]
    # 上下文字典
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"], # PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"], #"##"
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"], # PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]), #  PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"] # 继续提取的提示词
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"] # 如果循环的提示词

    already_processed = 0
    already_entities = 0
    already_relations = 0
    # 处理一个语段
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # 填充啦
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        # 调用大模型
        final_result = await use_llm_func(hint_prompt)
        # 打包数据
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        # 遍历
        for now_glean_index in range(entity_extract_max_gleaning):
            # 重复提取
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            # 打包历史
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
            # 判断是否结束
            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)

            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()

            if if_loop_result != "yes":
                break
        # 分割数据，抽取掉不必要的字段
        records = split_string_by_multi_markers(final_result,[context_base["record_delimiter"], context_base["completion_delimiter"]],)

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            # \( 和 \)：\( 和 \) 是转义字符，用来匹配普通的括号 ( 和 )。 匹配括号里的内容
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            # 捕获括号中的内容，可以通过 group(1) 提取。
            record = record.group(1) 
            # 分割数据
            record_attributes = split_string_by_multi_markers( 
                record, [context_base["tuple_delimiter"]] # "<|>"
            )
            # 抽取一个实体
            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key)

            if if_entities is not None:
                # 添加进节点数据
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            # 如果不是节点就处理关系
            if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key)

            if if_relation is not None:
                # 添加进边
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]

        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings 限制了异步调用
    results = await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        # 对于边进行排序
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    # 所有实体数据合并和插入
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    # 关系插入
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None
    # 如果实体向量 数据库不为空
    if entity_vdb is not None:
        # 计算数据哈希
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        # 将实体数据插入到实体向量数据中
        await entity_vdb.upsert(data_for_vdb)
    # 如果关系向量数据不为空，插入到关系向量数据库中
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                            "src_id": dp["src_id"],
                            "tgt_id": dp["tgt_id"],
                            "content": dp["keywords"]
                            + dp["src_id"]
                            + dp["tgt_id"]
                            + dp["description"],
                        }
                for dp in all_relationships_data
            }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

# local 模式
async def local_query(
    query,    
    knowledge_graph_inst: BaseGraphStorage, # 知识图谱数据
    entities_vdb: BaseVectorStorage, # 实体向量数据
    relationships_vdb: BaseVectorStorage, # 关系向量数据
    text_chunks_db: BaseKVStorage[TextChunkSchema], # 文本块本地缓存数据
    query_param: QueryParam, #查询参数
    global_config: dict, #全局配置
) -> str:
    context = None
    # llm配置函数，这里是个函数
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query) # 填入数据
    result = await use_model_func(kw_prompt) # 调用模型
    json_text = locate_json_string_body_from_string(result) #解析数据
    # 尝试加载json
    try:
        keywords_data = json.loads(json_text) 
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            # 重新进行字符串解析
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[-1].split("}")[0] + "}"
            # 再次尝试加载
            keywords_data = json.loads(result)
            # 重新尝试
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    # 如果解析成功啦！   
    if keywords:
        # 构建一个 local 的上下文
        context = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response

# 构建完整的上下文
async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # 查询实体的向量数据库，拿到所有存储
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    # 是否查询成功
    if not len(results):
        return None
    # 遍历实体查询结果，拿到节点，在图数据库进行查询 # 图数据库的节点就是实体本身
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    # all() 函数检查该布尔列表中的所有值是否均为 True
    if not all([n is not None for n in node_datas]):
        # 弹出空值警告
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    # 查询节点的度
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    # 构建实体数据 ，略过空数据
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram. 
    # 这个text_chunks_db在做什么？不要记得在airvx。检查图表。
    # 查询相关性的文本单元，查找直接相关的文本单元
    #  text_chunks_db：{"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # 找到最相关的边的详细数据
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    # 表头 
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    # 这个函数将一个二维列表（列表的列表）转换为 CSV 格式的字符串。
    entities_context = list_of_list_to_csv(entites_section_list)
    # 构建关系语段
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    # 构建相关性文本语段
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

# 从给定的节点数据中提取文本单元，并通过知识图谱找到与这些节点直接关联的一度节点。
# 计算每个文本单元的关联次数，即该文本单元在一度邻居节点中出现的次数。
# 对文本单元进行排序，优先返回与初始节点关联性更高的文本单元。
# 根据令牌数限制，截断文本单元列表，确保不超过指定的最大令牌数。
# 返回有效的文本数据列表，供后续处理或响应。
async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
): 
    # 文本单元？  GRAPH_FIELD_SEP = "<SEP>"
    # 将节点[source_id]的数据根据 GRAPH_FIELD_SEP = "<SEP>"分块 这个source_id是来源哪个，暂时存疑
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    # 查询实体的边 edges 是一个列表，包含了每个节点的边列表。
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    # 从边信息中提取所有与初始节点直接相连的一度节点。
    all_one_hop_nodes = set() 
    # all_one_hop_nodes 是所有一度邻居节点的集合。
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges]) # e[1] 是邻居节点，就是一度节点

    all_one_hop_nodes = list(all_one_hop_nodes)
    # 并发地获取所有一度节点的详细数据。 
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    # 构建一度节点文本单元查找表 键为节点 ID，值为其文本单元集合。
    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data) # 这一步对比主要是为了防止空值
        if v is not None and "source_id" in v  # Add source_id check
    }
    # 初始化文本单元查找表并计算关联次数 键为文本单元 ID，值为包含其数据、顺序和关联次数的字典。
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        # c_id 是分块后的文本列表
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                # 文本单元 c_id
                all_text_units_lookup[c_id] = { # 这里的data 的返回值要满住TextChunkSchema的要求
                                                # 这意味着data是在jsonKV库查询的的 
                                                # {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
                                                "data": await text_chunks_db.get_by_id(c_id),
                                                "order": index, # order可能会重复！！
                                                "relation_counts": 0,
                                                }
            # this_edges 当前节点的边列表
            if this_edges: # 如果当前节点有边（this_edges 非空），则进行进一步处理。
                for e in this_edges: # 对于当前节点的每条边 e
                    if (
                        e[1] in all_one_hop_text_units_lookup # 如果邻居在 即与当前节点直接相连的节点。 
                        and c_id in all_one_hop_text_units_lookup[e[1]] # c_id是否存在于一度邻居节点的文本单元查找表中。
                    ): 
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content # 过滤控制数据
    all_text_units = [
        {"id": k, **v} # 对V解包
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]
    # 确保至少有一个有效文本单元
    if not all_text_units:
        logger.warning("No valid text units found")
        return []
    # 主排序键： 按照 "order" 字段的值进行升序排序。即，"order" 值较小的元素排在前面。
    # 次排序键： 在 "order" 值相同的情况下，按照 "relation_counts" 字段的值进行降序排序。由于使用了负号（-x["relation_counts"]），因此 "relation_counts" 值较大的元素排在前面。
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    # 截断
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    # 返回最后的文本数据
    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    # 查询借点对应的边 查找节点数据中所有的相关边
    all_related_edges = await asyncio.gather(*[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas])
    all_edges = []
    seen = set()
    # 
    for this_edges in all_related_edges:
        # 遍历这条变
        for e in this_edges:
            # 确保排序
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)
    # 获取边数据 ，以下是 示例
            # async def get_edge(
            #     self, source_node_id: str, target_node_id: str
            # ) -> Union[dict, None]:
            #     return self._graph.edges.get((source_node_id, target_node_id))
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    # 获取度
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )

    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    # 根据 rank 和 weight 进行排序，逆序
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    # 截断描述
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data

# 全局模式
async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]
    # 调用 LLM 抽取关键词
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    # 获取高级特征
    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[-1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)

        except json.JSONDecodeError as e:
            # Handle parsing error
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_global_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=context, response_type=query_param.response_type)
    # 调用大模型
    response = await use_model_func(query, system_prompt=sys_prompt,)
    # very 暴力！
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_global_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # 高级特征查询需要先查询关系 根据关键词的相似性匹配查询
    #   低级特征是查询实体
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return None
    # 根据（点，点）关系查询边数据
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    # 查询边的度
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )

    edge_datas = [
                    {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
                    for k, v, d in zip(results, edge_datas, edge_degree)
                    if v is not None
                ]
    # 
    edge_datas = sorted(edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True)
    # 截断边的详细数据
    edge_datas = truncate_list_by_token_size(
                                                edge_datas,
                                                key=lambda x: x["description"],
                                                max_token_size=query_param.max_token_for_global_context,
                                            )
    # 查找边的相关实体
    use_entities = await _find_most_related_entities_from_relationships(edge_datas, query_param, knowledge_graph_inst)
    # 查找边的节点对应的文本数据
    use_text_units = await _find_related_text_unit_from_relationships(edge_datas, query_param, text_chunks_db, knowledge_graph_inst)

    logger.info(f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units")
    # 构建表头
    relations_section_list = [["id", "source", "target", "description", "keywords", "weight", "rank"]]
    # 构建边的关系
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
                                        [
                                            i,
                                            e["src_id"],
                                            e["tgt_id"],
                                            e["description"],
                                            e["keywords"],
                                            e["weight"],
                                            e["rank"],
                                        ]
                                    )
    relations_context = list_of_list_to_csv(relations_section_list)
    # 构建实体的关系
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    # 构建文本的关系
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""

# 找到边对应的节点的详细数据
async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set() 
    # 构建一个节点相关的集合和列表，防止重复
    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])
    # 找到节点的详细数据
    node_datas = await asyncio.gather(
                                    *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
                                )
    # 查找节点的度
    node_degrees = await asyncio.gather(*[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names])
    # 
    node_datas = [
                    {**n, "entity_name": k, "rank": d}
                    for k, n, d in zip(entity_names, node_datas, node_degrees)
                ]
    # 截断节点的描述，最后返回节点详细数据
    node_datas = truncate_list_by_token_size(
                                                node_datas,
                                                key=lambda x: x["description"],
                                                max_token_size=query_param.max_token_for_local_context,
                                            )

    return node_datas

# 从关系中查找相关文本数据
async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    # 文本单元
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id), # 查找文本数据
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    # 截断文本内容
    all_text_units = truncate_list_by_token_size(
                                                    all_text_units,
                                                    key=lambda x: x["data"]["content"],
                                                    max_token_size=query_param.max_token_for_text_unit,
                                                )
    # 规范化返回数据模式
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units

# 混合模式
async def hybrid_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[-1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    # 同时构建高级和低级提示词
    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    # 混合提示
    context = combine_contexts(high_level_context, low_level_context)

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response

# 去重并重新组合
def combine_contexts(high_level_context, low_level_context):
    # Function to extract entities, relationships, and sources from context strings

    def extract_sections(context):
        entities_match = re.search(r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL)
        relationships_match = re.search(r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL)
        sources_match = re.search(r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL)

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts
    # 重新抽取高级特征
    if high_level_context is None:
        warnings.warn("High Level context is None. Return empty High entity/relationship/source")
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn("Low Level context is None. Return empty Low entity/relationship/source")
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)
    # --------------------去重！！！！！！！！！！------------------
    # Combine and deduplicate the entities 合并和去重实体
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships 合并并去重这些关系
    combined_relationships = process_combine_contexts(hl_relationships, ll_relationships)

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    # Format the combined context 重新拼接
    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""

# 普通查询
async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage, # 文本向量数据库
    text_chunks_db: BaseKVStorage[TextChunkSchema], # 文本语段数据
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k) # 查询文本向量数据库
    if not len(results):
        return PROMPTS["fail_response"]
    
    chunks_ids = [r["id"] for r in results] # 重构文本语段ids
    chunks = await text_chunks_db.get_by_ids(chunks_ids) # 查询实际对应的chunk
    # 截断content
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    # 
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    # 仅需要上下文，直接返回
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    # 传入参考的上下文，和回复格式
    sys_prompt = sys_prompt_temp.format(content_data=section, response_type=query_param.response_type)
    # 调用大模型
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    # 好通用的接口，very 暴力
    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response
