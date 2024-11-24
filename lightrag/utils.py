import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

ENCODER = None

logger = logging.getLogger("lightrag")

# 设置日志文件路径
def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

# 嵌入函数
@dataclass
class EmbeddingFunc:
    embedding_dim: int  # 嵌入维度
    max_token_size: int # 最大的token size
    func: callable
    # 应该是传入完成嵌入的函数
    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

# 从给定的字符串 content 中提取一个可能的 JSON 格式的字符串，并返回该 JSON 字符串（如果找到的话）
def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None

# 解析 LLM 输出的 JSON
def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    # 是否成功解析 json
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None

# 计算参数的哈希值
def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()

# 计算给定字符串 content 的 MD5 哈希值，并在前面添加一个可选的 prefix（前缀），最终返回生成的字符串。
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

# 这个函数的作用是限制异步函数的并发调用次数，即在同一时间只能有最多 max_size 个调用正在执行。
# 超出并发限制的调用会被阻塞一段时间（由 waiting_time 参数指定），直到有空闲的执行“配额”。
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        # 不使用async。信号量到aovid使用nest-asyncio
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            # 使用了一个局部变量 __current_size 来记录当前正在执行的函数调用数。
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro

# 用于将一个普通函数封装为一个 EmbeddingFunc 对象，并同时为该对象添加额外的属性（通过 kwargs 传递）。
# 它主要是为了便捷地将一个普通的函数转化为 EmbeddingFunc 类型的对象
def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

# 从文件名加载 JSON 数据
def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)

# 将 JSON 数据写入文件
def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

# 使用 tiktoken 编码字符串
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

# 使用tiktoken 解码 token
def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content

# 打包数据
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(args) ]

# 这个函数将输入字符串 content 按照多个标记（markers）分隔，并返回分隔后的子字符串列表。
def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    # re.escape(marker): 将 marker 中的特殊字符（如 .、* 等）转义，以确保它们在正则表达式中按字面值匹配。

    results = re.split("|".join(re.escape(marker) for marker in markers), content)

    # content = "Hello, world! This is Python."
    # markers = [",", ".", "!"]
    #     # print(results)
    # print(split_string_by_multi_markers(content, markers))
    # ['Hello', ' world', ' This is Python', '']
    # ['Hello', 'world', 'This is Python']
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
# 清理字符串
def clean_str(input: Any) -> str:
    # 通过删除HTML转义字符、控制字符和其他不需要的字符来清理输入字符串。
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

# 这个函数通过正则表达式检查一个字符串是否可以被解析为浮点数。
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

# 根据token数量截断列表
def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

# 这个函数将一个二维列表（列表的列表）转换为 CSV 格式的字符串。
def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    # 返回一个字符串，内容是输入的二维列表按照 CSV 格式表示的结果
    return output.getvalue()

# 将字符串转换为 CSV 格式
def csv_string_to_list(csv_string: str) -> List[List[str]]:
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]

# 将数据保存到文件
def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# def xml_to_json(xml_file):
#     try:
#         tree = ET.parse(xml_file) #加载 XML 文件并提取根节点
#         root = tree.getroot()

#         # Print the root element's tag and attributes to confirm the file has been correctly loaded
#         print(f"Root element: {root.tag}")
#         print(f"Root attributes: {root.attrib}")
#         # 初始化结果数据结构
#         data = {"nodes": [], "edges": []}

#         # Use namespace findall 和 find 方法需要传入正确的命名空间，否则可能找不到目标节点。
#         namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

#         for node in root.findall(".//node", namespace):
#             node_data = {
#                 "id": node.get("id").strip('"'),
#                 "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
#                 if node.find("./data[@key='d0']", namespace) is not None
#                 else "",
#                 "description": node.find("./data[@key='d1']", namespace).text
#                 if node.find("./data[@key='d1']", namespace) is not None
#                 else "",
#                 "source_id": node.find("./data[@key='d2']", namespace).text
#                 if node.find("./data[@key='d2']", namespace) is not None
#                 else "",
#             }
#             data["nodes"].append(node_data)

#         for edge in root.findall(".//edge", namespace):
#             edge_data = {
#                 "source": edge.get("source").strip('"'),
#                 "target": edge.get("target").strip('"'),
#                 "weight": float(edge.find("./data[@key='d3']", namespace).text)
#                 if edge.find("./data[@key='d3']", namespace) is not None
#                 else 0.0,
#                 "description": edge.find("./data[@key='d4']", namespace).text
#                 if edge.find("./data[@key='d4']", namespace) is not None
#                 else "",
#                 "keywords": edge.find("./data[@key='d5']", namespace).text
#                 if edge.find("./data[@key='d5']", namespace) is not None
#                 else "",
#                 "source_id": edge.find("./data[@key='d6']", namespace).text
#                 if edge.find("./data[@key='d6']", namespace) is not None
#                 else "",
#             }
#             data["edges"].append(edge_data)

#         # Print the number of nodes and edges found
#         print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

#         return data
#     except ET.ParseError as e:
#         print(f"Error parsing XML file: {e}")
#         return None
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None
# 该函数将 XML 格式的图数据转换为 JSON 格式的字典结构。
def xml_to_json(xml_file: str):
    """
    Parse a GraphML XML file and convert its nodes and edges to a JSON-like dictionary structure.

    Args:
        xml_file (str): Path to the XML file to parse.

    Returns:
        dict: A dictionary with "nodes" and "edges" lists.
        None: If there is an error during parsing.
    """
    try:
        # Parse the XML file and get the root element
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Define the namespace for GraphML
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        # Initialize the output data structure
        data = {
            "nodes": [],
            "edges": []
        }

        # Helper function to safely extract text from a specific <data> tag
        def get_data_value(element, key, default=""):
            tag = element.find(f"./data[@key='{key}']", namespace)
            return tag.text.strip('"') if tag is not None and tag.text else default

        # Extract nodes
        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id", "").strip('"'),
                "entity_type": get_data_value(node, "d0"),
                "description": get_data_value(node, "d1"),
                "source_id": get_data_value(node, "d2")
            }
            data["nodes"].append(node_data)

        # Extract edges
        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source", "").strip('"'),
                "target": edge.get("target", "").strip('"'),
                "weight": float(get_data_value(edge, "d3", default="0.0")),
                "description": get_data_value(edge, "d4"),
                "keywords": get_data_value(edge, "d5"),
                "source_id": get_data_value(edge, "d6")
            }
            data["edges"].append(edge_data)

        # Print summary of extracted data
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
# 该函数的目的是从两个 CSV 字符串 hl 和 ll 中提取内容，合并它们的数据，同时去除重复项，重新生成一个带编号的 CSV 格式字符串。
def process_combine_contexts(hl, ll):
    header = None
    # 1. 提取 CSV 数据
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]  # 移除表头
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]  # 移除表头
    if header is None:
        return ""
    # 遍历 list_hl 和 list_ll 的每一行，跳过首列（通常是编号），只保留数据部分。
    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item] 
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()
    # 4. 合并数据，去重
    for item in list_hl + list_ll:
        # 塞进集合
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)
    # 第一行加入表头（用 ",\t".join(header) 拼接表头列）。
    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        # 填入数据 遍历 combined_sources，为每一行内容添加编号（从 1 开始）。
        combined_sources_result.append(f"{i},\t{item}")
    # 6. 返回最终字符串
    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result
