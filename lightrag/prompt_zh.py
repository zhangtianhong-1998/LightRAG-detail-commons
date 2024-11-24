GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

# 不完整，别使用
PROMPTS["entity_extraction"] = """-目标-
给定一个可能与该活动相关的文本文档以及一列实体类型，从文本中识别这些类型的所有实体及其关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- 实体名称：实体的名称，与输入文本使用相同的语言。如果是英文，则首字母大写。
- 实体类型：以下类型之一：[ {entity_types} ]
- 实体描述：全面描述实体的属性和活动。
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从第1步中识别出的实体中，识别出所有（source_entity, target_entity）对，这些对之间存在明确的关系。
对于每对相关的实体，提取以下信息：
- source_entity：第1步中识别出的源实体名称。
- target_entity：第1步中识别出的目标实体名称。
- relationship_description：解释为什么认为源实体和目标实体相关。
- relationship_strength：表示源实体和目标实体之间关系强度的数值评分。
- relationship_keywords：一个或多个高层次关键词，用于概括关系的总体性质，重点放在概念或主题上而非具体细节。
将每对关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别总结整篇文本主要概念、主题或话题的高层次关键词。这些关键词应捕捉文档中出现的总体思想。
将内容级关键词格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 按步骤1和步骤2中识别的所有实体和关系生成一个英文的列表作为输出。使用 **{record_delimiter}** 作为列表分隔符。

5. 输出结束时，返回 {completion_delimiter}

######################
-示例-
######################
示例1:

Entity_types: [person, technology, mission, organization, location]
Text:
当亚历克斯咬紧牙关时，泰勒权威自信的背景音下浮现出一丝钝化的挫败感。这种竞争性的暗流让他保持警觉，仿佛他和乔丹对发现的共同承诺是对克鲁兹控制和秩序的狭隘愿景的一种无声反叛。

然后，泰勒做了一件意想不到的事。他们停在乔丹旁边，片刻之间，以某种近乎敬畏的态度观察了设备。“如果这项技术能被理解……”泰勒的声音更低了一些，“这可能会改变我们的游戏规则。对我们所有人来说。”

之前潜藏的轻视似乎动摇了，取而代之的是对手中事物的重视与不情愿的尊重。乔丹抬起头，片刻间，他与泰勒对视，意志的无声冲突软化成一种不安的休战。

这是一个微小的变化，几乎察觉不到，但亚历克斯内心点头，他们都是由不同的道路带到这里。
#############
Output:
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人"{tuple_delimiter}"亚历克斯是一个对角色间动态变化感知敏锐的人，因泰勒的权威而有内心反应。"){record_delimiter}
("entity"{tuple_delimiter}"泰勒"{tuple_delimiter}"人"{tuple_delimiter}"泰勒展现出权威自信，但对设备表现出敬畏，体现了一种态度的转变。"){record_delimiter}
("entity"{tuple_delimiter}"乔丹"{tuple_delimiter}"人"{tuple_delimiter}"乔丹与泰勒有直接交互，对发现的共同承诺构成了故事的一部分。"){record_delimiter}
("entity"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"人"{tuple_delimiter}"克鲁兹代表控制和秩序的愿景，并影响了其他角色的动态。"){record_delimiter}
("entity"{tuple_delimiter}"该设备"{tuple_delimiter}"技术"{tuple_delimiter}"该设备是故事的核心，具有潜在改变规则的意义，并受到泰勒的尊重。"){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"泰勒"{tuple_delimiter}"亚历克斯因泰勒的权威发生内心反应，并注意到泰勒态度的变化。"{tuple_delimiter}"权力动态，态度转变"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"乔丹"{tuple_delimiter}"亚历克斯和乔丹在发现上的共同承诺，形成对克鲁兹愿景的反叛。"{tuple_delimiter}"共同目标，反叛"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"乔丹"{tuple_delimiter}"泰勒和乔丹在设备上有直接互动，体现出相互尊重与不安的休战。"{tuple_delimiter}"冲突化解，相互尊重"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"乔丹"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"乔丹在发现上的承诺是对克鲁兹控制和秩序愿景的反叛。"{tuple_delimiter}"意识形态冲突，反叛"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"该设备"{tuple_delimiter}"泰勒对设备表现出敬畏，体现其重要性和潜在影响。"{tuple_delimiter}"敬畏，技术意义"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"权力动态，意识形态冲突，发现，反叛"){completion_delimiter}
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """您是一个有用的助理，负责生成以下提供的数据的全面摘要。
给定一个或两个实体，以及一组描述，它们都与同一个实体或实体组相关。
请将所有这些连接成一个单一的、全面的描述。确保包含从所有描述中收集的信息。
如果提供的描述是矛盾的，请解决矛盾，并提供一个单一的，连贯的总结。
确保它是用第三人称写的，并包括实体名称，以便我们有完整的上下文。

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """许多实体在最后一次提取中被遗漏了。使用相同的格式在下面添加它们：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """似乎仍有一些实体被遗漏了。如果还有需要添加的实体，请回答YES | NO。
"""

PROMPTS["fail_response"] = "对不起，我无法回答这个问题。"

PROMPTS["rag_response"] = """---Role---

---角色---

您是一位帮助用户解答有关所提供表格数据问题的助手。

---目标---

生成一个目标长度和格式的回复，该回复需回答用户的问题，归纳输入数据表中的所有相关信息，并根据回复的长度和格式整合任何相关的一般知识。 如果您不知道答案，请直接说明。不要编造信息。 不要包含没有支持证据的信息。

---目标回复长度和格式---

{response_type}

---数据表格---

{context_data}

根据需要为回复添加适当的章节和注释，并以 Markdown 格式编写内容。
"""

PROMPTS["keywords_extraction"] = """---角色---

您是一名助手，任务是识别用户查询中的高层次和低层次关键词。

---目标---

根据查询，列出高层次和低层次关键词。高层次关键词关注总体概念或主题，而低层次关键词关注具体实体、细节或具体术语。

---指示---

- 以 JSON 格式输出关键词。
- JSON 应包含两个键：
  - "high_level_keywords"：用于总体概念或主题。
  - "low_level_keywords"：用于具体实体或细节。

######################
-示例-
######################
示例 1：

查询："国际贸易如何影响全球经济稳定？"
################
输出：
{
  "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
  "low_level_keywords": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}
#############################
示例 2：

查询："砍伐森林对生物多样性的环境影响是什么？"
################
输出：
{
  "high_level_keywords": ["环境影响", "砍伐森林", "生物多样性丧失"],
  "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
}
#############################
示例 3：

查询："教育在减少贫困中扮演什么角色？"
################
输出：
{
  "high_level_keywords": ["教育", "减少贫困", "社会经济发展"],
  "low_level_keywords": ["学校入学", "识字率", "职业培训", "收入不平等"]
}
#############################
-真实数据-
######################
查询：{query}
######################
输出：

"""

PROMPTS["naive_rag_response"] = """---角色---

您是一位帮助用户解答有关所提供文档问题的助手。

---目标---

生成一个目标长度和格式的回复，该回复需回答用户的问题，归纳输入数据表中的所有相关信息，并根据回复的长度和格式整合任何相关的一般知识。
如果您不知道答案，请直接说明。不要编造信息。
不要包含没有支持证据的信息。

---目标回复长度和格式---

{response_type}

---文档---

{content_data}

根据需要为回复添加适当的章节和注释，并以 Markdown 格式编写内容。
"""
