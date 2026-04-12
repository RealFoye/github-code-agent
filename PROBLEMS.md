# 问题记录文档

> 记录项目开发过程中遇到的问题和解决方案，持续更新。

---

## 阶段一：搜索推荐

### 问题 1：PyGithub 模块未安装

**错误信息：**
```
ModuleNotFoundError: No module named 'github'
```

**原因：** 未安装 PyGithub 依赖包。

**解决方法：**

安装所需的依赖包：

```bash
pip install PyGithub requests openai python-dotenv
```

| 包名 | 用途 |
|------|------|
| PyGithub | GitHub API Python 封装 |
| requests | HTTP 请求库 |
| openai | OpenAI API 调用 |
| python-dotenv | .env 环境变量管理 |

**发生时间：** 2026/04/09

---

### 问题 2：PyGithub 异常类名错误

**错误信息：**
```
ImportError: cannot import name 'RateLimitExceededError' from 'github'
Did you mean: 'RateLimitExceededException'?
```

**原因：** PyGithub 库中限流异常的类名是 `RateLimitExceededException`，不是 `RateLimitExceededError`。错误地参考了其他库的命名习惯。

**解决方法：**

修改 `src/tools/github_search.py` 中的导入语句（第 12-13 行）：

```python
# 修改前（错误）
from github import Github, GithubException
from github.RateLimit import RateLimitExceededError

# 修改后（正确）
from github import Github, GithubException, RateLimitExceededException
```

同时修改第 132 行抛出异常的位置：

```python
# 修改前（错误）
raise RateLimitExceededError()

# 修改后（正确）
raise RateLimitExceededException()
```

**验证修复：**
```python
python3 -c "from github import RateLimitExceededException; print('OK')"
```

**发生时间：** 2026/04/09

---

## 变更记录（2026/04/12）

### 变更 1：IntentRecognizer 升级为 LLM 驱动版本

**变更描述：**
将 `src/agent/search_agent.py` 中的 `IntentRecognizer` 从纯规则匹配升级为 LLM 驱动版本。

**变更内容：**

1. **`UserIntent` 增加新字段**（可选，向后兼容）：
   - `project_type`: 项目类型（框架/工具/库/教程/Demo）
   - `difficulty`: 难度（入门友好/中级/高级）
   - `purpose`: 用途（学习参考/生产使用/研究）
   - `tech_stack`: 技术栈列表（如 `["React", "Node.js"]`）
   - `summary`: LLM 生成的意图摘要（供用户确认）

2. **`IntentRecognizer` 新增 `__init__`**：
   - 接受可选的 `LLMService` 参数
   - 不传时自动创建 `LLMService()`，初始化失败则降级为 `None`

3. **新增方法**：
   - `_llm_parse()`: LLM 驱动解析，返回结构化 `UserIntent`
   - `_format_summary()`: 将 LLM JSON 结果格式化为可读摘要
   - `_rule_parse()`: 原规则匹配逻辑，重命名保留作为 fallback

4. **`parse()` 方法**：先调用 `_llm_parse()`，失败则自动降级到 `_rule_parse()`

5. **`SearchRecommendationAgent.run()`**：若 `intent.summary` 非空则打印摘要

**示例效果：**
```
用户：我想找一个适合新手学习的全栈项目，最好是用 React 和 Node.js 的

我理解你想找：
  - 类型：全栈应用
  - 技术栈：React + Node.js
  - 难度：入门友好
  - 目的：学习参考

正在为你搜索匹配的 GitHub 项目...
```

**测试覆盖：**
- `TestIntentRecognizerRuleBased`（13 个用例）：验证规则匹配 fallback 正常工作
- `TestIntentRecognizerLLM`（9 个用例）：验证 LLM 解析、语言标准化、fallback 触发
- 所有 33 个测试全部通过

**文件变更：**
- `src/agent/search_agent.py`：IntentRecognizer 升级
- `tests/test_search_agent.py`：新增 LLM 相关测试

---


## 阶段二：深度代码分析

### 问题 1：tree-sitter-languages 与 tree-sitter 版本不兼容

**错误信息：**
```
初始化 python 解析器失败: __init__() takes exactly 1 argument (2 given)
```

**原因：** tree-sitter 0.25.x 与 tree-sitter-languages 1.10.2 不兼容。tree-sitter-languages 需要 tree-sitter 0.21.x 版本。

**解决方法：**

方案一：降级 tree-sitter 到兼容版本
```bash
pip install 'tree-sitter<0.22'
```

方案二：修改代码，使用纯正则表达式切片（当前采用）
- `_chunk_by_regex()` 使用正则表达式匹配函数签名
- 缺点：只能匹配函数签名，不能获取完整的函数体范围
- 影响：RAG 召回时只能按整文件切片，粒度不够精细

**验证 tree-sitter 是否正常工作：**
```python
from tree_sitter_languages import get_parser
parser = get_parser('python')  # 如果报错则不兼容
```

**验证结果：** ✅ 已修复（2026/04/12 验证）
- 当前安装版本：tree-sitter 0.21.3, tree-sitter-languages 1.10.2
- AST 解析正常工作，函数提取正常
- 注意：tree-sitter 0.25.x 存在兼容性问题，已通过降级到 0.21.x 解决

**发生时间：** 2026/04/09

---

### 问题 2：OpenAI API Key 未配置

**错误信息：**
```
OpenAI Embedding 失败: Error code: 401 - {'error': {'message': "You didn't provide an API key...
openai.AuthenticationError: Error code: 401
```

**原因：** 未在 `.env` 文件中配置 `OPENAI_API_KEY`。

**解决方法：**

在项目根目录创建 `.env` 文件：
```bash
cp .env.example .env
```

然后编辑 `.env`，填入你的 API Key：
```
OPENAI_API_KEY=sk-your-api-key-here
```

**验证配置是否正确：**
```python
from src.config import get_llm_config
print(get_llm_config())  # 应该显示 api_key
```

**发生时间：** 2026/04/09

---

### 问题 3：HuggingFace 模型下载问题

**问题描述：**
- bge-m3 模型（2.5GB）从中国大陆下载极慢且经常中断
- hf-mirror.com 镜像对该模型支持不稳定

**解决方案：**
改用更小更稳定的模型 `all-MiniLM-L6-v2`：
- 大小：~90MB
- 维度：384
- 效果：MTEB 排行榜开源免费前五

**验证命令：**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
print('Dim:', model.get_sentence_embedding_dimension())  # 输出 384
```

**配置更新（config.py）：**
```python
HF_MODEL_NAME = "all-MiniLM-L6-v2"  # 原为 BAAI/bge-m3
```

**发生时间：** 2026/04/10

---

### 问题 4：`calls` / `called_by` 调用关系未填充

**问题描述：**
- `CodeChunk` 有 `calls` 和 `called_by` 字段，但始终为空
- 原因：tree-sitter query 语法错误 + 捕获位置错误

**原因 1：tree-sitter query 括号不平衡**
```
# 原代码 - 第一个 pattern 的 [ 没有闭合
[(call ... @call
(call ... @call]   <-- 这个 ] 实际上关闭了第一个 [
```

**原因 2：@capture 位置错误**
```
# 错误：@name 捕获的是 identifier 节点
(call function: (identifier) @name) @call

# 正确：@call 应该放在 call 节点的 CLOSING ) 之后
(call function: (identifier)) @call
```

**原因 3：JavaScript/TypeScript 使用 `call_expression` 不是 `call`**
```python
# JavaScript 语法树中，函数调用是 call_expression，不是 call
# 且 function 是直接子节点，不是字段
```

**解决方法：**
1. 修复所有语言的 query，平衡括号
2. 修正 @capture 位置，确保捕获 call 节点本身
3. JavaScript/TypeScript 使用 `call_expression` 并遍历子节点提取函数名

**验证修复：**
```python
from src.tools.chunker import Chunker
chunker = Chunker()
chunks = chunker.chunk_file('test.py')  # 包含函数调用
for c in chunks:
    print(f'{c.name}: calls={c.calls}, called_by={c.called_by}')
# foo: calls=['helper'], called_by=[]
# helper: calls=[], called_by=['foo']
```

**验证结果：** ✅ 已修复（2026/04/12 验证）
- 测试用例：定义 `helper()` 和 `foo()`，其中 `foo()` 调用 `helper()`
- 实际结果：`helper: called_by=['foo']`，`foo: calls=['helper']` —— 关系正确

**发生时间：** 2026/04/11

---

### 问题 5：`max_file_size` 阈值过小导致 README 被跳过

**问题描述：**
- 分析 `yzfly/Awesome-MCP-ZH` 时，问答回答不正确
- 原因：README.md（206KB）被 `max_file_size=100_000`（100KB）阈值跳过
- 只索引了 LICENSE 和 .gitignore，搜不到 README 内容

**解决方法：**
将 `chunk_directory` 的 `max_file_size` 从 100KB 提高到 500KB：

```python
# 修改前
def chunk_directory(self, dir_path: str, max_file_size: int = 100_000)

# 修改后
def chunk_directory(self, dir_path: str, max_file_size: int = 500_000)
```

**影响：**
- 500KB 以内的文档文件（如 README.md）都能被索引
- 正常源代码文件很少超过 500KB，不会有 OOM 风险

**验证修复：**
```python
# 100KB 阈值：只索引 2 个 chunks（缺 README）
# 500KB 阈值：索引 3 个 chunks（包含 README）
```

**发生时间：** 2026/04/11

---

### 问题 6：`RepoCloner._protected_paths` 类变量定义位置错误

**错误信息：**
```
AttributeError: type object 'RepoCloner' has no attribute '_protected_paths'
```

**原因：**
`_protected_paths` 定义在类外面的模块级别，但类方法中使用 `cls._protected_paths` 引用

**解决方法：**
将 `_protected_paths` 移入 `RepoCloner` 类内部作为类变量：

```python
class RepoCloner:
    # 类级别：正在被分析的仓库路径（跨实例共享）
    _protected_paths: set = set()
```

**发生时间：** 2026/04/11

---

## 新发现问题（2026/04/12）

### 问题 7：类提取功能缺失

**问题描述：**
- `ASTParser` 有 `extract_functions()` 方法，但没有 `extract_classes()` 方法
- 导致 `CodeParser.parse_file()` 返回的 `classes` 字段始终为空

**验证：**
```python
from src.tools.code_parser import CodeParser
parser = CodeParser()
parsed = parser.parse_file('sample.py')  # 包含 class MyClass:
print(parsed.classes)  # 输出: None
```

**发生时间：** 2026/04/12

---

### 问题 8：函数签名和 docstring 未提取

**问题描述：**
- `CodeChunk.signature` 和 `CodeChunk.docstring` 字段存在但始终为空
- `_chunk_source_code()` 方法没有从 AST 中提取签名和文档字符串

**发生时间：** 2026/04/12

---

### 问题 9：代码片段真实性校验不完整

**问题描述：**
- `_verify_sources()` 只检查行号范围是否在文件内
- 不校验检索返回的 `chunk.content` 是否与文件中实际内容一致
- 不校验 LLM 回答中提到的函数/类是否真实存在

**代码位置：** `analysis_agent.py` 第 347-376 行

**发生时间：** 2026/04/12

---

### 问题 10：GitHub API 限流处理无指数退避

**问题描述：**
- `github_search.py` 的 `_handle_rate_limit()` 只用固定等待时间
- 应该使用指数退避：`retry_delay * (attempt + 1)`

**代码位置：** `github_search.py` 第 50-58 行

**对比：** `llm_service.py` 已有正确的指数退避实现

**发生时间：** 2026/04/12

---

### 问题 11：依赖声明文件解析未实现

**问题描述：**
- `DependencyInfo` 数据模型存在但从未被填充
- `analysis_agent.py` 不解析 requirements.txt / package.json 等依赖文件

**发生时间：** 2026/04/12

---

### 问题 12：报告未生成 Mermaid 架构图

**问题描述：**
- 需求书要求报告包含 "Mermaid 架构图（自动生成）"
- `report_generator.py` 没有生成任何 Mermaid 图

**发生时间：** 2026/04/12

---

### 问题 13：LLM 提取关键词过多导致 GitHub 搜索返回 0 结果 ✅ 已修复

**错误现象：**
```
搜索 GitHub: agent AI agent LLM agent 新手入门 beginner tutorial stars:>=100
找到 0 个仓库，返回 0 个
```

**原因：**
- LLM 提取了 6 个关键词（含中文、多词短语）
- `SearchRecommendationAgent.run()` 直接将所有 keywords `" ".join()` 拼接作为搜索词
- GitHub Search API 将每个词都视为必须匹配的条件，词越多越找不到结果

**解决方法：**
在 `UserIntent` 新增 `search_query` 字段，专门存放面向 GitHub 搜索优化的精简词：

1. LLM prompt 中要求返回 `search_query`（2-3 个英文词，如 `"agent llm beginner"`）
2. 新增 `_build_search_query()` 方法作为 fallback（过滤中文、多词短语取首词、最多 3 个）
3. `SearchRecommendationAgent.run()` 改用 `intent.search_query` 而非 `" ".join(intent.keywords)`

```python
# 修改前（错误）
keywords = " ".join(intent.keywords)  # → "agent AI agent LLM agent 新手入门 beginner tutorial"

# 修改后（正确）
search_query = intent.search_query    # → "agent llm beginner"
```

**发生时间：** 2026/04/12

---

### 问题 14：MiniMax thinking 块导致意图解析降级

**错误现象：**
```
LLM 意图解析失败，降级到规则匹配
LLM 调用失败 [minimax]: 无法从响应中提取文本内容，响应类型: ['thinking']
```

**原因：**
- MiniMax-M2.7 启用 thinking 时返回 `thinking` 块 + `text` 块
- `llm_service.py` 的 `_call_minimax` 只处理 `text` 块，遇到只有 `thinking` 块时报错
- 意图解析 `max_tokens=500` 太小，JSON 输出被截断

**解决方法：**
1. `search_agent.py` 中意图解析 `max_tokens=500` → `2000`
2. `search_agent.py` 中 `_call_minimax` 增加 thinking 块重试逻辑（`max_tokens * 4`）
3. `llm_service.py` 中 `_call_minimax` 同步增加重试逻辑

**验证：** 意图解析成功返回三层结构，无降级

**发生时间：** 2026/04/12

---

### 问题 15：意图分类偏"框架/库"，搜索词含 framework 偏高门槛

**错误现象：**
```
输入：给我一个适合新手学习的 agent 项目
解析结果：type=框架/库, difficulty=入门友好, min_stars=100
搜索词：AI agent framework beginner tutorial stars:>=100  ← framework 不该出现
```

**原因：**
- LLM 将"新手学习 agent 项目"判定为"框架/库"，而用户实际想要的是 tutorial/demo/example 项目
- `min_stars=100` 对新手学习场景偏高，误伤小而清晰的教学仓库
- "framework" 进入搜索词后，候选偏向框架生态而非学习项目

**解决方法：**
1. `run()` 中新增判断：新手学习场景（difficulty="入门友好" 或 purpose="学习参考"）时，`min_stars` 从 100 降至 50
2. `_build_multi_queries()` 中：新手学习场景下将搜索词中的 "framework" 替换为 "example"

```python
# run() 中新增
is_learning_scenario = (
    intent.difficulty in ("入门友好", "入门") or
    intent.purpose in ("学习参考", "学习") or
    any(kw in user_query.lower() for kw in ["新手", "学习", "入门", "学着做", "教程", "示例"])
)
if is_learning_scenario and intent.min_stars > 50:
    intent.min_stars = 50

# _build_multi_queries() 中
if is_learning:
    main_query = main_query.replace("framework", "example")
```

**验证：** 新手学习场景搜索词不再含 framework，star 门槛降至 50

**发生时间：** 2026/04/12

---

### 问题 16：summary 显示与实际执行不一致（min_stars 显示 100 实际搜 50）

**错误现象：**
```
显示：最低 Stars：100
实际：stars:>=50
```

**原因：**
`run()` 中 `min_stars` 的调整发生在 `intent.summary` 打印**之后**，导致用户看到的摘要和实际搜索参数不一致。

**解决方法：**
将 `min_stars` 调整逻辑移到 `print(intent.summary)` **之前**，确保摘要反映实际使用的值。

**发生时间：** 2026/04/12

---

### 问题 17：排序偏好"可动手复刻"——结果偏大型教程集合而非最小可复刻项目

**错误现象：**
- 前两名是 33k stars 和 21k stars 的大型教程仓库（50+ tutorials、notebook 合集）
- 用户实际想找"最适合第一次模仿的小 demo"

**原因：**
排序的 `intent_fit` 对所有学习相关仓库一视同仁，没有对"可动手复刻"特征（demo/example/from-scratch）额外加分。

**解决方法：**
在 `_build_evaluation_prompt()` 中新增学习场景额外偏好引导：
- 对名称含 demo/example/from-scratch/build-your-own/step-by-step 的仓库额外加分（intent_fit +0.1~0.2）
- 对大型教程集合（50+ tutorials、notebook 合集）轻微降权（intent_fit -0.05~0.1）

**发生时间：** 2026/04/12

---

### 问题 18：意图分类偏"框架/工具/库"——非框架类学习项目被误判

**问题描述：**
- 用户说"给我一个适合新手学习的 agent 项目"
- LLM 解析结果：`type=框架/工具/库`
- 更理想的类型应该是：Tutorial/Demo/学习项目/from-scratch 项目

**影响：**
- 搜索词仍含 "framework"
- summary 展示类型不准确
- 排序倾向有偏差

**后续改进方向（未实现）：**
在 `_llm_parse` 的 prompt 中加强对"新手学习场景"的类型判断指导，例如：
- 用户提到"新手"、"学习"、"入门"时，强制映射到 tutorial/demo/project 类型
- 避免将"学习项目"误判为"框架"

**发生时间：** 2026/04/12

---

### 问题 19：用户否定约束完全失效（不要 LangChain / 不是框架库 / 本地跑）

**错误现象：**
- 查询 5："我只想找一个 agent demo，不要 LangChain" → LangChain 相关仓库未被排除
- 查询 4："给我一个适合学习的 agent 项目，不是框架库" → framework 类仓库未被排除
- 查询 3："推荐一个能本地跑的 agent demo" → "本地跑"约束未进入搜索词

**根因：**
1. LLM 把用户否定内容错误放入了 `tech_stack_domain`（如查询 5："tech_stack_domain: ['LangChain']"）
2. `tech_stack_domain` 规则是"不进入评分"，导致否定约束完全失效
3. 没有 `excluded_tech` 和 `excluded_categories` 字段来记录否定约束

**解决方法：**
1. UserIntent 新增 `excluded_tech: List[str]`（用户明确排除的技术/框架）和 `excluded_categories: List[str]`（用户明确排除的仓库类型）
2. `_llm_parse` prompt 新增规则：否定词（不要/不是/别/不含/排除）引导到 `excluded_tech` 或 `excluded_categories`，禁止进入任何 tech_stack 字段
3. `VerifiableGate` 新增 `excluded_tech` 硬过滤（topics → repo name → description 三级检查），日志输出 `dropped_by_excluded_tech=X (langchain=Y)`
4. `_build_evaluation_prompt` 新增 `excluded_categories` 软降权指导
5. 噪音过滤：过滤 `.github` 结尾仓库、描述为空且名称泛化的仓库

**实现文件：** `src/agent/search_agent.py`

**验证：** 查询 4/5/3 重新测试

**残留问题：**
- `excluded_tech` 硬过滤已生效（Q5 LangChain 排除验证通过）：**成立**
- `excluded_categories`（否定类型）软降权仍不可靠：LLM 意图解析时对"不是 X"的类型否定识别率低，即使识别了软 penalty 力度也不足以将无关仓库挤出前 5
- **不升级为硬过滤**：framework/demo/tutorial 在 GitHub 元数据里本身边界模糊，硬过滤会误杀真正适合学习的仓库，违背"只有能从元数据可靠验证的约束才进 Gate"原则
- 继续保留为排序层信号，先提高识别率与软降权效果，等确认有稳定、可验证的类别判定规则后再讨论部分硬化

**发生时间：** 2026/04/12

---

### 问题 20：JSON 解析失败率高，导致 fallback 到 star-based ranking

**错误现象：**
- 查询 Q1："Python 写的、适合新手学习的 agent 项目" → JSON 解析失败，`Expecting ',' delimiter`，fallback 到 star-based，结果全是 matplotlib/python 基础教程
- 查询 Q4（历史）：JSON 解析失败，`Expecting ',' delimiter`，fallback 到 star-based
- 离线模拟测试：Q2 风格（suitability_reasons 含特殊字符，数组截断）和 Q4 风格（尾随多余文字）是最常见的坏 JSON 模式

**根因：**
MiniMax LLM 返回的 suitability_reasons 数组中常含中文引号`""`、括号`（）`等特殊字符，导致 JSON 解析器在遇到不完整的字符串时报告 `Expecting ',' delimiter`。同时 LLM 输出有时会被截断，导致数组不完整。

**解决方法：**
新增 `_cleanup_json_string` 方法（`_parse_evaluation` 内），在 `json.loads` 失败后尝试：
1. 去除 markdown code block 标记（` ```json` / ` ``` `）
2. 截取最外层完整 `{...}` 对象
3. 从 `full_name` 字段位置反向查找对象起点
4. 仍失败才 fallback

**实现文件：** `src/agent/search_agent.py`

**离线验证结果：**

| 坏 JSON 模式 | 修复方法 | 结果 |
|-------------|---------|------|
| Q2 风格（suitability_reasons 含特殊字符，数组截断） | full_name_scan | ✅ 救回 |
| Q4 风格（尾随多余文字） | brace_crop | ✅ 救回 |
| Markdown code block 包裹 | direct_ok | ✅ 救回 |
| 不完整尾部（中文字符串截断） | no_fix | ❌ 仍失败 |

**Live Test 结果（3 查询）：**

| 查询 | 触发 cleanup | 解析成功 | fallback |
|------|-------------|---------|---------|
| Q1: Python/新手 agent | ❌ 否 | ✅ 是 | ❌ 否 |
| Q2: 不是框架库 | ❌ 否 | ✅ 是 | ✅ 是（评估阶段 529 overload 错误导致） |
| Q3: 小而美 | ❌ 否 | ✅ 是 | ❌ 否 |

**残留问题：**
- `_cleanup_json_string` 在本次 3 个查询中**未被触发**（JSON 解析本身就成功了）
- Q2 fallback 的根因是 LLM 评估阶段的 529 overload 错误，而非 JSON 格式问题
- **新发现**：Q2 出现"JSON 解析成功但仍 fallback"现象，说明存在未排查的后置降级路径
- cleanup 逻辑在本次测试中**未观察到实际救回**的真实 case

**下一步改进方向：**
1. 排查并修正"JSON 已解析成功但仍 fallback"的触发条件（可能是 evaluations 列表 key 与候选仓库不匹配、某字段缺失导致评分异常等）
2. 从源头降低 ranker 输出非标准 JSON 的概率（不限于中文引号，还可能有：代码块包裹、解释文字前后缀、末尾截断、漏逗号等）

**发生时间：** 2026/04/12

---

## 设计讨论记录（2026/04/12）

> **写给后来看这份文档的人：**
> 这一节记录的不是 Bug，而是在系统能跑起来之后，发现了一批更隐蔽的设计层问题。
> 这些问题不会报错，但会让系统悄悄朝错误方向运行，而且越用越偏。
> 每个问题都记录了：当时发现了什么现象、为什么觉得这是个问题、思考过程是什么、最终想到了什么改进方案。
> 截至记录时尚未落代码，留在这里作为后续实现的依据和设计备忘。

---

### 设计问题 A：LLM 把自己的"行业常识"当成了用户的需求

**起因——一次让人疑惑的意图解析结果**

用户在搜索框输入了一句很普通的话：

> 给我一个适合新手学习的 agent 项目

系统的 `IntentRecognizer` 用 LLM 解析这句话后，输出了：

```
技术栈：LLM + AI Agent + LangChain + OpenAI API
```

看到这个结果，第一反应是：**用户根本没提 LangChain，也没提 OpenAI API，这是哪来的？**

答案是：LLM 自己补进去的。在 LLM 的训练数据里，"AI Agent 项目"就是和 LangChain、OpenAI API 强关联的，所以它把这个领域的主流技术栈当成了用户的需求，顺手填进了结果里。

**为什么这件事比看起来更严重**

表面上看，LLM 只是"猜了一下用户可能想用什么框架"，好像无伤大雅。但这个推断一旦进入系统，会在两个地方造成真实伤害：

第一，**搜索层**：如果搜索词里带了 LangChain 相关的倾向，GitHub 搜索就会倾向于返回 LangChain 生态的仓库，直接漏掉那些更适合新手、不依赖大框架的轻量 demo 项目。

第二，**排序层**：如果 LLMRelevanceRanker 在评分时把 LangChain 当成了加分项，那些没用 LangChain 的仓库就会被无端扣分，哪怕它们在文档质量、代码清晰度上明显更适合新手。

更糟糕的是：**这两层的偏差用户完全看不见**。系统表面上还是在"帮用户找仓库"，但搜到的、排出来的，已经是被 LLM 的行业偏见悄悄过滤过的结果。

**想清楚根本原因之后**

这个问题的根本不是 LLM 太聪明或太笨，而是系统在设计上没有区分**三种性质完全不同的信息**：

| 信息类型 | 来源 | 举例 | 能用来做什么 |
|---------|------|------|------------|
| **用户明确说的**（explicit） | 用户原话直接包含 | "Python 项目"、"MIT 许可证" | 可以用来硬过滤、评分、展示给用户 |
| **从用户原话合理推断的**（strong inference） | 用户没直说，但原话有明确指向 | "本地跑" → 本地模型、"新手学习" → tutorial 类型 | 只能作为加分项，不能扣分，展示时要标注"系统推断" |
| **LLM 的领域常识**（domain knowledge） | LLM 训练数据中的行业默认 | "agent 项目" → LangChain | **禁止进入过滤和评分**，不展示给用户 |

当前系统把这三类信息全部混在一起塞进了 `tech_stack` 字段，LLM 输出什么就用什么，没有任何区分。

**一个关键的防腐规则**

把"从原话推断"和"领域常识"分开，有一个可执行的判断标准：

> **强推断必须能回指到用户原话里的某个具体词或短语。**

举例：
- ✅ 用户说"本地跑" → 推断倾向本地模型：合法，"本地跑"这三个字在原话里
- ✅ 用户说"新手学习" → 推断倾向 tutorial/demo 类型：合法，"新手学习"在原话里
- ❌ 用户说"agent 项目" → 推断倾向 LangChain：**非法**，"LangChain"在用户原话里完全找不到
- ❌ 用户说"AI 项目" → 推断需要 OpenAI API：**非法**，理由同上

凡是无法回指到原话的推断，一律归入"领域常识"层，禁止进入评分和过滤。这条规则不依赖人工逐一审核，本身就是一个可以写进 prompt 的可执行约束。

**改进方案**

1. `UserIntent` 将现有的单一 `tech_stack` 字段拆成三个：`tech_stack_explicit`（用户明说的）、`tech_stack_inferred`（强推断）、`tech_stack_domain`（领域常识）
2. `_llm_parse` 的 prompt 明确要求 LLM 按三层分别输出，且每条强推断必须附上它来源于用户原话的哪个片段
3. `LLMRelevanceRanker` 的评分 prompt 同步说明：explicit 层可用于评分，inferred 层只能加分，domain 层禁止进入评分

---

### 设计问题 B：所有用户需求都做成"硬门槛"，但有些根本没法在门口验证

**问题的来源**

在梳理排序系统的设计时，有一个合理的直觉：用户明确说了的要求，应该作为不可妥协的门槛（Gate）——不满足就直接淘汰，而不是参与打分可能被其他维度补回来。

但仔细一想，有一类用户需求在这里行不通：

> "适合新手"、"文档清楚"、"容易跑起来"、"不要太工程化"

这些需求用户说得非常明确，是真实的核心诉求。但 GitHub Search API 返回的信息只有：仓库名、描述、stars、forks、语言、topics、许可证、更新时间。

**从这些字段里，你根本无法可靠判断一个仓库是否"适合新手"或"容易跑起来"**——这需要看 README、看代码结构、看有没有安装指南，但这些内容搜索阶段根本拿不到。

如果强行把这类要求做成 Gate，结果是：**系统根据不充分的信息做了一个武断的二元判断，大量可能完全符合用户需求的仓库被提前淘汰，连排序的机会都没有。**

**区分两种"用户明确说的要求"**

这里的洞察是：**"用户明确说了"和"系统能可靠验证"是两件完全不同的事。**

由此把 Gate 细分为两类：

**A. Absolute Gate（可在门口可靠验证的硬约束）**

这类约束能直接从 GitHub 元数据判断，不满足就淘汰：
- 用户指定了编程语言 → 直接用 `language:Python` 过滤
- 用户指定了许可证类型 → 直接检查 license 字段
- 用户明确排除某个框架 → 过滤掉 topics 或描述里含该框架名的仓库

**B. Intent Gate（重要但当前无法可靠验证的核心需求）**

这类需求用户说得很明确，但目前手里的信号不足以在门口做可靠判断，应该转交给后续的排序层（`intent_fit`）来处理，让它作为排序的主要依据，而不是一刀切的硬过滤：
- "适合新手" → 排序时重点考察
- "文档清楚" → 排序时考察 description 的具体程度等代理信号
- "容易跑起来" → 排序时考察，但明确标注为"当前信号不足，中性评估"

**核心原则：** 能可靠从现有信号验证的明确约束才做 Gate，不能可靠验证的明确需求交给排序主分处理，两者不能混为一谈。

---

### 设计问题 C：让 LLM 评"质量"，但没告诉它手里只有有限的信息

**问题怎么被发现的**

在设计排序系统的评分维度时，"Quality（质量）"是一个自然会想到的维度：好的仓库应该有清晰的 README、完整的示例、稳定的维护。

但随即意识到一个问题：**排序时 LLM 手里的信息只有 GitHub 搜索返回的元数据**——名称、描述、stars、forks、语言、topics、许可证、更新时间。

README 的内容、代码结构、示例完整度、上手难度，这些在排序阶段**完全看不到**。

那么当你问 LLM "请评估这个仓库的质量"，它会怎么做？——它会凑合着用手里有的东西。而手里最醒目的数字是 **stars**。于是 LLM 很容易在没有更好信号的情况下，默默把 stars 数量当成质量的代理指标，退回到最原始的"高 star = 高质量"逻辑，把之前整套排序设计的意义基本抵消。

**更危险的地方在于：这个退化是隐性的。** Prompt 里写了"评估质量"，LLM 看起来也认真打了分，但打分依据是什么，外面完全看不出来。

**解决思路：让 LLM 只评它能评的，其余强制中性**

改进的核心不是给 LLM 更多信息（当前做不到），而是**明确告诉它哪些可以评、哪些不能评**。

将 `Quality` 重命名为 `Observable Quality`（可观测质量），强调这不是"完整质量"，只是"从当前可见信号推断的质量外显特征"。

在 prompt 里把每个子项的可信度写清楚：

| 质量子项 | 当前可用的信号 | 可信度评估 |
|---------|-------------|----------|
| 维护活跃度 | 最近更新时间 + open issues 数量 | 中（代理信号，不完全准确） |
| 表达清晰度 | description 是否具体 + topics 是否有意义 | 低（只能推测） |
| 项目成熟度 | 是否有 license + 元数据是否完整 | 中 |
| README 完整度 | **无信号** | 不可知，**强制给中性分** |
| 示例完整度 | **无信号** | 不可知，**强制给中性分** |
| 上手难度 | **无信号** | 不可知，**强制给中性分** |

在 prompt 里明确写：**不确定的项一律给中性分，不要用 stars 代替完整质量判断。**

这样 LLM 就不会因为"总要给个分"而乱猜。

---

### 设计问题 D：光靠 Prompt 约束 LLM"不要扣分"，但数学上还是可以绕过去

**问题的发现过程**

在讨论排序评分结构时，明确了一条原则：系统从用户原话推断出来的偏好（strong_inferences），只能用来给相关仓库加分，不能用来给不符合推断的仓库扣分。

比如推断"用户偏好 Python"，就只能给 Python 仓库加分，不能因为一个仓库是 Go 语言就扣分——毕竟用户从没说"不要 Go"。

但这条原则如果只写在 Prompt 里，会有一个漏洞：**如果 Preference bonus 这个分数的取值区间允许负数**（比如 -0.1 到 +0.1），LLM 完全可以在不违反 prompt 文字要求的情况下，对"不符合推断偏好"的仓库打一个负数。形式上是在打"preference bonus"，实质上变成了惩罚。

这相当于：你规定了规则，但没有关门，LLM 随时可以从侧门绕进去。

**解决思路：在结构上封死，不依赖 LLM 自觉**

最直接的修法：**把 Preference bonus 的分值区间锁定为 0 到 +0.15，不允许出现负值。**

这样不管 LLM 怎么理解 prompt，数学上它就无法通过这个维度给任何仓库打负分。"推断偏好不能扣分"这条原则变成了一个结构性约束，不再依赖 LLM 的"良心"。

这是一类很重要的设计思路：**对于系统里真正关键的约束，应该尽量在结构上实现，而不是依赖 prompt 说教。** Prompt 可以忘记，结构不会。

---

### 设计问题 E：搜索返回的候选太单一，而且"更多路搜索"不等于"更多样的候选"

**单路搜索的天花板**

当前系统只做一次 GitHub 搜索。问题在于 GitHub 搜索对关键词的措辞极度敏感——同样是找"新手 agent 学习项目"，不同的词组会让你进入完全不同的仓库池：

- 搜 `tutorial` → 主要返回各种教程和资源列表仓库
- 搜 `example` → 偏向小型 demo 仓库
- 搜 `framework` → 偏向基础设施类仓库
- 搜 `awesome` → 几乎全是资源聚合列表

单路搜索就像只开了一扇窗，只能看到仓库生态的一个切面。

**多路搜索：不是同义词改写，而是覆盖不同类型**

想到用多路搜索来扩大候选面。但这里有一个容易走偏的陷阱：如果多路搜索只是把同一个意思改写成几种不同的英文表达（`ai agent tutorial` / `llm agent guide` / `artificial intelligence agent tutorial`），搜回来的仓库大概率还是同一类——只是换了几种词汇入口，候选集仍然同质。

真正有效的多路搜索，是让每条查询**主动覆盖不同类型的仓库**，也就是"查询角色分工"：

- **主查询**：最贴近用户原话，找主流相关仓库（`agent llm beginner`）
- **教程查询**：专门找学习资源类（`ai agent tutorial example`）
- **项目查询**：专门找可运行的 demo 类（`agent demo starter simple`）

每路取前 10 个结果，合并去重后大约 20-30 个候选，再统一送入后续评分。串行执行（不并发），避免消耗 GitHub API 限额（认证后 30次/分钟）。

**多路之后还有一个问题：候选仍然可能扎堆**

即使查询角色分工做好了，评分排序完成后，前几名结果还是可能集中在同一类型——比如全是 LangChain tutorial，或者全是 awesome-* 资源列表。

这时需要在排序完成后加一个**去同质化重排**步骤：

- 资源聚合仓库（awesome-* 类）最多保留 1 个
- 同一框架生态的仓库不能连续占满前 3
- 最终展示的结果里保持"项目型 + 教程型 + 轻框架型"的混合分布
- 理想的最终展示形态：**1 个主推荐**（评分最高）+ **2-3 个不同风格的备选**（最容易跑的 / 代码最清晰的 / 最接近真实应用的）

有一点需要注意：去同质化之后，如果前端卡片没有标注每个仓库的"角色"或推荐理由，用户就感受不到多样性的价值，只会觉得"搜出来的东西有点奇怪"。所以去同质化需要和展示层配套设计。

---

### 设计问题 F：系统为什么这么排序？——没有人能回答这个问题

**问题在哪里**

系统设计越来越精细——三层意图、结构化评分、多路搜索——但有一个问题始终存在：最终输出的是一个仓库列表，每个仓库有一个 0-1 的综合得分。

当这个得分看起来有问题时（比如一个看起来不相关的仓库排在前面），**没有任何方式去追查原因**。是 intent_fit 打高了？还是 quality 给歪了？还是 preference bonus 意外加了很多？还是 Gate 没有过滤掉不该留下的？完全不知道。

这让调参、排查 bad case、迭代优化变得极为困难：你只能凭感觉猜，然后改 prompt，然后再猜。

**改进方案：让每个推荐结果都带着自己的"证据摘要"**

在评分的同时，要求 LLM 输出一份内部证据记录：为什么通过了 Gate、intent_fit 命中了哪些点、quality 用了哪些可观测信号、哪些信号因为不可知给了中性分、preference_bonus 来自哪条强推断、哪些领域常识被明确排除在评分之外。

示例结构：

```json
{
  "full_name": "owner/repo",
  "scores": {
    "intent_fit": 0.85,
    "observable_quality": 0.6,
    "preference_bonus": 0.1,
    "final": 0.82
  },
  "evidence": {
    "gate_pass_reason": "language=Python 匹配，无许可证限制",
    "intent_fit_hits": ["新手友好", "demo 类型", "有完整示例说明"],
    "quality_signals_used": {"updated_at": "2024-11", "has_license": true, "description_specific": true},
    "quality_unknown_neutral": ["README 完整度", "示例完整度", "上手难度"],
    "preference_bonus_source": ["strong_inference: Python 优先（来自用户原话'新手学习 agent'）"],
    "domain_knowledge_excluded": ["LangChain", "OpenAI API（领域常识层，不进评分）"]
  }
}
```

初期这份证据摘要只打日志，不展示给用户。但它带来两个立竿见影的好处：
1. **调试时可以快速定位**——看到一个奇怪的排序结果，打开日志马上知道是哪个评分维度出了问题
2. **为后续展示奠定基础**——将来如果要在推荐卡片上展示"为什么推荐这个"，证据摘要直接就是素材

这件事建议从第一版就做进去，不要留到"以后再说"。否则系统越复杂，调试越困难，而所有改进都只能靠凭感觉试错。

---

### 汇总：改进后的完整搜索推荐流程

以下是综合以上六个设计问题的改进方案，形成的新版推荐流程：

```
用户输入
  ↓
【意图解析——三层分法】
  explicit_constraints（用户明说的）   → 可用于硬过滤、评分、展示
  strong_inferences（原话可回指的推断）→ 只能加分不能扣分，展示时标注"系统推断"
  domain_knowledge（LLM 领域常识）    → 禁止进入过滤和评分，不展示
  ↓
【多路搜索——查询角色分工】
  主查询（贴近用户原话）
  教程查询（偏 tutorial / example）
  项目查询（偏 demo / starter）
  → 每路取前 10，合并去重，得到约 20-30 个候选
  ↓
【Absolute Gate 过滤】
  只对"能从元数据可靠验证"的明确约束做硬淘汰
  无法可靠验证的明确需求（如"适合新手"）→ 转交排序层处理，不在这里淘汰
  ↓
【结构化评分】
  intent_fit（最高权重）：用户明确需求 + 无法验证的明确需求
  observable_quality（次高）：只用可观测信号，不可知项强制中性，不用 stars 代替质量
  preference_bonus（只加分，区间锁定 0~+0.15）：来自 strong_inferences 层
  domain_knowledge：完全不进入评分
  → 同步输出每个仓库的内部证据摘要（打日志）
  ↓
【去同质化重排】
  避免候选扎堆同一类型
  目标：1 个主推荐 + 2-3 个不同风格的备选
  ↓
【展示】
  推荐卡片标注每个仓库的"角色"和推荐理由
```

**这套设计的核心出发点：**

> 这个系统不是在"让 LLM 帮用户找仓库"，而是在"限制 LLM 乱替用户做决定"。
>
> LLM 有很强的领域联想能力，但这种能力如果不加约束，就会悄悄把它自己的偏好当成用户的偏好，
> 把行业常识当成用户的需求，把"主流选择"当成"正确答案"。
> 这套设计的每一层，都是在把 LLM 的这种越界倾向挡回去。

---

## 技术实现指南（供开发者 / AI 直接参照修改）

> 本节是上方设计讨论的工程落地版本。
> 每个改动项均列出：需要修改的文件、具体位置、改动方式、验证方法。
> 所有改动集中在 `src/agent/search_agent.py`，不涉及其他文件。

---

### 改动一：UserIntent 数据结构重构（tech_stack 三层化）

**文件：** `src/agent/search_agent.py`
**位置：** `UserIntent` dataclass

**改动前结构（当前）：**
```python
@dataclass
class UserIntent:
    keywords: List[str]
    language: Optional[str]
    min_stars: int
    domain_hints: List[str]
    quality_focus: bool
    recent_preference: bool
    project_type: Optional[str] = None
    difficulty: Optional[str] = None
    purpose: Optional[str] = None
    tech_stack: List[str] = field(default_factory=list)   # ← 问题所在：三类信息混在一个字段
    summary: str = ""
    search_query: str = ""
```

**改动后结构：**
```python
@dataclass
class UserIntent:
    keywords: List[str]
    language: Optional[str]
    min_stars: int
    domain_hints: List[str]
    quality_focus: bool
    recent_preference: bool
    project_type: Optional[str] = None
    difficulty: Optional[str] = None
    purpose: Optional[str] = None
    # tech_stack 拆成三层
    tech_stack_explicit: List[str] = field(default_factory=list)   # 用户明确说的技术
    tech_stack_inferred: List[str] = field(default_factory=list)   # 强推断（可回指原话）
    inferred_sources: Dict[str, str] = field(default_factory=dict)  # {推断技术: 来源短语} 审计追踪用
    tech_stack_domain: List[str] = field(default_factory=list)     # 领域常识
    # 向后兼容字段（废弃但保留）
    tech_stack: List[str] = field(default_factory=list)
    summary: str = ""
    search_query: str = ""
```

**tech_stack_domain 使用规则（写入 docstring）：**
- 禁止进入 Gate 过滤
- 禁止进入主评分和扣分
- 默认不进入搜索词
- 未来可作为低优先级探索支路使用（不锁死）

**注意：** 原 `tech_stack` 字段保留向后兼容，如有其他地方引用，需同步修改。

---

### 改动二：_llm_parse 的 prompt 更新（三层提取 + 强推断回指）

**文件：** `src/agent/search_agent.py`
**位置：** `IntentRecognizer._llm_parse()` 方法内的 `prompt` 字符串

**将 prompt 中关于 tech_stack 的部分替换为以下内容：**

```python
prompt = f"""你是一个 GitHub 项目搜索助手。请分析用户的搜索需求，严格区分三类信息。

用户需求："{user_query}"

【重要规则】
- tech_stack_explicit：只放用户原话里明确出现的技术名称，一个字都没提的不放
- tech_stack_inferred：只放能从用户原话某个具体短语推断出的技术方向，必须在 inferred_sources 里注明来源短语
- **inferred_sources 的键必须和 tech_stack_inferred 中的值一一对应**，不允许出现键值不匹配的情况
- tech_stack_domain：你作为 LLM 知道的该领域主流技术，但用户完全没提的，放这里（这层不会用于评分）

请返回 JSON：
{{
  "project_type": "项目类型（框架/工具/库/教程/Demo/全栈应用/其他）",
  "keywords": ["核心展示关键词1", "关键词2"],
  "search_query": "用于 GitHub 搜索的精简英文词（2-3个单词，不含中文）",
  "language": "主要编程语言（用户明确说的才填，否则为 null）",
  "tech_stack_explicit": ["用户原话明确提到的技术"],
  "tech_stack_inferred": ["从原话推断的技术方向"],
  "inferred_sources": {{"技术名": "来源于用户原话的哪个短语"}},
  "tech_stack_domain": ["领域常识技术，用户未提及"],
  "difficulty": "难度（入门友好/中级/高级，用户说了才填，否则 null）",
  "purpose": "用途（学习参考/生产使用/研究/其他，用户说了才填，否则 null）",
  "quality_focus": true或false,
  "recent_preference": true或false,
  "min_stars": 最低star数（整数，默认{DEFAULT_MIN_STARS}），
  "domain_hints": ["领域标签"]
}}

只返回 JSON，不要有其他内容。"""
```

**同步更新 `_llm_parse` 中解析 JSON 后的赋值逻辑：**

```python
# 替换原来的 tech_stack 赋值
intent = UserIntent(
    ...
    tech_stack_explicit=data.get("tech_stack_explicit") or [],
    tech_stack_inferred=data.get("tech_stack_inferred") or [],
    inferred_sources=data.get("inferred_sources") or {},
    tech_stack_domain=data.get("tech_stack_domain") or [],
    ...
)
```

---

### 改动三：LLMRelevanceRanker 的评分 prompt 重构

**文件：** `src/agent/search_agent.py`
**位置：** `LLMRelevanceRanker._build_evaluation_prompt()` 方法

**当前 prompt 只传了用户原始查询字符串，需要改为传入结构化的 UserIntent。**

**第一步：修改方法签名，接收 UserIntent：**

```python
def _build_evaluation_prompt(
    self,
    repos: List[Dict[str, Any]],
    user_query: str,
    intent: UserIntent,          # 已有
) -> str:
```

**第二步：在 prompt 里明确各层的评分权限，替换当前的 prompt 字符串：**

```python
# 构建约束说明
explicit_desc = "、".join(intent.tech_stack_explicit) if intent.tech_stack_explicit else "无"
inferred_desc = "、".join(intent.tech_stack_inferred) if intent.tech_stack_inferred else "无"
domain_desc   = "、".join(intent.tech_stack_domain)   if intent.tech_stack_domain   else "无"

prompt = f"""你是一个 GitHub 项目推荐专家。用户正在寻找: "{user_query}"

【评分约束——必须严格遵守】
1. 用户明确要求的技术：{explicit_desc}
   → 可作为正向评分依据
2. 从用户原话推断的偏好：{inferred_desc}
   → 只能加分（0 到 +0.15），不能因为仓库不符合此项而扣分
3. 领域常识（用户未提及）：{domain_desc}
   → 禁止进入评分。不要因为仓库未使用这些技术而扣分，也不要因为使用了而加分

【评分维度说明】
- intent_fit（0~1）：仓库是否符合用户的核心意图，{intent.project_type or '项目类型不限'}，
  难度：{intent.difficulty or '不限'}，用途：{intent.purpose or '不限'}
- observable_quality（0~1）：只基于可观测信号评估。
  可用信号：更新时间、issue数量、description具体程度、是否有license。
  不可知项（README内容、示例完整度、上手难度）一律给 0.5 中性分，不要猜测，不要用stars代替质量。
- preference_bonus（0~0.15）：仅加分，不减分。
  如果仓库符合推断偏好（{inferred_desc}），可适当加分，上限 0.15。

以下是候选仓库：
{repos_text}

请返回 JSON：
{{
  "evaluations": [
    {{
      "full_name": "owner/repo",
      "intent_fit": 0.0~1.0,
      "observable_quality": 0.0~1.0,
      "preference_bonus": 0.0~0.15,
      "final_score": intent_fit*0.6 + observable_quality*0.3 + preference_bonus,
      "suitability_reasons": ["命中了哪些用户需求"],
      "potential_concerns": ["有哪些不确定项"],
      "tags": ["tag1"],
      "evidence": {{
        "gate_pass": true或false,
        "intent_fit_hits": ["具体命中点"],
        "quality_signals_used": ["用了哪些可观测信号"],
        "quality_unknown_neutral": ["哪些项因信息不足给了中性分"],
        "preference_bonus_reason": "为什么加了这个分（或'无'）",
        "domain_knowledge_excluded": ["哪些领域常识被排除在评分外"]
      }}
    }}
  ]
}}

只返回 JSON。"""
```

**注意最终分数计算公式：** `final_score = intent_fit * 0.6 + observable_quality * 0.3 + preference_bonus`
权重可在此处集中调整，不要分散到其他地方。

---

### 改动四：VerifiableGate（可验证约束 Gate）过滤层

**文件：** `src/agent/search_agent.py`
**位置：** `LLMRelevanceRanker._parse_evaluation()` 方法内，构建 RepoCard 之前

**新增 Gate 过滤逻辑（在调用 LLM 评分之前执行）：**

```python
def _apply_verifiable_gate(
    self,
    repos: List[Dict[str, Any]],
    intent: UserIntent,
) -> List[Dict[str, Any]]:
    """
    可验证约束 Gate：只过滤能从元数据可靠验证的明确硬约束。
    不适用于"新手友好""容易跑起来"等无法在元数据层验证的需求。
    当前实现：仅语言过滤（当用户明确指定 language 时）。
    """
    passed = []
    dropped = {"language": 0}
    for repo in repos:
        # 规则1：用户明确指定了语言，且语言不符
        if intent.language and repo.get("language"):
            if repo["language"].lower() != intent.language.lower():
                logger.debug(f"Gate 淘汰 {repo['full_name']}: 语言不符（{repo['language']} != {intent.language}）")
                dropped["language"] += 1
                continue

        # 其他 VerifiableGate 规则可在此扩展

        passed.append(repo)

    logger.info(f"VerifiableGate：{len(repos)} 个候选 → {len(passed)} 个通过（dropped_by_language={dropped['language']}）")
    return passed
```

**在 `rank_repos()` 中插入调用位置：**

```python
def rank_repos(self, repos, user_query, intent):
    if not repos:
        return []
    # ← 在这里插入 Gate
    repos = self._apply_verifiable_gate(repos, intent)
    if not repos:
        logger.warning("VerifiableGate 过滤后候选为空，跳过 LLM 评分")
        return []
    # 后续原有逻辑不变
    ...
```

---

### 改动五：多路搜索 + 查询角色分工

**文件：** `src/agent/search_agent.py`
**位置：** `SearchRecommendationAgent.run()` 方法

**当前代码（单路搜索）：**
```python
search_query = intent.search_query or " ".join(intent.keywords[:2])
search_result = self.searcher.search_repos(
    keywords=search_query,
    language=intent.language,
    min_stars=intent.min_stars,
    limit=30,
)
```

**改动后（三路搜索，串行执行）：**

```python
def _build_multi_queries(self, intent: UserIntent) -> List[Dict[str, Any]]:
    """
    构建三条查询，各有角色分工，不是同义词改写。
    返回格式：[{"role": "main", "query": "...", "limit": 10}, ...]
    """
    main_query = intent.search_query or " ".join(intent.keywords[:2])

    # 教程查询：偏 tutorial / example / beginner
    tutorial_suffix = "tutorial"
    if intent.difficulty in ("入门友好", "入门"):
        tutorial_suffix = "beginner tutorial"
    tutorial_query = f"{main_query.split()[0]} {tutorial_suffix}"

    # 项目查询：偏 demo / starter / example project
    project_query = f"{main_query.split()[0]} demo example"

    return [
        {"role": "main",     "query": main_query,     "limit": 10},
        {"role": "tutorial", "query": tutorial_query,  "limit": 10},
        {"role": "project",  "query": project_query,   "limit": 10},
    ]

def _multi_search(self, intent: UserIntent) -> List[Dict[str, Any]]:
    """串行执行多路搜索，合并去重。"""
    queries = self._build_multi_queries(intent)
    seen_full_names = set()
    all_repos = []

    for q in queries:
        try:
            result = self.searcher.search_repos(
                keywords=q["query"],
                language=intent.language,
                min_stars=intent.min_stars,
                limit=q["limit"],
            )
            for repo in result.repos:
                if repo["full_name"] not in seen_full_names:
                    seen_full_names.add(repo["full_name"])
                    repo["search_role"] = q["role"]   # 记录来自哪路搜索
                    all_repos.append(repo)
        except Exception as e:
            logger.warning(f"多路搜索 [{q['role']}] 失败，跳过: {e}")

    logger.info(f"多路搜索完成：{len(all_repos)} 个去重候选")
    return all_repos
```

**在 `run()` 中替换原来的搜索调用：**

```python
# 替换原来的 Step 2
repos = self._multi_search(intent)
if not repos:
    logger.warning("未找到任何仓库")
    return []
```

**设计原则：** 第一版保持简单，每路统一权重，不做复杂的 query 权重分配。变量已经够多，先验证多路召回能覆盖不同仓库类型即可。

---

### 改动六：去同质化重排

**文件：** `src/agent/search_agent.py`
**位置：** `SearchRecommendationAgent.run()` 中，LLM 排序完成后

**新增方法：**

```python
def _diversify(self, cards: List[RepoCard], limit: int) -> List[RepoCard]:
    """
    去同质化重排：避免前几名全是同类型仓库。

    可扩展规则（当前实现第一条，后续可按需添加）：
    - awesome-* 类最多保留 1 个
    - 后续可扩展：教程列表型最多 1 个、同生态不占满前 3 等
    """
    AWESOME_LIMIT = 1
    awesome_count = 0
    result = []
    deferred = []   # 暂缓加入的同质化仓库

    for card in cards:
        is_awesome = (
            "awesome" in card.repo_name.lower() or
            "awesome" in (card.description or "").lower()[:50] or
            any("awesome" in t for t in card.tags)
        )

        if is_awesome:
            if awesome_count < AWESOME_LIMIT:
                result.append(card)
                awesome_count += 1
            else:
                deferred.append(card)
        else:
            result.append(card)

        if len(result) >= limit:
            break

    # 如果结果不够，从 deferred 补充
    for card in deferred:
        if len(result) >= limit:
            break
        result.append(card)

    return result[:limit]
```

**在 `run()` 末尾插入调用：**

```python
# 原有代码
top_cards = cards[:limit]

# 改为
top_cards = self._diversify(cards, limit)
```

---

### 改动七：评分证据摘要输出

**文件：** `src/agent/search_agent.py`
**位置：** `LLMRelevanceRanker._parse_evaluation()` 方法

**证据最小 Schema（至少包含以下固定键）：**
```python
evidence: {
    "intent_fit_hits": List[str],           # 命中了哪些用户核心意图
    "quality_signals": List[str],            # 用了哪些可观测信号评估质量
    "preference_bonus_reasons": List[str],   # 加分来源（来自哪条强推断）
    "neutral_unknowns": List[str],           # 哪些项因信息不足给了中性分
    "gate_checks": Dict[str, bool],          # 各 Gate 检查结果
    "domain_excluded": List[str],             # 哪些领域常识被排除在评分外
}
```

**在解析 LLM JSON 结果时，把 evidence 字段存入 RepoCard：**

首先确认 `RepoCard`（`src/models/repo_card.py`）中有 `evidence` 字段，如没有则添加：

```python
# src/models/repo_card.py 中的 RepoCard dataclass
evidence: Dict[str, Any] = field(default_factory=dict)   # 内部评分证据摘要
```

然后在 `_parse_evaluation` 的 RepoCard 构建处加入：

```python
card = RepoCard(
    ...
    relevance_score=eval_data.get("final_score", eval_data.get("relevance_score", 0.0)),
    suitability_reasons=eval_data.get("suitability_reasons", []),
    potential_concerns=eval_data.get("potential_concerns", []),
    tags=eval_data.get("tags", []),
    evidence=eval_data.get("evidence", {}),   # ← 新增
)
```

**在 `rank_repos()` 中将 evidence 打入日志：**

```python
for card in cards:
    if card.evidence:
        logger.debug(
            f"[证据] {card.full_name} | "
            f"intent={card.evidence.get('intent_fit_hits')} | "
            f"quality_unknown={card.evidence.get('quality_unknown_neutral')} | "
            f"excluded={card.evidence.get('domain_knowledge_excluded')}"
        )
```

---

### 验证方法

完成以上改动后，用以下场景验证效果：

**场景 1：验证 tech_stack 三层正确分离 + inferred_sources 追溯**
```
输入：给我一个适合新手学习的 agent 项目
预期：
  tech_stack_explicit = []（用户没说任何具体技术）
  tech_stack_inferred = ["tutorial", "demo"]（来自"新手学习"）
  inferred_sources = {"tutorial": "新手学习", "demo": "新手学习"}
  tech_stack_domain = ["LangChain", "OpenAI API"]（领域常识，不进评分）
```

**场景 2：验证 VerifiableGate 正确过滤**
```
输入：找一个 Go 语言的 web 框架
预期：所有 language != "Go" 的仓库在 Gate 阶段被过滤掉
```

**场景 3：验证 Preference bonus 不出现负值**
```
检查方式：在日志中搜索 preference_bonus，确认所有值 >= 0
```

**场景 4：验证多路搜索返回多样候选**
```
输入：找一个适合新手的 agent 项目
预期日志：
  多路搜索完成：XX 个去重候选
  搜索角色分布：main=10, tutorial=X, project=X
```

**场景 5：验证证据摘要输出**
```
将日志级别设为 DEBUG，运行一次搜索，确认每个推荐仓库都有对应的 [证据] 日志行输出
```