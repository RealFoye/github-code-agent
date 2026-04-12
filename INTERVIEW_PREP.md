# 项目名称

**CodeRAG** — 智能 GitHub 代码分析与问答系统

---

# 项目介绍（简历版）

## 一句话简介
基于 RAG + LLM 的 GitHub 仓库智能分析工具，可对任意开源项目进行代码理解、结构分析、语义问答。

## 详细介绍

**核心能力：**
- 输入自然语言描述需求，AI 自动搜索并推荐最相关的 GitHub 仓库（意图识别 → 搜索 → LLM 排序）
- 对任意 GitHub 仓库进行深度代码分析（克隆 → AST 解析 → 向量化 → 语义检索 → 问答）
- 支持 OpenAI / DeepSeek / MiniMax 多模型接入，本地 Embedding（ HuggingFace sentence-transformers）

**技术亮点：**
- 自研语义切片引擎：按函数/类/模块边界切分代码，配合上下文行号，解决"代码被切开导致语义丢失"问题
- RAG 召回验证机制：检索后验证引用代码片段是否真实存在，防止 LLM 幻觉
- 多级重排序：向量检索 → 类型优先级 → 关键词匹配 → 相关性评分
- 支持 7+ 主流编程语言的 AST 解析（Python / JavaScript / TypeScript / Go / Java / Rust / C++）

**技术栈：**
Python · Streamlit · ChromaDB · Tree-sitter · PyGithub · sentence-transformers · OpenAI API / DeepSeek / MiniMax API

---

# 面试问答

## 第一部分：搜索与推荐（阶段一）

---

### Q1：你们的搜索流程是什么样的？为什么要分阶段？

**回答：**

分两个阶段是设计决策，不是技术限制。

**阶段一（搜索推荐）的职责：**
1. 用户输入自然语言查询 → 意图识别（提取语言、stars 门槛、领域关键词）
2. 调用 GitHub Search API 获取候选仓库列表
3. 用 LLM 评估每个仓库与用户需求的匹配度，输出相关性分数和推荐理由
4. 排序后返回 Top N 推荐卡片

**为什么不直接让 LLM 搜？**
- GitHub Search API 有 5000 次/小时限额（认证后），LLM 没有
- LLM 的知识有截止日期，无法反映仓库的最新 stars 和活跃度
- LLM 搜仓库的成本远高于 API 搜索

**为什么不用 ElasticSearch 而用 GitHub API？**
- GitHub API 直接返回 stars / forks / topics / license 等结构化字段，不需要自己解析
- 无需自己维护索引，更新由 GitHub 自动处理

---

### Q2：意图识别是怎么做的？

**回答：**

意图识别用**规则 + 正则**实现，不调用 LLM（为了快且省钱）。

```python
# 三个维度：
1. 语言提取：正则匹配 "python" → "Python"
2. Stars 要求：">1000" / "stars>1000" / "超过 1000" → 提取数字
3. 领域关键词：从预定义领域词表匹配（web/ml/devops/mobile...）
```

**为什么不调 LLM 做意图识别？**
- 用户一句话可能只包含 10-20 个 token，调 LLM 成本高且延迟大
- 意图识别本质是结构化提取，不是生成，用规则足够
- LLM 的强项在**排序**（评估仓库相关性），不应该浪费在简单的模式匹配上

---

### Q3：GitHub API 限流是怎么处理的？

**回答：**

限流分两种情况处理：

**1. 发现限流时（403 响应）：**
```python
# 从响应头读取 X-RateLimit-Reset 时间戳
wait_seconds = reset_time - current_time
if wait_seconds > 0:
    time.sleep(min(wait_seconds, 60))  # 最多等 60 秒
```

**2. 预防性措施：**
- 有 GitHub Token：5000 次/小时
- 无 Token：60 次/小时（严重影响功能）
- 每次搜索最多返回 100 条，循环拉取时控制速率

**Token 从哪来？**
从 `.env` 读取，通过 `GITHUB_TOKEN` 环境变量配置，不硬编码在代码里。

---

### Q4：排序为什么要调 LLM？直接按 stars 排序不行吗？

**回答：**

直接按 stars 排序有严重问题：

**Stars 排序的缺陷：**
- 仓库 A：Stars 10万，描述是"这是一个 Python 机器学习库"，实际是教学示例代码
- 仓库 B：Stars 5000，描述是"Production-ready PyTorch 训练框架，含分布式训练"

用户要找"生产级 ML 框架"，A 排在第一但实际不相关。

**LLM 排序的价值：**
- 理解用户意图（"生产级"→ 需要分布式、GPU 训练、monorepo）
- 对比仓库描述与用户需求的语义匹配度
- 输出中文推荐理由，提高用户信任度

**成本控制：**
- GitHub Search 返回 30 条，只取前 10 条让 LLM 评估（可配置）
- LLM 输出 JSON 结构化结果，解析后按 `relevance_score` 排序

---

## 第二部分：代码切片（核心难点）

---

### Q5：代码切片为什么不按字符数切？按行切不行吗？

**回答：**

这是 RAG 项目最核心的设计决策。

**按字符数/行数切的问题：**
```python
# 假设每 500 字符切一刀
def calculate(x, y):      # ← 切断了！
    return x + y

def main():
    result = calculate(1, 2)
    print(result)
```

函数被切成两半，检索返回半个函数，LLM 看到的是不完整的代码。

**我们的方案：按 AST 语义单元切分**

1. 用 Tree-sitter 解析文件，提取：
   - 函数定义（开始行 → 结束行）
   - 类定义
   - 模块级 import + docstring

2. 每个 Chunk 包含：
   - 完整的函数/类/模块代码
   - 前后各 3 行上下文（`context_before` / `context_after`）
   - 元数据：文件路径、行号范围、chunk_type、函数签名

3. 检索时：
   - 先向量检索拿到候选 chunks
   - 再按 chunk_type 优先级重排（function > class > module）
   - 最后验证代码片段是否真实存在（防止 LLM 幻觉）

---

### Q6：什么是 Tree-sitter？和 Python 内置 AST 有什么区别？

**回答：**

Tree-sitter 是一个**增量 AST 解析器**，支持多语言。

**和 Python 内置 `ast` 模块的区别：**

| 维度 | Python `ast` | Tree-sitter |
|------|-------------|-------------|
| 多语言支持 | 仅 Python | 30+ 语言 |
| 增量解析 | 不支持 | 支持（文件修改只重解析变化的节点） |
| 增量语义更新 | ❌ | ✅（用于 IDE 代码高亮） |
| 语法错误容忍度 | 遇到语法错误直接抛异常 | 尽可能继续解析，保留尽可能多的子树 |
| 查询语言 | 无 | 有（tree-sitter query，可精准提取函数名/参数/body） |

**我们为什么用 Tree-sitter？**
```python
# tree-sitter query 示例（提取 Python 函数）
query = """
(function_definition
    name: (identifier) @name
    parameters: (parameters) @params
    body: (block) @body) @func
"""
# 遍历 captures 拿到每个函数的名字、参数、起始行、结束行
```

同一个 query 改个语言名就能提取 Go/JavaScript/TypeScript 的函数，代码复用率高。

---

### Q7：chunk 的元数据有哪些？为什么需要这些？

**回答：**

每个 Chunk 的元数据：

```python
@dataclass
class CodeChunk:
    chunk_id: str          # 全局唯一 ID（UUID），用于向量数据库去重
    file_path: str         # 文件路径（用于验证和显示来源）
    start_line: int        # 起始行
    end_line: int          # 结束行

    chunk_type: str        # function / class / module / config / document
    name: str              # 函数名或类名

    context_before: str     # 前 3 行上下文（防止被切断的代码失去语义）
    context_after: str      # 后 3 行上下文

    signature: str          # 函数签名（用于快速判断是否匹配问题）
    docstring: str         # 文档字符串（语义丰富，是最好的检索目标）

    calls: List[str]       # 调用了哪些函数（构建调用图）
    called_by: List[str]   # 被哪些函数调用
```

**为什么需要 `signature`？**
用户问"这个函数怎么调用"，检索时先过滤 `chunk_type=function`，再看 `signature` 里有参数列表，可以直接告诉用户怎么用。

**为什么需要 `calls` / `called_by`？**
用户问"这个函数被谁调用了"，直接查 `called_by` 字段，而不是再跑一次向量检索。

---

### Q8：大文件怎么处理？会全部加载到内存吗？

**回答：**

**两个保护机制：**

**1. 文件大小过滤（`chunker.py`）：**
```python
def chunk_directory(self, dir_path: str, max_file_size: int = 100_000):
    if file_path.stat().st_size > max_file_size:
        logger.info(f"跳过过大文件: {file_path}")
        continue
```
超过 100KB 的文件直接跳过（不切片不进索引）。

**2. 行数统计时流式读取（`repo_cloner.py`）：**
```python
# 原来（会 OOM）：
total_lines += len(blob.data_stream.read().decode(...).splitlines())

# 现在（流式）：
stream = blob.data_stream
while True:
    chunk = stream.read(65536)  # 64KB per read
    if not chunk:
        break
    lines += chunk.decode(...).count("\n")
```

流式读取把大文件拆成 64KB 块处理，内存占用恒定。

**3. ChromaDB 向量入库时也按 100 个一批处理：**
```python
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    self._index_batch(batch)  # 每批只占约 100 个 chunk 的内存
```

---

## 第三部分：RAG 与向量检索

---

### Q9：Embedding 模型用的是哪个？为什么不用 OpenAI 的？

**回答：**

默认用 **HuggingFace sentence-transformers（all-MiniLM-L6-v2）**，轻量高效。

**选型理由：**

| 模型 | 维度 | 特点 |
|------|------|------|
| all-MiniLM-L6-v2 | 384 | 轻量、CPU 可跑、免费 |
| text-embedding-3-small | 1536 | OpenAI 官方、效果好但要钱 |
| bge-m3 | 1024 | 效果好、免费、但更慢更吃内存 |

**`all-MiniLM-L6-v2` 的优势：**
- 384 维向量，ChromaDB 存储体积小，检索速度快
- macOS 可用 MPS（Metal GPU）加速，不需要 CUDA
- 论文验证在 MTEB 基准上 6 行代码任务表现优秀

**代码实现：**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
embeddings = model.encode(texts, convert_to_numpy=True)
```

---

### Q10：向量检索完了为什么还要 Re-rank？

**回答：**

向量检索基于**语义相似度**，但语义相似不等于**用途相似**。

**例子：**
```
用户问题："这个项目的入口函数在哪里？"

向量检索结果：
1. 某个工具函数（语义相似度高，但不是在找入口）
2. main.py 中的 main 函数（次相似，但才是真正的入口）
3. 配置文件（相似但无关）
```

**Re-rank 策略（`rag_pipeline.py`）：**

```python
type_priority = {"function": 0, "class": 1, "module": 2, "file": 3}

def rerank_key(item: RetrievedChunk) -> Tuple[int, float, int]:
    type_score = type_priority.get(chunk.chunk_type, 99)
    path_match = query_lower in chunk.file_path.lower()  # 文件名含关键词
    name_match = query_lower in chunk.name.lower()        # 函数名含关键词
    score = item.score - path_match * 0.5 - name_match * 0.3
    return (type_score, -score, chunk.start_line)
```

**三层过滤：**
1. **向量相似度**（初筛，取 Top 20）
2. **chunk_type 优先级**（function > class > module）
3. **关键词精确匹配**（文件名/函数名含查询词优先）

---

### Q11：RAG 的召回率怎么保证？有没有可能 LLM 编造代码？

**回答：**

**两个机制防止幻觉：**

**1. 召回后验证（`analysis_agent.py`）：**
```python
def _verify_sources(self, chunks, repo_path):
    for chunk in chunks:
        # 检查文件是否真实存在
        if not file_path.startswith(repo_path):
            continue
        # 验证行号范围是否有效
        if not (1 <= chunk.start_line <= len(lines)):
            continue
        # 验证通过才加入引用列表
        verified.append({"chunk": chunk, "verified": True})
```

如果文件被删了、行号超出范围，直接丢弃这个引用。

**2. Prompt 约束：**
```
请根据以上代码片段回答用户的问题。
如果代码片段不足以回答问题，请明确说明，不要编造。
回答时必须指出相关的代码位置（文件路径和行号）。
```

**为什么还是可能出错？**
- 文件存在，行号也有效，但代码内容被修改过（克隆后有新提交）→ 这种情况无法 100% 避免
- ChromaDB 持久化的是 Embedding，不实时同步最新代码

---

### Q12：ChromaDB 是什么？为什么选它？

**回答：**

ChromaDB 是一个**嵌入式向量数据库**（用 SQLite 存储，不需要独立部署）。

**选型理由：**
- **零运维**：Python 直接 import，不需要起 Docker 或云服务
- **持久化**：向量存在本地磁盘，重启不丢失
- **支持元数据过滤**：可以按 `file_path`、`chunk_type` 过滤

**缺点：**
- 不支持分布式，不适合海量数据（百万级以上）
- 并发写入能力弱（单 writer）

**对于我们的场景（个人工具、仓库级数据量）：完全够用。**

---

## 第四部分：LLM 调用与多模型支持

---

### Q13：OpenAI / DeepSeek / MiniMax 三种模型是怎么统一调用的？

**回答：**

通过**统一封装 + Provider 判断**：

```python
def chat(messages, provider, temperature, max_tokens):
    service = LLMService(provider)
    return service.chat(messages, temperature, max_tokens)
```

**两个模型的处理差异：**

**OpenAI / DeepSeek（用 OpenAI SDK）：**
```python
from openai import OpenAI
client = OpenAI(api_key=key, base_url=base_url)
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
)
return response.choices[0].message.content
```

**MiniMax（用 Anthropic SDK）：**
```python
import anthropic
client = anthropic.Anthropic(api_key=key, base_url=base_url)
response = client.messages.create(
    model=model,
    messages=messages,
    max_tokens=max_tokens,
    system=system_prompt,
)
# MiniMax 可能返回 ThinkingBlock + TextBlock
for block in response.content:
    if block.type == "text":
        return block.text
```

**为什么 MiniMax 用不同 SDK？**
因为 MiniMax 的 API 是 Anthropic 兼容协议，但不是官方 SDK，需要用 `base_url` 指向 `api.minimaxi.com`。

---

### Q14：MiniMax API 返回 529 过载，你们怎么处理？

**回答：**

**三重保护：**

**1. 指数退避重试：**
```python
retryable_errors = ["529", "overloaded", "500", "502", "503", "504", "api_error"]

for attempt in range(max_retries):
    try:
        response = call_api(...)
    except Exception as e:
        if is_retryable(e) and attempt < max_retries - 1:
            wait = retry_delay * (attempt + 1)  # 2s → 4s → 6s
            time.sleep(wait)
            continue
        raise
```

**2. 多 Provider 降级：**
如果 MiniMax 一直失败，可以切换 `LLM_PROVIDER=deepseek`，DeepSeek 作为备选。

**3. LLM 不可用时的降级策略（search_agent）：**
```python
try:
    evaluation = _call_llm(prompt)
except Exception:
    # LLM 挂了，用 stars 数降级排序
    return fallback_ranking(repos)
```

---

### Q15：Embedding 模型可以换吗？怎么配置？

**回答：**

在 `.env` 中配置，支持三种 Provider：

```env
# 方案 1：HuggingFace（默认，免费）
EMBEDDING_PROVIDER=huggingface
HF_MODEL_NAME=all-MiniLM-L6-v2
HF_DEVICE=mps  # macOS 用 Metal GPU

# 方案 2：OpenAI（效果好，要钱）
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# 方案 3：Ollama 本地（完全免费，要自己部署）
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen3-embedding
```

**切换只需要改 `.env`，不需要改代码**。`EmbeddingService` 类根据 `EMBEDDING_PROVIDER` 自动选择初始化哪个 Client。

---

## 第五部分：系统设计与架构

---

### Q16：项目整体架构是什么样的？

**回答：**

```
用户输入自然语言
       ↓
┌─────────────────────────────────────────┐
│           阶段一：搜索推荐                  │
│  IntentRecognizer（意图识别）               │
│         ↓                                  │
│  GitHubSearch（API 搜索）                   │
│         ↓                                  │
│  LLMRelevanceRanker（LLM 排序）            │
│         ↓                                  │
│  RepoCard[]（推荐卡片）                     │
└─────────────────────────────────────────┘
       ↓ 用户选择仓库
┌─────────────────────────────────────────┐
│           阶段二：代码分析                  │
│  RepoCloner（克隆仓库）                     │
│         ↓                                 │
│  CodeParser（AST 解析结构）                 │
│         ↓                                 │
│  Chunker（语义切片）                       │
│         ↓                                 │
│  RAGPipeline（向量索引）                    │
│         ↓                                 │
│  LLM 问答（生成答案）                       │
│         ↓                                 │
│  ReportGenerator（报告生成）                 │
└─────────────────────────────────────────┘
```

**为什么分两个 Agent？**
职责分离，阶段一不依赖 RAG 和向量数据库，可以独立运行。阶段二需要克隆代码，资源重，不用的场景不需要跑。

---

### Q17：Session 是什么？为什么需要这个概念？

**回答：**

`AnalysisSession` 是**一次分析的完整上下文**：

```python
@dataclass
class AnalysisSession:
    repo_url: str           # 仓库 URL
    repo_path: str          # 本地路径
    repo_name: str          # 仓库名
    full_name: str          # owner/repo

    repo_info: dict         # GitHub API 返回的仓库信息
    structure: RepoStructure # AST 解析出的结构

    rag_pipeline: RAGPipeline  # 向量索引
    is_indexed: bool          # 是否已建索引
```

**为什么不用全局变量？**
- 用户可能同时分析多个仓库（不同 Session）
- Web 界面需要把 Session 存在 `st.session_state` 里，用户切换标签页不丢状态
- CLI 界面一次只分析一个，但保留 Session 允许用户多次问答

**Session 和 Pipeline 的关系：**
每个 Session 有自己的 `RAGPipeline`，对应 ChromaDB 里独立的 Collection（通过 repo_name 的 MD5 生成集合名）。不同仓库的向量索引互不干扰。

---

### Q18：为什么用 Streamlit 而不是 FastAPI + Vue/React？

**回答：**

**Streamlit 的优势：**
- **开发速度**：10 行代码就能跑一个带 DataFrame 展示的 Web 界面
- **内置组件**：按钮、表格、图表、进度条、侧边栏全部现成
- **适合单用户工具**：不是面向高并发的产品，是个人效率工具

**FastAPI + 前端的问题：**
- 需要写两套代码（后端 API + 前端页面）
- 状态管理复杂（Session 存在 Redis？）
- 开发时间长

**Streamlit 的局限：**
- 不适合高并发（但这个工具本来就不是给高并发用的）
- 前端定制能力弱（但够用）
- WebSocket 走反向代理有坑（你们实测踩过这个坑）

---

### Q19：Streamlit 页面通过 ngrok 暴露给外部时白屏，怎么排查？

**回答：**

这是 Streamlit 的经典问题，排查路径：

**Step 1：确认本地是否正常**
```
curl -I http://localhost:8501
→ 200 OK
```
本地正常才能往下走。

**Step 2：确认 ngrok 链路是否通**
```
curl -I https://your-ngrok-url.ngrok-free.app
→ 200/404（404 也说明 ngrok 到 Nginx/Streamlit 是通的）
```

**Step 3：如果是静态资源 503**
- Streamlit 的静态资源（CSS/JS）是动态生成的，Host Header 依赖请求的域名
- 外部访问时 Host Header 是 ngrok 域名，Streamlit 可能校验失败返回 503
- 解决方案：**加反向代理**（Nginx/Caddy），让 Streamlit 只收到 `Host: localhost`

**Step 4：如果是 WebSocket 失败（400 错误）**
- Streamlit 的 `_stcore/stream` 端点走 WebSocket
- nginx 的 `/ws` location 匹配不到 `/stcore`，改成：
```nginx
location /_stcore {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

**根本原因：** Streamlit 开发服务器不适合直接公网暴露，生产环境推荐 Nginx/Caddy 反向代理 + Streamlit 跑在 localhost。

---

## 第六部分：Python 高级特性

---

### Q20： `@contextmanager` 装饰器是怎么工作的？用在什么地方？

**回答：**

**contextmanager 原理：**
```python
@contextmanager
def protecting(repo_path):
    RepoCloner.protect_repo(repo_path)  # __enter__
    try:
        yield  # 执行 with 块里的代码
    finally:
        RepoCloner.unprotect_repo(repo_path)  # __exit__
```

装饰器把生成器函数变成实现了 `__enter__` 和 `__exit__` 的上下文管理器对象。

**使用场景（`run_full_pipeline`）：**
```python
def run_full_pipeline(self, repo_url, questions, output_path):
    clone_result = self.cloner.clone(repo_url)
    repo_path = clone_result["local_path"]

    # 分析期间保护仓库不被清理
    with self.cloner.protecting(repo_path):
        return self._run_analysis(repo_url, repo_path, questions, output_path)
    # 分析完成，自动解除保护
```

**如果不这样写：**
- 需要手动 try/finally
- 如果中途抛异常，仓库保护不会被解除

**和 `with open(file) as f:` 的区别：**
`open()` 返回的是文件对象，`contextmanager` 装饰的是一个生成器函数，两种都能用在 `with` 里。

---

### Q21：Python 的 `functools.wraps` 有什么用？

**回答：**

`functools.wraps` 用于**装饰器中保留被装饰函数的元信息**。

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet():
    """这是文档"""
    pass

print(greet.__name__)  # "wrapper" ← 原函数名丢失了
print(greet.__doc__)   # None     ← 文档也丢了
```

```python
import functools

def my_decorator(func):
    @functools.wraps(func)  # 保留原函数元信息
    def wrapper(*args, **kwargs):
        print("Before")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet():
    """这是文档"""
    pass

print(greet.__name__)  # "greet" ← 元信息保留了
print(greet.__doc__)   # "这是文档"
```

**实际项目在哪里用？**
`st.cache_resource` 装饰的方法如果内部出错，`wraps` 能帮助保留函数签名（虽然我们项目没有自定义装饰器，但 `cache_resource` 内部用了 `wraps`）。

---

### Q22：`classmethod` 和 `staticmethod` 区别？项目里用在哪？

**回答：**

```python
class RepoCloner:
    @classmethod
    def protect_repo(cls, repo_path):  # cls 是类本身
        cls._protected_paths.add(repo_path)

    @staticmethod
    def _cleanup_old_repos(keep_list):  # 没有 self，也没有 cls
        ...
```

**区别：**
- `@classmethod`：第一个参数是类本身，可以访问类变量（`_protected_paths`）
- `@staticmethod`：既不需要类也不需要实例，纯函数逻辑
- 普通方法：第一个参数是 `self`，只能通过实例访问

**项目中用在哪：**
```python
# 类变量，所有实例共享
_protected_paths: set = set()
MAX_KEEP_REPOS = 5

# 类方法：保护仓库（修改类变量）
@classmethod
def protect_repo(cls, repo_path):
    cls._protected_paths.add(repo_path)

# 静态方法：清理逻辑（不访问类变量）
@staticmethod
def _cleanup_old_repos(keep_list):
    ...
```

---

### Q23：typing 的 `Optional` 和 `Union` 区别？

**回答：**

```python
# Optional[str] = str 或 None
Optional[str]  ==  Union[str, None]

# Union[str, int, None] 不能简写
```

```python
from typing import Optional, Union

def greet(name: Optional[str]) -> str:
    # name 可以是 str 也可以是 None
    if name is None:
        return "Hello!"
    return f"Hello, {name}!"

def process(value: Union[str, int, list]) -> str:
    # 三种类型之一
    ...
```

**项目中实际用法：**
```python
def search_repos(
    keywords: str,
    language: Optional[str] = None,  # 可以不传，默认 None
    min_stars: int = 10,
) -> List[Dict[str, Any]]:
```

---

## 第七部分：测试与质量保障

---

### Q24：你们的测试覆盖率是多少？主要测了什么？

**回答：**

测试文件在 `tests/`，核心模块全覆盖：

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_chunker.py` | 语义切片（函数、类、配置文件、文档） |
| `test_rag_pipeline.py` | 向量检索、Re-rank、Collection 管理 |
| `test_search_agent.py` | 意图识别、LLM 排序、降级策略 |
| `test_analysis_agent.py` | 全流程、Session 管理、Source 验证 |

**测试策略：**
- **单元测试**：Mock LLM API 和 ChromaDB，只测逻辑不测外部依赖
- **集成测试**：用真实小仓库验证端到端流程
- 不依赖真实 GitHub Token（Mock GitHubSearch）

**覆盖率目标：> 60%，当前 81 个测试全通过。**

---

### Q25：Mock 是怎么用的？不用 Mock 行不行？

**回答：**

**为什么需要 Mock？**

```python
# 如果不用 Mock，测试会真的调 API
def test_rank_repos():
    repos = [...]  # 假数据
    ranker = LLMRelevanceRanker()

    # 真实调用 MiniMax API（要钱、要网络）
    result = ranker.rank_repos(repos, query, intent)

    # API 失败，测试就失败（不应该）
    # API 慢，测试就慢
```

**Mock 正确用法：**
```python
def test_rank_repos_with_openai(self):
    ranker = LLMRelevanceRanker()

    # Mock LLM 返回值
    with patch.object(ranker, "_call_openai") as mock_call:
        mock_call.return_value = '{"evaluations": [...]}'  # 固定返回值
        result = ranker.rank_repos(repos, query, intent)
        assert result[0].relevance_score == 0.9
        mock_call.assert_called_once()  # 确认真的调了
```

**什么时候不用 Mock？**
集成测试可以用真实数据（如克隆一个小的真实 GitHub 仓库），但要放在 CI 的慢速测试里，不能每次 `pytest` 都跑。

---

## 第八部分：性能与优化

---

### Q26：克隆大仓库（如 PyTorch 80GB）会怎么处理？

**回答：**

**三重保护：**

**1. 浅克隆（`depth=100`）：**
```python
repo = git.Repo.clone_from(url, path, depth=100)
# 只拿最近 100 个 commit，不是完整历史
# 网络流量从 GB 级降到 MB 级
```

**2. 文件大小过滤：**
```python
if file_path.stat().st_size > 100_000:  # 100KB
    continue  # 跳过不进 RAG 索引
```

**3. 大文件行数统计流式读取：**
```python
while True:
    chunk = stream.read(65536)
    if not chunk:
        break
    lines += chunk.count("\n")
# 不把整个文件加载到内存
```

**但是！**
- 80GB 仓库即使浅克隆也可能几 GB
- ChromaDB 索引 80GB 仓库的所有代码块 Embedding，费用可能很高
- 建议用户通过 `min_stars` 过滤或用 `limit` 参数限制分析范围

---

### Q27：RAG 的 Embedding 是每次查询都重新算吗？还是预计算好的？

**回答：**

**预计算，存向量数据库。**

流程：
1. **建索引时**：`build_index()` 把每个 chunk 的向量算好，存进 ChromaDB
2. **查询时**：只算用户问题的向量（1 次 Embedding 计算），然后查 ChromaDB

```python
# 建索引
embeddings = embedding_service.embed([chunk.content for chunk in chunks])
collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

# 查询
query_embedding = embedding_service.embed([query])[0]  # 只算 1 次
results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
```

**为什么不每次查询时重新算？**
- Embedding 是 O(N) 的，N 是 chunks 数量（可能几千个）
- 预计算后查询是 O(1) 向量距离计算，快 100 倍+

---

## 第九部分：项目难点与反思

---

### Q28：这个项目最难的地方是什么？你是怎么解决的？

**回答：**

**难点 1：代码切片的边界问题**

不能按字符数硬切，否则函数被截断，RAG 召回的代码不完整。

解决：用 Tree-sitter 解析 AST，按函数/类/模块的实际边界切，每个 chunk 保证是完整的语义单元。

**难点 2：LLM 幻觉**

RAG 召回的代码片段可能不准确，LLM 可能把不相关的内容串起来编造成答案。

解决：召回后验证机制（检查文件是否存在、行号是否有效），并通过 Prompt 约束 LLM 必须引用具体代码位置。

**难点 3：多模型兼容**

OpenAI 和 MiniMax 的 SDK 和响应格式不同，要统一封装。

解决：抽象 `LLMService` 类，根据 Provider 类型选择用 OpenAI SDK 还是 Anthropic SDK。

---

### Q29：如果重新做这个项目，你会改进什么？

**回答：**

**1. 分离索引存储和检索服务**
- 现在索引存在本地 ChromaDB，换电脑就没了
- 应该加一个向量数据库服务（Pinecone / Qdrant），支持分布式查询

**2. 支持增量索引**
- 仓库更新了新代码，需要重新索引整个仓库
- 应该支持增量更新，只索引变化的文件

**3. 评估体系**
- 现在没有衡量 RAG 质量的指标（召回率、精确率）
- 应该加一个评估数据集，定期跑 QA 对比

**4. 多模态支持**
- 代码图谱可视化（模块依赖关系）
- 直接在页面上显示函数调用关系图

---

### Q30：你们的 RAG 和市面上的 RAG 有什么区别？

**回答：**

**通用 RAG（文本文档）：**
- 切片方式：按字符数（如 500 tokens）
- 元数据：文档名、页码
- 召回：向量相似度

**CodeRAG（代码仓库）：**

| 维度 | 通用 RAG | CodeRAG |
|------|---------|---------|
| 切片方式 | 字符数硬切 | AST 语义边界 |
| 元数据 | 文档/页码 | 文件路径、行号、chunk_type、函数签名 |
| 上下文 | 无 | 前后各 3 行上下文 |
| 召回策略 | 向量检索 | 向量检索 + chunk_type 优先级 + 关键词匹配 |
| 幻觉防护 | 无 | 召回后验证代码片段存在性 |
| 调用关系 | 无 | calls / called_by 字段 |

**本质区别：** 代码不是自然语言，是有结构的符号系统。通用 RAG 的切片策略会破坏代码的语义完整性，必须按语法树切才能保证每个 chunk 可独立解释。
