# GitHub Code Analysis Agent

> 智能 GitHub 项目发现与深度分析 Agent

## 项目概述

用户用自然语言描述需求 → Agent 搜索筛选 GitHub 项目 → 用户选择感兴趣的项目 → Agent 进行代码级深度分析 + RAG 问答。

**两阶段设计：**
- 阶段一：智能搜索与项目推荐（意图识别 → GitHub 搜索 → LLM 相关性排序 → 推荐卡片）
- 阶段二：深度代码分析（克隆 → AST 解析 → RAG 构建 → 问答 + 报告生成）

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | GPT-4o-mini / DeepSeek-V3 / MiniMax-M2.7 |
| Embedding | text-embedding-3-small / bge-m3（开源免费） |
| 向量数据库 | ChromaDB（嵌入式，无需部署） |
| AST 解析 | Tree-sitter（多语言支持） |
| GitHub API | PyGithub |
| Web 界面 | Streamlit |
| 编程语言 | Python 3.10+ |

## 项目结构

```
github-code-agent/
├── src/
│   ├── agent/
│   │   ├── search_agent.py      # 阶段一：搜索推荐 Agent
│   │   └── analysis_agent.py    # 阶段二：代码分析 Agent
│   ├── tools/
│   │   ├── github_search.py     # GitHub API 封装
│   │   ├── repo_cloner.py       # 仓库克隆与管理
│   │   ├── code_parser.py       # 代码解析（AST/Tree-sitter）
│   │   ├── chunker.py           # 代码切片器（语义切片）
│   │   ├── rag_pipeline.py      # RAG 流程
│   │   ├── llm_service.py       # LLM 统一调用（OpenAI/DeepSeek/MiniMax）
│   │   └── report_generator.py  # 报告生成
│   ├── models/
│   │   ├── repo_card.py         # 仓库推荐卡片数据模型
│   │   └── analysis_result.py   # 分析结果数据模型
│   ├── ui/
│   │   ├── cli.py               # 命令行交互界面
│   │   └── web.py               # Streamlit Web 界面
│   └── config.py                # 配置管理（API Key 等）
├── data/repos/                  # 克隆的仓库临时存放
├── tests/                       # 单元测试
├── requirements.txt
└── .env.example                # 环境变量模板
```

## RAG 策略（核心工程难点）

### 切片原则
- **不按字符数切**，按语义单元切
- 函数/方法 → 独立 Chunk
- 类定义 → 独立 Chunk
- 模块级（imports + docstring）→ 独立 Chunk
- 配置文件 → 特殊处理

### 元数据
每个 Chunk 附带：文件路径、行号范围、函数签名、调用关系

### 召回策略
1. 用户问题 → Embedding → 向量检索 Top K
2. 对检索结果做 Re-rank（按相关性排序）
3. 验证：检查引用的代码片段是否真实存在（防止幻觉）

## 工程规范

### 配置管理
- 所有 API Key（OpenAI/GitHub/Serper）通过 `.env` 管理
- 禁止硬编码任何敏感信息

### 错误处理
- GitHub API 限流 → 缓存 + 限流控制
- 网络异常 → 重试机制 + 优雅降级
- LLM 调用失败 → 降级回答 + 错误日志

### 日志
- 使用 `logging` 模块，分级输出（DEBUG/INFO/WARNING/ERROR）

### 测试要求
- 核心模块（chunker、rag_pipeline）必须有单元测试
- 覆盖率 > 60%

## 已知技术难点

1. **GitHub API 限流**：未认证 60次/分，认证 5000次/时（Search API 30次/分）
2. **代码切片边界**：函数被拆两半、跨文件类处理
3. **LLM 幻觉**：Agent 可能编造不存在的函数名 → 必须有验证机制
4. **大仓库处理**：万行代码全部 Embedding 成本高 → 文件大小限制 + 选择性解析
5. **Token 上下文**：超长代码超出窗口 → 摘要策略 + 重点切片召回
