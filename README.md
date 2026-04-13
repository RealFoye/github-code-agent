# GitHub Code Analysis Agent

> 智能 GitHub 项目发现与深度代码分析 Agent

[English](#english) | [中文](#中文)

---

## English

### What is this?

A two-stage AI Agent for GitHub repository discovery and deep code analysis. Users describe what they're looking for in natural language, and the Agent:

1. **Stage 1 - Search & Recommend**: Searches GitHub, filters candidates, ranks by relevance using LLM
2. **Stage 2 - Code Analysis**: Clones repo, parses AST via Tree-sitter, builds RAG index, supports Q&A

### Key Features

- **Two-stage pipeline**: Search recommendation → Deep code analysis
- **Tree-sitter AST parsing**: Chunk by function/class/module boundaries, not character count
- **Multi-level Rerank**: Vector recall + Type priority + Keyword matching
- **Code authenticity verification**: Reduces LLM hallucination
- **Multi-Provider LLM**: OpenAI / DeepSeek / MiniMax support
- **Rate limit handling**: Exponential backoff retry

### Tech Stack

| Component | Choice |
|-----------|--------|
| LLM | GPT-4o-mini / DeepSeek-V3 / MiniMax-M2.7 |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (embedded, no deployment) |
| AST Parsing | Tree-sitter (multi-language) |
| GitHub API | PyGithub |
| Web UI | Streamlit |

### Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/RealFoye/github-code-agent.git
cd github-code-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and fill in your API keys

# 4. Run
python -m src.ui.web    # Web interface
# or
python -m src.ui.cli    # Command line
```

### Project Structure

```
github-code-agent/
├── src/
│   ├── agent/
│   │   ├── search_agent.py      # Stage 1: Search & Recommend
│   │   └── analysis_agent.py    # Stage 2: Code Analysis
│   ├── tools/
│   │   ├── github_search.py     # GitHub API wrapper
│   │   ├── repo_cloner.py       # Repo clone & management
│   │   ├── code_parser.py       # Tree-sitter AST parsing
│   │   ├── chunker.py           # Semantic code chunking
│   │   ├── rag_pipeline.py      # RAG pipeline
│   │   └── report_generator.py  # Markdown report generation
│   ├── models/
│   │   ├── repo_card.py         # Repo recommendation card
│   │   └── analysis_result.py  # Analysis result model
│   ├── ui/
│   │   ├── cli.py               # CLI interface
│   │   └── web.py               # Streamlit web interface
│   └── config.py                # Configuration
├── tests/                       # Unit tests
├── data/repos/                  # Cloned repos (temp)
├── PROBLEMS.md                  # Problem solving log
└── requirements.txt
```

### Core RAG Strategy

1. **Chunking by semantics**: Function/Class/Module boundaries via Tree-sitter AST
2. **Metadata**: File path, line numbers, function signature, call relationships
3. **Multi-level recall**: Vector search → Type priority rerank → Keyword boost
4. **Authenticity check**: Verify code chunks exist in actual files

### Configuration

Copy `.env.example` to `.env` and configure:

```bash
LLM_PROVIDER=minimax          # openai / deepseek / minimax
MINIMAX_API_KEY=your_key_here
# or
OPENAI_API_KEY=your_key_here
# or
DEEPSEEK_API_KEY=your_key_here
```

---

## 中文

### 这是什么？

一个两阶段 AI Agent，用于 GitHub 仓库搜索和深度代码分析。用户用自然语言描述需求，Agent 自动：

1. **阶段一 - 搜索推荐**：GitHub 搜索 → LLM 相关性评分 → 结构化推荐
2. **阶段二 - 代码分析**：克隆仓库 → Tree-sitter AST 解析 → RAG 索引 → 问答

### 核心功能

- **两阶段设计**：搜索推荐与深度分析解耦
- **Tree-sitter 语义切片**：按函数/类/模块边界切分，非字符乱切
- **多级 Rerank**：向量召回 + 类型优先级 + 关键词匹配
- **真实性校验**：验证检索到的代码片段是否真实存在
- **多 Provider LLM**：OpenAI / DeepSeek / MiniMax
- **限流重试**：指数退避机制

### 技术栈

| 组件 | 选型 |
|------|------|
| LLM | GPT-4o-mini / DeepSeek-V3 / MiniMax-M2.7 |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) |
| 向量数据库 | ChromaDB（嵌入式，无需部署） |
| AST 解析 | Tree-sitter（多语言支持） |
| GitHub API | PyGithub |
| Web 界面 | Streamlit |

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/RealFoye/github-code-agent.git
cd github-code-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 4. 运行
python -m src.ui.web    # Web 界面
# 或
python -m src.ui.cli    # 命令行
```

### 项目结构

```
github-code-agent/
├── src/
│   ├── agent/
│   │   ├── search_agent.py      # 阶段一：搜索推荐
│   │   └── analysis_agent.py    # 阶段二：代码分析
│   ├── tools/
│   │   ├── github_search.py     # GitHub API 封装
│   │   ├── repo_cloner.py       # 仓库克隆管理
│   │   ├── code_parser.py       # Tree-sitter AST 解析
│   │   ├── chunker.py           # 语义代码切片
│   │   ├── rag_pipeline.py      # RAG 流程
│   │   └── report_generator.py  # Markdown 报告生成
│   ├── models/
│   │   ├── repo_card.py         # 仓库推荐卡片
│   │   └── analysis_result.py   # 分析结果模型
│   ├── ui/
│   │   ├── cli.py               # 命令行界面
│   │   └── web.py               # Streamlit Web 界面
│   └── config.py                # 配置管理
├── tests/                       # 单元测试
├── data/repos/                  # 克隆的仓库（临时）
├── PROBLEMS.md                  # 问题记录文档
└── requirements.txt
```

### RAG 策略核心

1. **语义切片**：通过 Tree-sitter AST 按函数/类/模块边界切分
2. **元数据**：文件路径、行号范围、函数签名、调用关系
3. **多级召回**：向量检索 → 类型优先级重排 → 关键词增强
4. **真实性校验**：检查检索到的代码片段是否在文件中真实存在

### 配置说明

复制 `.env.example` 为 `.env`，填入以下配置：

```bash
LLM_PROVIDER=minimax          # openai / deepseek / minimax
MINIMAX_API_KEY=你的密钥
# 或者
OPENAI_API_KEY=你的密钥
# 或者
DEEPSEEK_API_KEY=你的密钥
```

### 环境要求

- Python 3.10+
- 需要网络连接（API 调用）

---

## License

MIT