"""
配置管理模块
所有 API Key 和配置项通过 .env 管理，不硬编码
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # 尝试默认位置


# ==================== API 配置 ====================

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# DeepSeek API（备选）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# MiniMax API
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

# GitHub API
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Serper API（搜索引擎，备选）
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")


# ==================== Embedding 配置 ====================

# Embedding 模型选择：openai / ollama / huggingface
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")

# OpenAI Embedding
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Ollama（本地Embedding）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding")

# HuggingFace Embedding（默认使用，免费开源）
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "all-MiniLM-L6-v2")  # 轻量高效，384维
HF_DEVICE = os.getenv("HF_DEVICE", "mps")  # macOS: mps (Metal GPU), Linux/Windows: cuda, CPU: cpu
HF_TOKEN = os.getenv("HF_TOKEN", "")  # HuggingFace API Token（必填，无默认值）


# ==================== LLM 配置 ====================

# LLM 提供者：openai / deepseek / ollama / minimax
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


# ==================== 路径配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 仓库克隆存放目录
REPOS_DIR = PROJECT_ROOT / "data" / "repos"
REPOS_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB 持久化目录
CHROMADB_DIR = PROJECT_ROOT / "data" / "chromadb"
CHROMADB_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 日志配置 ====================

def setup_logging(level: str = "INFO"):
    """配置日志系统"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "agent.log", encoding="utf-8")
        ]
    )

    return logging.getLogger(__name__)


# 默认日志级别
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ==================== 搜索配置 ====================

# GitHub 搜索默认参数
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_MIN_STARS = 10

# 推荐仓库数量
RECOMMENDED_REPO_COUNT = 5


# ==================== RAG 配置 ====================

# 向量检索参数
RAG_TOP_K = 5
RAG_RERANK_TOP_K = 3

# 代码切片大小限制
MAX_CHUNK_LINES = 500

# 支持的编程语言（用于 AST 解析）
SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "go", "java", "rust", "cpp"]


# ==================== 工具函数 ====================

def get_llm_config() -> dict:
    """获取当前 LLM 配置"""
    if LLM_PROVIDER == "openai":
        return {
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_BASE_URL,
            "model": OPENAI_MODEL,
        }
    elif LLM_PROVIDER == "deepseek":
        return {
            "provider": "deepseek",
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
        }
    elif LLM_PROVIDER == "ollama":
        return {
            "provider": "ollama",
            "base_url": OLLAMA_BASE_URL,
            "model": os.getenv("OLLAMA_MODEL", "qwen3"),
        }
    elif LLM_PROVIDER == "minimax":
        return {
            "provider": "minimax",
            "api_key": MINIMAX_API_KEY,
            "base_url": MINIMAX_BASE_URL,
            "model": MINIMAX_MODEL,
        }
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def get_embedding_config() -> dict:
    """获取当前 Embedding 配置"""
    if EMBEDDING_PROVIDER == "openai":
        return {
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
            "model": OPENAI_EMBEDDING_MODEL,
        }
    elif EMBEDDING_PROVIDER == "ollama":
        return {
            "provider": "ollama",
            "base_url": OLLAMA_BASE_URL,
            "model": OLLAMA_EMBEDDING_MODEL,
        }
    elif EMBEDDING_PROVIDER == "huggingface":
        return {
            "provider": "huggingface",
            "model": HF_MODEL_NAME,
            "device": HF_DEVICE,
        }
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")
