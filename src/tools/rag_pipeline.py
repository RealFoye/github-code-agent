"""
RAG 流程
负责：向量索引构建、检索、召回
"""

import logging
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from src.tools.chunker import CodeChunk, Chunker
from src.config import (
    CHROMADB_DIR,
    EMBEDDING_PROVIDER,
    get_embedding_config,
    RAG_TOP_K,
    RAG_RERANK_TOP_K,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """检索结果"""
    chunk: CodeChunk
    distance: float = 0.0
    score: float = 0.0


class EmbeddingService:
    """Embedding 服务"""

    def __init__(self):
        """初始化 Embedding 服务"""
        self.config = get_embedding_config()
        self.provider = EMBEDDING_PROVIDER
        self._init_client()

    def _init_client(self):
        """初始化客户端"""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"不支持的 embedding provider: {self.provider}")

    def _init_openai(self):
        """初始化 OpenAI Embedding"""
        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.config.get("api_key"),
        )
        self.model = self.config.get("model", "text-embedding-3-small")
        self.dimension = 1536  # text-embedding-3-small 默认维度

    def _init_ollama(self):
        """初始化 Ollama Embedding"""
        import requests

        self.client = requests
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.model = self.config.get("model", "qwen3-embedding")

        # 获取实际维度
        try:
            response = self.client.get(f"{self.base_url}/api/show", params={"name": self.model})
            if response.status_code == 200:
                # ollama 默认 768 维
                self.dimension = 768
            else:
                self.dimension = 768
        except Exception:
            self.dimension = 768

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        生成 Embedding

        Args:
            texts: 文本列表

        Returns:
            Embedding 向量列表
        """
        if self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "ollama":
            return self._embed_ollama(texts)
        elif self.provider == "huggingface":
            return self._embed_huggingface(texts)
        return [[0.0] * self.dimension]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """OpenAI Embedding"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI Embedding 失败: {e}")
            return [[0.0] * self.dimension for _ in texts]

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """Ollama Embedding"""
        embeddings = []
        for text in texts:
            try:
                response = self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30,
                )
                if response.status_code == 200:
                    embeddings.append(response.json().get("embedding", []))
                else:
                    embeddings.append([0.0] * self.dimension)
            except Exception as e:
                logger.error(f"Ollama Embedding 失败: {e}")
                embeddings.append([0.0] * self.dimension)
        return embeddings

    def _init_huggingface(self):
        """初始化 HuggingFace Embedding (sentence-transformers)"""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.config.get("model", "BAAI/bge-m3")
            device = self.config.get("device", "cpu")

            logger.info(f"加载 HuggingFace 模型: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)

            # bge-m3 输出 1024 维
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"HuggingFace 模型加载成功，维度: {self.dimension}")

        except ImportError:
            logger.error("sentence-transformers 未安装，请运行: pip install sentence-transformers")
            raise

    def _embed_huggingface(self, texts: List[str]) -> List[List[float]]:
        """HuggingFace Embedding"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"HuggingFace Embedding 失败: {e}")
            return [[0.0] * self.dimension for _ in texts]


class RAGPipeline:
    """RAG 流程"""

    def __init__(self, repo_name: str):
        """
        初始化 RAG Pipeline

        Args:
            repo_name: 仓库名称（用于区分不同仓库的索引）
        """
        self.repo_name = repo_name
        self.chunker = Chunker()
        self.embedding_service = EmbeddingService()

        # ChromaDB 客户端
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMADB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # 集合名称（基于 repo_name 生成安全名称）
        safe_name = self._generate_safe_name(repo_name)
        self.collection_name = f"repo_{safe_name}"

        # 获取或创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"repo_name": repo_name},
        )

        logger.info(f"RAG Pipeline 初始化完成: {self.collection_name}")

    def _generate_safe_name(self, name: str) -> str:
        """生成安全的集合名称"""
        safe = hashlib.md5(name.encode()).hexdigest()[:16]
        return safe

    def build_index(self, repo_path: str, batch_size: int = 100) -> int:
        """
        构建 RAG 索引

        Args:
            repo_path: 仓库路径
            batch_size: 批处理大小

        Returns:
            索引的 Chunk 数量
        """
        logger.info(f"开始构建索引: {repo_path}")

        # 切片整个仓库
        chunks = self.chunker.chunk_directory(repo_path)

        if not chunks:
            logger.warning("没有找到任何代码切片")
            return 0

        logger.info(f"共 {len(chunks)} 个切片，开始向量化...")

        # 批量处理
        total_indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._index_batch(batch)
            total_indexed += len(batch)
            logger.info(f"已索引 {total_indexed}/{len(chunks)}")

        logger.info(f"索引构建完成: {total_indexed} 个 chunks")
        return total_indexed

    def _index_batch(self, chunks: List[CodeChunk]):
        """索引一批 chunks"""
        if not chunks:
            return

        # 准备数据
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "language": chunk.language or "",
            }
            for chunk in chunks
        ]

        # 生成 embedding
        embeddings = self.embedding_service.embed(documents)

        # 存储到 ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        检索相关代码块

        Args:
            query: 查询文本
            top_k: 返回数量
            filter_metadata: 元数据过滤条件

        Returns:
            RetrievedChunk 列表
        """
        # 生成 query embedding
        query_embedding = self.embedding_service.embed([query])[0]

        # 向量检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # 构建 CodeChunk
                chunk = CodeChunk(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i],
                    file_path=metadata.get("file_path", ""),
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                    chunk_type=metadata.get("chunk_type", "unknown"),
                    name=metadata.get("name", ""),
                    language=metadata.get("language") or None,
                )

                # 计算相似度分数 (distance 越小越相似)
                score = 1.0 / (1.0 + distance)

                retrieved.append(RetrievedChunk(
                    chunk=chunk,
                    distance=distance,
                    score=score,
                ))

        return retrieved

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = RAG_RERANK_TOP_K,
    ) -> List[RetrievedChunk]:
        """
        对检索结果重排序

        Args:
            query: 查询文本
            chunks: 检索结果
            top_k: 返回数量

        Returns:
            重排序后的结果
        """
        if not chunks:
            return []

        # 简单重排序：结合相关性和位置
        # 优先级：
        # 1. chunk_type 匹配（function > class > module > file）
        # 2. 文件路径包含关键词
        # 3. 原始相似度分数

        type_priority = {"function": 0, "class": 1, "module": 2, "file": 3, "config": 4, "document": 5}

        def rerank_key(item: RetrievedChunk) -> Tuple[int, float, int]:
            chunk = item.chunk

            # 类型优先级
            type_score = type_priority.get(chunk.chunk_type, 99)

            # 关键词匹配
            query_lower = query.lower()
            path_match = 0
            if chunk.file_path and query_lower in chunk.file_path.lower():
                path_match = 1
            if chunk.name and query_lower in chunk.name.lower():
                path_match += 1

            # 位置偏好（代码靠前的略优先）
            line_penalty = chunk.start_line / 10000

            return (type_score, -item.score - path_match * 0.5 + line_penalty, chunk.start_line)

        chunks.sort(key=rerank_key)
        return chunks[:top_k]

    def delete_index(self):
        """删除索引"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            logger.info(f"已删除索引: {self.collection_name}")
        except Exception as e:
            logger.error(f"删除索引失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "chunk_count": count,
                "repo_name": self.repo_name,
            }
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return {}


# ==================== 全局索引管理 ====================

def get_all_indexes() -> List[str]:
    """列出所有仓库索引"""
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMADB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        return [col.name for col in client.list_collections()]
    except Exception as e:
        logger.error(f"列出索引失败: {e}")
        return []


def delete_repo_index(repo_name: str) -> bool:
    """删除指定仓库的索引"""
    try:
        pipeline = RAGPipeline(repo_name)
        pipeline.delete_index()
        return True
    except Exception as e:
        logger.error(f"删除索引失败: {e}")
        return False


# 便捷函数
def build_index(repo_path: str, repo_name: str = "default") -> int:
    """
    构建索引（便捷函数）

    Args:
        repo_path: 仓库路径
        repo_name: 仓库名称

    Returns:
        索引的 Chunk 数量
    """
    pipeline = RAGPipeline(repo_name)
    return pipeline.build_index(repo_path)


def search(query: str, repo_name: str = "default", top_k: int = RAG_TOP_K) -> List[RetrievedChunk]:
    """
    检索代码块（便捷函数）

    Args:
        query: 查询文本
        repo_name: 仓库名称
        top_k: 返回数量

    Returns:
        RetrievedChunk 列表
    """
    pipeline = RAGPipeline(repo_name)
    return pipeline.search(query, top_k)