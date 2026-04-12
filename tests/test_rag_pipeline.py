"""
测试 RAG 流程 (rag_pipeline)
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.tools.rag_pipeline import (
    RAGPipeline,
    EmbeddingService,
    RetrievedChunk,
    build_index,
    search,
    get_all_indexes,
    delete_repo_index,
)
from src.tools.chunker import CodeChunk


class TestRetrievedChunk:
    """测试 RetrievedChunk 数据类"""

    def test_creation(self):
        """测试创建 RetrievedChunk"""
        chunk = CodeChunk(
            chunk_id="test_id",
            content="def test(): pass",
            file_path="/test/test.py",
            start_line=1,
            end_line=2,
        )
        retrieved = RetrievedChunk(chunk=chunk, distance=0.5, score=0.8)

        assert retrieved.chunk == chunk
        assert retrieved.distance == 0.5
        assert retrieved.score == 0.8


class TestEmbeddingService:
    """测试 Embedding 服务"""

    @patch("src.tools.rag_pipeline.EMBEDDING_PROVIDER", "openai")
    def test_init_openai(self):
        """测试初始化 OpenAI Embedding"""
        with patch("src.tools.rag_pipeline.get_embedding_config") as mock_config:
            mock_config.return_value = {"api_key": "test_key", "model": "text-embedding-3-small"}

            service = EmbeddingService()

            assert service.provider == "openai"
            assert service.dimension == 1536

    @patch("src.tools.rag_pipeline.EMBEDDING_PROVIDER", "huggingface")
    def test_init_huggingface(self):
        """测试初始化 HuggingFace Embedding"""
        with patch("src.tools.rag_pipeline.get_embedding_config") as mock_config:
            mock_config.return_value = {"model": "all-MiniLM-L6-v2", "device": "cpu"}

            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st.return_value = mock_model

                service = EmbeddingService()

                assert service.provider == "huggingface"
                assert service.dimension == 384

    @patch("src.tools.rag_pipeline.EMBEDDING_PROVIDER", "openai")
    def test_embed_openai(self):
        """测试 OpenAI Embedding"""
        with patch("src.tools.rag_pipeline.get_embedding_config") as mock_config:
            mock_config.return_value = {"api_key": "test_key", "model": "text-embedding-3-small"}

            service = EmbeddingService()

            with patch.object(service.client, "embeddings") as mock_embeddings:
                mock_response = MagicMock()
                mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
                mock_embeddings.create.return_value = mock_response

                result = service.embed(["test text"])

                assert len(result) == 1
                assert len(result[0]) == 1536

    @patch("src.tools.rag_pipeline.EMBEDDING_PROVIDER", "huggingface")
    def test_embed_huggingface(self):
        """测试 HuggingFace Embedding"""
        with patch("src.tools.rag_pipeline.get_embedding_config") as mock_config:
            mock_config.return_value = {"model": "all-MiniLM-L6-v6", "device": "cpu"}

            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_model.encode.return_value = [[0.1] * 384]
                mock_st.return_value = mock_model

                service = EmbeddingService()

                result = service.embed(["test text"])

                assert len(result) == 1
                assert len(result[0]) == 384


class TestRAGPipeline:
    """测试 RAG Pipeline"""

    def setup_method(self):
        """每个测试方法前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_name = "test_repo"

    def teardown_method(self):
        """清理临时文件"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_file(self, filename: str, content: str) -> str:
        """创建测试文件"""
        file_path = os.path.join(self.temp_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def test_pipeline_init(self):
        """测试 Pipeline 初始化"""
        with patch("src.tools.rag_pipeline.EmbeddingService"):
            pipeline = RAGPipeline(self.repo_name)

            assert pipeline.repo_name == self.repo_name
            assert pipeline.collection_name is not None

    def test_generate_safe_name(self):
        """测试生成安全的集合名称"""
        with patch("src.tools.rag_pipeline.EmbeddingService"):
            pipeline = RAGPipeline("owner/repo-name")

            # 验证名称只包含字母数字
            assert pipeline.collection_name.replace("repo_", "").isalnum()

    @patch("src.tools.rag_pipeline.EmbeddingService")
    def test_build_index_empty_directory(self, mock_embedding):
        """测试构建空目录索引"""
        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        # 创建一个空目录
        empty_dir = os.path.join(self.temp_dir, "empty_repo")
        os.makedirs(empty_dir)

        pipeline = RAGPipeline("empty_repo")

        with patch.object(pipeline.chroma_client, "get_or_create_collection") as mock_collection:
            mock_col = MagicMock()
            mock_collection.return_value = mock_col

            count = pipeline.build_index(empty_dir)

            assert count == 0

    @patch("src.tools.rag_pipeline.EmbeddingService")
    def test_build_index_with_files(self, mock_embedding):
        """测试构建包含文件的索引"""
        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        # 创建测试文件
        self._create_test_file("test.py", "def hello(): print('hello')")

        pipeline = RAGPipeline("test_repo")

        with patch.object(pipeline.chroma_client, "get_or_create_collection") as mock_collection:
            mock_col = MagicMock()
            mock_collection.return_value = mock_col

            count = pipeline.build_index(self.temp_dir)

            # 应该有至少一个 chunk
            assert count >= 1

    def test_search_without_results(self):
        """测试无结果的检索"""
        with patch("src.tools.rag_pipeline.EmbeddingService") as mock_embedding:
            mock_embedding_instance = MagicMock()
            mock_embedding_instance.embed.return_value = [[0.1] * 384]
            mock_embedding.return_value = mock_embedding_instance

            pipeline = RAGPipeline("test_repo")

            with patch.object(pipeline.collection, "query") as mock_query:
                mock_query.return_value = {"ids": [], "documents": [], "metadatas": [], "distances": []}

                results = pipeline.search("test query")

                assert len(results) == 0

    def test_search_with_results(self):
        """测试有结果的检索"""
        with patch("src.tools.rag_pipeline.EmbeddingService") as mock_embedding:
            mock_embedding_instance = MagicMock()
            mock_embedding_instance.embed.return_value = [[0.1] * 384]
            mock_embedding.return_value = mock_embedding_instance

            pipeline = RAGPipeline("test_repo")

            with patch.object(pipeline.collection, "query") as mock_query:
                mock_query.return_value = {
                    "ids": [["chunk_1"]],
                    "documents": [["def hello(): pass"]],
                    "metadatas": [[{
                        "file_path": "/test/test.py",
                        "start_line": 1,
                        "end_line": 2,
                        "chunk_type": "function",
                        "name": "hello",
                    }]],
                    "distances": [[0.5]],
                }

                results = pipeline.search("test query")

                assert len(results) == 1
                assert results[0].chunk.name == "hello"
                assert results[0].score > 0

    def test_rerank_empty(self):
        """测试重排序空列表"""
        with patch("src.tools.rag_pipeline.EmbeddingService"):
            pipeline = RAGPipeline("test_repo")

            results = pipeline.rerank("test", [], top_k=3)

            assert len(results) == 0

    def test_rerank_with_chunks(self):
        """测试重排序有内容的列表"""
        with patch("src.tools.rag_pipeline.EmbeddingService"):
            pipeline = RAGPipeline("test_repo")

            # 创建测试 chunks
            chunks = []
            for i in range(5):
                chunk = CodeChunk(
                    chunk_id=f"chunk_{i}",
                    content=f"def func_{i}(): pass",
                    file_path=f"/test/test{i}.py",
                    start_line=i * 10,
                    end_line=i * 10 + 5,
                    chunk_type="function",
                    name=f"func_{i}",
                )
                retrieved = RetrievedChunk(chunk=chunk, distance=0.5, score=0.5)
                chunks.append(retrieved)

            # 按 func_0 查询，应该优先返回包含 func_0 的 chunk
            results = pipeline.rerank("func_0", chunks, top_k=2)

            assert len(results) <= 2

    def test_delete_index(self):
        """测试删除索引"""
        with patch("src.tools.rag_pipeline.EmbeddingService"):
            pipeline = RAGPipeline("test_repo")

            with patch.object(pipeline.chroma_client, "delete_collection") as mock_delete:
                pipeline.delete_index()

                mock_delete.assert_called_once()

    def test_get_stats(self):
        """测试获取统计信息"""
        with patch("src.tools.rag_pipeline.EmbeddingService"):
            pipeline = RAGPipeline("test_repo")

            with patch.object(pipeline.collection, "count") as mock_count:
                mock_count.return_value = 100

                stats = pipeline.get_stats()

                assert stats["chunk_count"] == 100
                assert stats["repo_name"] == "test_repo"


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_build_index_convenience(self):
        """测试 build_index 便捷函数"""
        with patch("src.tools.rag_pipeline.RAGPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.build_index.return_value = 10
            mock_pipeline_class.return_value = mock_pipeline

            with tempfile.TemporaryDirectory() as temp_dir:
                result = build_index(temp_dir, "test_repo")

                assert result == 10
                mock_pipeline.build_index.assert_called_once_with(temp_dir)

    def test_search_convenience(self):
        """测试 search 便捷函数"""
        with patch("src.tools.rag_pipeline.RAGPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline.search.return_value = []
            mock_pipeline_class.return_value = mock_pipeline

            result = search("test query", "test_repo", top_k=5)

            assert result == []
            mock_pipeline.search.assert_called_once_with("test query", 5)

    def test_get_all_indexes(self):
        """测试获取所有索引"""
        with patch("src.tools.rag_pipeline.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client

            mock_col1 = MagicMock()
            mock_col1.name = "repo_abc123"
            mock_col2 = MagicMock()
            mock_col2.name = "repo_def456"
            mock_client.list_collections.return_value = [mock_col1, mock_col2]

            indexes = get_all_indexes()

            assert len(indexes) == 2
            assert "repo_abc123" in indexes

    def test_delete_repo_index(self):
        """测试删除仓库索引"""
        with patch("src.tools.rag_pipeline.RAGPipeline") as mock_pipeline_class:
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline

            result = delete_repo_index("test_repo")

            assert result is True
            mock_pipeline.delete_index.assert_called_once()
