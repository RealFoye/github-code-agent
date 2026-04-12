"""
测试代码分析 Agent (analysis_agent)
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.agent.analysis_agent import (
    CodeAnalysisAgent,
    AnalysisSession,
)


class TestAnalysisSession:
    """测试 AnalysisSession 数据类"""

    def test_session_creation(self):
        """测试创建会话"""
        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
        )

        assert session.repo_url == "https://github.com/test/repo"
        assert session.repo_path == "/tmp/repo"
        assert session.is_indexed is False

    def test_session_with_repo_info(self):
        """测试带仓库信息的会话"""
        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
            repo_info={"stars": 1000, "description": "Test repo"},
        )

        assert session.repo_info["stars"] == 1000


class TestCodeAnalysisAgent:
    """测试 CodeAnalysis Agent"""

    def setup_method(self):
        """每个测试前创建 Agent"""
        self.agent = CodeAnalysisAgent()

    def test_agent_initialization(self):
        """测试 Agent 初始化"""
        assert self.agent.cloner is not None
        assert self.agent.parser is not None
        assert self.agent.report_generator is not None

    def test_extract_repo_name_github(self):
        """测试从 GitHub URL 提取仓库名"""
        name = self.agent._extract_repo_name("https://github.com/owner/repo")
        assert name == "owner/repo"

    def test_extract_repo_name_github_with_git(self):
        """测试从 GitHub URL（带 .git）提取仓库名"""
        name = self.agent._extract_repo_name("https://github.com/owner/repo.git")
        assert name == "owner/repo"

    def test_extract_repo_name_github_with_trailing_slash(self):
        """测试从 GitHub URL（带斜杠）提取仓库名"""
        name = self.agent._extract_repo_name("https://github.com/owner/repo/")
        assert name == "owner/repo"

    def test_extract_repo_name_invalid(self):
        """测试无效 URL"""
        name = self.agent._extract_repo_name("not a github url")
        assert name == ""

    @patch("src.agent.analysis_agent.RepoCloner")
    def test_start_session(self, mock_cloner_class):
        """测试启动会话"""
        mock_cloner = MagicMock()
        mock_cloner.clone.return_value = {
            "local_path": "/tmp/test_repo"
        }
        mock_cloner.get_repo_info.return_value = {
            "name": "test_repo",
            "full_name": "test/test_repo"
        }
        mock_cloner_class.return_value = mock_cloner

        agent = CodeAnalysisAgent()
        session = agent.start_session("https://github.com/test/test_repo")

        assert session.repo_path == "/tmp/test_repo"
        assert session.repo_name == "test_repo"
        assert session.full_name == "test/test_repo"

    @patch("src.agent.analysis_agent.CodeParser")
    def test_analyze_structure(self, mock_parser_class):
        """测试分析仓库结构"""
        mock_parser = MagicMock()
        mock_parser.scan_structure.return_value = MagicMock(
            total_files=100,
            total_lines=10000,
        )
        mock_parser_class.return_value = mock_parser

        agent = CodeAnalysisAgent()
        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
        )

        structure = agent.analyze_structure(session)

        assert structure.total_files == 100
        assert structure.total_lines == 10000

    @patch("src.agent.analysis_agent.RAGPipeline")
    def test_build_rag_index(self, mock_pipeline_class):
        """测试构建 RAG 索引"""
        mock_pipeline = MagicMock()
        mock_pipeline.build_index.return_value = 50
        mock_pipeline_class.return_value = mock_pipeline

        agent = CodeAnalysisAgent()
        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
        )

        count = agent.build_rag_index(session)

        assert count == 50
        assert session.is_indexed is True
        assert session.rag_pipeline is not None

    def test_ask_without_index(self):
        """测试未构建索引时问答"""
        agent = CodeAnalysisAgent()
        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
            is_indexed=False,
        )

        with pytest.raises(ValueError, match="RAG 索引未构建"):
            agent.ask(session, "test question")

    def test_ask_no_results(self):
        """测试无检索结果"""
        agent = CodeAnalysisAgent()

        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = []

        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
            is_indexed=True,
            rag_pipeline=mock_pipeline,
        )

        result = agent.ask(session, "test question")

        assert result["answer"] == "没有找到相关的代码片段来回答这个问题。"
        assert result["sources"] == []

    @patch("src.agent.analysis_agent.get_llm_config")
    def test_ask_with_results_openai(self, mock_get_config):
        """测试有检索结果的问答（OpenAI）"""
        mock_get_config.return_value = {
            "provider": "openai",
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "model": "gpt-4o-mini",
        }

        from src.tools.chunker import CodeChunk
        from src.tools.rag_pipeline import RetrievedChunk

        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = [
            RetrievedChunk(
                chunk=CodeChunk(
                    chunk_id="test",
                    content="def hello(): print('world')",
                    file_path="/tmp/repo/hello.py",
                    start_line=1,
                    end_line=2,
                    chunk_type="function",
                    name="hello",
                    language="python",
                ),
                distance=0.1,
                score=0.9,
            )
        ]
        mock_pipeline.rerank.return_value = mock_pipeline.search.return_value

        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
            is_indexed=True,
            rag_pipeline=mock_pipeline,
        )

        agent = CodeAnalysisAgent()

        # Mock OpenAI call
        with patch.object(agent, "_call_openai") as mock_call:
            mock_call.return_value = "这是答案"
            result = agent.ask(session, "test question")

            assert result["question"] == "test question"
            assert result["answer"] == "这是答案"

    @patch("src.agent.analysis_agent.get_llm_config")
    def test_ask_with_results_minimax(self, mock_get_config):
        """测试有检索结果的问答（MiniMax）"""
        mock_get_config.return_value = {
            "provider": "minimax",
            "api_key": "test_key",
            "base_url": "https://api.minimaxi.com/anthropic",
            "model": "MiniMax-M2.7",
        }

        from src.tools.chunker import CodeChunk
        from src.tools.rag_pipeline import RetrievedChunk

        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = [
            RetrievedChunk(
                chunk=CodeChunk(
                    chunk_id="test",
                    content="def hello(): print('world')",
                    file_path="/tmp/repo/hello.py",
                    start_line=1,
                    end_line=2,
                    chunk_type="function",
                    name="hello",
                    language="python",
                ),
                distance=0.1,
                score=0.9,
            )
        ]
        mock_pipeline.rerank.return_value = mock_pipeline.search.return_value

        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
            is_indexed=True,
            rag_pipeline=mock_pipeline,
        )

        agent = CodeAnalysisAgent()

        # Mock MiniMax call
        with patch.object(agent, "_call_minimax") as mock_call:
            mock_call.return_value = "MiniMax 生成的答案"
            result = agent.ask(session, "test question")

            assert result["question"] == "test question"
            assert result["answer"] == "MiniMax 生成的答案"

    def test_build_context(self):
        """测试构建上下文"""
        from src.tools.chunker import CodeChunk

        chunks = [
            CodeChunk(
                chunk_id="test1",
                content="def hello(): pass",
                file_path="/tmp/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                name="hello",
                language="python",
                signature="def hello()",
            ),
        ]

        context = self.agent._build_context(chunks)

        assert "/tmp/test.py" in context
        assert "def hello():" in context
        assert "代码片段 1" in context

    @patch("src.agent.analysis_agent.get_llm_config")
    def test_generate_answer_openai(self, mock_get_config):
        """测试 OpenAI 生成答案"""
        mock_get_config.return_value = {
            "provider": "openai",
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "model": "gpt-4o-mini",
        }

        with patch.object(self.agent, "_call_openai") as mock_call:
            mock_call.return_value = "OpenAI 答案"
            answer = self.agent._generate_answer("问题", "上下文")

            assert answer == "OpenAI 答案"

    @patch("src.agent.analysis_agent.get_llm_config")
    def test_generate_answer_deepseek(self, mock_get_config):
        """测试 DeepSeek 生成答案"""
        mock_get_config.return_value = {
            "provider": "deepseek",
            "api_key": "test_key",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        }

        with patch.object(self.agent, "_call_deepseek") as mock_call:
            mock_call.return_value = "DeepSeek 答案"
            answer = self.agent._generate_answer("问题", "上下文")

            assert answer == "DeepSeek 答案"

    @patch("src.agent.analysis_agent.get_llm_config")
    def test_generate_answer_unsupported(self, mock_get_config):
        """测试不支持的 provider"""
        mock_get_config.return_value = {
            "provider": "unsupported",
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "model": "test-model",
        }

        with pytest.raises(ValueError, match="不支持的 LLM provider"):
            self.agent._generate_answer("问题", "上下文")

    def test_verify_sources(self):
        """测试验证代码来源"""
        from src.tools.chunker import CodeChunk

        # 创建临时文件 (2行代码)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
            f.write("def hello(): pass\n")
            f.write("print('hello')\n")
            temp_path = f.name

        try:
            chunks = [
                CodeChunk(
                    chunk_id="test",
                    content="def hello(): pass\nprint('hello')",
                    file_path=temp_path,
                    start_line=1,
                    end_line=2,
                    chunk_type="function",
                    name="hello",
                    language="python",
                ),
            ]

            # repo_path 应该是文件所在的目录
            repo_path = os.path.dirname(temp_path)
            verified = self.agent._verify_sources(chunks, repo_path)

            assert len(verified) == 1
            assert verified[0]["verified"] is True
        finally:
            os.unlink(temp_path)

    def test_verify_sources_file_not_found(self):
        """测试验证不存在的文件"""
        from src.tools.chunker import CodeChunk

        chunks = [
            CodeChunk(
                chunk_id="test",
                content="def hello(): pass",
                file_path="/nonexistent/test.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                name="hello",
                language="python",
            ),
        ]

        verified = self.agent._verify_sources(chunks, "/nonexistent")

        assert len(verified) == 0

    @patch("src.agent.analysis_agent.ReportGenerator")
    def test_generate_report(self, mock_generator_class):
        """测试生成报告"""
        mock_generator = MagicMock()
        mock_generator.generate_report.return_value = "# Test Report"
        mock_generator_class.return_value = mock_generator

        agent = CodeAnalysisAgent()
        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
        )

        report = agent.generate_report(session)

        assert report == "# Test Report"
        mock_generator.generate_report.assert_called_once()

    def test_cleanup_session(self):
        """测试清理会话"""
        agent = CodeAnalysisAgent()
        session = AnalysisSession(
            repo_url="https://github.com/test/repo",
            repo_path="/tmp/repo",
            repo_name="repo",
            full_name="test/repo",
            rag_pipeline=MagicMock(),
        )

        # 不应抛出异常
        agent.cleanup_session(session)

    @patch("src.agent.analysis_agent.RepoCloner")
    @patch("src.agent.analysis_agent.CodeParser")
    @patch("src.agent.analysis_agent.RAGPipeline")
    @patch("src.agent.analysis_agent.ReportGenerator")
    def test_run_full_pipeline(self, mock_generator_class, mock_pipeline_class, mock_parser_class, mock_cloner_class):
        """测试完整流程"""
        # Mock cloner
        mock_cloner = MagicMock()
        mock_cloner.clone.return_value = {"local_path": "/tmp/repo"}
        mock_cloner.get_repo_info.return_value = {"name": "repo"}
        mock_cloner_class.return_value = mock_cloner

        # Mock parser
        mock_parser = MagicMock()
        mock_parser.scan_structure.return_value = MagicMock(total_files=10, total_lines=100)
        mock_parser_class.return_value = mock_parser

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.build_index.return_value = 5
        mock_pipeline_class.return_value = mock_pipeline

        # Mock report generator
        mock_generator = MagicMock()
        mock_generator.generate_report.return_value = "# Test Report"
        mock_generator_class.return_value = mock_generator

        agent = CodeAnalysisAgent()

        with patch.object(agent, "ask") as mock_ask:
            mock_ask.return_value = {"question": "q1", "answer": "a1", "sources": []}

            result = agent.run_full_pipeline(
                "https://github.com/test/repo",
                questions=["q1"],
            )

            assert result["chunk_count"] == 5
            assert len(result["qa_results"]) == 1
            assert result["report"] == "# Test Report"


class TestCodeAnalysisAgentIntegration:
    """测试 CodeAnalysis Agent 集成（更高级别的测试）"""

    @patch("src.agent.analysis_agent.get_llm_config")
    def test_minimax_thinking_block_handling(self, mock_get_config):
        """测试 MiniMax ThinkingBlock 处理"""
        mock_get_config.return_value = {
            "provider": "minimax",
            "api_key": "test_key",
            "base_url": "https://api.minimaxi.com/anthropic",
            "model": "MiniMax-M2.7",
        }

        agent = CodeAnalysisAgent()

        # 模拟包含 ThinkingBlock 的响应
        mock_response = MagicMock()
        mock_block1 = MagicMock()
        mock_block1.type = "thinking"
        mock_block1.text = "思考过程..."
        mock_block2 = MagicMock()
        mock_block2.type = "text"
        mock_block2.text = "最终答案"
        mock_response.content = [mock_block1, mock_block2]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            result = agent._call_minimax("test prompt", mock_get_config.return_value)

            assert result == "最终答案"
