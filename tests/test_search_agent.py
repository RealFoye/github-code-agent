"""
测试搜索推荐 Agent (search_agent)
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.agent.search_agent import (
    IntentRecognizer,
    LLMRelevanceRanker,
    SearchRecommendationAgent,
    UserIntent,
)
from src.models.repo_card import RepoCard
from src.tools.llm_service import LLMService


# ==================== 辅助工厂 ====================

def _make_intent(**kwargs) -> UserIntent:
    defaults = dict(
        keywords=["test"],
        language=None,
        min_stars=10,
        domain_hints=[],
        quality_focus=False,
        recent_preference=False,
    )
    defaults.update(kwargs)
    return UserIntent(**defaults)


class TestIntentRecognizerRuleBased:
    """测试规则匹配（禁用 LLM）"""

    def setup_method(self):
        mock_llm = Mock(spec=LLMService)
        # 让 complete 抛出异常，强制走规则匹配
        mock_llm.complete.side_effect = Exception("LLM disabled")
        self.recognizer = IntentRecognizer(llm_service=mock_llm)

    def test_parse_simple_query(self):
        """测试简单查询解析"""
        intent = self.recognizer.parse("找一个 Python web 框架")

        assert intent.keywords is not None
        assert len(intent.keywords) > 0
        assert intent.language == "Python"

    def test_parse_with_stars_requirement(self):
        """测试带 stars 要求的查询"""
        intent = self.recognizer.parse("找个 stars > 1000 的机器学习项目")

        assert intent.min_stars == 1000

    def test_parse_quality_focus(self):
        """测试高质量偏好"""
        intent = self.recognizer.parse("推荐高质量的 JavaScript 库")

        assert intent.quality_focus is True
        assert intent.language == "JavaScript"

    def test_parse_recent_preference(self):
        """测试近期偏好"""
        intent = self.recognizer.parse("找最新的 Rust 项目")

        assert intent.recent_preference is True
        assert intent.language == "Rust"

    def test_extract_language_python(self):
        """测试提取 Python 语言"""
        assert self.recognizer._extract_language("python web framework") == "Python"
        assert self.recognizer._extract_language("find a python project") == "Python"

    def test_extract_language_javascript(self):
        """测试提取 JavaScript 语言"""
        assert self.recognizer._extract_language("javascript library") == "JavaScript"
        assert self.recognizer._extract_language("find a javascript library") == "JavaScript"

    def test_extract_language_go(self):
        """测试提取 Go 语言"""
        assert self.recognizer._extract_language("go microservice") == "Go"

    def test_extract_language_rust(self):
        """测试提取 Rust 语言"""
        assert self.recognizer._extract_language("rust web") == "Rust"

    def test_extract_stars_explicit(self):
        """测试显式 stars 要求"""
        assert self.recognizer._extract_stars_requirement("stars > 5000") == 5000
        assert self.recognizer._extract_stars_requirement("> 1000") == 1000
        assert self.recognizer._extract_stars_requirement("5000+") == 5000

    def test_extract_stars_chinese(self):
        """测试中文 stars 要求"""
        assert self.recognizer._extract_stars_requirement("超过 1000 stars") == 1000
        assert self.recognizer._extract_stars_requirement("至少 500") == 500

    def test_extract_keywords(self):
        """测试关键词提取"""
        keywords = self.recognizer._extract_keywords("找一个 Python 机器学习项目")

        assert "python" not in [k.lower() for k in keywords]
        assert any("机器学习" in k or "machine" in k.lower() for k in keywords)

    def test_extract_domain_hints(self):
        """测试领域提示提取"""
        intent = self.recognizer.parse("机器学习 TensorFlow 项目")

        assert "ml" in intent.domain_hints

    def test_parse_empty_query(self):
        """测试空查询"""
        intent = self.recognizer.parse("")

        assert intent.keywords is not None


class TestIntentRecognizerLLM:
    """测试 LLM 驱动的意图识别"""

    def _make_llm_response(self, **override) -> str:
        data = {
            "project_type": "全栈应用",
            "keywords": ["fullstack", "react", "nodejs"],
            "language": "JavaScript",
            "tech_stack_explicit": ["React", "Node.js"],
            "tech_stack_inferred": [],
            "inferred_sources": {},
            "tech_stack_domain": [],
            "difficulty": "入门友好",
            "purpose": "学习参考",
            "quality_focus": False,
            "recent_preference": False,
            "min_stars": 10,
            "domain_hints": ["web"],
        }
        data.update(override)
        return json.dumps(data, ensure_ascii=False)

    def test_llm_parse_fullstack_query(self):
        """测试 LLM 解析全栈项目查询"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.return_value = self._make_llm_response()

        recognizer = IntentRecognizer(llm_service=mock_llm)
        intent = recognizer.parse("我想找一个适合新手学习的全栈项目，最好是用 React 和 Node.js 的")

        assert intent.project_type == "全栈应用"
        assert intent.language == "JavaScript"
        assert "React" in intent.tech_stack
        assert "Node.js" in intent.tech_stack
        assert intent.difficulty == "入门友好"
        assert intent.purpose == "学习参考"

    def test_llm_parse_summary_contains_type(self):
        """测试 summary 包含项目类型"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.return_value = self._make_llm_response()

        recognizer = IntentRecognizer(llm_service=mock_llm)
        intent = recognizer.parse("找全栈项目")

        assert "全栈应用" in intent.summary
        assert "React" in intent.summary or "Node.js" in intent.summary
        assert "入门友好" in intent.summary

    def test_llm_parse_language_normalization(self):
        """测试语言名标准化（小写 -> 标准格式）"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.return_value = self._make_llm_response(language="python")

        recognizer = IntentRecognizer(llm_service=mock_llm)
        intent = recognizer.parse("找 python 项目")

        assert intent.language == "Python"

    def test_llm_parse_quality_focus(self):
        """测试 LLM 识别高质量偏好"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.return_value = self._make_llm_response(
            quality_focus=True, min_stars=1000
        )

        recognizer = IntentRecognizer(llm_service=mock_llm)
        intent = recognizer.parse("找高质量的 Python 机器学习框架")

        assert intent.quality_focus is True
        assert intent.min_stars == 1000

    def test_llm_fallback_on_invalid_json(self):
        """测试 LLM 返回无效 JSON 时降级到规则匹配"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.return_value = "这不是有效的 JSON 响应"

        recognizer = IntentRecognizer(llm_service=mock_llm)
        # 不应该抛出异常，而是降级到规则匹配
        intent = recognizer.parse("找 Python 机器学习项目")

        assert intent is not None
        assert intent.keywords is not None
        # 规则匹配应该能识别出 Python
        assert intent.language == "Python"

    def test_llm_fallback_on_exception(self):
        """测试 LLM 调用异常时降级到规则匹配"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.side_effect = Exception("API error")

        recognizer = IntentRecognizer(llm_service=mock_llm)
        intent = recognizer.parse("找 Rust 性能工具")

        assert intent is not None
        assert intent.language == "Rust"

    def test_llm_parse_no_language(self):
        """测试 LLM 返回 null 语言时不崩溃"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.return_value = self._make_llm_response(language=None)

        recognizer = IntentRecognizer(llm_service=mock_llm)
        intent = recognizer.parse("找一个数据可视化工具")

        assert intent.language is None

    def test_llm_parse_empty_tech_stack(self):
        """测试 LLM 返回空技术栈时不崩溃"""
        mock_llm = Mock(spec=LLMService)
        mock_llm.complete.return_value = self._make_llm_response(
            tech_stack_explicit=[], tech_stack_inferred=[], tech_stack_domain=[]
        )

        recognizer = IntentRecognizer(llm_service=mock_llm)
        intent = recognizer.parse("找一个 Go 微服务框架")

        assert intent.tech_stack == []

    def test_no_llm_uses_rule_parse(self):
        """测试不传 LLM 服务时走规则匹配（LLM 初始化失败场景）"""
        with patch("src.agent.search_agent.LLMService", side_effect=Exception("no config")):
            recognizer = IntentRecognizer()

        assert recognizer._llm is None

        intent = recognizer.parse("找 Java Spring Boot 项目")
        assert intent is not None
        assert intent.language == "Java"




class TestLLMRelevanceRanker:
    """测试 LLM 相关性排序"""

    def setup_method(self):
        """每个测试前创建排序器"""
        self.mock_config = {
            "provider": "openai",
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "model": "gpt-4o-mini",
        }

    def test_rank_repos_empty(self):
        """测试空仓库列表"""
        with patch("src.agent.search_agent.get_llm_config", return_value=self.mock_config):
            ranker = LLMRelevanceRanker()
            result = ranker.rank_repos([], "test query", UserIntent(
                keywords=["test"],
                language=None,
                min_stars=10,
                domain_hints=[],
                quality_focus=False,
                recent_preference=False,
            ))

            assert result == []

    @patch("src.agent.search_agent.get_llm_config")
    def test_rank_repos_with_openai(self, mock_get_config):
        """测试使用 OpenAI 排序"""
        mock_get_config.return_value = self.mock_config

        repos = [
            {
                "full_name": "test/repo1",
                "name": "repo1",
                "author": "test",
                "url": "https://github.com/test/repo1",
                "description": "A test repo",
                "language": "Python",
                "stars": 1000,
                "forks": 100,
                "open_issues": 10,
                "license": "MIT",
                "last_updated": "2024-01-01",
                "topics": ["ml"],
            }
        ]

        ranker = LLMRelevanceRanker()

        # Mock OpenAI response
        with patch.object(ranker, "_call_openai") as mock_call:
            mock_call.return_value = '{"evaluations": [{"full_name": "test/repo1", "relevance_score": 0.9, "suitability_reasons": ["popular"], "tags": ["ml"]}]}'

            result = ranker.rank_repos(repos, "test query", UserIntent(
                keywords=["test"],
                language=None,
                min_stars=10,
                domain_hints=[],
                quality_focus=False,
                recent_preference=False,
            ))

            assert len(result) == 1
            assert result[0].relevance_score == 0.9

    @patch("src.agent.search_agent.get_llm_config")
    def test_rank_repos_fallback(self, mock_get_config):
        """测试降级排序"""
        mock_get_config.return_value = self.mock_config

        repos = [
            {
                "full_name": "test/repo1",
                "name": "repo1",
                "author": "test",
                "url": "https://github.com/test/repo1",
                "description": "A test repo",
                "language": "Python",
                "stars": 5000,
                "forks": 100,
                "open_issues": 10,
                "license": "MIT",
                "last_updated": "2024-01-01",
                "topics": [],
            }
        ]

        ranker = LLMRelevanceRanker()

        # Mock LLM failure
        with patch.object(ranker, "_call_llm") as mock_call:
            mock_call.side_effect = Exception("LLM error")

            result = ranker.rank_repos(repos, "test query", UserIntent(
                keywords=["test"],
                language=None,
                min_stars=10,
                domain_hints=[],
                quality_focus=False,
                recent_preference=False,
            ))

            # 应该使用降级排序
            assert len(result) == 1
            assert result[0].relevance_score > 0

    @patch("src.agent.search_agent.get_llm_config")
    def test_build_evaluation_prompt(self, mock_get_config):
        """测试 prompt 构建"""
        mock_get_config.return_value = self.mock_config

        ranker = LLMRelevanceRanker()

        repos = [
            {
                "full_name": "test/repo1",
                "name": "repo1",
                "author": "test",
                "url": "https://github.com/test/repo1",
                "description": "A test repo",
                "language": "Python",
                "stars": 1000,
                "forks": 100,
                "open_issues": 10,
                "license": "MIT",
                "last_updated": "2024-01-01",
                "topics": ["ml"],
            }
        ]

        prompt = ranker._build_evaluation_prompt(repos, "test query", UserIntent(
            keywords=["test"],
            language="Python",
            min_stars=10,
            domain_hints=["ml"],
            quality_focus=True,
            recent_preference=False,
        ))

        assert "test/repo1" in prompt
        assert "test query" in prompt
        assert "evaluations" in prompt

    @patch("src.agent.search_agent.get_llm_config")
    def test_parse_evaluation_valid_json(self, mock_get_config):
        """测试解析有效 JSON"""
        mock_get_config.return_value = self.mock_config

        ranker = LLMRelevanceRanker()

        repos = [
            {
                "full_name": "test/repo1",
                "name": "repo1",
                "author": "test",
                "url": "https://github.com/test/repo1",
                "description": "A test repo",
                "language": "Python",
                "stars": 1000,
                "forks": 100,
                "open_issues": 10,
                "license": "MIT",
                "last_updated": "2024-01-01",
                "topics": [],
            }
        ]

        evaluation = '{"evaluations": [{"full_name": "test/repo1", "relevance_score": 0.8, "suitability_reasons": ["popular"], "tags": ["ml"]}]}'

        result = ranker._parse_evaluation(repos, evaluation)

        assert len(result) == 1
        assert result[0].relevance_score == 0.8

    @patch("src.agent.search_agent.get_llm_config")
    def test_parse_evaluation_invalid_json(self, mock_get_config):
        """测试解析无效 JSON"""
        mock_get_config.return_value = self.mock_config

        ranker = LLMRelevanceRanker()

        repos = [
            {
                "full_name": "test/repo1",
                "name": "repo1",
                "author": "test",
                "url": "https://github.com/test/repo1",
                "description": "A test repo",
                "language": "Python",
                "stars": 1000,
                "forks": 100,
                "open_issues": 10,
                "license": "MIT",
                "last_updated": "2024-01-01",
                "topics": [],
            }
        ]

        evaluation = "not valid json"

        result = ranker._parse_evaluation(repos, evaluation)

        # 应该返回空评估但保留原始分数
        assert len(result) == 1


class TestSearchRecommendationAgent:
    """测试搜索推荐 Agent"""

    def setup_method(self):
        """每个测试前创建 Agent"""
        self.mock_config = {
            "provider": "openai",
            "api_key": "test_key",
            "base_url": "https://api.test.com",
            "model": "gpt-4o-mini",
        }
        # 禁用 IntentRecognizer 的 LLM，避免测试中真实调用 API
        self.mock_llm_instance = Mock(spec=LLMService)
        self.mock_llm_instance.complete.side_effect = Exception("LLM disabled in test")

    @patch("src.agent.search_agent.get_llm_config")
    @patch("src.agent.search_agent.GitHubSearch")
    @patch("src.agent.search_agent.LLMService")
    def test_run_no_results(self, mock_llm_class, mock_search_class, mock_get_config):
        """测试无搜索结果"""
        mock_get_config.return_value = self.mock_config
        mock_llm_class.return_value = self.mock_llm_instance

        mock_searcher = MagicMock()
        mock_searcher.search_repos.return_value = MagicMock(repos=[])
        mock_search_class.return_value = mock_searcher

        agent = SearchRecommendationAgent()
        result = agent.run("xyznonexistent123", limit=5)

        assert result == []

    @patch("src.agent.search_agent.get_llm_config")
    @patch("src.agent.search_agent.GitHubSearch")
    @patch("src.agent.search_agent.LLMService")
    def test_run_with_results(self, mock_llm_class, mock_search_class, mock_get_config):
        """测试有搜索结果"""
        mock_get_config.return_value = self.mock_config
        mock_llm_class.return_value = self.mock_llm_instance

        mock_searcher = MagicMock()
        mock_searcher.search_repos.return_value = MagicMock(repos=[
            {
                "full_name": "test/repo1",
                "name": "repo1",
                "author": "test",
                "url": "https://github.com/test/repo1",
                "description": "A test repo",
                "language": "Python",
                "stars": 1000,
                "forks": 100,
                "open_issues": 10,
                "license": "MIT",
                "last_updated": "2024-01-01",
                "topics": [],
            }
        ])
        mock_search_class.return_value = mock_searcher

        agent = SearchRecommendationAgent()

        with patch.object(agent.ranker, "rank_repos") as mock_rank:
            mock_rank.return_value = [
                RepoCard(
                    repo_name="repo1",
                    author="test",
                    full_name="test/repo1",
                    url="https://github.com/test/repo1",
                    stars=1000,
                    forks=100,
                    open_issues=10,
                    language="Python",
                    description="A test repo",
                    license="MIT",
                    relevance_score=0.9,
                    suitability_reasons=["popular"],
                    tags=["test"],
                )
            ]

            result = agent.run("test query", limit=5)

            assert len(result) == 1
            assert result[0].full_name == "test/repo1"

    @patch("src.agent.search_agent.get_llm_config")
    @patch("src.agent.search_agent.GitHubSearch")
    @patch("src.agent.search_agent.LLMService")
    def test_search_only(self, mock_llm_class, mock_search_class, mock_get_config):
        """测试仅搜索"""
        mock_get_config.return_value = self.mock_config
        mock_llm_class.return_value = self.mock_llm_instance

        mock_searcher = MagicMock()
        mock_searcher.search_repos.return_value = MagicMock(repos=[
            {"full_name": "test/repo1", "name": "repo1"}
        ])
        mock_search_class.return_value = mock_searcher

        agent = SearchRecommendationAgent()
        result = agent.search_only("test", limit=10)

        assert len(result) == 1
        assert result[0]["full_name"] == "test/repo1"

    @patch("src.agent.search_agent.get_llm_config")
    @patch("src.agent.search_agent.GitHubSearch")
    @patch("src.agent.search_agent.LLMService")
    def test_intent_recognition(self, mock_llm_class, mock_search_class, mock_get_config):
        """测试意图识别"""
        mock_get_config.return_value = self.mock_config
        mock_llm_class.return_value = self.mock_llm_instance

        mock_searcher = MagicMock()
        mock_searcher.search_repos.return_value = MagicMock(repos=[])
        mock_search_class.return_value = mock_searcher

        agent = SearchRecommendationAgent()

        assert agent.intent_recognizer is not None
        assert isinstance(agent.intent_recognizer, IntentRecognizer)

    @patch("src.agent.search_agent.get_llm_config")
    @patch("src.agent.search_agent.GitHubSearch")
    @patch("src.agent.search_agent.LLMService")
    def test_run_prints_llm_summary(self, mock_llm_class, mock_search_class, mock_get_config, capsys):
        """测试 LLM 意图摘要在 run() 中被打印"""
        import json as _json
        mock_get_config.return_value = self.mock_config

        # LLM 意图解析返回包含 summary 的结果
        mock_llm_inst = Mock(spec=LLMService)
        mock_llm_inst.complete.return_value = _json.dumps({
            "project_type": "教程",
            "keywords": ["react"],
            "language": "JavaScript",
            "tech_stack": ["React"],
            "difficulty": "入门友好",
            "purpose": "学习参考",
            "quality_focus": False,
            "recent_preference": False,
            "min_stars": 10,
            "domain_hints": ["web"],
        }, ensure_ascii=False)
        mock_llm_class.return_value = mock_llm_inst

        mock_searcher = MagicMock()
        mock_searcher.search_repos.return_value = MagicMock(repos=[])
        mock_search_class.return_value = mock_searcher

        agent = SearchRecommendationAgent()
        agent.run("找 React 入门教程", limit=5)

        captured = capsys.readouterr()
        assert "我理解你想找" in captured.out
        assert "教程" in captured.out
