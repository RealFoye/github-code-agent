"""
搜索推荐 Agent（阶段一）
负责：意图识别 → GitHub 搜索 → LLM 相关性排序 → 推荐卡片
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from src.tools.github_search import GitHubSearch, search_repos, SearchResult
from src.tools.llm_service import LLMService
from src.models.repo_card import RepoCard
from src.config import (
    get_llm_config,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_MIN_STARS,
    RECOMMENDED_REPO_COUNT,
)

logger = logging.getLogger(__name__)


# ==================== 意图识别 ====================

@dataclass
class UserIntent:
    """用户搜索意图"""
    keywords: List[str]           # 核心关键词
    language: Optional[str]       # 偏好语言
    min_stars: int                # 最低 stars 要求
    domain_hints: List[str]       # 领域提示（从 topics 推断）
    quality_focus: bool           # 是否强调高质量（高 stars）
    recent_preference: bool       # 是否偏好近期更新
    # LLM 驱动的增强字段（可选，向后兼容）
    project_type: Optional[str] = None           # 项目类型（框架/工具/库/教程/Demo）
    difficulty: Optional[str] = None             # 难度（入门友好/中级/高级）
    purpose: Optional[str] = None                # 用途（学习参考/生产使用/研究）
    # tech_stack 三层分离
    tech_stack_explicit: List[str] = field(default_factory=list)   # 用户明确说的技术
    tech_stack_inferred: List[str] = field(default_factory=list)   # 强推断（可回指原话）
    inferred_sources: Dict[str, str] = field(default_factory=dict)  # {推断技术: 来源短语} 审计追踪用
    tech_stack_domain: List[str] = field(default_factory=list)     # 领域常识
    # 排除约束（用户明确否定）
    excluded_tech: List[str] = field(default_factory=list)        # 用户明确排除的技术/框架
    excluded_categories: List[str] = field(default_factory=list)   # 用户明确排除的仓库类型
    # 向后兼容字段（废弃但保留）
    tech_stack: List[str] = field(default_factory=list)  # 技术栈（如 React, Node.js）
    summary: str = ""                            # 意图摘要（供用户确认）
    search_query: str = ""                       # GitHub 搜索用的精简关键词（2-3 个英文词）


class IntentRecognizer:
    """意图识别器（LLM 驱动 + 规则匹配 fallback）"""

    # 常见编程语言映射
    LANGUAGE_MAP = {
        "python": "Python",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "java": "Java",
        "go": "Go",
        "rust": "Rust",
        "cpp": "C++",
        "c++": "C++",
        "csharp": "C#",
        "c#": "C#",
        "ruby": "Ruby",
        "php": "PHP",
        "swift": "Swift",
        "kotlin": "Kotlin",
        "scala": "Scala",
        "shell": "Shell",
        "bash": "Bash",
    }

    # 高质量项目的 star 门槛
    HIGH_QUALITY_THRESHOLDS = {
        "small": 100,
        "medium": 500,
        "large": 1000,
        "enterprise": 5000,
    }

    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        初始化意图识别器

        Args:
            llm_service: LLM 服务实例（不传则自动创建；传 None 则禁用 LLM）
        """
        if llm_service is not None:
            self._llm: Optional[LLMService] = llm_service
        else:
            try:
                self._llm = LLMService()
            except Exception as e:
                logger.warning(f"LLM 服务初始化失败，将使用规则匹配: {e}")
                self._llm = None

    def parse(self, user_query: str) -> UserIntent:
        """
        解析用户查询：优先使用 LLM，失败时降级到规则匹配

        Args:
            user_query: 用户自然语言查询

        Returns:
            UserIntent 对象（包含 summary 摘要字段）
        """
        logger.info(f"解析用户意图: {user_query}")

        if self._llm is not None:
            try:
                intent = self._llm_parse(user_query)
                logger.info(f"LLM 意图解析完成: type={intent.project_type}, lang={intent.language}, keywords={intent.keywords}")
                return intent
            except Exception as e:
                logger.warning(f"LLM 意图解析失败，降级到规则匹配: {e}")

        return self._rule_parse(user_query)

    def _llm_parse(self, user_query: str) -> UserIntent:
        """LLM 驱动的意图解析"""
        prompt = f"""你是一个 GitHub 项目搜索助手。请分析用户的搜索需求，严格区分三类信息。

用户需求："{user_query}"

【重要规则】
- tech_stack_explicit：只放用户原话里明确出现的技术名称，一个字都没提的不放
- tech_stack_inferred：只放能从用户原话某个具体短语推断出的技术方向，必须在 inferred_sources 里注明来源短语
- inferred_sources 的键必须和 tech_stack_inferred 中的值一一对应，不允许出现键值不匹配
- tech_stack_domain：你作为 LLM 知道的该领域主流技术，但用户完全没提的，放这里（这层不会用于评分）
- **否定表达**：如果用户说"不要X"、"不是X"、"别用X"、"不含X"、"排除X"，X必须放入 excluded_tech 或 excluded_categories，绝对不能放入任何 tech_stack 相关字段

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
  "excluded_tech": ["用户明确排除的技术/框架（如 LangChain）"],
  "excluded_categories": ["用户明确排除的仓库类型（如 framework、library）"],
  "difficulty": "难度（入门友好/中级/高级，用户说了才填，否则 null）",
  "purpose": "用途（学习参考/生产使用/研究/其他，用户说了才填，否则 null）",
  "quality_focus": true或false,
  "recent_preference": true或false,
  "min_stars": 最低star数（整数，默认{DEFAULT_MIN_STARS}），
  "domain_hints": ["领域标签"]
}}

只返回 JSON，不要有其他内容。"""

        raw = self._llm.complete(prompt, temperature=0.1, max_tokens=2000)

        # 提取 JSON
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            raise ValueError(f"LLM 未返回有效 JSON: {raw[:100]}")

        data = json.loads(json_match.group())

        keywords = data.get("keywords") or []
        if not keywords:
            keywords = self._extract_keywords(user_query.lower())

        language = data.get("language")
        # 标准化语言名
        if language:
            language = self.LANGUAGE_MAP.get(language.lower(), language)

        tech_stack_explicit = data.get("tech_stack_explicit") or []
        tech_stack_inferred = data.get("tech_stack_inferred") or []
        inferred_sources = data.get("inferred_sources") or {}
        tech_stack_domain = data.get("tech_stack_domain") or []
        excluded_tech = data.get("excluded_tech") or []
        excluded_categories = data.get("excluded_categories") or []

        # search_query：LLM 给出的精简英文搜索词，无则从 keywords 提取
        search_query = (data.get("search_query") or "").strip()
        if not search_query:
            search_query = self._build_search_query(keywords)

        intent = UserIntent(
            keywords=keywords,
            language=language,
            min_stars=int(data.get("min_stars", DEFAULT_MIN_STARS)),
            domain_hints=data.get("domain_hints") or [],
            quality_focus=bool(data.get("quality_focus", False)),
            recent_preference=bool(data.get("recent_preference", False)),
            project_type=data.get("project_type"),
            difficulty=data.get("difficulty"),
            purpose=data.get("purpose"),
            tech_stack_explicit=tech_stack_explicit,
            tech_stack_inferred=tech_stack_inferred,
            inferred_sources=inferred_sources,
            tech_stack_domain=tech_stack_domain,
            excluded_tech=excluded_tech,
            excluded_categories=excluded_categories,
            tech_stack=tech_stack_explicit + tech_stack_inferred,  # 向后兼容
            summary=self._format_summary(data, user_query),
            search_query=search_query,
        )

        return intent

    def _build_search_query(self, keywords: List[str]) -> str:
        """从 keywords 提取适合 GitHub 搜索的精简词（英文为主，最多 3 个单词）"""
        result = []
        for kw in keywords:
            # 跳过含中文字符的词
            if any('\u4e00' <= c <= '\u9fff' for c in kw):
                continue
            # 多词短语只取第一个词，避免组合搜索词过长
            first_word = kw.split()[0] if kw.split() else kw
            if first_word and first_word not in result:
                result.append(first_word)
            if len(result) >= 3:
                break
        # 如果全是中文，至少保留第一个关键词的第一个词
        if not result and keywords:
            first = keywords[0].split()
            if first:
                result = [first[0]]
        return " ".join(result)

    def _format_summary(self, data: Dict[str, Any], user_query: str) -> str:
        """将 LLM 解析结果格式化为可读摘要"""
        lines = ["我理解你想找："]

        project_type = data.get("project_type")
        if project_type:
            lines.append(f"  - 类型：{project_type}")

        language = data.get("language")
        tech_stack_explicit = data.get("tech_stack_explicit") or []
        tech_stack_inferred = data.get("tech_stack_inferred") or []

        if tech_stack_explicit:
            stack_str = " + ".join(tech_stack_explicit)
            lines.append(f"  - 技术栈：{stack_str}")
        elif tech_stack_inferred:
            stack_str = " + ".join(tech_stack_inferred) + " (推断)"
            lines.append(f"  - 技术栈：{stack_str}")
        elif language:
            lines.append(f"  - 语言：{language}")

        difficulty = data.get("difficulty")
        if difficulty:
            lines.append(f"  - 难度：{difficulty}")

        purpose = data.get("purpose")
        if purpose:
            lines.append(f"  - 目的：{purpose}")

        min_stars = data.get("min_stars", DEFAULT_MIN_STARS)
        if int(min_stars) > DEFAULT_MIN_STARS:
            lines.append(f"  - 最低 Stars：{min_stars:,}")

        if data.get("recent_preference"):
            lines.append("  - 偏好：近期活跃")

        lines.append("\n正在为你搜索匹配的 GitHub 项目...")
        return "\n".join(lines)

    def _rule_parse(self, user_query: str) -> UserIntent:
        """规则匹配解析（作为 LLM 的 fallback）"""
        query = user_query.lower().strip()

        language = self._extract_language(query)
        min_stars = self._extract_stars_requirement(query)

        quality_focus = any(keyword in query for keyword in [
            "高质量", "高星", "popular", "best", "top", "优秀", "知名", "主流"
        ])
        recent_preference = any(keyword in query for keyword in [
            "最新", "最近", "recent", "new", "updated", "活跃", "active"
        ])

        keywords = self._extract_keywords(query)
        domain_hints = self._extract_domain_hints(query)

        if quality_focus and min_stars == DEFAULT_MIN_STARS:
            min_stars = self.HIGH_QUALITY_THRESHOLDS["medium"]

        intent = UserIntent(
            keywords=keywords,
            language=language,
            min_stars=min_stars,
            domain_hints=domain_hints,
            quality_focus=quality_focus,
            recent_preference=recent_preference,
            search_query=self._build_search_query(keywords),
        )

        logger.info(f"规则匹配解析结果: {intent}")
        return intent

    def _extract_language(self, query: str) -> Optional[str]:
        """提取编程语言偏好"""
        for alias, language in self.LANGUAGE_MAP.items():
            # 匹配语言关键词（词边界）
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, query):
                return language
        return None

    def _extract_stars_requirement(self, query: str) -> int:
        """提取 stars 要求"""
        # 匹配模式：">1000", "stars>1000", "1000+", "超过1000" 等
        patterns = [
            r">\s*(\d+)",
            r"stars?\s*>\s*(\d+)",
            r"(\d+)\s*\+",
            r"超过\s*(\d+)",
            r"高于\s*(\d+)",
            r"至少\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return int(match.group(1))

        return DEFAULT_MIN_STARS

    def _extract_keywords(self, query: str) -> List[str]:
        """提取核心关键词"""
        # 移除常见停用词和修饰词
        stop_words = {
            # 中文停用词
            "我想要", "想要", "需要", "找", "搜索", "查找", "看看",
            "的", "一个", "一些", "有关", "关于", "这个", "那个",
            "请", "给我", "推荐", "介绍",
            # 英文停用词
            "i", "want", "need", "look", "find", "search",
            "a", "an", "the", "some", "any",
            # 项目相关（太泛化）
            "project", "repo", "repository", "projects", "repos",
            "项目", "仓库",
            # 修饰词（单独提取，不作为搜索关键词）
            "高质量", "高星", "popular", "best", "top", "优秀", "知名", "主流",
            "最新", "最近", "recent", "new", "updated", "活跃", "active",
        }

        # 移除数字（stars 数量等）
        words = re.findall(r"[\w]+", query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2 and not w.isdigit()]

        # 过滤掉语言名称（单独提取）
        language_names = set(l.lower() for l in self.LANGUAGE_MAP.values())
        keywords = [w for w in keywords if w not in language_names]

        # 如果没有提取到关键词，使用整个查询
        if not keywords:
            keywords = [query]

        return keywords

    def _extract_domain_hints(self, query: str) -> List[str]:
        """提取领域提示"""
        domain_keywords = {
            "web": ["web", "网站", "前端", "后端", "http", "api", "rest"],
            "ml": ["machine learning", "ml", "ai", "人工智能", "深度学习", "deep learning", "tensorflow", "pytorch"],
            "data": ["data", "数据", "analytics", "分析", "pipeline", "etl"],
            "devops": ["devops", "ci", "cd", "docker", "kubernetes", "k8s", "部署"],
            "mobile": ["mobile", "移动", "ios", "android", "react native"],
            "blockchain": ["blockchain", "区块链", "crypto", "web3", "solidity"],
            "gui": ["gui", "桌面", "desktop", "qt", "gtk", "tkinter"],
            "game": ["game", "游戏", "unity", "unreal", "godot"],
            "iot": ["iot", "物联网", "embedded", "嵌入式", "arduino"],
            "security": ["security", "安全", "crypto", "加密", "auth"],
        }

        hints = []
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    hints.append(domain)
                    break

        return hints


# ==================== LLM 相关性评估 ====================

class LLMRelevanceRanker:
    """基于 LLM 的相关性排序"""

    def __init__(self):
        """初始化 LLM 配置"""
        llm_config = get_llm_config()
        self.provider = llm_config["provider"]
        self.api_key = llm_config.get("api_key", "")
        self.base_url = llm_config.get("base_url", "")
        self.model = llm_config.get("model", "")

    def rank_repos(
        self,
        repos: List[Dict[str, Any]],
        user_query: str,
        intent: UserIntent,
    ) -> List[RepoCard]:
        """
        使用 LLM 评估仓库相关性并排序

        Args:
            repos: GitHub 搜索返回的仓库列表
            user_query: 用户原始查询
            intent: 解析后的用户意图

        Returns:
            排序后的 RepoCard 列表
        """
        if not repos:
            return []

        # VerifiableGate 过滤
        repos = self._apply_verifiable_gate(repos, intent)
        if not repos:
            logger.warning("VerifiableGate 过滤后候选为空，跳过 LLM 评分")
            return []

        logger.info(f"使用 LLM 评估 {len(repos)} 个仓库的相关性")

        # 构建 prompt
        prompt = self._build_evaluation_prompt(repos, user_query, intent)

        # 调用 LLM
        try:
            evaluation = self._call_llm(prompt)
            logger.info(f"LLM 返回原始内容长度: {len(evaluation)} 字符")

            # 记录 LLM 返回内容前 500 字符（用于调试 JSON 解析问题）
            logger.debug(f"LLM 原始返回（前500字符）: {evaluation[:500]}")

            cards, matched_count = self._parse_evaluation(repos, evaluation)
            logger.info(f"LLM 评估完成，返回 {len(cards)} 个推荐，matched_count={matched_count}")

            # 记录评分证据
            for card in cards:
                if card.evidence:
                    logger.debug(
                        f"[证据] {card.full_name} | "
                        f"intent={card.evidence.get('intent_fit_hits')} | "
                        f"quality_unknown={card.evidence.get('neutral_unknowns')} | "
                        f"excluded={card.evidence.get('domain_excluded')}"
                    )

            # 显式 fallback 条件：matched_count == 0 或 所有 score 都是 0.0
            all_zero = all(c.relevance_score == 0.0 for c in cards) if cards else True
            if matched_count == 0 or all_zero:
                logger.warning(
                    f"matched_count={matched_count}, all_zero={all_zero}，"
                    f"显式触发 fallback 到 star-based ranking"
                )
                return self._fallback_ranking(repos, intent)

            return cards

        except Exception as e:
            logger.error(f"LLM 评估失败: {e}，触发 fallback 到 star-based ranking")
            # 降级：使用基础分数排序
            return self._fallback_ranking(repos, intent)

    def _apply_verifiable_gate(
        self,
        repos: List[Dict[str, Any]],
        intent: UserIntent,
    ) -> List[Dict[str, Any]]:
        """
        可验证约束 Gate：只过滤能从元数据可靠验证的明确硬约束。
        不适用于"新手友好""容易跑起来"等无法在元数据层验证的需求。
        当前实现：语言过滤 + excluded_tech 过滤 + 噪音过滤。
        """
        passed = []
        dropped = {"language": 0, "excluded_tech": {}, "noise": 0}

        for repo in repos:
            full_name = repo.get("full_name", "")

            # 规则0：噪音过滤（高置信规则）
            # .github 类泛化仓库
            if full_name.endswith(".github"):
                logger.debug(f"噪音过滤 {full_name}: .github 类泛化仓库")
                dropped["noise"] += 1
                continue
            # 描述为空且名称异常泛化
            desc = repo.get("description") or ""
            name = repo.get("name", "").lower()
            if not desc and name in ("resources", "awesome", "list", "collection", "home", "index"):
                logger.debug(f"噪音过滤 {full_name}: 描述为空且名称泛化")
                dropped["noise"] += 1
                continue

            # 规则1：用户明确指定了语言，且语言不符
            if intent.language and repo.get("language"):
                if repo["language"].lower() != intent.language.lower():
                    logger.debug(f"Gate 淘汰 {full_name}: 语言不符（{repo['language']} != {intent.language}）")
                    dropped["language"] += 1
                    continue

            # 规则2：excluded_tech 过滤（topics → repo name → description）
            if intent.excluded_tech:
                matched_excluded = self._match_excluded_tech(repo, intent.excluded_tech)
                if matched_excluded:
                    logger.debug(f"Gate 淘汰 {full_name}: excluded_tech={matched_excluded}")
                    dropped["excluded_tech"][matched_excluded] = dropped["excluded_tech"].get(matched_excluded, 0) + 1
                    continue

            passed.append(repo)

        # 记录汇总日志
        dropped_tech_str = ", ".join(f"{k}={v}" for k, v in dropped["excluded_tech"].items())
        logger.info(
            f"VerifiableGate：{len(repos)} 个候选 → {len(passed)} 个通过"
            f"（dropped_by_language={dropped['language']}, "
            f"dropped_by_excluded_tech={sum(dropped['excluded_tech'].values())}"
            + (f" [{dropped_tech_str}]" if dropped_tech_str else "")
            + f", dropped_by_noise={dropped['noise']}）"
        )
        return passed

    def _match_excluded_tech(
        self,
        repo: Dict[str, Any],
        excluded_tech: List[str],
    ) -> Optional[str]:
        """
        检查仓库是否命中 excluded_tech。
        检查顺序：topics → repo name → description。
        返回匹配到的第一个 excluded tech，否则返回 None。
        """
        topics = repo.get("topics", []) or []
        name = repo.get("name", "").lower()
        desc = (repo.get("description") or "").lower()

        for tech in excluded_tech:
            tech_lower = tech.lower()
            # 1. topics 精确匹配
            if tech_lower in [t.lower() for t in topics]:
                return tech
            # 2. repo name 匹配
            if tech_lower in name:
                return tech
            # 3. description 中明确提及
            if tech_lower in desc:
                return tech
        return None

    def _build_evaluation_prompt(
        self,
        repos: List[Dict[str, Any]],
        user_query: str,
        intent: UserIntent,
    ) -> str:
        """构建评估 prompt"""
        # 限制评估数量，避免 token 超出
        repos_to_evaluate = repos[:10]

        repo_summaries = []
        for i, repo in enumerate(repos_to_evaluate, 1):
            summary = f"""
{i}. {repo['full_name']}
   描述: {repo.get('description') or '无'}
   语言: {repo.get('language') or '未指定'}
   Stars: {repo.get('stars', 0):,}
   Forks: {repo.get('forks', 0):,}
   Topics: {', '.join(repo.get('topics', [])[:5]) or '无'}
   许可证: {repo.get('license') or '未指定'}
   更新: {repo.get('last_updated', '未知')}
"""
            repo_summaries.append(summary)

        repos_text = "\n".join(repo_summaries)

        # 构建约束说明
        explicit_desc = "、".join(intent.tech_stack_explicit) if intent.tech_stack_explicit else "无"
        inferred_desc = "、".join(intent.tech_stack_inferred) if intent.tech_stack_inferred else "无"
        domain_desc   = "、".join(intent.tech_stack_domain)   if intent.tech_stack_domain   else "无"
        excluded_cat_desc = "、".join(intent.excluded_categories) if intent.excluded_categories else "无"

        # 检测学习场景
        is_learning = (
            intent.difficulty in ("入门友好", "入门") or
            intent.purpose in ("学习参考", "学习")
        )
        learning_guidance = ""
        if is_learning:
            learning_guidance = """
【学习场景额外偏好——优先推荐可动手复刻的小项目】
用户想找的是"最适合第一次上手模仿的项目"，而非大型教程仓库。
请额外给以下特征加分（intent_fit 可提升 0.1~0.2）：
- 项目名称含：demo、example、from-scratch、build-your-own、step-by-step、tutorial、starter
- description 强调"从零开始"、"手把手"、"自己构建"、"简单清晰"
- 不是巨大的资源集合（50+ tutorials、notebook 合集、大型课程仓库）

同时对以下类型轻微降权（intent_fit 可降低 0.05~0.1）：
- 大型教程集合（50+ tutorials、20+ notebooks）
- 过于通用的"资源列表"型仓库
- 不适合第一次动手的综合框架样例
"""

        prompt = f"""你是一个 GitHub 项目推荐专家。用户正在寻找: "{user_query}"

【评分约束——必须严格遵守】
1. 用户明确要求的技术：{explicit_desc}
   → 可作为正向评分依据
2. 从用户原话推断的偏好：{inferred_desc}
   → 只能加分（0 到 +0.15），不能因为仓库不符合此项而扣分
3. 领域常识（用户未提及）：{domain_desc}
   → 禁止进入评分。不要因为仓库未使用这些技术而扣分，也不要因为使用了而加分
4. 用户明确排除的类型：{excluded_cat_desc}
   → 如果仓库属于这些类型（如 framework、library），intent_fit 降低 0.1~0.2{learning_guidance}

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

【输出规则——必须严格遵守】
- full_name 必须从候选仓库列表中选取，不允许编造或使用模板占位符（如 "owner/repo"）
- 每个 full_name 必须逐字匹配列表中的某一仓库
- 如果某个仓库无法评估，可以省略，但不要使用任何未在列表中出现过的 full_name
- 只输出 JSON，不要包含任何解释或说明文字

请返回 JSON：
{{
  "evaluations": [
    {{
      "full_name": "候选仓库的完整名称（必须与上述列表中的 full_name 完全一致）",
      "intent_fit": 0.0~1.0,
      "observable_quality": 0.0~1.0,
      "preference_bonus": 0.0~0.15,
      "final_score": intent_fit*0.6 + observable_quality*0.3 + preference_bonus,
      "suitability_reasons": ["命中了哪些用户需求"],
      "potential_concerns": ["有哪些不确定项"],
      "tags": ["tag1"],
      "evidence": {{
        "intent_fit_hits": ["具体命中点"],
        "quality_signals": ["用了哪些可观测信号"],
        "preference_bonus_reasons": ["加分来源（来自哪条强推断）"],
        "neutral_unknowns": ["哪些项因信息不足给了中性分"],
        "gate_checks": {{"language_match": true或false}},
        "domain_excluded": ["哪些领域常识被排除在评分外"]
      }}
    }}
  ]
}}

只返回 JSON。"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API"""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "deepseek":
            return self._call_deepseek(prompt)
        elif self.provider == "minimax":
            return self._call_minimax(prompt)
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        """调用 OpenAI API"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的 GitHub 项目推荐助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    def _call_deepseek(self, prompt: str) -> str:
        """调用 DeepSeek API"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的 GitHub 项目推荐助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    def _call_minimax(self, prompt: str) -> str:
        """调用 MiniMax API (使用 Anthropic SDK，带重试和 thinking 块处理)"""
        import anthropic
        import time

        max_retries = 3
        retry_delay = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )

                response = client.messages.create(
                    model=self.model,
                    max_tokens=8000,
                    system="你是一个专业的 GitHub 项目推荐助手。",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )

                # 遍历 content 找到 TextBlock（MiniMax-M2.7 可能返回 ThinkingBlock + TextBlock）
                for block in response.content:
                    if block.type == "text":
                        return block.text

                # 如果只有 thinking 块没有 text 块，说明 max_tokens 不够，重试一次
                block_types = [b.type for b in response.content]
                if "thinking" in block_types and "text" not in block_types:
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=8000 * 4,
                        system="你是一个专业的 GitHub 项目推荐助手。",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                    )
                    for block in response.content:
                        if block.type == "text":
                            return block.text

                # 如果没有找到 TextBlock，返回错误信息
                raise ValueError(f"无法从响应中提取文本内容，响应类型: {[b.type for b in response.content]}")

            except Exception as e:
                last_error = e
                error_str = str(e)

                # 判断是否可重试：529 过载 / 500/502/503/504 服务器错误
                retryable = (
                    "529" in error_str or
                    "overloaded" in error_str.lower() or
                    "500" in error_str or
                    "502" in error_str or
                    "503" in error_str or
                    "504" in error_str or
                    "api_error" in error_str.lower() or
                    "unknown error" in error_str.lower()
                )

                if retryable and attempt < max_retries - 1:
                    wait = retry_delay * (attempt + 1)
                    logger.warning(f"MiniMax API 可重试错误: {e}，{wait}秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    continue

                raise

        raise last_error if last_error else Exception("Max retries exceeded")

    def _cleanup_json_string(self, raw: str) -> str:
        """
        清理 JSON 字符串中的常见问题，尝试恢复可解析的 JSON。
        1. 去除 markdown code block 标记
        2. 去除尾部的截断内容（不完整的数组/对象）
        """
        import json
        # 去除 markdown 代码块标记
        cleaned = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned)

        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

        # 尝试找到 {...} 范围并截取
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # 尝试从每个 full_name 位置往前找对象起点
        for match in re.finditer(r'"full_name"\s*:\s*"[^"]*"', cleaned):
            obj_start = cleaned.rfind("{", 0, match.end())
            if obj_start != -1:
                candidate = cleaned[obj_start:end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        return cleaned

    def _parse_evaluation(
        self,
        repos: List[Dict[str, Any]],
        evaluation: str,
    ) -> Tuple[List[RepoCard], int]:
        """
        解析 LLM 评估结果。

        Returns:
            Tuple of (cards, matched_count) — matched_count 用于诊断
        """
        import json

        # 尝试提取 JSON
        evaluations = {}
        repo_full_names = {repo["full_name"] for repo in repos}  # 候选仓库 full_name 集合
        repo_full_names_lower = {fn.lower(): fn for fn in repo_full_names}  # 大小写不敏感查找

        # 检测模板占位符的 pattern
        PLACEHOLDER_PATTERNS = (
            "owner/repo", "username/reponame", "user/repo",
            "owner/name", "user/repository", "某用户/某仓库",
        )
        is_placeholder = lambda fn: fn.lower() in PLACEHOLDER_PATTERNS

        matched_count = 0
        placeholder_detected = False

        try:
            # 查找 JSON 块
            json_match = re.search(r"\{[\s\S]*\}", evaluation)
            if json_match:
                raw_json = json_match.group()
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError as e:
                    # 尝试清理后重新解析
                    logger.debug(f"JSON 解析首次失败，尝试 cleanup: {e}")
                    cleaned_json = self._cleanup_json_string(raw_json)
                    data = json.loads(cleaned_json)
            else:
                data = json.loads(evaluation)

            # 构建 evaluations 字典
            raw_evaluations = data.get("evaluations", [])
            logger.debug(f"LLM 返回 {len(raw_evaluations)} 个 evaluations")

            # 详细诊断：检查 full_name 匹配情况
            evaluations = {}
            unmatched_full_names = []  # LLM 返回但不在候选中的 full_name
            for e in raw_evaluations:
                fn = e.get("full_name", "")

                # 步骤1：检测模板占位符
                if is_placeholder(fn):
                    logger.warning(f"[placeholder_full_name] 检测到模板占位符 full_name: '{fn}'")
                    placeholder_detected = True
                    continue

                # 步骤2：先不做精确匹配，先尝试大小写不敏感匹配
                fn_lower = fn.lower()
                actual_fn = repo_full_names_lower.get(fn_lower)

                if actual_fn:
                    # 候选中存在（大小写不敏感匹配成功）
                    matched_count += 1
                    evaluations[actual_fn] = e
                else:
                    # 不在候选列表中 → 可能是 gibberish 或乱码
                    unmatched_full_names.append(fn)
                    logger.warning(
                        f"[gibberish_full_name] LLM 返回了不在候选列表中的 full_name: '{fn}'"
                    )

            logger.debug(
                f"Evaluations 匹配诊断: "
                f"候选仓库数={len(repos)}, "
                f"LLM 返回数={len(raw_evaluations)}, "
                f"匹配数={matched_count}, "
                f"未匹配LLM返回={unmatched_full_names[:5]}"
            )


        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}，使用降级排序")
            evaluations = {}

        # 构建 RepoCard 列表
        cards = []
        for repo in repos:
            full_name = repo["full_name"]

            # 匹配评估结果
            eval_data = evaluations.get(full_name, {})

            card = RepoCard(
                repo_name=repo.get("name", ""),
                author=repo.get("author", ""),
                full_name=full_name,
                url=repo.get("url", ""),
                stars=repo.get("stars", 0),
                forks=repo.get("forks", 0),
                open_issues=repo.get("open_issues", 0),
                language=repo.get("language"),
                languages=[],  # 搜索 API 不返回完整语言列表
                description=repo.get("description"),
                license=repo.get("license"),
                last_updated=repo.get("last_updated", "")[:10] if repo.get("last_updated") else None,
                topics=repo.get("topics", []),
                relevance_score=eval_data.get("final_score", eval_data.get("relevance_score", 0.0)),
                suitability_reasons=eval_data.get("suitability_reasons", []),
                potential_concerns=eval_data.get("potential_concerns", []),
                tags=eval_data.get("tags", []),
                evidence=eval_data.get("evidence", {}),
            )
            cards.append(card)

        # 按相关性分数排序
        cards.sort(key=lambda c: c.relevance_score, reverse=True)

        return cards, matched_count

    def _fallback_ranking(
        self,
        repos: List[Dict[str, Any]],
        intent: UserIntent,
    ) -> List[RepoCard]:
        """降级排序策略（不使用 LLM）"""
        logger.info("使用降级排序策略")

        cards = []
        for repo in repos:
            # 基础分数 = stars 归一化
            base_score = min(repo.get("stars", 0) / 10000, 1.0)

            # 语言匹配加分
            if intent.language and repo.get("language") == intent.language:
                base_score += 0.2

            card = RepoCard(
                repo_name=repo.get("name", ""),
                author=repo.get("author", ""),
                full_name=repo.get("full_name", ""),
                url=repo.get("url", ""),
                stars=repo.get("stars", 0),
                forks=repo.get("forks", 0),
                open_issues=repo.get("open_issues", 0),
                language=repo.get("language"),
                description=repo.get("description"),
                license=repo.get("license"),
                last_updated=repo.get("last_updated", "")[:10] if repo.get("last_updated") else None,
                topics=repo.get("topics", []),
                relevance_score=min(base_score, 1.0),
                suitability_reasons=[f"Stars: {repo.get('stars', 0):,}"],
                tags=intent.domain_hints[:3] if intent.domain_hints else [],
            )
            cards.append(card)

        cards.sort(key=lambda c: c.relevance_score, reverse=True)
        return cards


# ==================== 搜索推荐 Agent ====================

class SearchRecommendationAgent:
    """搜索推荐 Agent（阶段一）"""

    def __init__(self):
        """初始化 Agent"""
        self.searcher = GitHubSearch()
        self.intent_recognizer = IntentRecognizer()
        self.ranker = LLMRelevanceRanker()

    def _build_multi_queries(self, intent: UserIntent) -> List[Dict[str, Any]]:
        """
        构建三条查询，各有角色分工，不是同义词改写。
        返回格式：[{"role": "main", "query": "...", "limit": 10}, ...]
        """
        main_query = intent.search_query or " ".join(intent.keywords[:2])

        # 新手学习场景：把 "framework" 替换为 "example"，避免偏向框架生态
        is_learning = (
            intent.difficulty in ("入门友好", "入门") or
            intent.purpose in ("学习参考", "学习")
        )
        if is_learning:
            main_query = main_query.replace("framework", "example").replace("Framework", "example")

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

    def run(self, user_query: str, limit: int = RECOMMENDED_REPO_COUNT) -> List[RepoCard]:
        """
        运行搜索推荐流程

        Args:
            user_query: 用户自然语言查询
            limit: 返回推荐数量

        Returns:
            推荐仓库列表（RepoCard）
        """
        logger.info(f"=== 开始搜索推荐: {user_query} ===")

        # Step 1: 意图识别
        intent = self.intent_recognizer.parse(user_query)
        logger.info(f"识别到意图: keywords={intent.keywords}, language={intent.language}, min_stars={intent.min_stars}")

        # 新手学习场景：降低 star 门槛，避免误伤小而清晰的教学仓库
        is_learning_scenario = (
            intent.difficulty in ("入门友好", "入门") or
            intent.purpose in ("学习参考", "学习") or
            any(kw in user_query.lower() for kw in ["新手", "学习", "入门", "学着做", "教程", "示例"])
        )
        if is_learning_scenario and intent.min_stars > 50:
            intent.min_stars = 50

        # 显示意图摘要（LLM 驱动时有内容）
        if intent.summary:
            print(f"\n{intent.summary}\n")

        # Step 2: 多路搜索
        repos = self._multi_search(intent)

        if not repos:
            logger.warning("未找到任何仓库")
            return []

        logger.info(f"多路搜索返回 {len(repos)} 个仓库")

        # Step 3: LLM 相关性排序
        cards = self.ranker.rank_repos(repos, user_query, intent)

        # Step 4: 去同质化重排
        top_cards = self._diversify(cards, limit)

        logger.info(f"=== 搜索推荐完成，返回 {len(top_cards)} 个推荐 ===")
        return top_cards

    def search_only(self, keywords: str, language: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        仅执行 GitHub 搜索（不进行 LLM 排序）

        Args:
            keywords: 搜索关键词
            language: 语言筛选
            limit: 结果数量

        Returns:
            原始搜索结果
        """
        result = self.searcher.search_repos(keywords, language, limit=limit)
        return result.repos


# 便捷函数
def recommend_repos(user_query: str, limit: int = RECOMMENDED_REPO_COUNT) -> List[RepoCard]:
    """
    推荐仓库（便捷函数）

    Args:
        user_query: 用户查询
        limit: 返回数量

    Returns:
        RepoCard 列表
    """
    agent = SearchRecommendationAgent()
    return agent.run(user_query, limit)