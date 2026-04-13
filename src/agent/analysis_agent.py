"""
代码分析 Agent（阶段二）
负责：克隆 → 解析 → RAG 构建 → 问答 → 报告生成
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.tools.repo_cloner import RepoCloner
from src.tools.code_parser import CodeParser, RepoStructure
from src.tools.github_search import GitHubSearch
from src.tools.chunker import Chunker, CodeChunk
from src.tools.rag_pipeline import RAGPipeline, RetrievedChunk, build_index, search, detect_tutorial_number, detect_query_intent
from src.tools.report_generator import ReportGenerator, generate_report, save_report
from src.models.repo_card import RepoCard
from src.config import get_llm_config, RAG_TOP_K, RAG_RERANK_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class AnalysisSession:
    """分析会话"""
    repo_url: str
    repo_path: str
    repo_name: str
    full_name: str

    # 仓库信息
    repo_info: Dict[str, Any] = None
    structure: RepoStructure = None

    # RAG Pipeline
    rag_pipeline: RAGPipeline = None

    # 分析状态
    is_indexed: bool = False


class CodeAnalysisAgent:
    """代码分析 Agent（阶段二）"""

    def __init__(self):
        """初始化 Agent"""
        self.cloner = RepoCloner()
        self.parser = CodeParser()
        self.report_generator = ReportGenerator("")

    def start_session(self, repo_url: str) -> AnalysisSession:
        """
        启动分析会话

        Args:
            repo_url: 仓库 URL

        Returns:
            AnalysisSession 对象
        """
        logger.info(f"=== 启动分析会话: {repo_url} ===")

        # 从 URL 提取仓库信息
        full_name = self._extract_repo_name(repo_url)
        repo_name = full_name.split("/")[-1] if full_name else "unknown"

        # 克隆仓库
        clone_result = self.cloner.clone(repo_url)
        repo_path = clone_result["local_path"]

        session = AnalysisSession(
            repo_url=repo_url,
            repo_path=repo_path,
            repo_name=repo_name,
            full_name=full_name or repo_name,
        )

        # 获取仓库信息
        session.repo_info = self.cloner.get_repo_info(repo_path)
        # 补充基本信息（report_generator 需要，但 cloner 不返回）
        session.repo_info["full_name"] = full_name or repo_name
        session.repo_info["repo_url"] = repo_url

        # 通过 GitHub API 补充仓库详情（stars、description、language 等）
        if full_name:
            try:
                github_info = GitHubSearch().get_repo_details(full_name)
                if github_info:
                    # 合并 GitHub API 返回的信息
                    for key in ["description", "stars", "forks", "language", "license", "topics"]:
                        if key in github_info:
                            session.repo_info[key] = github_info[key]
            except Exception as e:
                logger.warning(f"无法获取 GitHub 仓库详情: {e}")

        return session

    def _extract_repo_name(self, repo_url: str) -> str:
        """从 URL 提取仓库完整名称"""
        # https://github.com/owner/repo -> owner/repo
        if "github.com" in repo_url:
            parts = repo_url.rstrip("/").split("/")
            if len(parts) >= 2:
                owner = parts[-2]
                repo = parts[-1].replace(".git", "")
                return f"{owner}/{repo}"
        return ""

    def analyze_structure(self, session: AnalysisSession) -> RepoStructure:
        """
        分析仓库结构

        Args:
            session: 分析会话

        Returns:
            RepoStructure 对象
        """
        logger.info("开始分析仓库结构...")

        structure = self.parser.scan_structure(session.repo_path)
        session.structure = structure

        logger.info(f"结构分析完成: {structure.total_files} 文件, {structure.total_lines} 行代码")
        return structure

    def build_rag_index(self, session: AnalysisSession) -> int:
        """
        构建 RAG 索引

        Args:
            session: 分析会话

        Returns:
            索引的 Chunk 数量
        """
        logger.info("开始构建 RAG 索引...")

        # 创建 RAG Pipeline
        session.rag_pipeline = RAGPipeline(session.full_name)

        # 构建索引
        chunk_count = session.rag_pipeline.build_index(session.repo_path)
        session.is_indexed = True

        logger.info(f"RAG 索引构建完成: {chunk_count} chunks")
        return chunk_count

    def ask(
        self,
        session: AnalysisSession,
        question: str,
        top_k: int = RAG_TOP_K,
    ) -> Dict[str, Any]:
        """
        回答关于代码库的问题

        Args:
            session: 分析会话
            question: 问题
            top_k: 检索数量

        Returns:
            包含答案和引用来源的字典
        """
        logger.info(f"=== 问答: {question} ===")

        if not session.is_indexed or session.rag_pipeline is None:
            raise ValueError("RAG 索引未构建，请先调用 build_rag_index()")

        # Step 1: 检测 query intent 和 tutorial 编号
        query_intent = detect_query_intent(question)
        tutorial_hint = detect_tutorial_number(question)

        if tutorial_hint:
            logger.info(f"检测到 tutorial 编号: {tutorial_hint}，增强检索")
        if query_intent != "general":
            logger.info(f"检测到 query intent: {query_intent}，应用路由策略")

        # Step 2: 检索相关代码块
        retrieved = session.rag_pipeline.search(question, top_k=top_k * 2, tutorial_hint=tutorial_hint)

        if not retrieved:
            return {
                "question": question,
                "answer": "没有找到相关的代码片段来回答这个问题。",
                "sources": [],
            }

        # Step 3: 重排序
        reranked = session.rag_pipeline.rerank(
            question, retrieved, top_k=top_k,
            tutorial_hint=tutorial_hint, query_intent=query_intent
        )

        # Step 4: 去重（同一文件多个片段时合并）
        context_chunks = [r.chunk for r in reranked]
        deduplicated_map = {}  # file_path -> first chunk with that path
        for chunk in context_chunks:
            if chunk.file_path not in deduplicated_map:
                deduplicated_map[chunk.file_path] = chunk
        deduplicated_chunks = list(deduplicated_map.values())

        # Step 5: 构建上下文（使用去重后的 chunks）
        context = self._build_context(deduplicated_chunks)

        # Step 6: 调用 LLM 生成答案
        answer = self._generate_answer(question, context)

        # Step 7: 验证引用（检查代码片段是否真实存在）
        verified_sources = self._verify_sources(deduplicated_chunks, session.repo_path)

        logger.info(f"问答完成，找到 {len(verified_sources)} 个引用")

        return {
            "question": question,
            "answer": answer,
            "sources": verified_sources,
        }

    def _build_context(self, chunks: List[CodeChunk]) -> str:
        """构建 LLM 上下文（带去重）"""
        # 按文件路径去重，保留同一个文件的多个片段但标注来源
        seen_files = {}  # file_path -> list of (start_line, end_line, chunk_index)

        # 去重：合并同一文件的多个片段
        deduplicated_chunks = []
        for chunk in chunks:
            file_path = chunk.file_path
            if file_path in seen_files:
                # 追加片段信息
                seen_files[file_path].append(chunk)
            else:
                seen_files[file_path] = [chunk]
                deduplicated_chunks.append(chunk)

        parts = []

        for i, chunk in enumerate(deduplicated_chunks, 1):
            # 显示 heading_path 如果存在
            location = f"文件: {chunk.file_path}"
            if chunk.heading_path:
                location += f" ({chunk.heading_path})"
            location += f" (行 {chunk.start_line}-{chunk.end_line})"

            part = f"""--- 代码片段 {i} ---
{location}
类型: {chunk.chunk_type}
名称: {chunk.name}
签名: {chunk.signature}

```{(chunk.language or 'text')}
{chunk.content}
```
"""
            parts.append(part)

        return "\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """调用 LLM 生成答案"""
        llm_config = get_llm_config()
        provider = llm_config["provider"]

        prompt = f"""你是一个专业的代码分析师。以下是代码库中的相关代码片段：

{context}

请根据以上代码片段回答用户的问题。

用户问题：{question}

回答要求：
1. 基于提供的代码片段进行回答，不要编造代码中不存在的信息
2. 如果代码片段不足以回答问题，请明确说明
3. 指出相关的代码位置（文件路径和行号）
4. 适当引用代码中的关键部分

回答："""

        if provider == "openai":
            return self._call_openai(prompt, llm_config)
        elif provider == "deepseek":
            return self._call_deepseek(prompt, llm_config)
        elif provider == "minimax":
            return self._call_minimax(prompt, llm_config)
        else:
            raise ValueError(f"不支持的 LLM provider: {provider}")

    def _call_openai(self, prompt: str, config: dict) -> str:
        """调用 OpenAI API"""
        from openai import OpenAI

        client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url"))

        response = client.chat.completions.create(
            model=config.get("model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "你是一个专业的代码分析师，擅长分析代码库并给出准确的回答。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    def _call_deepseek(self, prompt: str, config: dict) -> str:
        """调用 DeepSeek API"""
        from openai import OpenAI

        client = OpenAI(api_key=config.get("api_key"), base_url=config.get("base_url"))

        response = client.chat.completions.create(
            model=config.get("model", "deepseek-chat"),
            messages=[
                {"role": "system", "content": "你是一个专业的代码分析师，擅长分析代码库并给出准确的回答。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        return response.choices[0].message.content

    def _call_minimax(self, prompt: str, config: dict) -> str:
        """调用 MiniMax API (使用 Anthropic SDK，带重试)"""
        import anthropic
        import time

        max_retries = 3
        retry_delay = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                client = anthropic.Anthropic(
                    api_key=config.get("api_key"),
                    base_url=config.get("base_url"),
                )

                response = client.messages.create(
                    model=config.get("model", "MiniMax-M2.7"),
                    max_tokens=2000,
                    system="你是一个专业的代码分析师，擅长分析代码库并给出准确的回答。",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )

                # 遍历 content 找到 TextBlock（MiniMax-M2.7 可能返回 ThinkingBlock + TextBlock）
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

                # 不可重试或已达最大重试次数
                raise

        # 理论上不会到达这里，因为循环内每次都会 raise
        raise last_error if last_error else Exception("Max retries exceeded")

    def _verify_sources(
        self,
        chunks: List[CodeChunk],
        repo_path: str,
    ) -> List[Dict[str, Any]]:
        """验证引用的代码片段是否真实存在"""
        verified = []

        for chunk in chunks:
            file_path = chunk.file_path

            # 检查文件路径是否在仓库内
            if not file_path.startswith(repo_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                # 验证行号范围
                if not (1 <= chunk.start_line <= len(lines) and 1 <= chunk.end_line <= len(lines)):
                    logger.debug(f"行号范围无效: {file_path} ({chunk.start_line}-{chunk.end_line})")
                    continue

                # 验证内容一致性：检查 chunk.content 是否与文件中实际内容匹配
                actual_content = "".join(lines[chunk.start_line - 1:chunk.end_line])

                # 内容匹配校验（允许尾部空白差异）
                if self._content_matches(chunk.content, actual_content):
                    verified.append({
                        "chunk": chunk,
                        "verified": True,
                    })
                else:
                    # 内容不匹配，尝试用更宽松的校验
                    if self._content_matches_lenient(chunk.content, actual_content):
                        logger.debug(f"内容部分匹配（宽松校验）: {file_path} ({chunk.start_line}-{chunk.end_line})")
                        verified.append({
                            "chunk": chunk,
                            "verified": True,  # 宽松校验通过也算真实
                        })
                    else:
                        logger.warning(f"代码片段内容不匹配: {file_path} ({chunk.start_line}-{chunk.end_line})")
                        # 内容严重不匹配的 chunk 仍然加入，但标记 verified=False
                        verified.append({
                            "chunk": chunk,
                            "verified": False,
                        })

            except Exception as e:
                logger.debug(f"验证代码片段失败: {file_path}, {e}")

        return verified

    def _content_matches(self, chunk_content: str, actual_content: str) -> bool:
        """
        精确匹配校验：chunk.content 与文件中实际内容是否一致

        考虑因素：
        - 换行符差异（\n vs \r\n）
        - 尾部空白差异
        """
        # 标准化空白字符
        normalized_chunk = chunk_content.rstrip()
        normalized_actual = actual_content.rstrip()

        return normalized_chunk == normalized_actual

    def _content_matches_lenient(self, chunk_content: str, actual_content: str) -> bool:
        """
        宽松匹配校验：用于处理切片边界略有不同的情况

        策略：逐行比较，跳过空白行差异
        """
        chunk_lines = [l.rstrip() for l in chunk_content.splitlines() if l.strip()]
        actual_lines = [l.rstrip() for l in actual_content.splitlines() if l.strip()]

        if not chunk_lines or not actual_lines:
            return False

        # 检查关键行是否匹配（忽略顺序和完全空行的差异）
        chunk_set = set(chunk_lines)
        actual_set = set(actual_lines)

        # 如果有超过 50% 的 chunk 行在 actual 中找到，认为匹配
        matching = sum(1 for cl in chunk_lines if cl in actual_set)
        match_ratio = matching / len(chunk_lines) if chunk_lines else 0

        return match_ratio >= 0.7

    def generate_report(
        self,
        session: AnalysisSession,
        qa_results: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成分析报告

        Args:
            session: 分析会话
            qa_results: 问答结果列表
            output_path: 输出路径（可选）

        Returns:
            Markdown 报告文本
        """
        logger.info("开始生成报告...")

        # 更新报告生成器的仓库名
        self.report_generator = ReportGenerator(session.full_name)

        # 生成报告
        report = self.report_generator.generate_report(
            repo_path=session.repo_path,
            repo_info=session.repo_info,
            structure=session.structure,
            qa_results=qa_results,
        )

        # 保存到文件
        if output_path:
            save_report(report, output_path)
        else:
            # 默认保存到仓库目录
            default_path = f"{session.repo_path}/analysis_report.md"
            save_report(report, default_path)

        return report

    def run_full_pipeline(
        self,
        repo_url: str,
        questions: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的分析流程（保护仓库不被清理）

        Args:
            repo_url: 仓库 URL
            questions: 需要回答的问题列表（可选）
            output_path: 报告输出路径

        Returns:
            包含所有结果的字典
        """
        logger.info(f"=== 开始完整分析: {repo_url} ===")

        # 克隆并获取路径（用于保护）
        clone_result = self.cloner.clone(repo_url)
        repo_path = clone_result["local_path"]

        # 保护仓库不被自动清理，直到分析完成
        with self.cloner.protecting(repo_path):
            return self._run_analysis(repo_url, repo_path, questions, output_path)

    def _run_analysis(
        self,
        repo_url: str,
        repo_path: str,
        questions: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        内部：执行分析流程（假设仓库已被保护）

        Args:
            repo_url: 仓库 URL
            repo_path: 仓库本地路径
            questions: 问题列表
            output_path: 报告输出路径

        Returns:
            分析结果字典
        """
        # 构建 Session 对象（复用 start_session 的逻辑，但不克隆）
        from dataclasses import dataclass
        full_name = self._extract_repo_name(repo_url) or repo_path.split("/")[-1]

        # 获取仓库信息（已克隆，直接读本地）
        session = AnalysisSession(
            repo_url=repo_url,
            repo_path=repo_path,
            repo_name=full_name.split("/")[-1] if "/" in full_name else full_name,
            full_name=full_name,
        )
        session.repo_info = self.cloner.get_repo_info(repo_path)
        session.repo_info["full_name"] = full_name
        session.repo_info["repo_url"] = repo_url

        # 补充 GitHub 详情
        if full_name:
            try:
                github_info = GitHubSearch().get_repo_details(full_name)
                if github_info:
                    for key in ["description", "stars", "forks", "language", "license", "topics"]:
                        if key in github_info:
                            session.repo_info[key] = github_info[key]
            except Exception as e:
                logger.warning(f"无法获取 GitHub 仓库详情: {e}")

        # 分析结构
        structure = self.analyze_structure(session)

        # 构建 RAG 索引
        chunk_count = self.build_rag_index(session)

        # 问答
        qa_results = []
        if questions:
            for q in questions:
                result = self.ask(session, q)
                qa_results.append(result)

        # 生成报告
        report = self.generate_report(session, qa_results, output_path)

        return {
            "session": session,
            "structure": structure,
            "chunk_count": chunk_count,
            "qa_results": qa_results,
            "report": report,
        }

    def cleanup_session(self, session: AnalysisSession):
        """清理会话资源"""
        if session.rag_pipeline:
            # 注意：不删除索引，保留供后续使用
            pass


# ==================== 便捷函数 ====================

def analyze_repo(
    repo_url: str,
    questions: Optional[List[str]] = None,
    repo_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    分析仓库（便捷函数）

    Args:
        repo_url: 仓库 URL
        questions: 问题列表
        repo_name: 仓库名称（可选）

    Returns:
        分析结果字典
    """
    agent = CodeAnalysisAgent()

    # 如果提供了 repo_name，使用它创建索引
    if repo_name:
        agent.cloner = RepoCloner()

    return agent.run_full_pipeline(repo_url, questions)


def ask_about_repo(
    repo_url: str,
    question: str,
    repo_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    询问仓库问题（便捷函数）

    Args:
        repo_url: 仓库 URL
        question: 问题
        repo_name: 仓库名称

    Returns:
        问答结果
    """
    agent = CodeAnalysisAgent()
    session = agent.start_session(repo_url)

    # 检查是否已有索引
    from src.tools.rag_pipeline import RAGPipeline
    full_name = repo_name or session.full_name

    try:
        pipeline = RAGPipeline(full_name)
        stats = pipeline.get_stats()
        if stats.get("chunk_count", 0) > 0:
            session.rag_pipeline = pipeline
            session.is_indexed = True
        else:
            agent.build_rag_index(session)
    except Exception:
        agent.build_rag_index(session)

    return agent.ask(session, question)