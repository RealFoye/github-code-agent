"""
Streamlit Web 界面
Web 交互式界面，用于搜索 GitHub 仓库并进行代码分析
"""

import streamlit as st
import logging
from typing import Optional

from src.agent.search_agent import SearchRecommendationAgent, recommend_repos
from src.agent.analysis_agent import CodeAnalysisAgent, AnalysisSession
from src.models.repo_card import RepoCard
from src.config import setup_logging, get_llm_config

logger = logging.getLogger(__name__)


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="GitHub 代码分析 Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== 初始化 ====================

@st.cache_resource
def get_search_agent():
    """获取搜索 Agent（缓存）"""
    return SearchRecommendationAgent()


@st.cache_resource
def get_analysis_agent():
    """获取分析 Agent（缓存）"""
    return CodeAnalysisAgent()


# ==================== 辅助函数 ====================

def display_repo_card(repo: RepoCard, key: str):
    """显示仓库卡片"""
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**{repo.full_name}**")
            if repo.description:
                st.caption(repo.description[:200] + ("..." if len(repo.description) > 200 else ""))

        with col2:
            st.write(f"⭐ {repo.stars:,}")
            if repo.language:
                st.write(f"📝 {repo.language}")

        with st.expander("查看详情"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**作者**: {repo.author}")
                st.write(f"**Fork**: {repo.forks:,}")
                st.write(f"**Issues**: {repo.open_issues}")
            with col2:
                st.write(f"**许可证**: {repo.license or 'N/A'}")
                st.write(f"**更新**: {repo.last_updated or 'N/A'}")
                if repo.topics:
                    st.write(f"**Topics**: {', '.join(repo.topics[:5])}")

            if repo.suitability_reasons:
                st.write("**推荐理由**:")
                for reason in repo.suitability_reasons:
                    st.write(f"- {reason}")

            st.markdown(f"[访问仓库]({repo.url})")


def display_qa_result(result: dict):
    """显示问答结果"""
    st.markdown("### 答案")
    st.markdown(result["answer"])

    if result["sources"]:
        st.markdown("### 引用来源")
        for i, source in enumerate(result["sources"], 1):
            chunk = source["chunk"]
            with st.expander(f"来源 {i}: {chunk.file_path} (行 {chunk.start_line}-{chunk.end_line})"):
                st.write(f"**类型**: {chunk.chunk_type}")
                st.write(f"**名称**: {chunk.name}")
                st.code(chunk.content, language=chunk.language or "text")


# ==================== 主界面 ====================

def main():
    st.title("🔍 GitHub 代码分析 Agent")
    st.markdown("搜索 GitHub 仓库，获取代码分析报告")

    # 侧边栏配置
    with st.sidebar:
        st.header("配置")

        # LLM 配置信息
        llm_config = get_llm_config()
        st.write(f"**LLM Provider**: {llm_config.get('provider', 'N/A')}")
        st.write(f"**Model**: {llm_config.get('model', 'N/A')}")

        if not llm_config.get("api_key"):
            st.warning("⚠️ 未配置 API Key，请在 .env 文件中配置")

        st.divider()

        # 操作选择
        st.header("操作")
        operation = st.radio(
            "选择操作",
            ["🔍 搜索仓库", "📊 分析仓库", "💬 问答"],
            captions=[
                "搜索 GitHub 仓库并查看推荐",
                "深度分析指定仓库",
                "对已分析的仓库提问",
            ],
        )

    # ==================== 搜索仓库 ====================
    if operation == "🔍 搜索仓库":
        st.header("搜索 GitHub 仓库")

        search_query = st.text_input(
            "输入搜索关键词",
            placeholder="例如: Python web framework fastapi",
            help="输入要搜索的仓库关键词，支持自然语言",
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            limit = st.number_input("返回数量", min_value=1, max_value=20, value=5)

        if st.button("🔍 搜索", type="primary"):
            if not search_query:
                st.warning("请输入搜索关键词")
            else:
                with st.spinner("搜索中..."):
                    try:
                        repos = recommend_repos(search_query, limit=limit)

                        if not repos:
                            st.info("未找到相关仓库，请尝试其他关键词")
                        else:
                            st.success(f"找到 {len(repos)} 个相关仓库")

                            for i, repo in enumerate(repos):
                                display_repo_card(repo, f"repo_{i}")

                            # 保存到 session state
                            st.session_state["search_results"] = repos

                    except Exception as e:
                        st.error(f"搜索失败: {e}")
                        logger.error(f"搜索失败: {e}", exc_info=True)

    # ==================== 分析仓库 ====================
    elif operation == "📊 分析仓库":
        st.header("深度分析仓库")

        # 输入仓库 URL
        repo_url = st.text_input(
            "GitHub 仓库 URL",
            placeholder="例如: https://github.com/owner/repo",
            help="输入要分析的 GitHub 仓库 URL",
        )

        # 可选：预定义问题
        st.markdown("**预设问题** (可选):")
        preset_questions = st.text_area(
            "输入想要分析的问题，每行一个",
            placeholder="这个项目的架构是什么样的？\n有哪些主要的模块？\n如何运行这个项目？",
            height=100,
        )

        if st.button("📊 开始分析", type="primary"):
            if not repo_url:
                st.warning("请输入仓库 URL")
            else:
                with st.spinner("分析中，请稍候..."):
                    try:
                        # 初始化 agent
                        analysis_agent = get_analysis_agent()

                        # 启动会话
                        with st.status("正在克隆仓库...") as status:
                            session = analysis_agent.start_session(repo_url)
                            st.session_state["analysis_session"] = session
                            status.update(label=f"仓库克隆完成: {session.repo_path}")

                        # 分析结构
                        with st.status("正在分析仓库结构...") as status:
                            structure = analysis_agent.analyze_structure(session)
                            status.update(
                                label=f"结构分析完成: {structure.total_files} 文件, {structure.total_lines} 行代码"
                            )

                            # 显示结构概览
                            col1, col2, col3 = st.columns(3)
                            col1.metric("文件数", structure.total_files)
                            col2.metric("代码行数", f"{structure.total_lines:,}")
                            col3.metric("语言数", len(structure.by_language))

                            # 语言分布
                            if structure.by_language:
                                st.bar_chart(structure.by_language)

                        # 构建 RAG 索引
                        with st.status("正在构建 RAG 索引...") as status:
                            chunk_count = analysis_agent.build_rag_index(session)
                            st.session_state["is_indexed"] = True
                            status.update(label=f"RAG 索引构建完成: {chunk_count} 个代码块")

                        st.success("✅ 分析完成！现在可以在问答模式下提问")

                        # 处理预设问题
                        if preset_questions:
                            questions = [q.strip() for q in preset_questions.split("\n") if q.strip()]
                            if questions:
                                st.markdown("---")
                                st.header("预设问题回答")

                                for q in questions:
                                    with st.spinner(f"回答: {q[:50]}..."):
                                        result = analysis_agent.ask(session, q)
                                        st.markdown(f"**Q: {q}**")
                                        display_qa_result(result)
                                        st.divider()

                    except Exception as e:
                        st.error(f"分析失败: {e}")
                        logger.error(f"分析失败: {e}", exc_info=True)

    # ==================== 问答 ====================
    elif operation == "💬 问答":
        st.header("代码库问答")

        # 检查是否有已分析的仓库
        if "analysis_session" not in st.session_state or not st.session_state.get("is_indexed"):
            st.info("请先在「分析仓库」中分析一个仓库，然后再进行问答")
        else:
            session = st.session_state["analysis_session"]

            st.caption(f"当前分析仓库: **{session.full_name}**")

            # 问题输入
            question = st.text_input(
                "输入问题",
                placeholder="例如: 这个项目的主要功能是什么？",
                help="输入关于代码库的问题",
            )

            if st.button("💬 提问", type="primary"):
                if not question:
                    st.warning("请输入问题")
                else:
                    with st.spinner("思考中..."):
                        try:
                            result = session.rag_pipeline.search(question, top_k=10)
                            reranked = session.rag_pipeline.rerank(question, result, top_k=5)
                            context_chunks = [r.chunk for r in reranked]
                            context = build_context(context_chunks)
                            answer = generate_answer(question, context)

                            st.markdown("### 答案")
                            st.markdown(answer)

                            # 显示检索到的来源
                            with st.expander("📚 检索到的相关代码"):
                                for i, chunk in enumerate(context_chunks, 1):
                                    st.markdown(f"**{i}. {chunk.file_path} (行 {chunk.start_line}-{chunk.end_line})**")
                                    st.code(chunk.content[:500], language=chunk.language or "text")
                                    st.divider()

                        except Exception as e:
                            st.error(f"回答失败: {e}")
                            logger.error(f"问答失败: {e}", exc_info=True)

            # 快速问题
            st.markdown("---")
            st.markdown("**快速问题**")

            quick_questions = [
                "这个项目是做什么的？",
                "主要的模块有哪些？",
                "如何安装和运行？",
                "项目的架构是怎样的？",
            ]

            cols = st.columns(2)
            for i, q in enumerate(quick_questions):
                if cols[i % 2].button(q, key=f"quick_{i}"):
                    # 设置问题并触发回答
                    st.session_state["quick_question"] = q


def build_context(chunks) -> str:
    """构建 LLM 上下文"""
    parts = []

    for i, chunk in enumerate(chunks, 1):
        part = f"""--- 代码片段 {i} ---
文件: {chunk.file_path} (行 {chunk.start_line}-{chunk.end_line})
类型: {chunk.chunk_type}
名称: {chunk.name}
签名: {chunk.signature}

```{(chunk.language or 'text')}
{chunk.content}
```
"""
        parts.append(part)

    return "\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    """调用 LLM 生成答案"""
    from src.tools.llm_service import complete

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

    try:
        answer = complete(
            prompt=prompt,
            system_prompt="你是一个专业的代码分析师，擅长分析代码库并给出准确的回答。",
            max_tokens=2000,
        )
        return answer
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return f"生成回答失败: {e}"


if __name__ == "__main__":
    # 配置日志
    setup_logging()

    # 运行 web 界面
    main()