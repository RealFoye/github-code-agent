"""
CLI 界面
命令行交互式界面，用于搜索 GitHub 仓库并进行代码分析
"""

import logging
import sys
from typing import List, Optional

from src.agent.search_agent import SearchRecommendationAgent, recommend_repos
from src.agent.analysis_agent import CodeAnalysisAgent, AnalysisSession
from src.models.repo_card import RepoCard
from src.config import setup_logging, get_llm_config

logger = logging.getLogger(__name__)


class CLIInterface:
    """CLI 交互界面"""

    def __init__(self):
        """初始化 CLI"""
        self.search_agent = SearchRecommendationAgent()
        self.analysis_agent = CodeAnalysisAgent()
        self.current_session: Optional[AnalysisSession] = None

    def run(self):
        """运行 CLI 主循环"""
        print("=" * 60)
        print("GitHub 代码分析 Agent")
        print("=" * 60)
        print()

        while True:
            print("\n请选择操作：")
            print("1. 搜索 GitHub 仓库")
            print("2. 分析已有仓库")
            print("3. 退出")
            print()

            choice = input("请输入选项 (1-3): ").strip()

            if choice == "1":
                self._search_and_analyze()
            elif choice == "2":
                self._analyze_existing()
            elif choice == "3":
                print("再见!")
                break
            else:
                print("无效选项，请重新输入")

    def _search_and_analyze(self):
        """搜索并分析仓库"""
        # 获取搜索查询
        query = input("\n请输入搜索关键词: ").strip()
        if not query:
            print("搜索关键词不能为空")
            return

        # 执行搜索
        print(f"\n正在搜索: {query} ...")
        try:
            repos = recommend_repos(query, limit=5)
        except Exception as e:
            print(f"搜索失败: {e}")
            return

        if not repos:
            print("未找到相关仓库")
            return

        # 显示搜索结果
        print("\n搜索结果:")
        print("-" * 60)
        for i, repo in enumerate(repos, 1):
            print(f"{i}. {repo.full_name}")
            print(f"   Stars: {repo.stars:,} | Forks: {repo.forks:,} | Language: {repo.language or 'N/A'}")
            print(f"   {repo.description or '无描述'}")
            print()

        # 选择仓库
        choice = input("请选择要分析的仓库编号 (1-5)，或按 Enter 跳过: ").strip()
        if not choice:
            return

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(repos):
                print("无效的选择")
                return
            selected_repo = repos[idx]
        except ValueError:
            print("无效的输入")
            return

        # 分析仓库
        self._analyze_repo(selected_repo.url, selected_repo.full_name)

    def _analyze_existing(self):
        """分析已有仓库"""
        repo_url = input("\n请输入 GitHub 仓库 URL: ").strip()
        if not repo_url:
            print("仓库 URL 不能为空")
            return

        repo_name = input("请输入仓库名称 (可选，按 Enter 使用 URL 中的名称): ").strip()

        self._analyze_repo(repo_url, repo_name or None)

    def _analyze_repo(self, repo_url: str, repo_name: Optional[str] = None):
        """分析指定仓库"""
        print(f"\n正在分析仓库: {repo_url} ...")

        try:
            # 启动会话
            session = self.analysis_agent.start_session(repo_url)
            self.current_session = session
            print(f"仓库克隆完成: {session.repo_path}")

            # 分析结构
            structure = self.analysis_agent.analyze_structure(session)
            print(f"仓库结构: {structure.total_files} 文件, {structure.total_lines} 行代码")

            # 构建 RAG 索引
            print("正在构建 RAG 索引...")
            chunk_count = self.analysis_agent.build_rag_index(session)
            print(f"RAG 索引构建完成: {chunk_count} 个代码块")

            # 问答循环
            self._qa_loop(session)

        except Exception as e:
            print(f"分析失败: {e}")
            logger.error(f"分析失败: {e}", exc_info=True)

    def _qa_loop(self, session: AnalysisSession):
        """问答循环"""
        print("\n" + "=" * 60)
        print("问答模式 - 输入问题进行分析，输入 'quit' 退出")
        print("=" * 60)

        while True:
            question = input("\n问题: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("退出问答模式")
                break

            if not question:
                print("问题不能为空")
                continue

            print("正在思考...")

            try:
                result = self.analysis_agent.ask(session, question)
                print(f"\n答案:\n{result['answer']}")

                if result["sources"]:
                    print(f"\n引用来源 ({len(result['sources'])} 个):")
                    for i, source in enumerate(result["sources"], 1):
                        chunk = source["chunk"]
                        print(f"  {i}. {chunk.file_path} (行 {chunk.start_line}-{chunk.end_line})")
                        print(f"     类型: {chunk.chunk_type} | 名称: {chunk.name}")

            except Exception as e:
                print(f"回答失败: {e}")
                logger.error(f"问答失败: {e}", exc_info=True)

    def cleanup(self):
        """清理资源"""
        if self.current_session:
            self.analysis_agent.cleanup_session(self.current_session)


def run_cli():
    """运行 CLI 界面"""
    # 配置日志
    setup_logging()

    # 检查 LLM 配置
    llm_config = get_llm_config()
    if not llm_config.get("api_key"):
        print("警告: 未配置 API Key，部分功能可能无法使用")

    cli = CLIInterface()
    try:
        cli.run()
    finally:
        cli.cleanup()


if __name__ == "__main__":
    run_cli()