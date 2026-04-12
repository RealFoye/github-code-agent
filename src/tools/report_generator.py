"""
报告生成器
负责生成 Markdown 格式的代码分析报告
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.tools.code_parser import CodeParser, RepoStructure
from src.tools.rag_pipeline import RetrievedChunk

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, repo_name: str):
        """
        初始化报告生成器

        Args:
            repo_name: 仓库名称
        """
        self.repo_name = repo_name

    def generate_report(
        self,
        repo_path: str,
        repo_info: Dict[str, Any],
        structure: RepoStructure,
        qa_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        生成完整分析报告

        Args:
            repo_path: 仓库路径
            repo_info: 仓库基本信息
            structure: 仓库结构分析
            qa_results: 问答结果列表

        Returns:
            Markdown 格式报告
        """
        sections = []

        # 标题
        sections.append(self._generate_header(repo_info))

        # 概览
        sections.append(self._generate_overview(repo_info, structure))

        # 仓库结构
        sections.append(self._generate_structure_section(structure))

        # 语言分布
        sections.append(self._generate_language_section(structure))

        # 关键文件
        sections.append(self._generate_key_files_section(structure))

        # 问答结果（如果有）
        if qa_results:
            sections.append(self._generate_qa_section(qa_results))

        # 页脚
        sections.append(self._generate_footer())

        return "\n\n".join(sections)

    def _generate_header(self, repo_info: Dict[str, Any]) -> str:
        """生成报告头部"""
        title = repo_info.get("full_name", self.repo_name)
        description = repo_info.get("description", "无描述")
        stars = repo_info.get("stars", 0)
        forks = repo_info.get("forks", 0)
        language = repo_info.get("language", "未知")
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""# {title}

## 基本信息

| 项目 | 值 |
|------|-----|
| **仓库名** | {title} |
| **描述** | {description} |
| **主语言** | {language} |
| **Stars** | ⭐ {stars:,} |
| **Forks** | 🍴 {forks:,} |
| **生成时间** | {generated_at} |

---"""

    def _generate_overview(self, repo_info: Dict[str, Any], structure: RepoStructure) -> str:
        """生成概览部分"""
        total_files = structure.total_files
        total_lines = structure.total_lines
        current_branch = repo_info.get("current_branch", "unknown")
        latest_commit = repo_info.get("latest_commit", "unknown")

        return f"""## 概览

### 代码规模

| 指标 | 数值 |
|------|------|
| **总文件数** | {total_files:,} |
| **总代码行数** | {total_lines:,} |
| **当前分支** | {current_branch} |
| **最新提交** | {latest_commit} |

### 仓库状态

- **分支**: {current_branch}
- **最新提交**: `{latest_commit}`
- **代码行数**: {total_lines:,} 行
"""

    def _generate_structure_section(self, structure: RepoStructure) -> str:
        """生成仓库结构部分"""
        lines = ["## 仓库结构"]

        # 文件类型分布
        if structure.by_type:
            lines.append("\n### 文件类型分布\n")
            lines.append("| 文件类型 | 数量 |")
            lines.append("|----------|------|")
            for ftype, count in sorted(structure.by_type.items(), key=lambda x: -x[1]):
                lines.append(f"| {ftype} | {count} |")

        # 目录树（限制深度）
        lines.append("\n### 目录结构\n")
        lines.append("```")
        lines.append(self._generate_dir_tree(structure.root, max_depth=3))
        lines.append("```")

        return "\n".join(lines)

    def _generate_dir_tree(self, root: str, max_depth: int = 3, prefix: str = "", depth: int = 0) -> str:
        """生成目录树"""
        if depth >= max_depth:
            return ""

        lines = []
        root_path = Path(root)

        try:
            items = sorted(root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            dirs = []
            files = []

            for item in items:
                # 忽略的目录
                if item.is_dir() and item.name in {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", "target"}:
                    continue
                if item.is_file() and item.name.startswith("."):
                    continue

                if item.is_dir():
                    dirs.append(item)
                else:
                    files.append(item)

            # 先显示文件
            for f in files[:10]:  # 每层最多 10 个文件
                connector = "└── " if depth == 0 and f == files[-1] else "│   "
                lines.append(f"{prefix}{connector}{f.name}")

            if len(files) > 10:
                lines.append(f"{prefix}... 还有 {len(files) - 10} 个文件")

            # 再显示目录
            for d in dirs:
                if depth < max_depth - 1:
                    new_prefix = prefix + "    " if prefix else ""
                    lines.append(f"{prefix}└── {d.name}/")
                    subtree = self._generate_dir_tree(d, max_depth, new_prefix, depth + 1)
                    if subtree:
                        lines.append(subtree)
                else:
                    lines.append(f"{prefix}└── {d.name}/")

        except PermissionError:
            pass

        return "\n".join(lines)

    def _generate_language_section(self, structure: RepoStructure) -> str:
        """生成语言分布部分"""
        lines = ["## 语言分布"]

        if structure.by_language:
            total = sum(structure.by_language.values())
            lines.append(f"\n共检测到 **{len(structure.by_language)}** 种编程语言，总计 {total:,} 行代码：\n")

            lines.append("| 语言 | 行数 | 占比 |")
            lines.append("|------|------|------|")

            for lang, count in sorted(structure.by_language.items(), key=lambda x: -x[1]):
                percentage = count / total * 100 if total > 0 else 0
                bar = "█" * int(percentage / 5)
                lines.append(f"| {lang} | {count:,} | {percentage:.1f}% {bar} |")
        else:
            lines.append("\n未检测到代码文件。")

        return "\n".join(lines)

    def _generate_key_files_section(self, structure: RepoStructure) -> str:
        """生成关键文件部分"""
        lines = ["## 关键文件"]

        # 找出最大的源代码文件
        source_files = [
            f for f in structure.files
            if f.file_type.value == "source_code" and f.line_count > 0
        ]
        source_files.sort(key=lambda x: x.line_count, reverse=True)

        if source_files:
            lines.append("\n### 代码行数最多的文件\n")
            lines.append("| 文件 | 行数 | 语言 |")
            lines.append("|------|------|------|")

            for f in source_files[:15]:
                lines.append(f"| `{f.path}` | {f.line_count:,} | {f.language or '?'} |")

        # 配置文件列表
        config_files = [f for f in structure.files if f.file_type.value == "config"]
        if config_files:
            lines.append("\n### 配置文件\n")
            for f in config_files[:20]:
                lines.append(f"- `{f.path}`")

        return "\n".join(lines)

    def _generate_qa_section(self, qa_results: List[Dict[str, Any]]) -> str:
        """生成问答部分"""
        lines = ["## 问答分析"]

        for i, qa in enumerate(qa_results, 1):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            sources = qa.get("sources", [])

            lines.append(f"\n### Q{i}: {question}\n")
            lines.append(f"**Answer:**\n\n{answer}")

            if sources:
                lines.append("\n**参考代码片段:**\n")
                for src in sources[:5]:
                    chunk = src.get("chunk")
                    if chunk:
                        lines.append(f"\n**文件:** `{chunk.file_path}` (行 {chunk.start_line}-{chunk.end_line})\n")
                        # 显示代码片段（限制行数）
                        code_lines = chunk.content.split("\n")[:10]
                        code_preview = "\n".join(code_lines)
                        if len(chunk.content.split("\n")) > 10:
                            code_preview += "\n..."
                        lines.append(f"```\n{code_preview}\n```")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """生成页脚"""
        return f"""---

*此报告由 GitHub Code Analysis Agent 自动生成*
*生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*"""


def generate_report(
    repo_path: str,
    repo_info: Dict[str, Any],
    structure: RepoStructure,
    qa_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    生成报告（便捷函数）

    Args:
        repo_path: 仓库路径
        repo_info: 仓库信息
        structure: 仓库结构
        qa_results: 问答结果

    Returns:
        Markdown 报告
    """
    repo_name = repo_info.get("full_name", "unknown")
    generator = ReportGenerator(repo_name)
    return generator.generate_report(repo_path, repo_info, structure, qa_results)


def save_report(markdown_text: str, output_path: str) -> bool:
    """
    保存报告到文件

    Args:
        markdown_text: Markdown 文本
        output_path: 输出文件路径

    Returns:
        是否成功
    """
    try:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        logger.info(f"报告已保存: {output_path}")
        return True

    except Exception as e:
        logger.error(f"保存报告失败: {e}")
        return False