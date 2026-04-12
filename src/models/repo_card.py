"""
仓库推荐卡片数据模型
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class RepoCard:
    """GitHub 仓库推荐卡片"""

    # 基本信息
    repo_name: str                    # 仓库名（不含作者）
    author: str                       # 作者/组织
    full_name: str                    # 完整名称（author/repo_name）
    url: str                          # GitHub URL

    # 统计数据
    stars: int                        # Stars 数
    forks: int                        # Forks 数
    open_issues: int                  # Open Issues 数
    language: Optional[str] = None    # 主语言
    languages: List[str] = field(default_factory=list)  # 所有语言

    # 仓库状态
    description: Optional[str] = None # 仓库描述
    license: Optional[str] = None     # 许可证
    last_updated: Optional[str] = None # 最后更新时间
    topics: List[str] = field(default_factory=list)       # GitHub Topics

    # Agent 分析结果
    relevance_score: float = 0.0     # 与用户需求的相关性分数 (0-1)
    suitability_reasons: List[str] = field(default_factory=list)  # 适合的理由
    potential_concerns: List[str] = field(default_factory=list)  # 潜在问题
    tags: List[str] = field(default_factory=list)        # 标签
    evidence: Dict[str, Any] = field(default_factory=dict)  # 内部评分证据摘要

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "repo_name": self.repo_name,
            "author": self.author,
            "full_name": self.full_name,
            "url": self.url,
            "stars": self.stars,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "language": self.language,
            "languages": self.languages,
            "description": self.description,
            "license": self.license,
            "last_updated": self.last_updated,
            "topics": self.topics,
            "relevance_score": round(self.relevance_score, 2),
            "suitability_reasons": self.suitability_reasons,
            "potential_concerns": self.potential_concerns,
            "tags": self.tags,
            "evidence": self.evidence,
        }

    def format_for_display(self) -> str:
        """格式化输出"""
        lines = [
            f"## {self.full_name}",
            f"⭐ {self.stars:,} | 🍴 {self.forks:,} | 📝 {self.open_issues}",
            f"Language: {self.language or 'N/A'}",
            f"License: {self.license or 'N/A'}",
            f"Updated: {self.last_updated or 'N/A'}",
            f"",
            f"**Description:** {self.description or 'No description'}",
            f"",
            f"**Relevance Score:** {self.relevance_score:.0%}",
            f"",
        ]

        if self.suitability_reasons:
            lines.append("**Why this fits your request:**")
            for reason in self.suitability_reasons:
                lines.append(f"  - {reason}")
            lines.append("")

        if self.potential_concerns:
            lines.append("**Potential concerns:**")
            for concern in self.potential_concerns:
                lines.append(f"  - {concern}")
            lines.append("")

        if self.topics:
            lines.append(f"**Topics:** {', '.join(self.topics)}")

        return "\n".join(lines)
