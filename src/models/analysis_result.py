"""
分析结果数据模型
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime


@dataclass
class ModuleInfo:
    """模块信息"""

    name: str                        # 模块名
    path: str                        # 路径
    description: Optional[str] = None # 描述
    functions: List[str] = field(default_factory=list)   # 函数列表
    classes: List[str] = field(default_factory=list)       # 类列表
    submodules: List[str] = field(default_factory=list)    # 子模块


@dataclass
class DependencyInfo:
    """依赖信息"""

    runtime_deps: List[str] = field(default_factory=list)  # 运行时依赖
    dev_deps: List[str] = field(default_factory=list)      # 开发依赖
    language: Optional[str] = None                         # 编程语言


@dataclass
class AnalysisResult:
    """代码分析结果"""

    # 基本信息
    repo_name: str
    repo_url: str
    analysis_time: str = field(default_factory=lambda: datetime.now().isoformat())

    # 结构信息
    directory_tree: Optional[Dict] = None  # 目录树
    modules: List[ModuleInfo] = field(default_factory=list)  # 模块列表
    dependencies: Optional[DependencyInfo] = None  # 依赖信息

    # 代码统计
    total_files: int = 0
    total_lines: int = 0
    language_stats: Dict[str, int] = field(default_factory=dict)  # 各语言行数

    # 代码切片（已移除，RAG 使用 chunker.CodeChunk）
    chunks: List = field(default_factory=list)  # 所有切片

    # 分析元数据
    entry_points: List[str] = field(default_factory=list)  # 入口文件
    main_modules: List[str] = field(default_factory=list)  # 主要模块

    def to_dict(self) -> dict:
        return {
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "analysis_time": self.analysis_time,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "language_stats": self.language_stats,
            "entry_points": self.entry_points,
            "main_modules": self.main_modules,
            "chunks_count": len(self.chunks),
        }
