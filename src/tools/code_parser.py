"""
代码解析器
负责：目录扫描、文件分类、AST 解析
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class FileType(Enum):
    """文件类型枚举"""
    SOURCE_CODE = "source_code"
    TEST = "test"
    CONFIG = "config"
    DOCUMENT = "document"
    BUILD = "build"
    DATA = "data"
    NOTEBOOK = "notebook"       # Jupyter Notebook (.ipynb)
    MARKDOWN = "markdown"       # Markdown 文档 (.md)
    OTHER = "other"


@dataclass
class ParsedFile:
    """解析后的文件"""
    path: str                    # 文件路径（相对仓库根目录）
    absolute_path: str           # 绝对路径
    file_type: FileType          # 文件类型
    language: Optional[str]      # 编程语言
    size: int                    # 文件大小（字节）
    line_count: int              # 行数
    functions: List[Dict] = None  # 解析出的函数列表
    classes: List[Dict] = None    # 解析出的类列表
    imports: List[str] = None     # 导入语句
    exports: List[str] = None      # 导出语句
    ast: Any = None               # AST 树（原始）


@dataclass
class RepoStructure:
    """仓库结构"""
    root: str
    total_files: int
    total_lines: int
    files: List[ParsedFile]
    by_language: Dict[str, int]
    by_type: Dict[str, int]


# ==================== 文件分类 ====================

class FileClassifier:
    """文件分类器"""

    # 配置文件模式
    CONFIG_PATTERNS = {
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".env", ".gitignore", ".dockerignore", ".editorconfig",
        "Makefile", "CMakeLists.txt", "setup.py", "setup.cfg",
        "pyproject.toml", "poetry.lock", "package.json", "webpack.config",
        ".eslintrc", ".prettierrc", ".babelrc",
    }

    # 测试文件模式
    TEST_PATTERNS = {
        "test_", "_test.py", "tests/", "spec/", "__tests__/",
        ".test.", ".spec.", "Test.", "Tests.",
    }

    # 构建文件模式
    BUILD_PATTERNS = {
        "Dockerfile", "docker-compose", ".dockerfile",
        "Makefile", "CMakeLists", "build.gradle", "pom.xml",
    }

    # 文档文件模式
    DOC_PATTERNS = {
        ".md", ".rst", ".txt", ".adoc", ".tex",
        "README", "CHANGELOG", "LICENSE", "CONTRIBUTING",
    }

    # 数据文件模式
    # 注意：.json/.yaml/.yml 已在上方 CONFIG_PATTERNS 中处理，不会到达此处
    DATA_PATTERNS = {
        ".csv", ".tsv", ".xml",
        ".sql", ".db", ".sqlite", ".parquet",
    }

    # 语言映射
    EXT_TO_LANGUAGE = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".java": "java",
        ".rs": "rust",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".ps1": "powershell",
        ".r": "r",
        ".lua": "lua",
    }

    def classify(self, file_path: str) -> Tuple[FileType, Optional[str]]:
        """
        分类文件

        Args:
            file_path: 文件路径

        Returns:
            (文件类型, 语言) 元组
        """
        name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        # 检查是否配置
        if name in self.CONFIG_PATTERNS or ext in self.CONFIG_PATTERNS:
            return FileType.CONFIG, None

        # 检查是否构建
        if name in self.BUILD_PATTERNS or "Dockerfile" in name:
            return FileType.BUILD, None

        # 检查是否测试
        for pattern in self.TEST_PATTERNS:
            if pattern in name:
                return FileType.TEST, None

        # 检查是否 Notebook
        if ext == ".ipynb":
            return FileType.NOTEBOOK, "python"  # Notebook 通常用 Python

        # 检查是否 Markdown
        if ext == ".md":
            return FileType.MARKDOWN, None

        # 检查是否文档（非 Markdown）
        if ext in self.DOC_PATTERNS or name in self.DOC_PATTERNS:
            return FileType.DOCUMENT, None

        # 检查是否数据
        if ext in self.DATA_PATTERNS:
            return FileType.DATA, None

        # 检查是否源代码
        language = self.EXT_TO_LANGUAGE.get(ext)
        if language:
            return FileType.SOURCE_CODE, language

        return FileType.OTHER, None


# ==================== AST 解析 ====================

class ASTParser:
    """AST 解析器（使用 tree-sitter）"""

    def __init__(self):
        """初始化 AST 解析器"""
        self.parsers = {}  # 缓存各语言的解析器
        self._init_parsers()

    def _init_parsers(self):
        """初始化各语言的解析器"""
        try:
            from tree_sitter_languages import get_parser
            for lang in SUPPORTED_LANGUAGES:
                try:
                    self.parsers[lang] = get_parser(lang)
                    logger.debug(f"初始化 {lang} 解析器成功")
                except Exception as e:
                    logger.warning(f"初始化 {lang} 解析器失败: {e}")
        except ImportError as e:
            logger.warning(f"tree-sitter-languages 未安装: {e}")

    def parse(self, file_path: str, content: str, language: str) -> Optional[Any]:
        """
        解析文件生成 AST

        Args:
            file_path: 文件路径
            content: 文件内容
            language: 编程语言

        Returns:
            AST 树或 None
        """
        if language not in self.parsers:
            return None

        try:
            parser = self.parsers[language]
            return parser.parse(bytes(content, "utf8"))
        except Exception as e:
            logger.debug(f"AST 解析失败 {file_path}: {e}")
            return None

    def extract_functions(self, ast: Any, language: str) -> List[Dict[str, Any]]:
        """从 AST 提取函数信息"""
        if language == "python":
            return self._extract_python_functions(ast)
        elif language == "javascript" or language == "typescript":
            return self._extract_js_functions(ast)
        elif language == "go":
            return self._extract_go_functions(ast)
        return []

    def _extract_python_functions(self, ast: Any) -> List[Dict[str, Any]]:
        """提取 Python 函数"""
        functions = []
        if ast is None:
            return functions

        # 使用 tree-sitter 查询
        try:
            from tree_sitter import Language, Node
            from tree_sitter_languages import get_language

            language = get_language("python")

            # 查找函数定义
            query = language.query("""
                (function_definition
                    name: (identifier) @name
                    parameters: (parameters) @params
                    body: (block) @body) @func
            """)

            captures = query.captures(ast.root_node)
            for node, capture_type in captures:
                if capture_type == "name":
                    func_node = node.parent
                    if func_node and func_node.type == "function_definition":
                        start = func_node.start_point
                        end = func_node.end_point
                        functions.append({
                            "name": node.text.decode("utf8"),
                            "start_line": start[0] + 1,
                            "end_line": end[0] + 1,
                            "type": "function",
                        })
        except Exception as e:
            logger.debug(f"提取 Python 函数失败: {e}")

        return functions

    def _extract_js_functions(self, ast: Any) -> List[Dict[str, Any]]:
        """提取 JavaScript/TypeScript 函数"""
        functions = []
        if ast is None:
            return functions

        try:
            from tree_sitter_languages import get_language

            language = get_language("javascript")

            # 查找函数定义
            query = language.query("""
                (function_declaration
                    name: (identifier) @name
                    parameters: (formal_parameters) @params) @func
                [(method_definition
                    name: (property_identifier) @name
                    parameters: (formal_parameters) @params) @func]
            """)

            captures = query.captures(ast.root_node)
            for node, capture_type in captures:
                if capture_type == "name":
                    func_node = node.parent
                    if func_node:
                        start = func_node.start_point
                        end = func_node.end_point
                        func_type = "method" if func_node.type == "method_definition" else "function"
                        functions.append({
                            "name": node.text.decode("utf8"),
                            "start_line": start[0] + 1,
                            "end_line": end[0] + 1,
                            "type": func_type,
                        })
        except Exception as e:
            logger.debug(f"提取 JS 函数失败: {e}")

        return functions

    def _extract_go_functions(self, ast: Any) -> List[Dict[str, Any]]:
        """提取 Go 函数"""
        functions = []
        if ast is None:
            return functions

        try:
            from tree_sitter_languages import get_language

            language = get_language("go")

            query = language.query("""
                (function_declaration
                    name: (identifier) @name
                    parameters: (parameter_list) @params
                    body: (block) @body) @func
            """)

            captures = query.captures(ast.root_node)
            for node, capture_type in captures:
                if capture_type == "name":
                    func_node = node.parent
                    if func_node and func_node.type == "function_declaration":
                        start = func_node.start_point
                        end = func_node.end_point
                        functions.append({
                            "name": node.text.decode("utf8"),
                            "start_line": start[0] + 1,
                            "end_line": end[0] + 1,
                            "type": "function",
                        })
        except Exception as e:
            logger.debug(f"提取 Go 函数失败: {e}")

        return functions


# ==================== 代码解析器主类 ====================

class CodeParser:
    """代码解析器主类"""

    def __init__(self):
        """初始化解析器"""
        self.classifier = FileClassifier()
        self.ast_parser = ASTParser()

        # 忽略的目录
        self.ignore_dirs = {
            ".git", ".svn", ".hg",
            "node_modules", "bower_components",
            "__pycache__", ".pytest_cache", ".tox",
            "venv", ".venv", "env", ".env",
            "dist", "build", "out", "target",
            ".idea", ".vscode", ".vs",
            "coverage", ".coverage",
            "site-packages", "lib", "include",
        }

        # 忽略的文件
        self.ignore_files = {
            ".DS_Store", "Thumbs.db", "desktop.ini",
            "package-lock.json", "yarn.lock", "poetry.lock",
            ".gitignore", ".gitattributes",
        }

    def scan_structure(self, repo_path: str) -> RepoStructure:
        """
        扫描仓库目录结构

        Args:
            repo_path: 仓库根目录路径

        Returns:
            RepoStructure 对象
        """
        repo_path = Path(repo_path)
        files = []
        by_language: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        total_lines = 0

        for root, dirs, filenames in os.walk(repo_path):
            # 过滤忽略的目录
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for filename in filenames:
                if filename in self.ignore_files:
                    continue

                file_path = Path(root) / filename
                rel_path = str(file_path.relative_to(repo_path))

                try:
                    # 获取文件信息
                    stat = file_path.stat()
                    size = stat.st_size

                    # 分类文件
                    file_type, language = self.classifier.classify(rel_path)

                    # 统计
                    by_type[file_type.value] = by_type.get(file_type.value, 0) + 1
                    if language:
                        by_language[language] = by_language.get(language, 0) + 1

                    # 读取文件内容（源代码、Notebook、Markdown 都统计行数）
                    line_count = 0
                    if file_type in (FileType.SOURCE_CODE, FileType.NOTEBOOK, FileType.MARKDOWN):
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                line_count = len(content.splitlines())
                                total_lines += line_count
                        except Exception:
                            pass

                    files.append(ParsedFile(
                        path=rel_path,
                        absolute_path=str(file_path),
                        file_type=file_type,
                        language=language,
                        size=size,
                        line_count=line_count,
                    ))

                except Exception as e:
                    logger.debug(f"处理文件失败 {file_path}: {e}")

        return RepoStructure(
            root=str(repo_path),
            total_files=len(files),
            total_lines=total_lines,
            files=files,
            by_language=by_language,
            by_type=by_type,
        )

    def parse_file(self, file_path: str) -> ParsedFile:
        """
        解析单个文件

        Args:
            file_path: 文件路径

        Returns:
            ParsedFile 对象
        """
        file_path = Path(file_path)
        rel_path = str(file_path.relative_to(file_path.parent.parent))
        stat = file_path.stat()
        file_type, language = self.classifier.classify(str(file_path))

        result = ParsedFile(
            path=rel_path,
            absolute_path=str(file_path),
            file_type=file_type,
            language=language,
            size=stat.st_size,
            line_count=0,
        )

        # 只解析源代码文件
        if file_type != FileType.SOURCE_CODE or not language:
            return result

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                result.line_count = len(content.splitlines())

                # 解析 AST
                ast = self.ast_parser.parse(str(file_path), content, language)
                result.ast = ast

                # 提取函数和类
                if ast:
                    result.functions = self.ast_parser.extract_functions(ast, language)

        except Exception as e:
            logger.debug(f"解析文件失败 {file_path}: {e}")

        return result

    def get_language_stats(self, repo_path: str) -> Dict[str, int]:
        """
        获取仓库语言统计

        Args:
            repo_path: 仓库路径

        Returns:
            {语言: 行数} 字典
        """
        stats: Dict[str, int] = {}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for filename in files:
                file_path = Path(root) / filename
                file_type, language = self.classifier.classify(str(file_path))

                if language:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            lines = len(f.read().splitlines())
                            stats[language] = stats.get(language, 0) + lines
                    except Exception:
                        pass

        return stats


# 便捷函数
def scan_structure(repo_path: str) -> RepoStructure:
    """扫描仓库结构（便捷函数）"""
    parser = CodeParser()
    return parser.scan_structure(repo_path)


def classify_file(file_path: str) -> Tuple[FileType, Optional[str]]:
    """分类文件（便捷函数）"""
    classifier = FileClassifier()
    return classifier.classify(file_path)


def parse_file(file_path: str, language: Optional[str] = None) -> ParsedFile:
    """解析文件（便捷函数）"""
    parser = CodeParser()

    # 自动检测语言
    if language is None:
        _, language = classify_file(file_path)

    return parser.parse_file(file_path)