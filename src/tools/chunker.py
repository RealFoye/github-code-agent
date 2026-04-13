"""
代码切片器
负责将代码文件按语义单元切分为独立的 Chunk
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.tools.code_parser import CodeParser, FileType, ParsedFile
from src.config import MAX_CHUNK_LINES

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """代码切片"""
    chunk_id: str                 # 唯一标识
    content: str                  # 代码内容
    file_path: str                # 所属文件
    start_line: int               # 起始行号
    end_line: int                 # 结束行号

    # 元数据
    chunk_type: str = "unknown"   # 类型：function, class, module, config, test, readme, tutorial_markdown, notebook_markdown, notebook_code, document
    name: str = ""                # 名称（如函数名、类名）
    language: Optional[str] = None

    # 语义路径（用于教程仓库的标题层级，如 "Tutorial 8 > Complex Workflow"）
    heading_path: str = ""

    # Notebook 专用
    notebook_cell_index: Optional[int] = None  # cell 序号

    # 代码切片专用
    imports: List[str] = field(default_factory=list)  # 导入语句（轻量提取）

    # 上下文（用于召回后验证）
    context_before: str = ""       # 前 n 行上下文
    context_after: str = ""        # 后 n 行上下文

    # 签名信息
    signature: str = ""            # 函数签名
    docstring: str = ""            # 文档字符串

    # 调用关系
    calls: List[str] = field(default_factory=list)       # 调用了哪些函数
    called_by: List[str] = field(default_factory=list)  # 被哪些函数调用

    # embedding 用
    embedding: List[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "language": self.language,
            "heading_path": self.heading_path,
            "notebook_cell_index": self.notebook_cell_index,
            "imports": self.imports,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "signature": self.signature,
            "docstring": self.docstring,
            "calls": self.calls,
            "called_by": self.called_by,
        }


class Chunker:
    """代码切片器"""

    # 上下文行数
    CONTEXT_LINES = 3

    # 忽略的文件名模式（系统派生产物）
    DERIVED_ARTIFACT_NAMES = {
        "analysis_report.md",
        "analysis_report.json",
        "search_results.json",
        "repo_cards.json",
    }

    # 忽略的目录模式（系统输出目录）
    DERIVED_ARTIFACT_DIRS = {
        ".claude-plugin",
        ".codex-plugin",
        "mempalace",
    }

    def __init__(self):
        """初始化切片器"""
        self.parser = CodeParser()

    def is_derived_artifact(self, file_path: Path) -> bool:
        """
        判断文件是否为系统派生产物

        Args:
            file_path: 文件路径

        Returns:
            True 如果是派生产物，应被跳过
        """
        name = file_path.name.lower()

        # 文件名黑名单
        if name in self.DERIVED_ARTIFACT_NAMES:
            return True

        # analysis_report_*.md 模式
        if name.startswith("analysis_report") and name.endswith(".md"):
            return True

        # 目录黑名单
        for part in file_path.parts:
            if part in self.DERIVED_ARTIFACT_DIRS:
                return True

        return False

    def chunk_file(self, file_path: str, parsed_file: Optional[ParsedFile] = None) -> List[CodeChunk]:
        """
        将代码文件切分为语义块

        Args:
            file_path: 文件路径
            parsed_file: 已解析的文件信息（可选）

        Returns:
            CodeChunk 列表
        """
        file_path = Path(file_path)
        filename = file_path.name

        # 分类文件
        file_type, language = self.parser.classifier.classify(str(file_path))

        # 配置文件特殊处理
        if file_type == FileType.CONFIG:
            return self._chunk_config(file_path, language)

        # Notebook 处理
        if file_type == FileType.NOTEBOOK:
            return self._chunk_notebook(file_path)

        # Markdown 处理
        if file_type == FileType.MARKDOWN:
            # 区分 README 和教程文档
            if filename.lower().startswith("readme"):
                return self._chunk_markdown(file_path, chunk_type="readme")
            else:
                return self._chunk_markdown(file_path, chunk_type="tutorial_markdown")

        # 文档文件处理（.txt, .rst 等非 Markdown 文档）
        if file_type == FileType.DOCUMENT:
            return self._chunk_document(file_path)

        # 源代码处理
        if file_type == FileType.SOURCE_CODE and language:
            return self._chunk_source_code(file_path, language, parsed_file)

        return []

    def _chunk_config(self, file_path: Path, language: Optional[str]) -> List[CodeChunk]:
        """配置文件切片"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return [CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, "config", 1),
                content=content,
                file_path=str(file_path),
                start_line=1,
                end_line=len(content.splitlines()),
                chunk_type="config",
                name=file_path.name,
                language=language,
            )]
        except Exception as e:
            logger.debug(f"配置文件切片失败 {file_path}: {e}")
            return []

    def _chunk_document(self, file_path: Path) -> List[CodeChunk]:
        """文档文件切片"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return [CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, "doc", 1),
                content=content,
                file_path=str(file_path),
                start_line=1,
                end_line=len(content.splitlines()),
                chunk_type="document",
                name=file_path.name,
            )]
        except Exception as e:
            logger.debug(f"文档切片失败 {file_path}: {e}")
            return []

    def _chunk_markdown(self, file_path: Path, chunk_type: str = "tutorial_markdown") -> List[CodeChunk]:
        """
        Markdown 文件切片（按标题层级切分）

        Args:
            file_path: 文件路径
            chunk_type: chunk 类型（readme, tutorial_markdown）

        Returns:
            CodeChunk 列表
        """
        import re

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.splitlines()
            chunks = []

            # 标题行模式：# ## ###
            heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

            # 找到所有标题的位置和层级
            headings = []
            for i, line in enumerate(lines):
                match = heading_pattern.match(line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    headings.append({
                        "line": i + 1,
                        "level": level,
                        "title": title,
                    })

            # 如果没有标题，整个文件作为一个 chunk
            if not headings:
                return [CodeChunk(
                    chunk_id=self._generate_chunk_id(file_path, chunk_type, 1),
                    content=content,
                    file_path=str(file_path),
                    start_line=1,
                    end_line=len(lines),
                    chunk_type=chunk_type,
                    name=file_path.stem,
                    heading_path="",
                )]

            # 按标题切分
            for i, heading in enumerate(headings):
                start_line = heading["line"]
                # 计算 heading_path（从最高级标题到当前标题的路径）
                heading_path_parts = []
                for h in headings[:i + 1]:
                    heading_path_parts.append(h["title"])
                heading_path = " > ".join(heading_path_parts)

                # 确定结束行（下一个标题前一行，或文件末尾）
                if i + 1 < len(headings):
                    end_line = headings[i + 1]["line"] - 1
                else:
                    end_line = len(lines)

                # 提取内容
                section_lines = lines[start_line - 1:end_line]
                section_content = "\n".join(section_lines)

                # 跳过太短的章节
                if len(section_lines) < 2:
                    continue

                chunks.append(CodeChunk(
                    chunk_id=self._generate_chunk_id(file_path, chunk_type, start_line),
                    content=section_content,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    name=heading["title"],
                    heading_path=heading_path,
                ))

            return chunks

        except Exception as e:
            logger.debug(f"Markdown 切片失败 {file_path}: {e}")
            return []

    def _chunk_notebook(self, file_path: Path) -> List[CodeChunk]:
        """
        Notebook 文件切片（按 cell 切分）

        Args:
            file_path: 文件路径

        Returns:
            CodeChunk 列表
        """
        import re
        import json

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                nb = json.load(f)

            chunks = []
            cells = nb.get("cells", [])

            # 跟踪当前标题路径（用于 markdown cell 的 heading_path）
            current_heading_path = file_path.stem  # 默认用文件名

            # 标题模式：# ## ###（用于跟踪当前标题）
            heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

            for cell_idx, cell in enumerate(cells):
                cell_type = cell.get("cell_type", "")
                source = cell.get("source", [])
                if isinstance(source, list):
                    source = "".join(source)
                elif not isinstance(source, str):
                    source = str(source)

                if not source.strip():
                    continue

                lines = source.splitlines()
                start_line = cell.get("metadata", {}).get("start_line", 1)
                end_line = start_line + len(lines) - 1

                if cell_type == "markdown":
                    # 检查是否有标题
                    for line in lines:
                        match = heading_pattern.match(line)
                        if match:
                            level = len(match.group(1))
                            title = match.group(2).strip()
                            # 构建 heading_path（只保留同级及更高级标题）
                            if level == 1:
                                current_heading_path = title
                            else:
                                current_heading_path = f"{current_heading_path} > {title}"
                            break

                    chunks.append(CodeChunk(
                        chunk_id=self._generate_chunk_id(file_path, "notebook_markdown", start_line),
                        content=source,
                        file_path=str(file_path),
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="notebook_markdown",
                        name=self._extract_title_from_markdown(source) or f"Cell {cell_idx}",
                        heading_path=current_heading_path,
                        notebook_cell_index=cell_idx,
                    ))

                elif cell_type == "code":
                    # 提取 imports（轻量正则）
                    imports = self._extract_imports_from_code(source)

                    # 确定代码语言（通常从 notebook metadata 获取）
                    language = "python"  # Notebook 默认 Python

                    chunks.append(CodeChunk(
                        chunk_id=self._generate_chunk_id(file_path, "notebook_code", start_line),
                        content=source,
                        file_path=str(file_path),
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="notebook_code",
                        name=f"Code Cell {cell_idx}",
                        heading_path=current_heading_path,
                        notebook_cell_index=cell_idx,
                        imports=imports,
                        language=language,
                    ))

            return chunks

        except Exception as e:
            logger.debug(f"Notebook 切片失败 {file_path}: {e}")
            return []

    def _extract_title_from_markdown(self, content: str) -> str:
        """从 markdown 内容中提取第一个标题"""
        import re
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_imports_from_code(self, code: str) -> List[str]:
        """
        从代码中提取 import 语句（轻量正则，不上 tree-sitter）

        支持：
        - import xxx
        - from xxx import yyy
        - require(xxx)  # Node.js
        """
        import re

        imports = []

        # import xxx
        import_pattern = re.compile(r'^import\s+([^\s;#]+)', re.MULTILINE)
        for match in import_pattern.finditer(code):
            imports.append(match.group(0).strip())

        # from xxx import yyy
        from_import_pattern = re.compile(r'^from\s+([^\s]+)\s+import', re.MULTILINE)
        for match in from_import_pattern.finditer(code):
            imports.append(match.group(0).strip())

        # require(xxx)
        require_pattern = re.compile(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)')
        for match in require_pattern.finditer(code):
            imports.append(match.group(0).strip())

        return imports

    def _chunk_source_code(
        self,
        file_path: Path,
        language: str,
        parsed_file: Optional[ParsedFile] = None
    ) -> List[CodeChunk]:
        """
        源代码切片

        按语义单元切分：
        - 函数/方法 → 独立 Chunk
        - 类定义 → 独立 Chunk
        - 模块级（imports + docstring）→ 独立 Chunk
        """
        chunks = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                content = "".join(lines)

            # 解析 AST 获取函数和类
            if parsed_file is None:
                parsed_file = self.parser.parse_file(str(file_path))

            functions = parsed_file.functions or []
            classes = parsed_file.classes or []

            # 如果没有 AST 信息，使用简单正则切片
            if not functions and not classes:
                return self._chunk_by_regex(file_path, language, lines)

            # 按行号排序所有定义
            definitions = []

            # 添加函数（包含签名和 docstring）
            for func in functions:
                definitions.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "start": func.get("start_line", 1) - 1,
                    "end": func.get("end_line", 1),
                    "signature": func.get("signature", ""),
                    "docstring": func.get("docstring", ""),
                })

            # 添加类（包含 docstring）
            for cls in classes:
                bases = cls.get("bases", [])
                sig = cls.get("name", "") + "(" + ", ".join(bases) + ")" if bases else cls.get("name", "")
                definitions.append({
                    "type": "class",
                    "name": cls.get("name", ""),
                    "start": cls.get("start_line", 1) - 1,
                    "end": cls.get("end_line", 1),
                    "signature": sig,
                    "docstring": cls.get("docstring", ""),
                })

            # 按起始行排序
            definitions.sort(key=lambda x: x["start"])

            # 生成切片
            prev_end = 0
            for i, defn in enumerate(definitions):
                start = defn["start"]
                end = defn["end"]
                chunk_type = defn["type"]
                name = defn["name"]

                # 模块级代码（imports + docstring + 空白）作为开场
                if i == 0 and start > 0:
                    module_content = "".join(lines[:start])
                    if module_content.strip():
                        chunks.append(CodeChunk(
                            chunk_id=self._generate_chunk_id(file_path, "module", 1),
                            content=module_content,
                            file_path=str(file_path),
                            start_line=1,
                            end_line=start,
                            chunk_type="module",
                            name=file_path.stem,
                            language=language,
                            context_after="".join(lines[start:min(start + self.CONTEXT_LINES, len(lines))]),
                        ))

                # 提取定义代码块
                def_content = "".join(lines[start:end])
                context_before = "".join(lines[max(0, start - self.CONTEXT_LINES):start])
                context_after = "".join(lines[end:min(end + self.CONTEXT_LINES, len(lines))])

                chunks.append(CodeChunk(
                    chunk_id=self._generate_chunk_id(file_path, chunk_type, start + 1),
                    content=def_content,
                    file_path=str(file_path),
                    start_line=start + 1,
                    end_line=end,
                    chunk_type=chunk_type,
                    name=name,
                    language=language,
                    signature=defn.get("signature", ""),
                    docstring=defn.get("docstring", ""),
                    context_before=context_before,
                    context_after=context_after,
                ))

                prev_end = end

            # 处理文件末尾（如果有大块未归类的代码）
            if prev_end < len(lines):
                tail_content = "".join(lines[prev_end:])
                if tail_content.strip() and len(tail_content.splitlines()) > 5:
                    chunks.append(CodeChunk(
                        chunk_id=self._generate_chunk_id(file_path, "module", prev_end + 1),
                        content=tail_content,
                        file_path=str(file_path),
                        start_line=prev_end + 1,
                        end_line=len(lines),
                        chunk_type="module",
                        name=file_path.stem,
                        language=language,
                        context_before="".join(lines[max(0, prev_end - self.CONTEXT_LINES):prev_end]),
                    ))

        except Exception as e:
            logger.debug(f"源代码切片失败 {file_path}: {e}")
            # 降级处理：整个文件作为一个 chunk
            return self._chunk_whole_file(file_path, language)

        # 填充函数调用关系（calls / called_by）
        if chunks and language in ("python", "javascript", "typescript", "go", "java", "rust"):
            self._populate_call_relationships(chunks, content, language)

        return chunks

    def _populate_call_relationships(
        self,
        chunks: List[CodeChunk],
        content: str,
        language: str,
    ) -> None:
        """
        填充 chunks 的 calls 和 called_by 字段
        通过 tree-sitter 提取函数调用关系
        """
        try:
            from tree_sitter_languages import get_parser

            parser = get_parser(language)
            tree = parser.parse(bytes(content, "utf8"))

            # 构建 "函数名 → chunk" 的映射（只处理 function 和 class 类型的 chunk）
            name_to_chunk: Dict[str, CodeChunk] = {}
            for chunk in chunks:
                if chunk.name and chunk.chunk_type in ("function", "class"):
                    name_to_chunk[chunk.name] = chunk

            if not name_to_chunk:
                return

            # 提取所有函数调用的 tree-sitter query（支持多语言）
            call_query_str = self._get_call_query(language)
            if not call_query_str:
                return

            from tree_sitter_languages import get_language
            lang = get_language(language)
            query = lang.query(call_query_str)
            captures = query.captures(tree.root_node)

            # 收集每个 chunk 内的调用
            # captures: (node, capture_type) 列表
            # capture_type 可以是 "call", "attr_call", "nested" 等
            for node, capture_type in captures:
                if "call" in capture_type:  # 匹配 @call, @attr_call, @nested 等
                    # 获取调用的函数名
                    func_name = self._extract_call_target_name(node, language)
                    if func_name and func_name in name_to_chunk:
                        # 找到这个调用发生在哪个 chunk 里
                        call_line = node.start_point[0] + 1
                        caller_chunk = self._find_chunk_containing(chunks, call_line)
                        if caller_chunk and caller_chunk != name_to_chunk[func_name]:
                            # 设置 calls
                            if func_name not in caller_chunk.calls:
                                caller_chunk.calls.append(func_name)
                            # 设置 called_by
                            called_chunk = name_to_chunk[func_name]
                            if caller_chunk.name not in called_chunk.called_by:
                                called_chunk.called_by.append(caller_chunk.name)

        except Exception as e:
            logger.debug(f"提取调用关系失败: {e}")

    def _get_call_query(self, language: str) -> str:
        """
        获取指定语言的函数调用 tree-sitter query

        注意: @capture 必须放在匹配节点的 CLOSING ) 之后，以捕获该节点本身
        例如: (call function: (identifier)) @call 捕获 call 节点
        而不是: (call function: (identifier) @call) 捕获 identifier 节点
        """
        queries = {
            "python": """[(call
    function: (identifier) @name) @call
(call
    function: (attribute
        object: (identifier) @obj
        attribute: (identifier) @method)) @attr_call
(call
    function: (attribute
        object: (call
            function: (identifier) @fn)
        attribute: (identifier) @method)) @nested]""",
            # JavaScript/TypeScript 使用 call_expression，不是 call
            # 且 function 是直接子节点，不是字段
            "javascript": """[(call_expression (identifier) @name) @simple_call
(call_expression (member_expression
    object: (identifier) @obj
    property: (property_identifier) @prop)) @member_call]""",
            "typescript": """[(call_expression (identifier) @name) @simple_call
(call_expression (member_expression
    object: (identifier) @obj
    property: (property_identifier) @prop)) @member_call]""",
            "go": """[(call
    function: (identifier) @name) @call
(call
    function: (selector_expression
        object: (identifier) @obj
        field: (field_identifier) @field)) @call]""",
            "java": """[(call
    function: (identifier) @name) @call
(call
    function: (method_invocation
        name: (identifier) @name)) @call]""",
            "rust": """[(call
    function: (identifier) @name) @call
(call
    function: (field_expression
        field: (field_identifier) @field)) @call]""",
        }
        return queries.get(language, "")

    def _extract_call_target_name(self, node, language: str) -> Optional[str]:
        """从 call 节点提取被调用的函数名"""
        try:
            if language == "python":
                # 支持 identifier 调用和 attribute 调用
                func = node.child_by_field_name("function")
                if func is None:
                    return None
                if func.type == "identifier":
                    return func.text.decode("utf8")
                elif func.type == "attribute":
                    obj = func.child_by_field_name("object")
                    method = func.child_by_field_name("attribute")
                    if obj and method:
                        obj_name = obj.text.decode("utf8")
                        method_name = method.text.decode("utf8")
                        # 过滤掉 self/cls/this 等关键字
                        if obj_name not in ("self", "cls", "this"):
                            return f"{obj_name}.{method_name}"
                        return method_name
            elif language in ("javascript", "typescript"):
                # JavaScript/TypeScript 使用 call_expression，function 是直接子节点，不是字段
                # 遍历子节点找 identifier 或 member_expression
                func_node = None
                for child in node.children:
                    if child.type in ("identifier", "member_expression"):
                        func_node = child
                        break
                if func_node is None:
                    return None
                if func_node.type == "identifier":
                    return func_node.text.decode("utf8")
                elif func_node.type == "member_expression":
                    obj = func_node.child_by_field_name("object")
                    prop = func_node.child_by_field_name("property")
                    if obj and prop:
                        return f"{obj.text.decode('utf8')}.{prop.text.decode('utf8')}"
            elif language == "go":
                func = node.child_by_field_name("function")
                if func is None:
                    return None
                if func.type == "identifier":
                    return func.text.decode("utf8")
                elif func.type == "selector_expression":
                    obj = func.child_by_field_name("object")
                    field = func.child_by_field_name("field")
                    if obj and field:
                        return f"{obj.text.decode('utf8')}.{field.text.decode('utf8')}"
            elif language == "java":
                if node.type == "method_invocation":
                    name = node.child_by_field_name("name")
                    if name:
                        return name.text.decode("utf8")
                else:
                    func = node.child_by_field_name("function")
                    if func:
                        return func.text.decode("utf8")
            elif language == "rust":
                func = node.child_by_field_name("function")
                if func and func.type == "field_expression":
                    field = func.child_by_field_name("field")
                    if field:
                        return field.text.decode("utf8")
                elif func:
                    return func.text.decode("utf8")
        except Exception:
            pass
        return None

    def _find_chunk_containing(self, chunks: List[CodeChunk], line: int) -> Optional[CodeChunk]:
        """找到包含指定行号的 chunk"""
        for chunk in chunks:
            if chunk.chunk_type in ("function", "class") and chunk.start_line <= line <= chunk.end_line:
                return chunk
        return None

    def _chunk_by_regex(self, file_path: Path, language: str, lines: List[str]) -> List[CodeChunk]:
        """使用正则表达式进行简单切片（降级方案）"""
        import re

        # 简单函数定义模式
        patterns = {
            "python": [
                r"^def\s+(\w+)\s*\(",
                r"^class\s+(\w+)",
                r"^async\s+def\s+(\w+)\s*\(",
            ],
            "javascript": [
                r"^function\s+(\w+)\s*\(",
                r"^const\s+(\w+)\s*=\s*function",
                r"^(\w+)\s*\([^)]*\)\s*\{",
                r"^async\s+function\s+(\w+)",
            ],
            "typescript": [
                r"^function\s+(\w+)\s*\(",
                r"^const\s+(\w+)\s*:\s*\([^)]*\)\s*=>",
                r"^(\w+)\s*\([^)]*\)\s*:\s*\w+\s*\{",
            ],
            "go": [
                r"^func\s+(\w+)\s*\(",
                r"^func\s+\([^)]+\)\s*(\w+)\s*\(",
            ],
        }

        lang_patterns = patterns.get(language, [])
        if not lang_patterns:
            return self._chunk_whole_file(file_path, language)

        combined_pattern = "|".join(f"({p})" for p in lang_patterns)
        regex = re.compile(combined_pattern, re.MULTILINE)

        # 找到所有匹配
        matches = []
        for match in regex.finditer("".join(lines)):
            for group_idx, group in enumerate(match.groups(), 1):
                if group:
                    matches.append({
                        "start": match.start(),
                        "end": match.end(),
                        "name": group.strip().split("(")[0].split(":")[0].strip(),
                    })
                    break

        if not matches:
            return self._chunk_whole_file(file_path, language)

        # 转换为行号
        line_starts = [0]
        for i, line in enumerate(lines[:-1]):
            line_starts.append(line_starts[-1] + len(line))

        def pos_to_line(pos):
            """位置转行号（二分查找）"""
            idx = 0
            while idx < len(line_starts) - 1 and line_starts[idx + 1] <= pos:
                idx += 1
            return idx

        chunks = []
        prev_line = 0

        for match in matches:
            start_line = pos_to_line(match["start"])
            end_line = pos_to_line(match["end"])

            # 避免太小的切片
            if end_line - start_line < 3:
                continue

            chunk_content = "".join(lines[start_line:end_line])

            chunks.append(CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, "function", start_line + 1),
                content=chunk_content,
                file_path=str(file_path),
                start_line=start_line + 1,
                end_line=end_line,
                chunk_type="function",
                name=match["name"],
                language=language,
                context_before="".join(lines[max(0, start_line - self.CONTEXT_LINES):start_line]),
                context_after="".join(lines[end_line:min(end_line + self.CONTEXT_LINES, len(lines))]),
            ))

            prev_line = end_line

        return chunks if chunks else self._chunk_whole_file(file_path, language)

    def _chunk_whole_file(self, file_path: Path, language: Optional[str]) -> List[CodeChunk]:
        """整个文件作为一个 Chunk（最后降级方案）"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return [CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, "file", 1),
                content=content,
                file_path=str(file_path),
                start_line=1,
                end_line=len(content.splitlines()),
                chunk_type="file",
                name=file_path.stem,
                language=language,
            )]
        except Exception as e:
            logger.debug(f"文件切片失败 {file_path}: {e}")
            return []

    def _generate_chunk_id(self, file_path: Path, chunk_type: str, line: int) -> str:
        """生成 Chunk ID（使用 UUID 确保全局唯一）"""
        # 使用 UUID 确保唯一性，前缀保留文件信息便于调试
        unique_id = uuid.uuid4().hex[:12]
        return f"{unique_id}_{file_path.name}_{chunk_type}_{line}"

    def chunk_directory(self, dir_path: str, max_file_size: int = 500_000) -> Tuple[List[CodeChunk], Dict[str, Any]]:
        """
        切片整个目录

        Args:
            dir_path: 目录路径
            max_file_size: 最大文件大小（字节），超过则跳过

        Returns:
            (所有文件的 Chunk 列表, 统计信息字典)
            统计信息包含: total_chunks, markdown_files, notebook_files, notebook_cells, tutorial_sections
        """
        all_chunks = []
        seen_ids = set()  # 用于去重

        # 统计信息
        stats = {
            "total_chunks": 0,
            "markdown_files": 0,
            "notebook_files": 0,
            "notebook_cells": 0,
            "tutorial_sections": 0,
        }

        for root, dirs, files in os.walk(dir_path):
            # 过滤忽略的目录
            dirs[:] = [d for d in dirs if d not in {
                ".git", "node_modules", "__pycache__", ".venv",
                "venv", "dist", "build", "target", "site-packages",
            }]

            for filename in files:
                file_path = Path(root) / filename

                # 跳过系统派生产物
                if self.is_derived_artifact(file_path):
                    logger.info(f"跳过派生产物: {file_path}")
                    continue

                # 跳过太大的文件
                try:
                    if file_path.stat().st_size > max_file_size:
                        logger.info(f"跳过过大文件: {file_path}")
                        continue
                except Exception:
                    pass

                # 统计文件类型
                file_type, _ = self.parser.classifier.classify(str(file_path))
                if file_type == FileType.MARKDOWN:
                    stats["markdown_files"] += 1
                elif file_type == FileType.NOTEBOOK:
                    stats["notebook_files"] += 1

                chunks = self.chunk_file(str(file_path))

                # 更新统计
                for chunk in chunks:
                    if chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        all_chunks.append(chunk)

                        if chunk.chunk_type == "notebook_code":
                            stats["notebook_cells"] += 1
                        elif chunk.chunk_type in ("tutorial_markdown", "readme"):
                            stats["tutorial_sections"] += 1

        stats["total_chunks"] = len(all_chunks)

        logger.info(f"目录切片完成: {dir_path}，共 {len(all_chunks)} 个 chunks，统计: {stats}")
        return all_chunks, stats


# 便捷函数
def chunk_file(file_path: str, language: Optional[str] = None) -> List[CodeChunk]:
    """
    切片文件（便捷函数）

    Args:
        file_path: 文件路径
        language: 编程语言（可选，自动检测）

    Returns:
        CodeChunk 列表
    """
    chunker = Chunker()

    # 自动检测语言
    if language is None:
        _, language = Chunker().parser.classifier.classify(file_path)

    return chunker.chunk_file(file_path)