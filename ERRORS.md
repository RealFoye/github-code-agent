# 错误记录

## 1. Streamlit PYTHONPATH 问题

**症状：** `ModuleNotFoundError: No module named 'src'`

**原因：** `main.py` 用 subprocess 启动 Streamlit 时没设置 PYTHONPATH

**修复：** `main.py` 中添加环境变量设置
```python
env = os.environ.copy()
env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
subprocess.run([...], env=env)
```

**时间：** 2026-04-10

---

## 2. ChromaDB chunk_id 重复

**症状：** `DuplicateIDError: Expected IDs to be unique, found duplicates of: jquery.js_function_2`

**原因：** `_generate_chunk_id` 使用 `file_path.name` + 行号，同一文件内 tree-sitter 解析出重复匹配时冲突

**修复：** `chunker.py` 中改用 UUID 确保全局唯一性
```python
def _generate_chunk_id(self, file_path: Path, chunk_type: str, line: int) -> str:
    unique_id = uuid.uuid4().hex[:12]
    return f"{unique_id}_{file_path.name}_{chunk_type}_{line}"
```

同时在 `chunk_directory` 添加去重逻辑作为双重保障。

**时间：** 2026-04-10

---

## 3. GitHub API 限流显示错误

**症状：** `get_rate_limit_status()` 返回 remaining=limit（都是30）

**原因：** 错误使用了 `limit` 字段而非 `remaining`

**修复：** `github_search.py:209` 改为 `"remaining": search_limit.get("remaining")`

**时间：** 2026-04-10

---

## 4. config.py MiniMax 配置重复

**症状：** .env 配置的 MINIMAX_BASE_URL 被覆盖

**原因：** config.py 有两个 MINIMAX 配置块，第二个覆盖第一个

**修复：** 删除重复的配置块，保留正确的值

**时间：** 2026-04-10

---

## 5. search_agent MiniMax ThinkingBlock 处理

**症状：** MiniMax-M2.7 返回 ThinkingBlock 时报错

**原因：** 直接访问 `response.content[0].text`，没考虑 ThinkingBlock

**修复：** 遍历 content 查找 TextBlock 类型

**时间：** 2026-04-10

---

## 6. MiniMax API 兼容性问题

**症状：** `Error code: 529 - overloaded_error` 或 `404 Not Found`

**原因：** `llm_service.py` 使用 OpenAI SDK 调用 MiniMax，但 MiniMax 的 OpenAI 兼容端点 URL 不同

**修复：** 修改 `llm_service.py`，对 MiniMax 使用 Anthropic SDK：
```python
if self._provider == "minimax":
    import anthropic
    self._anthropic_client = anthropic.Anthropic(
        api_key=self._config.get("api_key"),
        base_url=self._config.get("base_url"),  # api.minimaxi.com/anthropic
    )
```
同时保留自动重试机制应对 529 过载。

**时间：** 2026-04-10
