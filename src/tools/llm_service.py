"""
LLM 服务
统一封装 OpenAI/DeepSeek/MiniMax 等 LLM 调用
MiniMax 使用 Anthropic SDK，其他使用 OpenAI SDK
"""

import logging
import time
from typing import List, Dict, Any, Optional

from src.config import get_llm_config

logger = logging.getLogger(__name__)

# 重试配置
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # 秒


class LLMService:
    """LLM 服务（支持 OpenAI/DeepSeek/MiniMax 等 LLM）"""

    def __init__(self, provider: Optional[str] = None):
        """
        初始化 LLM 服务

        Args:
            provider: LLM 提供者，不指定则使用配置文件中的默认提供者
        """
        if provider:
            # 临时使用指定的 provider
            self._provider = provider
            self._config = self._get_provider_config(provider)
        else:
            # 使用默认配置
            self._config = get_llm_config()
            self._provider = self._config.get("provider", "openai")

        self._init_client()

    def _get_provider_config(self, provider: str) -> dict:
        """获取指定 provider 的配置"""
        from src.config import (
            OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
            DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
            MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL,
            OLLAMA_BASE_URL,
        )

        configs = {
            "openai": {
                "api_key": OPENAI_API_KEY,
                "base_url": OPENAI_BASE_URL,
                "model": OPENAI_MODEL,
            },
            "deepseek": {
                "api_key": DEEPSEEK_API_KEY,
                "base_url": DEEPSEEK_BASE_URL,
                "model": DEEPSEEK_MODEL,
            },
            "minimax": {
                "api_key": MINIMAX_API_KEY,
                "base_url": MINIMAX_BASE_URL,
                "model": MINIMAX_MODEL,
            },
            "ollama": {
                "base_url": OLLAMA_BASE_URL,
                "model": "qwen3",
            },
        }

        if provider not in configs:
            raise ValueError(f"Unknown LLM provider: {provider}")

        return configs[provider]

    def _init_client(self):
        """初始化 LLM 客户端"""
        self.model = self._config.get("model", "")

        if self._provider == "minimax":
            # MiniMax 使用 Anthropic SDK
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self._config.get("api_key"),
                base_url=self._config.get("base_url"),
            )
            self._openai_client = None
        else:
            # 其他使用 OpenAI SDK
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self._config.get("api_key", ""),
                base_url=self._config.get("base_url", ""),
            )
            self._anthropic_client = None

    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否应该重试"""
        error_str = str(error).lower()

        retryable_keywords = [
            "529", "overloaded", "rate limit", "429",
            "500", "502", "503", "504", "timeout", "connection",
        ]

        return any(keyword in error_str for keyword in retryable_keywords)

    def _call_minimax(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        """调用 MiniMax API（使用 Anthropic SDK）"""
        # 提取 system message
        system_content = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_content or "你是一个有用的助手。",
            messages=user_messages,
        )

        # 遍历 content 找到 TextBlock（MiniMax-M2.7 可能返回 ThinkingBlock + TextBlock）
        for block in response.content:
            if block.type == "text":
                return block.text

        # 如果只有 thinking 块没有 text 块，说明 max_tokens 不够，重试一次
        block_types = [b.type for b in response.content]
        if "thinking" in block_types and "text" not in block_types:
            response = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=max_tokens * 4,  # 大幅增加 token 上限
                system=system_content or "你是一个有用的助手。",
                messages=user_messages,
            )
            for block in response.content:
                if block.type == "text":
                    return block.text

        raise ValueError(f"无法从响应中提取文本内容，响应类型: {[b.type for b in response.content]}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        **kwargs,
    ) -> str:
        """
        发送对话请求（带自动重试）

        Args:
            messages: 消息列表，格式为 [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            **kwargs: 其他参数

        Returns:
            LLM 回复内容
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                if self._provider == "minimax":
                    return self._call_minimax(messages, max_tokens)
                else:
                    response = self._openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                last_error = e
                error_str = str(e)

                # 检查是否是 529 overloaded 错误
                if "529" in error_str or "overloaded" in error_str.lower():
                    logger.warning(f"API 过载 (529)，尝试重试 ({attempt + 1}/{max_retries})...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue

                # 检查是否是其他可重试错误
                if self._is_retryable_error(e):
                    logger.warning(f"API 可重试错误: {e}，尝试重试 ({attempt + 1}/{max_retries})...")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue

                logger.error(f"LLM 调用失败 [{self._provider}]: {e}")
                raise

        logger.error(f"LLM 调用失败，已重试 {max_retries} 次: {last_error}")
        raise last_error

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """
        补全请求（兼容旧接口）

        Args:
            prompt: 用户 prompt
            system_prompt: 系统提示（可选）
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            LLM 回复内容
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature, max_tokens, **kwargs)

    @property
    def provider(self) -> str:
        """获取当前 provider"""
        return self._provider


# ==================== 便捷函数 ====================

def get_llm_service(provider: Optional[str] = None) -> LLMService:
    """
    获取 LLM 服务实例

    Args:
        provider: LLM 提供者（可选）

    Returns:
        LLMService 实例
    """
    return LLMService(provider)


def chat(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """
    发送对话请求（便捷函数）

    Args:
        messages: 消息列表
        provider: LLM 提供者
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        LLM 回复内容
    """
    service = get_llm_service(provider)
    return service.chat(messages, temperature, max_tokens)


def complete(
    prompt: str,
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """
    补全请求（便捷函数）

    Args:
        prompt: 用户 prompt
        provider: LLM 提供者
        system_prompt: 系统提示
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        LLM 回复内容
    """
    service = get_llm_service(provider)
    return service.complete(prompt, system_prompt, temperature, max_tokens)