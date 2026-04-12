"""
GitHub 搜索工具
封装 GitHub API，提供仓库搜索能力
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import requests
from github import Github, GithubException, RateLimitExceededException

from src.config import GITHUB_TOKEN, DEFAULT_SEARCH_LIMIT, DEFAULT_MIN_STARS

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    total_count: int
    repos: List[Dict[str, Any]]
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[int] = None


class GitHubSearch:
    """GitHub 搜索工具"""

    def __init__(self, token: Optional[str] = None):
        """
        初始化 GitHub 搜索工具

        Args:
            token: GitHub Personal Access Token（可选，提高 API 限额）
        """
        self.token = token or GITHUB_TOKEN
        self.session = requests.Session()

        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            logger.info("GitHub API 已认证（5000 次/小时限额）")
        else:
            logger.warning("GitHub API 未认证（60 次/小时限额）")

        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "GitHub-Code-Analysis-Agent"

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """处理 API 限流"""
        if response.status_code == 403:
            reset_time = response.headers.get("X-RateLimit-Reset")
            if reset_time:
                wait_seconds = int(reset_time) - int(time.time())
                if wait_seconds > 0:
                    logger.warning(f"GitHub API 限流，等待 {wait_seconds} 秒...")
                    time.sleep(min(wait_seconds, 60))  # 最多等待 60 秒

    def search_repos(
        self,
        keywords: str,
        language: Optional[str] = None,
        min_stars: int = DEFAULT_MIN_STARS,
        limit: int = DEFAULT_SEARCH_LIMIT,
        sort: str = "stars",  # stars, forks, updated
    ) -> SearchResult:
        """
        搜索 GitHub 仓库

        Args:
            keywords: 搜索关键词
            language: 编程语言筛选（如 "Python", "JavaScript"）
            min_stars: 最小 stars 数
            limit: 返回结果数量上限
            sort: 排序方式（stars, forks, updated）

        Returns:
            SearchResult 对象
        """
        if not keywords:
            raise ValueError("搜索关键词不能为空")

        # 构建搜索查询
        query_parts = [keywords]
        if language:
            query_parts.append(f"language:{language}")
        if min_stars > 0:
            query_parts.append(f"stars:>={min_stars}")

        query = " ".join(query_parts)
        logger.info(f"搜索 GitHub: {query}")

        # GitHub Search API
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": min(limit, 100),  # 最多 100
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._handle_rate_limit(response)

            if response.status_code == 200:
                data = response.json()
                rate_limit = response.headers.get("X-RateLimit-Remaining")
                rate_reset = response.headers.get("X-RateLimit-Reset")

                repos = []
                for item in data.get("items", [])[:limit]:
                    repo = self._parse_repo(item)
                    repos.append(repo)

                logger.info(f"找到 {data.get('total_count', 0)} 个仓库，返回 {len(repos)} 个")

                return SearchResult(
                    total_count=data.get("total_count", 0),
                    repos=repos,
                    rate_limit_remaining=int(rate_limit) if rate_limit else None,
                    rate_limit_reset=int(rate_reset) if rate_reset else None,
                )

            elif response.status_code == 422:
                logger.error("搜索查询无效")
                raise ValueError("搜索查询无效")

            elif response.status_code == 403:
                logger.error("API 限额已用尽")
                raise RateLimitExceededException()

            else:
                logger.error(f"GitHub API 错误: {response.status_code}")
                response.raise_for_status()
                raise Exception(f"GitHub API 错误: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error("GitHub API 请求超时")
            raise Exception("GitHub API 请求超时，请检查网络连接")

        except requests.exceptions.RequestException as e:
            logger.error(f"网络错误: {e}")
            raise

    def _parse_repo(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """解析仓库数据"""
        owner = item.get("owner", {})

        return {
            "full_name": item.get("full_name", ""),
            "name": item.get("name", ""),
            "author": owner.get("login", ""),
            "description": item.get("description", ""),
            "url": item.get("html_url", ""),
            "stars": item.get("stargazers_count", 0),
            "forks": item.get("forks_count", 0),
            "open_issues": item.get("open_issues_count", 0),
            "language": item.get("language", ""),
            "license": item.get("license", {}).get("name") if item.get("license") else None,
            "topics": item.get("topics", []),
            "last_updated": item.get("updated_at", ""),
            "created_at": item.get("created_at", ""),
            "pushed_at": item.get("pushed_at", ""),
            "homepage": item.get("homepage", ""),
        }

    def get_repo_details(self, full_name: str) -> Optional[Dict[str, Any]]:
        """
        获取仓库详细信息

        Args:
            full_name: 仓库完整名称（author/repo）

        Returns:
            仓库详细信息字典
        """
        url = f"https://api.github.com/repos/{full_name}"

        try:
            response = self.session.get(url, timeout=30)
            self._handle_rate_limit(response)

            if response.status_code == 200:
                return self._parse_repo(response.json())
            elif response.status_code == 404:
                logger.warning(f"仓库不存在: {full_name}")
                return None
            else:
                response.raise_for_status()
                return None

        except Exception as e:
            logger.error(f"获取仓库详情失败: {e}")
            return None

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """获取 API 限额状态"""
        url = "https://api.github.com/rate_limit"

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                search_limit = data.get("resources", {}).get("search", {})
                return {
                    "remaining": search_limit.get("remaining"),
                    "limit": search_limit.get("limit"),
                    "reset": search_limit.get("reset"),
                    "used": search_limit.get("used"),
                }
            return {}
        except Exception as e:
            logger.error(f"获取限额状态失败: {e}")
            return {}


# 便捷函数
def search_repos(
    keywords: str,
    language: Optional[str] = None,
    min_stars: int = DEFAULT_MIN_STARS,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> SearchResult:
    """
    搜索 GitHub 仓库（便捷函数）

    Args:
        keywords: 搜索关键词
        language: 编程语言筛选
        min_stars: 最小 stars 数
        limit: 返回结果数量

    Returns:
        SearchResult 对象
    """
    searcher = GitHubSearch()
    return searcher.search_repos(keywords, language, min_stars, limit)