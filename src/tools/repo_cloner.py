"""
仓库克隆工具
负责 Git 仓库的克隆、更新、清理
"""

import logging
import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager

import git
from git import GitCommandError, Repo

from src.config import REPOS_DIR

logger = logging.getLogger(__name__)

# 最大保留仓库数量
MAX_KEEP_REPOS = 5


class RepoCloner:
    # 类级别：正在被分析的仓库路径（跨实例共享）
    # 这样不同 Agent 实例也能知道哪些仓库正在被分析
    _protected_paths: set = set()

    def __init__(self, repos_dir: Optional[Path] = None):
        """
        初始化仓库管理器

        Args:
            repos_dir: 仓库存储目录，默认使用 config 中的 REPOS_DIR
        """
        self.repos_dir = repos_dir or REPOS_DIR
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 仓库保护（防止误删正在分析的仓库）====================

    @classmethod
    def protect_repo(cls, repo_path: str) -> None:
        """
        标记一个仓库为"正在分析中"，不会被自动清理

        Args:
            repo_path: 仓库本地路径
        """
        cls._protected_paths.add(repo_path)
        logger.debug(f"仓库已保护: {repo_path}")

    @classmethod
    def unprotect_repo(cls, repo_path: str) -> None:
        """
        取消仓库保护，允许被清理

        Args:
            repo_path: 仓库本地路径
        """
        cls._protected_paths.discard(repo_path)
        logger.debug(f"仓库已解除保护: {repo_path}")

    @classmethod
    @contextmanager
    def protecting(cls, repo_path: str):
        """
        上下文管理器：自动保护/解除保护仓库
        用法:
            with cloner.protecting("/path/to/repo"):
                # repo 被保护
                do_something()
            # repo 自动解除保护
        """
        cls.protect_repo(repo_path)
        try:
            yield
        finally:
            cls.unprotect_repo(repo_path)

    # ==================== 自动清理旧仓库 ====================

    @classmethod
    def _get_repo_dirs(cls) -> List[Path]:
        """获取所有仓库目录，按修改时间排序（最老的在前）"""
        repos_dir = REPOS_DIR
        if not repos_dir.exists():
            return []
        dirs = []
        for item in repos_dir.iterdir():
            if item.is_dir():
                try:
                    # 使用目录的修改时间
                    mtime = item.stat().st_mtime
                    dirs.append((mtime, item))
                except OSError:
                    continue
        dirs.sort(key=lambda x: x[0])  # 最老的在前
        return [d for _, d in dirs]

    def _cleanup_old_repos(self, keep_list: Optional[List[str]] = None) -> int:
        """
        清理旧仓库（保守策略）

        Args:
            keep_list: 额外要保留的路径列表（正在分析的 + 即将克隆的）

        Returns:
            清理的仓库数量
        """
        all_dirs = self._get_repo_dirs()
        if len(all_dirs) <= MAX_KEEP_REPOS:
            return 0

        keep_set = set(keep_list or [])
        keep_set.update(self.__class__._protected_paths)  # 合并正在被保护的仓库

        dirs_to_delete = all_dirs[:-MAX_KEEP_REPOS]  # 保留最近的 MAX_KEEP_REPOS 个
        cleaned = 0
        for dir_path in dirs_to_delete:
            path_str = str(dir_path.resolve())
            if path_str in keep_set:
                continue
            try:
                shutil.rmtree(dir_path)
                logger.info(f"自动清理旧仓库: {path_str}")
                cleaned += 1
            except Exception as e:
                logger.warning(f"清理旧仓库失败: {path_str}, {e}")

        if cleaned:
            logger.info(f"自动清理完成，共清理 {cleaned} 个旧仓库")
        return cleaned

    # ==================== 克隆逻辑 ====================

    def _generate_local_name(self, repo_url: str) -> str:
        """
        根据 URL 生成本地目录名

        Args:
            repo_url: GitHub 仓库 URL

        Returns:
            本地目录名（安全、唯一）
        """
        # 使用 SHA1 哈希确保唯一性，避免特殊字符问题
        name_hash = hashlib.sha1(repo_url.encode()).hexdigest()[:12]

        # 从 URL 提取 owner/repo 部分作为可读名称
        # 例如: https://github.com/owner/repo -> owner-repo
        if "github.com" in repo_url:
            parts = repo_url.rstrip("/").split("/")
            if len(parts) >= 2:
                # 取最后两个部分: owner 和 repo
                owner = parts[-2]
                repo_name = parts[-1].replace(".git", "")
                return f"{owner}-{repo_name}-{name_hash}"

        # 降级处理：直接使用哈希
        return name_hash

    def clone(self, repo_url: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """
        克隆仓库到本地

        Args:
            repo_url: 仓库 URL（支持 HTTPS 和 SSH）
            branch: 指定分支，默认 main/master

        Returns:
            包含本地路径和仓库信息的字典
        """
        local_name = self._generate_local_name(repo_url)
        local_path = self.repos_dir / local_name

        # 自动清理旧仓库（保留最多 MAX_KEEP_REPOS 个，保护正在使用的和即将克隆的）
        self._cleanup_old_repos(keep_list=[str(local_path)])

        # 如果已存在，检查是否需要更新
        if local_path.exists():
            logger.info(f"仓库已存在: {local_path}")
            try:
                repo = Repo(local_path)
                # 尝试 fetch 更新
                origin = repo.remote("origin")
                origin.fetch()
                logger.info(f"仓库已更新: {local_path}")
                return {
                    "local_path": str(local_path),
                    "repo_url": repo_url,
                    "is_updated": True,
                    "is_existing": True,
                }
            except GitCommandError as e:
                logger.warning(f"更新失败，将重新克隆: {e}")
                shutil.rmtree(local_path)

        # 克隆新仓库
        logger.info(f"正在克隆仓库: {repo_url}")

        try:
            # 自动检测默认分支
            if branch is None:
                branch = self._detect_default_branch(repo_url)

            repo = git.Repo.clone_from(
                repo_url,
                local_path,
                branch=branch,
                depth=100,  # 浅克隆，加速
            )

            logger.info(f"克隆完成: {local_path}")

            return {
                "local_path": str(local_path),
                "repo_url": repo_url,
                "branch": branch,
                "is_updated": False,
                "is_existing": False,
            }

        except GitCommandError as e:
            logger.error(f"克隆失败: {e}")
            # 清理失败的克隆
            if local_path.exists():
                shutil.rmtree(local_path)
            raise

    def _detect_default_branch(self, repo_url: str) -> str:
        """检测仓库默认分支"""
        # 优先尝试 main，然后是 master
        for branch in ["main", "master"]:
            try:
                # 使用 ls-remote 检测分支是否存在
                # 注意：ls-remote 即使分支不存在也返回 exit code 0，需要检查输出
                result = git.Git().ls_remote("--heads", repo_url, branch)
                if result.strip():  # 输出非空说明分支存在
                    logger.info(f"检测到默认分支: {branch}")
                    return branch
            except GitCommandError:
                continue

        # 降级使用 main
        logger.warning("无法检测默认分支，使用 main")
        return "main"

    def get_repo_info(self, local_path: str) -> Dict[str, Any]:
        """
        获取仓库信息

        Args:
            local_path: 本地仓库路径

        Returns:
            仓库信息字典
        """
        try:
            repo = Repo(local_path)

            # 获取分支信息
            branches = [str(b) for b in repo.branches]
            current_branch = repo.active_branch.name if not repo.head.is_detached else "detached"

            # 获取最新提交
            if repo.head.is_valid:
                latest_commit = repo.head.commit.hexsha[:8]
                commit_date = datetime.fromtimestamp(repo.head.commit.committed_date).isoformat()
            else:
                latest_commit = None
                commit_date = None

            # 统计代码行数（流式读取，避免大文件 OOM）
            total_lines = 0
            file_count = 0
            for ext in [".py", ".js", ".ts", ".go", ".java", ".rs", ".cpp"]:
                try:
                    for blob in repo.tree().traverse():
                        if blob.path.endswith(ext) and blob.type == "blob":
                            # 跳过超大文件（> 5MB），避免内存溢出
                            if blob.size > 5_000_000:
                                continue
                            # 流式读取，逐块计算行数
                            lines = 0
                            stream = blob.data_stream
                            while True:
                                chunk = stream.read(65536)  # 64KB per read
                                if not chunk:
                                    break
                                lines += chunk.decode("utf-8", errors="ignore").count("\n")
                            total_lines += lines
                            file_count += 1
                except Exception:
                    pass

            return {
                "local_path": local_path,
                "current_branch": current_branch,
                "branches": branches,
                "latest_commit": latest_commit,
                "commit_date": commit_date,
                "is_dirty": repo.is_dirty(),
                "total_lines": total_lines,
                "file_count": file_count,
            }

        except Exception as e:
            logger.error(f"获取仓库信息失败: {e}")
            return {"local_path": local_path, "error": str(e)}

    def cleanup(self, local_path: Optional[str] = None, keep_list: Optional[list] = None) -> int:
        """
        清理仓库目录

        Args:
            local_path: 指定清理的仓库路径（单个）
            keep_list: 保留的仓库路径列表（其他全部删除）

        Returns:
            清理的仓库数量
        """
        cleaned = 0

        if local_path:
            # 清理指定仓库
            path = Path(local_path)
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                logger.info(f"已清理仓库: {local_path}")
                cleaned = 1

        elif keep_list:
            # 保留列表之外的仓库
            keep_set = set(str(Path(p).resolve()) for p in keep_list)
            for item in self.repos_dir.iterdir():
                if item.is_dir() and str(item.resolve()) not in keep_set:
                    shutil.rmtree(item)
                    cleaned += 1

        else:
            # 清理所有仓库（危险操作）
            for item in self.repos_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                    cleaned += 1

        logger.info(f"共清理 {cleaned} 个仓库")
        return cleaned

    def list_repos(self) -> list:
        """
        列出所有本地仓库

        Returns:
            仓库路径列表
        """
        repos = []
        for item in self.repos_dir.iterdir():
            if item.is_dir():
                try:
                    # 验证是有效的 Git 仓库
                    Repo(item)
                    repos.append(str(item))
                except Exception:
                    # 跳过无效目录
                    pass
        return repos


# 便捷函数
def clone_repo(repo_url: str, branch: Optional[str] = None) -> Dict[str, Any]:
    """
    克隆仓库（便捷函数）

    Args:
        repo_url: 仓库 URL
        branch: 分支名

    Returns:
        仓库信息字典
    """
    cloner = RepoCloner()
    return cloner.clone(repo_url, branch)


def cleanup_repo(local_path: str) -> bool:
    """
    清理指定仓库（便捷函数）

    Args:
        local_path: 本地仓库路径

    Returns:
        是否成功清理
    """
    try:
        cloner = RepoCloner()
        cloner.cleanup(local_path)
        return True
    except Exception as e:
        logger.error(f"清理失败: {e}")
        return False