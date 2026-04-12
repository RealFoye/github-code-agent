#!/usr/bin/env python3
"""
GitHub Code Analysis Agent - 主入口

用法:
    python main.py cli    # 运行 CLI 界面
    python main.py web    # 运行 Web 界面 (Streamlit)
"""

import sys
import argparse

from src.config import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="GitHub 代码分析 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py cli                    # 运行 CLI 界面
    python main.py web                    # 运行 Web 界面
    python main.py web --port 8502       # 指定端口运行 Web 界面
        """,
    )

    parser.add_argument(
        "interface",
        choices=["cli", "web"],
        nargs="?",
        default="web",
        help="选择界面类型 (默认: web)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit 端口 (默认: 8501)",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging()

    if args.interface == "cli":
        print("启动 CLI 界面...")
        from src.ui.cli import run_cli
        run_cli()
    else:
        print(f"启动 Web 界面 (端口 {args.port})...")
        import subprocess
        import os
        # 确保从项目根目录运行
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
        subprocess.run([
            "streamlit", "run",
            "src/ui/web.py",
            "--server.address", "127.0.0.1",
            "--server.port", str(args.port),
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.enableWebsocketCompression", "false",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
        ], env=env)


if __name__ == "__main__":
    main()