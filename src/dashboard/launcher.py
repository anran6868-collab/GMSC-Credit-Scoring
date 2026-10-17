"""从 Windows 桌面启动或重新打开 GMSC 本地网页。"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "app" / "main.py"
URL = "http://127.0.0.1:8501/"
HEALTH_URL = f"{URL}_stcore/health"
PORT = 8501


def _listener_pid() -> int | None:
    """返回占用固定端口的进程号；无人监听时返回 None。"""
    for connection in psutil.net_connections(kind="tcp"):
        if (
            connection.status == psutil.CONN_LISTEN
            and connection.laddr
            and connection.laddr.port == PORT
        ):
            return connection.pid or -1
    return None


def _is_our_app(pid: int) -> bool:
    if pid < 0:
        return False
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        if "run" not in command:
            return False
        index = command.index("run")
        if index + 1 >= len(command):
            return False
        entrypoint = Path(command[index + 1])
        if not entrypoint.is_absolute():
            entrypoint = Path(process.cwd()) / entrypoint
        return entrypoint.resolve() == APP_PATH.resolve()
    except (psutil.Error, OSError, ValueError):
        return False


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1) as response:
            return response.status == 200 and response.read(32).strip() == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def _open_browser() -> None:
    print(f"正在打开 GMSC 网页：{URL}", flush=True)
    if not webbrowser.open(URL, new=2):
        print(f"浏览器未自动打开，请手动访问：{URL}", flush=True)


def main() -> int:
    if not APP_PATH.is_file():
        print("未找到 Streamlit 页面入口 app/main.py。", flush=True)
        return 1

    try:
        listener = _listener_pid()
    except (psutil.Error, OSError, RuntimeError) as exc:
        print(f"无法检查本地 {PORT} 端口：{exc}。请稍后重试。", flush=True)
        return 2
    if listener is not None:
        if not _is_our_app(listener):
            print(f"{PORT} 端口已被其他程序占用，请先关闭该程序后重试。", flush=True)
            return 2
        for _ in range(15):
            if _healthy():
                break
            time.sleep(1)
        else:
            print("GMSC 服务正在启动或暂时无响应，请稍后再次双击快捷方式。", flush=True)
            return 2
        print("GMSC 网页已经在运行。", flush=True)
        _open_browser()
        return 0

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.showEmailPrompt",
        "false",
    ]
    print("正在启动 GMSC 本地网页，请稍候……", flush=True)
    try:
        server = subprocess.Popen(command, cwd=PROJECT_ROOT)
    except OSError as exc:
        print(f"GMSC 网页启动失败：{exc}", flush=True)
        return 1

    for _ in range(60):
        if server.poll() is not None:
            print(f"GMSC 网页启动失败，退出码：{server.returncode}。", flush=True)
            return server.returncode or 1
        if _healthy():
            _open_browser()
            print("网页运行中；关闭此窗口即可停止服务。", flush=True)
            try:
                return server.wait()
            except KeyboardInterrupt:
                server.terminate()
                server.wait()
                return 0
        time.sleep(1)

    server.terminate()
    server.wait()
    print("GMSC 网页在 60 秒内未能启动，请检查上方错误信息。", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
