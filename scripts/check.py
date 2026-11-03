#!/usr/bin/env python3
"""静态自检：不装 torch、不下模型，也能在 CI 里跑。

检查项：
  1. web/index.html 标签闭合、关键元素 id 齐备、确实调用了后端接口
  2. .gitignore 挡住了模型权重等大目录
  3. 前后端接口名对得上（HTML 里 fetch 的路径都能在 server.py 里找到）
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
errors: list[str] = []


def check_html() -> None:
    f = ROOT / "web/index.html"
    if not f.exists():
        errors.append("web/index.html 不存在")
        return
    html = f.read_text(encoding="utf-8")

    for tag in ("html", "head", "body", "script", "style"):
        opens = len(re.findall(rf"<{tag}[\s>]", html))
        closes = len(re.findall(rf"</{tag}>", html))
        if opens != closes:
            errors.append(f"<{tag}> 开闭不匹配: {opens} vs {closes}")

    # 三个模式的输入框与按钮
    for ident in ("cText", "dText", "kText", "cBtn", "dBtn", "kBtn"):
        if f'id="{ident}"' not in html:
            errors.append(f"缺少元素 #{ident}")

    if "/api/" not in html:
        errors.append("前端没有调用后端接口")


def check_gitignore() -> None:
    f = ROOT / ".gitignore"
    if not f.exists():
        errors.append(".gitignore 不存在")
        return
    lines = {line.strip() for line in f.read_text(encoding="utf-8").splitlines()}
    for d in ("models/", "_src/", ".venv/", "profiles/"):
        if d not in lines:
            errors.append(f".gitignore 未忽略 {d}")


def check_api_match() -> None:
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    html = (ROOT / "web/index.html").read_text(encoding="utf-8")
    routes = set(re.findall(r"@app\.(?:get|post)\(\"([^\"]+)\"\)", server))
    # 前端写死的接口路径（去掉模板变量部分）
    used = {m.split("?")[0] for m in re.findall(r"[\"'`](/api/[a-z_/.]+)", html)}
    for path in sorted(used):
        normalized = re.sub(r"\$\{[^}]+\}", "{pid}", path)
        if normalized in routes:
            continue
        # 允许 /api/profiles/xxx/ref.wav 这类子路径
        if any(r.split("{")[0] in normalized for r in routes if "{" in r):
            continue
        errors.append(f"前端调用 {path}，但 server.py 里没有对应路由")


def main() -> int:
    check_html()
    check_gitignore()
    check_api_match()
    if errors:
        print("自检未通过：")
        for e in errors:
            print("  ✗", e)
        return 1
    print("✓ 自检通过：HTML 结构、.gitignore、前后端接口一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
