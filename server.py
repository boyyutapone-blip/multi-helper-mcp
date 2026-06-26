"""
Multi-Helper MCP Server
=======================

让纯文本主控模型（如 GLM-5.2）通过 MCP 工具调用获得多项外援能力：
视觉理解、世界知识、联网搜索、网页读取、开源仓库访问。

核心思路：主控模型始终只处理文本，图片、知识查询、搜索、抓网页等通过工具调用
委托给专门模型或本地实现，工具返回纯文本结果给主控。主控人格稳定、上下文连续，
外援可插拔。

提供六个工具：
  vision_describe(image_path, question)        → 视觉模型看图返回文字描述
  deepseek_knowledge(query)                    → 快速知识模型回答客观事实
  deepseek_knowledge_deep(query)               → 深度推理模型处理复杂分析
  web_search(query, count)                     → 联网搜索（博查 Bocha API）
  web_reader(url, max_chars)                   → 抓取网页正文（本地 httpx + selectolax）
  github_repo(action, repo, path, query)       → 读取 GitHub 仓库（公开 API 免 token）

特性：
  - 自动压缩大图（最长边可配，默认 1024px，转 JPEG），加速视觉 API 调用
  - 图片编码缓存，同图多次问不同问题省去重复编码
  - 支持 image_path="latest" 自动定位最新截图
  - 完整错误处理，工具失败返回结构化错误文本，不抛裸 traceback
  - 所有配置通过环境变量传入，不硬编码任何密钥

兼容任何 OpenAI 兼容网关（火山云 coding plan / 智谱官方 / OpenRouter / 本地 Ollama 等）。
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Pillow 用于图片压缩/缩放，加速视觉 API 调用
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    sys.stderr.write("[multi-helper] 警告：未安装Pillow，图片不会被压缩，大图可能超时。\n")

# httpx 用于 web_reader 抓取网页；selectolax 用于解析 HTML 提取正文
try:
    import httpx
    from selectolax.parser import HTMLParser
    HAS_WEB_READER = True
except ImportError:
    HAS_WEB_READER = False
    sys.stderr.write("[multi-helper] 警告：未安装httpx/selectolax，web_reader 工具不可用。\n")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")  # 必填，main() 启动时会校验
ARK_BASE_URL = os.environ.get(
    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"
)
VISION_MODEL_ID = os.environ.get("VISION_MODEL_ID", "doubao-seed-2.0-lite")
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))  # 最长边1024px，识别截图足够清晰且快
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "800"))  # 详细描述需要更长输出
# 截图目录：vision_describe 支持 image_path="latest" 自动取最新截图，
# 此目录是扫描范围。默认 Windows 标准截图路径，可通过环境变量覆盖。
SCREENSHOT_DIR = os.environ.get(
    "SCREENSHOT_DIR",
    os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots"),
)

# ============ 联网搜索（博查 Bocha）============
# 火山云 coding plan 套餐不含任何搜索/browsing 模型（实测全部报 UnsupportedModel），
# 因此联网搜索走博查 API（open.bochaai.com，当前完全免费，无需信用卡）。
# 留空则 web_search 工具返回"未配置"提示，不会报错。
BOCHA_API_KEY = os.environ.get("BOCHA_API_KEY", "")
BOCHA_API_URL = os.environ.get(
    "BOCHA_API_URL", "https://api.bochaai.com/v1/web-search"
)

# ============ 网页读取 ============
# web_reader 抓取网页的超时（秒）和最大返回字符数（避免超长页面撑爆主控上下文）
WEB_READER_TIMEOUT = int(os.environ.get("WEB_READER_TIMEOUT", "20"))
WEB_READER_MAX_CHARS = int(os.environ.get("WEB_READER_MAX_CHARS", "8000"))
WEB_READER_USER_AGENT = os.environ.get(
    "WEB_READER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) multi-helper-mcp/0.3 "
    "(+https://github.com/local; like crawler)",
)

# ============ GitHub 仓库读取 ============
# 公开 API 免 token，限流 60 次/小时/IP。配了 token 提到 5000 次/小时。
# 留空则走匿名模式，对偶尔查公开仓库足够。
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_API_URL = "https://api.github.com"
GITHUB_RAW_URL = "https://raw.githubusercontent.com"

# 客户端默认超时55s（留5s余量给MCP外层的60s）；单个工具调用可在 create() 时覆盖
# max_retries=2：openai SDK 原生支持，自动重试瞬时抖动（429/5xx/连接错误），指数退避
# 注意：api_key 用 ARK_API_KEY 或占位符初始化，main() 会校验真实 key；
# 占位符避免模块加载时因 key 为空直接抛错，让 main() 能给出友好的错误提示
client = OpenAI(
    api_key=ARK_API_KEY or "placeholder-pending-validation",
    base_url=ARK_BASE_URL,
    timeout=55,
    max_retries=2,
)

mcp = FastMCP("glm-router")

# 图片编码缓存：同一张图反复问不同问题时，避免每次都重新压缩+编码。
# key = (路径, mtime, max_side, jpeg_quality)，参数变了缓存自动失效。
# 进程级缓存（MCP server 是长驻进程），不持久化，重启后清空——符合预期。
_IMAGE_CACHE: dict[tuple, tuple[str, str]] = {}
_IMAGE_CACHE_MAX = 50  # 最多缓存50张，避免内存爆炸


def _cache_image(cache_key: tuple, b64: str, mime: str) -> tuple[str, str]:
    """写入图片缓存，返回 (b64, mime)。超过上限时简单清空最早项（FIFO 近似）。"""
    if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX:
        # 弹出最早插入的一项（dict 保持插入顺序）
        oldest = next(iter(_IMAGE_CACHE))
        _IMAGE_CACHE.pop(oldest, None)
    _IMAGE_CACHE[cache_key] = (b64, mime)
    return b64, mime


def _resolve_image_path(image_path: str) -> str:
    """解析图片路径：支持 "latest" / "latest:N" 关键字自动取最新截图，
    否则原样返回路径。

    - "latest"      → SCREENSHOT_DIR 下最新的图片
    - "latest:3"    → 第 3 新的图片（latest:1 等价于 latest）
    - 正常路径       → 原样返回
    """
    if not image_path.startswith("latest"):
        return image_path

    # 解析 latest 或 latest:N
    parts = image_path.split(":", 1)
    n = 1
    if len(parts) == 2:
        try:
            n = int(parts[1])
            if n < 1:
                n = 1
        except ValueError:
            pass  # 忽略非法数字，按 latest=1 处理

    shot_dir = Path(SCREENSHOT_DIR)
    if not shot_dir.exists():
        raise FileNotFoundError(
            f"截图目录不存在: {SCREENSHOT_DIR}。"
            f"可通过环境变量 SCREENSHOT_DIR 配置正确路径，或直接传完整图片路径。"
        )

    # 扫描目录下所有支持的图片格式，按修改时间倒序
    img_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    candidates = [p for p in shot_dir.iterdir() if p.suffix.lower() in img_exts and p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"截图目录 {SCREENSHOT_DIR} 下没有图片文件。")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if n > len(candidates):
        raise FileNotFoundError(
            f"截图目录只有 {len(candidates)} 张图片，无法取第 {n} 新的。"
            f"最新的是: {candidates[0]}"
        )

    return str(candidates[n - 1])


def _encode_image(path: str) -> tuple[str, str]:
    """读取并编码图片：有Pillow时自动压缩缩放，没有时直接读原文件。
    返回 (base64字符串, mime类型)。带缓存：同一张图+同样压缩参数只处理一次。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"图片不存在: {path}")
    suffix = p.suffix.lower()

    # 构造缓存 key：路径+修改时间+压缩参数。任一变化都视为新图。
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0
    cache_key = (str(p), mtime, MAX_IMAGE_SIDE, JPEG_QUALITY)
    cached = _IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if HAS_PIL:
        # 打开图片，自动处理方向EXIF
        img = Image.open(p)
        try:
            img = Image.open(p)
            # 处理 EXIF 方向
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            img = Image.open(p)

        # 转RGB（处理RGBA/PNG透明通道）
        if img.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 缩放：最长边不超过 MAX_IMAGE_SIDE
        w, h = img.size
        if max(w, h) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # 输出为JPEG到内存
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return _cache_image(cache_key, b64, "image/jpeg")
    else:
        # 没装Pillow，直接读原文件
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(suffix, "image/png")
        b64 = base64.b64encode(p.read_bytes()).decode()
        return _cache_image(cache_key, b64, mime)


@mcp.tool()
def vision_describe(
    image_path: str,
    question: str = "请详细描述这张图片：包括主体内容、布局结构、配色、可见文字，以及任何值得注意的细节。",
) -> str:
    """⚠️ 【唯一正确的图片分析方法】分析本地图片文件并返回文字描述。
    当用户发送图片、截图、要求分析UI/图表/OCR/报错截图/照片时，**必须调用此工具**。

    🚨 最高优先级规则——用户提到图片/截图时，直接调本工具，不要走其他路径：
    - 用户说"看最新截图"/"刚刚截的图"/"我发的图"→ 直接调本工具，image_path 传 "latest"
    - 用户给了图片路径 → 直接调本工具，image_path 传该路径
    - ❌ 不要先 Glob 搜索图片文件再决定怎么处理——Glob 找到文件后你可能会忍不住 Read，会废掉会话
    - ❌ 不要先 Read 图片文件"看看是什么"——Read 对图片会注入二进制乱码到上下文，主模型（GLM-5.2）是纯文本模型，会话会永久损坏
    - ❌ 不要把图片当附件直接传给主模型——主模型不支持图片输入，会报 "Model only support text input"
    - 正确路径只有一条：调本工具，由本工具调用视觉模型分析图片，返回文字给你

    ⛔ 绝对禁止：
    - 绝对不要使用 Read 工具读取图片文件！
    - 绝对不要尝试把图片内容直接粘贴到消息里，会被拦截。
    - 如果此工具第一次调用超时（超过60秒），请直接告诉用户"读图失败"并询问是否重试或换方式，
      不要换用Read，不要自己兜底处理。

    ✅ 正确做法：调用本工具，传入图片绝对路径，或用关键字自动定位最新截图。

    参数：
        image_path: 图片路径，支持三种形式：
                    1. 绝对路径：如 C:\\Users\\xxx\\Pictures\\screenshot.png
                    2. "latest"：自动取截图目录下最新的图片（用户说"看最新截图"时用这个）
                    3. "latest:N"：取第 N 新的截图，如 "latest:2" 取第二新的，"latest:3" 取第三新的
                    用户没给路径但提到"最新截图"/"刚刚截的图"时，直接用 "latest"，不要先 Glob。
        question:   想从图片获取什么信息，可针对性指定，例如：
                    - "找出这个UI界面的设计问题和改进建议"
                    - "提取图中所有可见文字（OCR），原样输出"
                    - "这是报错截图，请说明错误信息和可能原因"
                    - "描述这个图表的数据趋势和关键数字"
    """
    t0 = time.time()
    try:
        # 先解析路径：latest / latest:N 关键字 → 实际文件路径
        resolved_path = _resolve_image_path(image_path)
        # _encode_image 也纳入 try：图片不存在/损坏/格式不支持等错误也要被结构化处理，
        # 否则会抛裸 traceback 给 MCP 框架，主模型看到后可能误判去用 Read 兜底。
        b64, mime = _encode_image(resolved_path)
        resp = client.chat.completions.create(
            model=VISION_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": question + "\n\n请简洁回答，控制在500字以内。"},
                    ],
                }
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=28,  # ZCode 工具执行硬超时 30s，留 2s 余量主动放弃，避免被外层硬掐导致错误信息不清
            # 关闭思考模式：视觉识别任务不需要推理链，关闭后响应速度大幅提升（实测2~3s）
            extra_body={"thinking": {"type": "disabled"}},
        )
        elapsed = time.time() - t0
        sys.stderr.write(
            f"[vision] ok req={image_path} actual={resolved_path} img_bytes~{len(b64)*3//4} "
            f"tokens={getattr(resp, 'usage', None) and resp.usage.total_tokens} "
            f"elapsed={elapsed:.1f}s\n"
        )
        return resp.choices[0].message.content
    except Exception as e:
        elapsed = time.time() - t0
        sys.stderr.write(f"[vision] FAIL req={image_path} elapsed={elapsed:.1f}s err={type(e).__name__}: {e}\n")
        # 返回结构化错误文本，让主模型能理解并告知用户，而不是看到裸 traceback 去尝试 Read 兜底
        return (
            f"[vision_describe 调用失败] 类型={type(e).__name__} 信息={e} 耗时={elapsed:.1f}s\n"
            "可能原因：图片路径不存在/格式不支持、视觉模型超时、网络抖动、API 限流。\n"
            "建议处理：直接告诉用户\"读图失败\"，询问是否重试、换图片或换方式。"
            "⛔ 不要用 Read 工具读取该图片，会损坏纯文本主模型的会话。"
        )


# ---------------------------------------------------------------------------
# 网页读取 / GitHub 仓库读取：纯本地实现，复用 httpx/selectolax
# ---------------------------------------------------------------------------
# 这两个工具不走 AI 模型，纯本地 HTTP 抓取 + 解析，零额外成本。
# 设计原则和上面的 AI 工具一致：超时主动放弃、结构化错误、stderr 日志。

# web_reader 要剔除的标签：导航/页脚/脚本/样式/广告等噪声，保留正文
_NOISE_TAGS = (
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "footer", "header", "aside", "form", "button",
)
# 正文候选标签：按优先级从高到低，命中第一个就提取它
_ARTICLE_TAGS = ("article", "main", "[role='main']", ".post-content", ".article-content", ".entry-content", "#content")


def _extract_main_text(html: str) -> tuple[str, str]:
    """从 HTML 中提取正文文本和标题。

    用 selectolax 解析：先按优先级找 article/main 等语义标签，命中则只取其内容；
    都没命中则退到 body 全文。剔除 script/nav/footer 等噪声标签后，按段落聚合。

    返回 (title, body_text)。title 取 <title> 或第一个 <h1>，都没有则空串。
    """
    tree = HTMLParser(html)

    # 提取标题
    title = ""
    title_node = tree.css_first("title")
    if title_node and title_node.text():
        title = title_node.text().strip()
    if not title:
        h1 = tree.css_first("h1")
        if h1 and h1.text():
            title = h1.text().strip()

    # 先删噪声标签（无论后面取哪个容器，都不该有脚本/样式混进来）
    for tag in _NOISE_TAGS:
        for node in tree.css(tag):
            node.decompose()

    # 按优先级找正文容器
    body_node = None
    for selector in _ARTICLE_TAGS:
        body_node = tree.css_first(selector)
        if body_node:
            break
    if body_node is None:
        body_node = tree.css_first("body") or tree

    # 按段落聚合：块级标签后加换行，内联标签直接拼文本
    # selectolax 的 text() 会丢结构，改用遍历：块级标签之间插 \n\n
    paragraphs: list[str] = []
    block_tags = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                  "section", "article", "blockquote", "pre", "tr", "br"}

    def walk(node) -> None:
        # selectolax 遍历：对每个子节点，先递归再判断是否块级
        for child in node.iter():
            tag = child.tag
            txt = child.text(strip=True) if hasattr(child, "text") else ""
            if tag in block_tags and txt:
                paragraphs.append(txt)
            elif txt and tag not in block_tags:
                # 内联文本（如直接在 div 里的裸文本节点），合并到最后一段或新建
                if paragraphs and not paragraphs[-1].endswith(("\n", " ")):
                    paragraphs[-1] = paragraphs[-1] + " " + txt
                else:
                    paragraphs.append(txt)

    walk(body_node)
    # 去重相邻空段、合并
    body_text = "\n\n".join(p for p in paragraphs if p)
    # 压缩多余空行
    while "\n\n\n" in body_text:
        body_text = body_text.replace("\n\n\n", "\n\n")
    return title, body_text.strip()


@mcp.tool()
def web_reader(url: str, max_chars: int = WEB_READER_MAX_CHARS) -> str:
    """【网页正文抓取】抓取指定 URL 的网页正文，返回干净的文本（非 AI 提炼，保真）。
    当用户需要读取某个网页、API 文档、博客文章、新闻内容时调用本工具。

    ✅ 适合场景：
    - "读一下这个网页说了啥" / "把这个 API 文档抓回来"
    - 需要网页原文细节（命令、配置、代码）—— 本工具不做 AI 提炼，原样返回正文
    - 替代 WebFetch（WebFetch 会二次模型提炼丢细节）

    ❌ 不适合场景：
    - JS 动态渲染的 SPA 页面（本工具只抓服务端返回的 HTML，不执行 JS）
    - 需要登录的页面（不带 cookie）
    - 图片/视频内容（只返回文本）

    参数：
        url:       要抓取的完整 URL（含 http:// 或 https://）
        max_chars: 返回正文的最大字符数，超出截断并标注。默认 8000（避免撑爆主控上下文）
                   长文档建议调大（如 20000），短页面可调小。
    """
    if not HAS_WEB_READER:
        return (
            "[web_reader 不可用] 未安装 httpx/selectolax 依赖。\n"
            "修复：在项目目录运行 `uv add httpx selectolax` 后重启 MCP server。"
        )

    # 简单 URL 校验，避免明显误用
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return f"[web_reader 参数错误] URL 不合法: {url}（需含 http:// 或 https:// 和域名）"
    if parsed.scheme not in ("http", "https"):
        return f"[web_reader 参数错误] 仅支持 http/https，收到: {parsed.scheme}"

    t0 = time.time()
    try:
        headers = {
            "User-Agent": WEB_READER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        # httpx 跟踪重定向，超时分别配置连接和读取
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(WEB_READER_TIMEOUT, connect=8.0),
            headers=headers,
        ) as http_client:
            resp = http_client.get(url)
            resp.raise_for_status()
            # selectolax 需要 bytes 或 str，按 charset 解码
            html = resp.text

        title, body = _extract_main_text(html)
        elapsed = time.time() - t0
        final_url = str(resp.url)  # 跟踪最终 URL（重定向后）
        redirect_note = ""
        if final_url != url:
            redirect_note = f"\n(重定向到: {final_url})"

        if not body:
            sys.stderr.write(f"[web_reader] WARN url={url} 提取正文为空\n")
            return (
                f"[web_reader] 抓取成功但正文为空。\n"
                f"URL: {url}{redirect_note}\n"
                f"标题: {title or '(无)'}\n"
                f"可能原因：页面是 JS 动态渲染、或正文在 iframe 中。建议改用 WebFetch 或浏览器查看。"
            )

        # 截断超长正文
        truncated = False
        if len(body) > max_chars:
            body = body[:max_chars] + f"\n\n...(已截断，原文共 {len(body)} 字，如需完整内容请调大 max_chars)"
            truncated = True

        sys.stderr.write(
            f"[web_reader] ok url={url} title={title[:40]!r} "
            f"chars={len(body)}{'(truncated)' if truncated else ''} "
            f"elapsed={elapsed:.1f}s\n"
        )
        # 输出格式：标题 + URL + 正文，方便主控引用
        header = f"# {title or '(无标题)'}\nURL: {url}{redirect_note}\n"
        return header + body
    except httpx.HTTPStatusError as e:
        elapsed = time.time() - t0
        sys.stderr.write(f"[web_reader] FAIL url={url} status={e.response.status_code} elapsed={elapsed:.1f}s\n")
        return (
            f"[web_reader 抓取失败] HTTP {e.response.status_code} {e.response.reason_phrase}\n"
            f"URL: {url}\n"
            f"可能原因：页面需登录、被反爬、或 URL 错误。建议检查 URL 或改用浏览器查看。"
        )
    except httpx.TimeoutException:
        elapsed = time.time() - t0
        sys.stderr.write(f"[web_reader] FAIL url={url} timeout elapsed={elapsed:.1f}s\n")
        return (
            f"[web_reader 抓取超时] URL: {url} 耗时={elapsed:.1f}s\n"
            "可能原因：目标站点慢、网络问题。建议重试或换网络。"
        )
    except Exception as e:
        elapsed = time.time() - t0
        sys.stderr.write(f"[web_reader] FAIL url={url} elapsed={elapsed:.1f}s err={type(e).__name__}: {e}\n")
        return (
            f"[web_reader 抓取失败] 类型={type(e).__name__} 信息={e}\n"
            f"URL: {url}\n"
            f"建议：检查 URL 是否正确、网络是否通畅。"
        )


# ---------------------------------------------------------------------------
# GitHub 仓库读取：复用 httpx，调公开 REST API
# ---------------------------------------------------------------------------
# 公开仓库免 token，限流 60 次/小时/IP；配了 GITHUB_TOKEN 提到 5000 次/小时。
# 三种 action 复用同一个工具，避免工具爆炸：
#   search_doc        → 搜索仓库的 README/issue/pr 概览（轻量，不抓全文）
#   get_repo_structure → 列出仓库目录树
#   read_file         → 读单个文件内容

def _github_headers() -> dict:
    """构造 GitHub API 请求头。配了 token 就带，没配走匿名。"""
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": WEB_READER_USER_AGENT,
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _parse_repo(repo: str) -> tuple[str, str]:
    """解析 owner/repo 形式。支持 'owner/repo'、'https://github.com/owner/repo'、
    'https://github.com/owner/repo/blob/main/...'(自动截取前两段)。
    """
    repo = repo.strip()
    if repo.startswith("http"):
        # 从 URL 提取 owner/repo
        parsed = urlparse(repo)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"无法从 URL 解析 owner/repo: {repo}")
        return parts[0], parts[1]
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"repo 格式应为 'owner/repo'，收到: {repo}")
    return parts[0], parts[1]


@mcp.tool()
def github_repo(action: str, repo: str, path: str = "", query: str = "") -> str:
    """【GitHub 仓库读取】读取 GitHub 公开仓库的结构与文件内容，无需 clone。
    替代智谱 ZRead MCP。当用户需要"看一下某开源项目结构"、"读某仓库的某文件"、
    "了解某开源库"时调用。

    ✅ 适合场景：
    - "看一下 facebook/react 的目录结构" → action="get_repo_structure"
    - "读 langchain-ai/langchain 的 pyproject.toml" → action="read_file", path="pyproject.toml"
    - "huggingface/transformers 这个仓库大概是干啥的" → action="search_doc"
    - "vuejs/core 最近有哪些 issue?" → action="search_doc", query="recent issues"

    参数：
        action: 要执行的操作，三选一：
                - "search_doc"        → 仓库概览：README 摘要 + 最近 issue/pr 标题（轻量，不抓全文）
                - "get_repo_structure" → 列出指定目录的文件/子目录列表
                - "read_file"         → 读取仓库中指定文件的完整内容
        repo:   仓库标识，三种格式都支持：
                - "owner/repo"（如 "facebook/react"）
                - 完整 URL（如 "https://github.com/facebook/react"）
                - 文件 URL 也会自动截取（如 "https://github.com/facebook/react/blob/main/README.md"）
        path:   仅 action="get_repo_structure" 或 "read_file" 时使用：
                - get_repo_structure: 目录路径，空串表示根目录（如 "src" 或 "packages/core"）
                - read_file: 文件完整路径（如 "src/index.ts" 或 "README.md"）
        query:  仅 action="search_doc" 时可选：额外筛选 issue/pr 的关键词。
                空串则返回最近的 issue/pr 概览。

    ⚠️ 限流说明：免 token 模式 60 次/小时/IP（偶尔查够用）。
       配置环境变量 GITHUB_TOKEN 可提到 5000 次/小时，且支持私有仓库。
    """
    if not HAS_WEB_READER:
        return (
            "[github_repo 不可用] 未安装 httpx 依赖。\n"
            "修复：运行 `uv add httpx` 后重启 MCP server。"
        )

    t0 = time.time()
    try:
        owner, name = _parse_repo(repo)
    except ValueError as e:
        return f"[github_repo 参数错误] {e}"

    headers = _github_headers()
    # 限制单次请求耗时，GitHub API 通常 1-3s 返回
    timeout = httpx.Timeout(20.0, connect=8.0)

    try:
        if action == "search_doc":
            # 仓库概览：基本信息 + README 前 3000 字 + 最近 5 个 open issue 标题
            with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
                # 1. 仓库基本信息（描述、star、语言等）
                meta_resp = c.get(f"{GITHUB_API_URL}/repos/{owner}/{name}")
                meta: dict = {}
                if meta_resp.status_code == 200:
                    meta = meta_resp.json()
                elif meta_resp.status_code == 404:
                    return f"[github_repo] 仓库不存在: {owner}/{name}"
                else:
                    meta_resp.raise_for_status()

                # 2. README（取前 3000 字，避免太长撑爆上下文）
                readme_resp = c.get(
                    f"{GITHUB_API_URL}/repos/{owner}/{name}/readme",
                    headers={**headers, "Accept": "application/vnd.github.raw"},
                )
                readme = ""
                if readme_resp.status_code == 200:
                    readme = readme_resp.text[:3000]
                    if len(readme_resp.text) > 3000:
                        readme += "\n\n...(README 已截断，完整内容用 read_file 读取)"

                # 3. 最近 5 个 open issue
                # 用 /repos/{owner}/{repo}/issues 端点（匿名可用），
                # 不用 /search/issues（匿名常返回 422，搜索 API 需认证）
                issues_params = {"state": "open", "per_page": 5, "sort": "updated", "direction": "desc"}
                if query:
                    # query 作为关键词：GitHub issues 端点不支持 q 参数，
                    # 用 labels 参数匹配；query 不是 label 时降级为不筛选
                    issues_params["labels"] = query
                issues_resp = c.get(
                    f"{GITHUB_API_URL}/repos/{owner}/{name}/issues",
                    params=issues_params,
                )
                issues_text = ""
                if issues_resp.status_code == 200:
                    items = issues_resp.json()
                    # issues 端点会同时返回 issue 和 PR（用 pull_request 字段区分）。
                    # 按 updated 排序时 PR 往往占多数——这是正常的，PR 活跃度高于 issue。
                    # 保留混合显示，标注类型，让用户看到仓库最近的真实活跃度。
                    if items:
                        lines = []
                        for it in items:
                            kind = "PR" if "pull_request" in it else "issue"
                            lines.append(f"  #{it['number']} [{kind}] {it['title']}")
                        issues_text = "\n最近活跃的 issue/PR:\n" + "\n".join(lines)
                    else:
                        issues_text = "\n最近活跃的 issue/PR: (无)"
                # 422/422 等错误静默降级：issue 查询失败不影响仓库概览主体
                elif issues_resp.status_code not in (422, 404):
                    issues_resp.raise_for_status()

            elapsed = time.time() - t0
            sys.stderr.write(
                f"[github_repo] ok action=search_doc repo={owner}/{name} "
                f"readme_chars={len(readme)} elapsed={elapsed:.1f}s\n"
            )
            # 组装概览
            return (
                f"# {owner}/{name}\n"
                f"描述: {meta.get('description', '(无)')}\n"
                f"语言: {meta.get('language', '(未知)')} | "
                f"Stars: {meta.get('stargazers_count', '?')} | "
                f"Forks: {meta.get('forks_count', '?')} | "
                f"Open issues: {meta.get('open_issues_count', '?')}\n"
                f"主页: {meta.get('html_url', '')}\n"
                f"默认分支: {meta.get('default_branch', 'main')}\n"
                f"\n--- README (前3000字) ---\n{readme or '(无 README)'}"
                f"{issues_text}"
            )

        elif action == "get_repo_structure":
            # 列目录：用 contents API（path 空表示根目录）
            target_path = path.strip("/")
            url = f"{GITHUB_API_URL}/repos/{owner}/{name}/contents/{target_path}"
            with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
                resp = c.get(url)
                if resp.status_code == 404:
                    return f"[github_repo] 路径不存在: {owner}/{name}/{target_path or '(根)'}"
                resp.raise_for_status()
                entries = resp.json()

            # contents API 返回数组=目录，对象=文件
            if isinstance(entries, dict):
                # 单文件，返回其元信息
                return (
                    f"path 是文件不是目录: {entries.get('path')}\n"
                    f"如需读取内容，改用 action='read_file'。"
                )
            # 排序：目录在前，文件在后，各自按名字
            entries.sort(key=lambda e: (e["type"] != "dir", e["name"]))
            lines = [f"{'📁' if e['type']=='dir' else '📄'} {e['name']}" for e in entries]
            elapsed = time.time() - t0
            sys.stderr.write(
                f"[github_repo] ok action=get_repo_structure repo={owner}/{name} "
                f"path={target_path or '(根)'} count={len(entries)} elapsed={elapsed:.1f}s\n"
            )
            return (
                f"# {owner}/{name}/{target_path or ''}\n"
                f"共 {len(entries)} 项:\n" + "\n".join(lines)
            )

        elif action == "read_file":
            # 读文件：用 raw URL 直接拿原文，比 contents API 的 base64 省事
            if not path:
                return "[github_repo 参数错误] read_file 必须传 path（文件路径）"
            # 先拿默认分支（contents API 会自动处理分支，raw 需要显式分支）
            with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
                # 用 contents API 一次拿到内容和默认分支信息
                file_resp = c.get(
                    f"{GITHUB_API_URL}/repos/{owner}/{name}/contents/{path.strip('/')}"
                )
                if file_resp.status_code == 404:
                    return f"[github_repo] 文件不存在: {owner}/{name}/{path}"
                file_resp.raise_for_status()
                data = file_resp.json()
                if isinstance(data, list):
                    return (
                        f"[github_repo 参数错误] path 是目录不是文件: {path}\n"
                        f"如需列目录，改用 action='get_repo_structure'。"
                    )
                # data['content'] 是 base64，data.get('encoding')=='base64'；解码
                import base64 as b64mod
                content = ""
                if data.get("encoding") == "base64" and data.get("content"):
                    content = b64mod.b64decode(data["content"]).decode("utf-8", errors="replace")
                # 截断超长文件
                truncated = False
                if len(content) > 30000:
                    content = content[:30000] + f"\n\n...(文件已截断，原文共 {len(content)} 字符)"
                    truncated = True
                size_kb = data.get("size", 0) / 1024

            elapsed = time.time() - t0
            sys.stderr.write(
                f"[github_repo] ok action=read_file repo={owner}/{name} path={path} "
                f"chars={len(content)}{'(truncated)' if truncated else ''} "
                f"elapsed={elapsed:.1f}s\n"
            )
            return (
                f"# {owner}/{name}/{path}\n"
                f"大小: {size_kb:.1f} KB | 类型: {data.get('type','?')}\n"
                f"--- 内容 ---\n{content}"
            )

        else:
            return (
                f"[github_repo 参数错误] action 必须是 'search_doc' / 'get_repo_structure' / 'read_file'，"
                f"收到: {action}"
            )

    except httpx.HTTPStatusError as e:
        elapsed = time.time() - t0
        # GitHub 限流返回 403 + X-RateLimit-Remaining: 0
        remaining = e.response.headers.get("X-RateLimit-Remaining", "?")
        sys.stderr.write(
            f"[github_repo] FAIL action={action} repo={repo} status={e.response.status_code} "
            f"rate_remaining={remaining} elapsed={elapsed:.1f}s\n"
        )
        if e.response.status_code == 403 and remaining == "0":
            return (
                "[github_repo 限流] GitHub API 免 token 模式限流 60 次/小时/IP，已用尽。\n"
                "修复：配置环境变量 GITHUB_TOKEN（个人访问令牌）可提到 5000 次/小时。\n"
                f"或等待限流重置后重试。"
            )
        return (
            f"[github_repo 失败] HTTP {e.response.status_code}\n"
            f"URL: {e.request.url}\n"
            f"响应: {e.response.text[:200]}"
        )
    except Exception as e:
        elapsed = time.time() - t0
        sys.stderr.write(f"[github_repo] FAIL action={action} repo={repo} elapsed={elapsed:.1f}s err={type(e).__name__}: {e}\n")
        return (
            f"[github_repo 失败] 类型={type(e).__name__} 信息={e}\n"
            f"建议：检查 repo 格式是否为 'owner/repo'，网络是否通畅。"
        )


# ---------------------------------------------------------------------------
# 联网搜索：博查 Bocha API（火山云 coding plan 套餐不含搜索模型，走第三方）
# ---------------------------------------------------------------------------
# 探测结论：火山云 coding plan 套餐下，doubao-search / bot-res-search / browsing
# 全部报 "does not support the coding plan feature"，enable_search 参数也无效。
# 因此联网搜索走博查 API（open.bochaai.com），当前完全免费，独立于云厂商。
# 博查 API 格式（官方文档 + 官方 MCP 源码交叉验证）：
#   POST https://api.bochaai.com/v1/web-search
#   headers: Authorization: Bearer <key>, Content-Type: application/json
#   body: {"query": "...", "summary": true, "count": N, "freshness": "noLimit|oneDay|oneWeek|oneMonth|oneYear|YYYY-MM-DD|YYYY-MM-DD..YYYY-MM-DD"}
#   返回: {"data": {"webPages": {"value": [{name,url,snippet,summary,siteName,datePublished}]}}}

@mcp.tool()
def web_search(query: str, count: int = 8, freshness: str = "noLimit") -> str:
    """【联网搜索】搜索互联网实时信息，返回网页标题、URL、摘要、发布时间。
    当用户问"最新"、"最近"、"今天"、"搜一下"等需要实时信息的问题时调用。

    ✅ 适合场景：
    - "搜一下 XXX 的最新消息"
    - "今天 AI 圈有什么" / "XXX 最新版本是什么"
    - 需要时效性信息（deepseek_knowledge 的训练数据有截止日期）

    ❌ 不要调用的场景：
    - 纯客观常识（用 deepseek_knowledge）
    - 代码/项目相关问题（主模型自己答）
    - 你已经知道答案的问题（不要为搜而搜）

    参数：
        query:      搜索关键词或完整问题（博查会做语义理解）
        count:      返回结果数量，默认 8，范围 5-20。多不一定好，主控消化 8 条够用。
        freshness:  时间范围筛选，控制结果时效性。可选值：
                    - "noLimit"  不限时间（默认，搜全部历史）
                    - "oneDay"   最近一天（用户问"今天"/"昨天"时用）
                    - "oneWeek"  最近一周（用户问"最近"时用）
                    - "oneMonth" 最近一个月
                    - "oneYear"  最近一年
                    - "YYYY-MM-DD"               特定日期之后
                    - "YYYY-MM-DD..YYYY-MM-DD"   日期区间
                    用户明确要"最新"时建议传 oneWeek 或 oneDay。
    """
    if not BOCHA_API_KEY:
        return (
            "[web_search 未配置] 博查 API Key 未设置。\n"
            "联网搜索功能需要博查 API（当前完全免费，open.bochaai.com 注册即得，无需信用卡）。\n"
            "配置步骤：\n"
            "1. 访问 https://open.bochaai.com 注册并获取 API Key\n"
            "2. 在 MCP server 的 env 配置中加入 BOCHA_API_KEY=sk-xxxxx\n"
            "3. 重启 MCP server\n"
            "在配置完成前，对时效性问题请告知用户\"联网搜索未配置\"，或改用 deepseek_knowledge。"
        )

    # count 限幅（博查支持 1-50，限制在 5-20 对主控消化足够）
    count = max(5, min(20, int(count)))
    # freshness 白名单校验，非法值降级为 noLimit
    valid_freshness = {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}
    # 也支持 YYYY-MM-DD 和 YYYY-MM-DD..YYYY-MM-DD 格式（简单启发式判断）
    if freshness not in valid_freshness and not (
        len(freshness) == 10 and freshness[4] == "-" and freshness[7] == "-"
    ) and not (".." in freshness and len(freshness) >= 21):
        freshness = "noLimit"

    t0 = time.time()
    try:
        # 博查 API：POST，Authorization: Bearer <key>
        # body: {"query": ..., "summary": true, "count": N, "freshness": ...}
        # 返回: {"data": {"webPages": {"value": [{name,url,snippet,summary,siteName,datePublished}]}}}
        # 复用 httpx（web_reader 已经依赖它）
        with httpx.Client(
            timeout=httpx.Timeout(25.0, connect=8.0),
            headers={
                "Authorization": f"Bearer {BOCHA_API_KEY}",
                "Content-Type": "application/json",
            },
        ) as c:
            resp = c.post(
                BOCHA_API_URL,
                json={"query": query, "summary": True, "count": count, "freshness": freshness},
            )
            resp.raise_for_status()
            data = resp.json()

        # 博查返回结构：{"data": {"webPages": {"value": [...]}}}
        # 多重 fallback 以防字段名差异
        web_pages = (
            data.get("data", {}).get("webPages", {}).get("value")
            or data.get("data", {}).get("result")
            or data.get("result")
            or []
        )
        if not web_pages:
            sys.stderr.write(f"[web_search] WARN query={query!r} 无结果, resp_keys={list(data.keys())}\n")
            return (
                f"[web_search] 搜索无结果。\n"
                f"query: {query}\n"
                f"建议：换关键词重试，或改用 deepseek_knowledge 查客观事实。"
            )

        # 格式化结果：编号 + 标题 + URL + 来源 + 发布时间 + 摘要
        lines = []
        for i, item in enumerate(web_pages[:count], 1):
            title = item.get("name") or item.get("title") or "(无标题)"
            url = item.get("url") or item.get("link") or ""
            # 优先用 summary（长文摘要，需请求 summary:true），降级到 snippet（短摘要）
            snippet = (
                item.get("summary")
                or item.get("snippet")
                or item.get("description")
                or ""
            )
            # 摘要限长，单条太长会喧宾夺主
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            site = item.get("siteName") or item.get("source") or ""
            # 发布时间（博查返回 ISO 8601，截取日期部分让主控易读）
            date_pub = item.get("datePublished") or ""
            if date_pub and len(date_pub) >= 10:
                date_pub = date_pub[:10]  # 取 YYYY-MM-DD
            # 组装单条：标题 + URL + 元信息行 + 摘要
            meta_parts = []
            if site:
                meta_parts.append(f"来源: {site}")
            if date_pub:
                meta_parts.append(f"发布: {date_pub}")
            meta_line = f"   {' | '.join(meta_parts)}\n" if meta_parts else ""
            lines.append(
                f"{i}. {title}\n"
                f"   URL: {url}\n"
                f"{meta_line}"
                f"{f'   摘要: {snippet}' if snippet else ''}"
            )

        elapsed = time.time() - t0
        sys.stderr.write(
            f"[web_search] ok query={query!r} results={len(web_pages)} "
            f"freshness={freshness} elapsed={elapsed:.1f}s\n"
        )
        return f"# 搜索: {query}\n共 {len(web_pages)} 条结果:\n\n" + "\n\n".join(lines)

    except httpx.HTTPStatusError as e:
        elapsed = time.time() - t0
        sys.stderr.write(
            f"[web_search] FAIL query={query!r} status={e.response.status_code} "
            f"elapsed={elapsed:.1f}s body={e.response.text[:200]}\n"
        )
        if e.response.status_code == 401:
            return (
                "[web_search 认证失败] 博查 API Key 无效。\n"
                "请检查 BOCHA_API_KEY 是否正确，是否以 sk- 开头。"
            )
        if e.response.status_code == 429:
            return (
                "[web_search 限流] 博查免费额度每天 100 次已用尽。\n"
                "免费档次日自动重置；或升级博查付费套餐。"
            )
        return (
            f"[web_search 失败] HTTP {e.response.status_code}\n"
            f"响应: {e.response.text[:200]}"
        )
    except Exception as e:
        elapsed = time.time() - t0
        sys.stderr.write(f"[web_search] FAIL query={query!r} elapsed={elapsed:.1f}s err={type(e).__name__}: {e}\n")
        return (
            f"[web_search 失败] 类型={type(e).__name__} 信息={e}\n"
            "建议：检查网络、API Key 配置。"
        )


def main() -> None:
    """启动 MCP server。启动前校验必需环境变量，缺失时打印友好错误并退出。"""
    # 校验必需环境变量（ARK_API_KEY 在模块加载时已通过 os.environ[] 强制要求，
    # 这里再校验一次确保信息清晰）
    if not ARK_API_KEY:
        sys.stderr.write(
            "[multi-helper] 错误：未设置 ARK_API_KEY 环境变量。\n"
            "请通过环境变量传入你的 API key，例如：\n"
            "  Windows:   set ARK_API_KEY=your-key-here\n"
            "  Linux/Mac: export ARK_API_KEY=your-key-here\n"
            "或在 MCP 客户端的 env 配置里传入。详见 README.md。\n"
        )
        sys.exit(1)

    sys.stderr.write(
        f"[multi-helper] vision={VISION_MODEL_ID}, "
        f"max_side={MAX_IMAGE_SIDE}px, screenshot_dir={SCREENSHOT_DIR} @ {ARK_BASE_URL}\n"
    )
    if not HAS_PIL:
        sys.stderr.write("[multi-helper] Pillow 未安装，图片不会压缩。建议运行: uv add pillow\n")
    if not HAS_WEB_READER:
        sys.stderr.write("[multi-helper] httpx/selectolax 未安装，web_reader/github_repo 不可用。建议运行: uv add httpx selectolax\n")
    if not BOCHA_API_KEY:
        sys.stderr.write("[multi-helper] BOCHA_API_KEY 未配置，web_search 工具将返回未配置提示。\n")
    else:
        sys.stderr.write(f"[multi-helper] web_search 启用（博查 API @ {BOCHA_API_URL}）\n")
    mcp.run()


if __name__ == "__main__":
    main()
