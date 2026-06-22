"""
Multi-Helper MCP Server
=======================

让纯文本主控模型（如 GLM-5.2）通过 MCP 工具调用获得"视觉理解"和"世界知识"两项外援能力。

核心思路：主控模型始终只处理文本，图片和知识查询通过工具调用委托给专门模型，
工具返回纯文本结果给主控。主控人格稳定、上下文连续，外援可插拔。

提供三个工具：
  vision_describe(image_path, question)        → 视觉模型看图返回文字描述
  deepseek_knowledge(query)                    → 快速知识模型回答客观事实
  deepseek_knowledge_deep(query)               → 深度推理模型处理复杂分析

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
import os
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Pillow 用于图片压缩/缩放，加速视觉 API 调用
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    sys.stderr.write("[multi-helper] 警告：未安装Pillow，图片不会被压缩，大图可能超时。\n")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")  # 必填，main() 启动时会校验
ARK_BASE_URL = os.environ.get(
    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"
)
VISION_MODEL_ID = os.environ.get("VISION_MODEL_ID", "doubao-seed-2.0-lite")
KNOWLEDGE_MODEL_ID = os.environ.get("KNOWLEDGE_MODEL_ID", "deepseek-v4-flash")  # 日常知识：Flash 快
KNOWLEDGE_MODEL_DEEP_ID = os.environ.get("KNOWLEDGE_MODEL_DEEP_ID", "deepseek-v4-pro")  # 深度知识：Pro 强
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))  # 最长边1024px，识别截图足够清晰且快
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "800"))  # 详细描述需要更长输出
# 截图目录：vision_describe 支持 image_path="latest" 自动取最新截图，
# 此目录是扫描范围。默认 Windows 标准截图路径，可通过环境变量覆盖。
SCREENSHOT_DIR = os.environ.get(
    "SCREENSHOT_DIR",
    os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots"),
)

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


@mcp.tool()
def deepseek_knowledge(query: str) -> str:
    """【客观事实查询】查询独立于对话上下文的客观事实：人名、时间、定义、术语解释。
    使用 DeepSeek V4 Flash 模型，响应快（~2s）。

    ⚠️ 严格限定场景——只在以下情况调用，其他情况 GLM 自己答：
    - 纯客观事实：某人生卒年、某事件时间、某术语的准确定义、某概念的官方解释
    - 不依赖当前对话上下文：问题本身就是完整的，DeepSeek 不需要知道之前聊了什么

    ❌ 不要调用的场景（GLM 自己能答，且答得更好）：
    - 需要结合当前对话/项目上下文的推理分析（DeepSeek 看不到上下文，答得通用反而不如你）
    - 复杂分析题、对比题、因果推理题（你给足 reasoning 空间能答得一样深，且贴合场景）
    - 代码、调试、项目上下文问题（这是你的主场）
    - 最新时事、新闻（DeepSeek 训练数据也有截止日期，不一定比你知道得多；真要最新信息用联网搜索）

    调用原则：如果你（GLM）自己能在合理时间内给出准确答案，就不要调这个工具。
    只有当你确实不掌握某个客观事实、且该事实独立于上下文时才调。

    参数：
        query: 完整自洽的问题，不依赖任何未提供的上下文。
    """
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=KNOWLEDGE_MODEL_ID,  # Flash：快，日常知识足够
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是知识外援，被一个主控编程模型（GLM-5.2）调用，负责提供它不掌握的世界知识。"
                        "请遵守：\n"
                        "1. 聚焦事实：直接回答问题，给关键事实、数据、定义、时间、人名等具体信息。\n"
                        "2. 简洁：除非用户明确要求展开，否则控制在 300 字以内，不要寒暄、不要重复问题。\n"
                        "3. 拒绝越界：被问代码编写、调试、当前项目上下文时，直接说\"这属于主模型职责，请自行处理\"，不要尝试回答。\n"
                        "4. 不确定时明说：拿不准的事实标注\"不确定\"，不要编造。\n"
                        "5. 输出纯文本：不要用 markdown 标题/代码块包装，方便主模型直接消化。"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=600,
            timeout=28,
            # Flash 是非推理模型，不需要 thinking 参数
        )
        elapsed = time.time() - t0
        sys.stderr.write(
            f"[knowledge] ok model={KNOWLEDGE_MODEL_ID} "
            f"tokens={getattr(resp, 'usage', None) and resp.usage.total_tokens} "
            f"elapsed={elapsed:.1f}s\n"
        )
        return resp.choices[0].message.content
    except Exception as e:
        elapsed = time.time() - t0
        sys.stderr.write(f"[knowledge] FAIL elapsed={elapsed:.1f}s err={type(e).__name__}: {e}\n")
        return (
            f"[deepseek_knowledge 调用失败] 类型={type(e).__name__} 信息={e} 耗时={elapsed:.1f}s\n"
            "可能原因：知识模型超时、网络抖动、API 限流。\n"
            "建议处理：直接告诉用户\"知识查询失败\"，询问是否重试，或由主模型（你）基于自身知识作答。"
        )


@mcp.tool()
def deepseek_knowledge_deep(query: str) -> str:
    """【客观事实深度查询】需要深度推理的客观知识问题，但仍然独立于对话上下文。
    使用 DeepSeek V4 Pro 模型 + reasoning budget=3000，响应较慢（~16s）但更深。

    ⚠️ 严格限定场景——这个工具的存在价值非常窄，绝大多数情况下 GLM 自己答更好：
    - GLM 给足 reasoning 空间能答得一样深，且贴合对话上下文
    - DeepSeek 看不到对话历史，答得"通用"而非"针对你的具体场景"
    - DeepSeek 的回复经 GLM 转述会有损耗

    仅在这些情况调用（同时满足才调）：
    1. 问题是客观知识、独立于上下文（不依赖当前对话/项目）
    2. 需要深度推理/学术级分析（简单事实用 deepseek_knowledge 即可）
    3. GLM 自己确实不掌握相关知识（不是"答得慢"，是"真不知道"）

    ❌ 不要调用的场景（即使看起来像"复杂知识问题"也不要调）：
    - 需要结合当前对话上下文的分析（GLM 自己答更贴合）
    - 代码架构对比、技术选型分析（GLM 主场，自己答）
    - 任何能用联网搜索解决的最新信息问题

    调用原则：如果你（GLM）自己能在给足 reasoning 空间后答出同等深度的答案，
    就不要调这个工具。只在"客观、深度、你真不知道"三者同时满足时才调。

    参数：
        query: 完整自洽的问题，包含必要的背景信息和推理要求，不依赖任何未提供的上下文。
    """
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=KNOWLEDGE_MODEL_DEEP_ID,  # Pro：带 reasoning，深度强
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是深度知识外援，被一个主控编程模型（GLM-5.2）调用，处理需要深度推理的复杂知识问题。"
                        "请遵守：\n"
                        "1. 深度优先：可以展开推理过程，给出原理、机制、因果链。\n"
                        "2. 结构清晰：复杂分析可用编号列表，但不要用 markdown 标题。\n"
                        "3. 拒绝越界：被问代码编写、调试、当前项目上下文时，直接说\"这属于主模型职责\"。\n"
                        "4. 不确定时明说：拿不准的标注\"不确定\"，不要编造。\n"
                        "5. 长度适中：控制在 600 字以内，够深但不冗长。"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=1200,  # 深度版允许更长输出
            timeout=28,
            # reasoning budget：实测 budget=2000 复杂问题会超时，budget=4000 卡 25s 危险边缘，
            # budget=3000 实测 ~16s，既有信息论级别的深度分析，又留足安全余量
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 3000}},
        )
        elapsed = time.time() - t0
        sys.stderr.write(
            f"[knowledge_deep] ok model={KNOWLEDGE_MODEL_DEEP_ID} "
            f"tokens={getattr(resp, 'usage', None) and resp.usage.total_tokens} "
            f"elapsed={elapsed:.1f}s\n"
        )
        return resp.choices[0].message.content
    except Exception as e:
        elapsed = time.time() - t0
        sys.stderr.write(f"[knowledge_deep] FAIL elapsed={elapsed:.1f}s err={type(e).__name__}: {e}\n")
        return (
            f"[deepseek_knowledge_deep 调用失败] 类型={type(e).__name__} 信息={e} 耗时={elapsed:.1f}s\n"
            "可能原因：深度模型超时、网络抖动、API 限流。\n"
            "建议处理：直接告诉用户\"深度知识查询失败\"，询问是否重试，或改用 deepseek_knowledge（Flash 快速版）。"
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
        f"knowledge={KNOWLEDGE_MODEL_ID}(fast), knowledge_deep={KNOWLEDGE_MODEL_DEEP_ID}(pro), "
        f"max_side={MAX_IMAGE_SIDE}px, screenshot_dir={SCREENSHOT_DIR} @ {ARK_BASE_URL}\n"
    )
    if not HAS_PIL:
        sys.stderr.write("[multi-helper] Pillow 未安装，图片不会压缩。建议运行: uv add pillow\n")
    mcp.run()


if __name__ == "__main__":
    main()
