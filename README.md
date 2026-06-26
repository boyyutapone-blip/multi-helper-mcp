# Multi-Helper MCP

让纯文本主控模型（如 GLM-5.2）通过 MCP 工具调用获得**视觉理解、世界知识、联网搜索、网页读取、开源仓库访问**五项外援能力。

## 为什么需要它

很多强大的编程模型（GLM-5.2、DeepSeek-Coder 等）是**纯文本模型**，没有视觉能力。一旦图片内容进入对话上下文，会导致会话永久损坏（`Model only support text input` 报错）。

这个 MCP server 让纯文本主控模型通过"工具调用"的方式间接使用多模态模型和外部能力：**图片、知识查询、搜索、抓网页、读仓库都作为工具参数传给 MCP server，server 调用专门模型或本地实现处理，返回纯文本结果给主控**。主控始终只处理文本，安全且稳定。

## 架构

```
┌─────────────────────────────────────────────────────┐
│  主控模型（纯文本，如 GLM-5.2）                      │
│  遇到图片 → vision_describe                         │
│  遇到客观事实 → deepseek_knowledge                  │
│  遇到复杂分析 → deepseek_knowledge_deep             │
│  遇到最新信息 → web_search                          │
│  遇到读网页 → web_reader                            │
│  遇到读开源仓库 → github_repo                       │
└────────────────────┬────────────────────────────────┘
                     ↓ MCP (stdio)
   ┌────┬────┬────┬────┬────┐
   ▼    ▼    ▼    ▼    ▼
  视觉  知识  知识  博查  本地
  模型  (快)  (深)  搜索  httpx
```

核心思想：**主控模型始终是主大脑，外援只是"工具"**。主控人格稳定、上下文连续，外援可插拔。

## 提供的工具

| 工具 | 作用 | 背后实现 |
|------|------|---------|
| `vision_describe(image_path, question)` | 分析本地图片，返回文字描述 | 视觉模型（如 doubao-seed-2.0-lite） |
| `deepseek_knowledge(query)` | 快速查询客观事实（人名、时间、定义） | 快速知识模型（如 deepseek-v4-flash） |
| `deepseek_knowledge_deep(query)` | 深度推理复杂知识问题 | 深度知识模型（如 deepseek-v4-pro） |
| `web_search(query, count, freshness)` | 联网搜索实时信息，支持时间范围筛选 | 博查 Bocha API（独立于云厂商，当前免费） |
| `web_reader(url, max_chars)` | 抓取网页正文，保真不提炼 | 本地 httpx + selectolax |
| `github_repo(action, repo, ...)` | 读取 GitHub 仓库结构与文件 | GitHub REST API（公开仓库免 token） |

`vision_describe` 支持 `image_path="latest"` 自动定位最新截图，`"latest:2"` 取第二新的，无需手动复制文件路径。

### 为什么联网搜索用博查而不是火山云？

实测结论：火山云 coding plan 套餐**不含任何搜索/browsing 模型**——`doubao-search`、`bot-res-search`、`doubao-pro-32k-browsing` 全部返回 `UnsupportedModel: does not support the coding plan feature`，`enable_search=True` 参数也无效（模型只是把它当指令回吐）。因此联网搜索走第三方博查 API（当前完全免费，独立于云厂商）。（火山云的agent plan好像是有提供联网搜索的，大家有其他方法可以试一下）

## 前置要求

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)**（Python 包管理器，安装后无需手动管虚拟环境）
- **任一 OpenAI 兼容网关的 API key**：
  - 火山云 coding plan（推荐，套餐内免费）
  - 智谱官方 BigModel
  - OpenRouter
  - 本地 Ollama / vLLM 等

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url> multi-helper-mcp
cd multi-helper-mcp
```

### 2. 安装依赖

```bash
uv sync
```

uv 会自动创建隔离环境并安装 `mcp[cli]`、`openai`、`pillow`。

### 3. 验证启动

```bash
# Windows
set ARK_API_KEY=your-key-here
uv run server.py

# Linux/Mac
export ARK_API_KEY=your-key-here
uv run server.py
```

看到类似下面的输出说明启动成功：
```
[multi-helper] vision=doubao-seed-2.0-lite, knowledge=deepseek-v4-flash(fast), ...
Running stdio MCP server
```

## 接入客户端

这个 MCP server 兼容任何支持 MCP 协议的客户端。以下给出 ZCode 和 Claude Code 的配置示例。

### ZCode

编辑 `~/.zcode/cli/config.json`，在 `mcp.servers` 里加入：

```json
{
  "mcp": {
    "servers": {
      "glm-router": {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "--directory", "/path/to/multi-helper-mcp", "server.py"],
        "env": {
          "ARK_API_KEY": "<your-api-key>",
          "ARK_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding/v3",
          "VISION_MODEL_ID": "doubao-seed-2.0-lite",
          "KNOWLEDGE_MODEL_ID": "deepseek-v4-flash",
          "KNOWLEDGE_MODEL_DEEP_ID": "deepseek-v4-pro"
        }
      }
    }
  }
}
```

> ZCode 还支持全局指令文件 `~/.zcode/AGENTS.md`，可以写规则约束主控模型正确调用本工具（例如"用户提到截图时直接调 vision_describe('latest')，不要用 Read 读图片"）。

### Claude Code

Claude Code 支持两种配置方式：

**方式 A：全局配置**（所有项目可用）

编辑 `~/.claude.json`，加入 `mcpServers`：

```json
{
  "mcpServers": {
    "glm-router": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/multi-helper-mcp", "server.py"],
      "env": {
        "ARK_API_KEY": "<your-api-key>",
        "ARK_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "VISION_MODEL_ID": "doubao-seed-2.0-lite",
        "KNOWLEDGE_MODEL_ID": "deepseek-v4-flash",
        "KNOWLEDGE_MODEL_DEEP_ID": "deepseek-v4-pro"
      }
    }
  }
}
```

**方式 B：项目级配置**（只在该项目可用）

在项目根目录创建 `.mcp.json`，内容同上。

**配合 CLAUDE.md 使用**：Claude Code 会读取 `~/.claude/CLAUDE.md`（全局）或项目 `CLAUDE.md`，可以在里面写规则约束主控模型正确调用本工具。

### 通用 MCP 配置

任何 MCP 客户端只要支持 stdio transport，都可以用以下配置接入：

```
command: uv
args: ["run", "--directory", "<path-to-this-repo>", "server.py"]
env:
  ARK_API_KEY: <your-api-key>
  ARK_BASE_URL: <your-gateway-url>
  VISION_MODEL_ID: <vision-model-id>
  KNOWLEDGE_MODEL_ID: <fast-knowledge-model-id>
  KNOWLEDGE_MODEL_DEEP_ID: <deep-knowledge-model-id>
```

## 配置项详解

所有配置通过环境变量传入，无硬编码默认密钥。完整模板见 `.env.example`。

### 必填

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `ARK_API_KEY` | 网关的 API key | `ark-xxxxx` / `sk-xxxxx` |

### 模型配置

| 环境变量 | 说明 | 默认值（火山云） | 要求 |
|----------|------|-----------------|------|
| `ARK_BASE_URL` | 网关地址 | `https://ark.cn-beijing.volces.com/api/coding/v3` | OpenAI 兼容 |
| `VISION_MODEL_ID` | 视觉模型 ID | `doubao-seed-2.0-lite` | 必须支持图片输入 |
| `KNOWLEDGE_MODEL_ID` | 快速知识模型 ID | `deepseek-v4-flash` | 文本模型即可 |
| `KNOWLEDGE_MODEL_DEEP_ID` | 深度知识模型 ID | `deepseek-v4-pro` | 带 reasoning 更佳 |

> **其他网关怎么填**：智谱官方用 `glm-4v-flash` 做视觉；OpenRouter 用 `openai/gpt-4o-mini` 做视觉；本地 Ollama 用 `llava:7b` 做视觉。知识模型同理，按你的网关支持的模型 ID 填。

### 可选调优

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `MAX_IMAGE_SIDE` | 图片压缩最长边像素 | `1024` |
| `JPEG_QUALITY` | JPEG 压缩质量（0-100） | `85` |
| `MAX_OUTPUT_TOKENS` | 视觉模型单次返回最大 token | `800` |
| `SCREENSHOT_DIR` | 截图目录（`latest` 关键字扫描范围） | 系统默认（`~/Pictures/Screenshots`） |

### 联网搜索（博查 Bocha）

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `BOCHA_API_KEY` | 博查 API Key，留空则 `web_search` 返回未配置提示 | 空（未配置） |
| `BOCHA_API_URL` | 博查搜索接口地址 | `https://api.bochaai.com/v1/web-search` |

**获取博查 API Key**：访问 https://open.bochaai.com 注册，当前完全免费（无需信用卡），Individual 档含 web searches 和长文摘要。独立于云厂商。

### 网页读取（web_reader）

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `WEB_READER_TIMEOUT` | 抓取网页超时（秒） | `20` |
| `WEB_READER_MAX_CHARS` | 返回正文最大字符数，超出截断 | `8000` |
| `WEB_READER_USER_AGENT` | 请求 UA，部分站点拒无 UA 请求 | 内置合理 UA |

### GitHub 仓库读取（github_repo）

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `GITHUB_TOKEN` | GitHub Personal Access Token，留空走匿名模式 | 空（匿名，60 次/小时） |

**匿名模式**：公开仓库免 token，限流 60 次/小时/IP，偶尔查够用。
**配 Token**：申请 https://github.com/settings/tokens（classic token 勾 `public_repo`），提到 5000 次/小时，且支持私有仓库。

## 使用方式

接入客户端后，正常和主控模型对话即可。主控模型会自动判断何时调用工具：

- **发图片/截图/问看图问题**：主控调用 `vision_describe`，图片在 MCP server 里被视觉模型分析，返回文字给主控
- **问客观事实**（人名、时间、定义）：主控调用 `deepseek_knowledge`
- **问复杂分析**（多步推理、对比）：主控调用 `deepseek_knowledge_deep`
- **问最新信息**（"最近"、"今天"、"搜一下"）：主控调用 `web_search`（需配博查 Key）
- **读网页正文**（"读一下这个 URL"）：主控调用 `web_reader`，原样抓正文不经过 AI 提炼
- **读开源仓库**（"看下某项目结构"、"读某仓库某文件"）：主控调用 `github_repo`
- **写代码/项目相关**：主控自己处理，不绕工具

### `latest` 关键字用法

`vision_describe` 的 `image_path` 参数支持：
- 绝对路径：`C:\Users\xxx\screenshot.png`
- `"latest"`：自动取截图目录下最新的图片
- `"latest:N"`：取第 N 新的截图（`latest:2` = 第二新，`latest:3` = 第三新）

用户说"看最新截图"时，主控模型应直接调 `vision_describe("latest")`，无需手动复制路径。

## 常见问题

### Q: 工具调用超时怎么办？

MCP 客户端通常有 30 秒工具执行超时。server.py 内部超时设为 28 秒主动放弃，避免被外层硬掐。如果频繁超时：
- 视觉超时：检查 `VISION_MODEL_ID` 是否选了快速模型（如 `doubao-seed-2.0-lite` 关闭思考模式）
- 知识超时：`deepseek_knowledge_deep` 的 reasoning budget 默认 3000，复杂问题约 16 秒，已留足余量

### Q: 提示 "Model only support text input"？

说明主控模型直接接触到了图片字节，会话已损坏。这通常是因为主控用了 Read 工具读图片而非调用 `vision_describe`。解决：
- 在客户端的全局指令文件里写规则禁止 Read 读图片（ZCode: `~/.zcode/AGENTS.md`，Claude Code: `~/.claude/CLAUDE.md`）
- 当前会话只能废弃，开新会话

### Q: 截图目录在哪？

`SCREENSHOT_DIR` 环境变量配置。默认值：
- Windows: `C:\Users\<用户名>\Pictures\Screenshots`
- macOS: `~/Pictures/Screenshots`
- Linux: `~/Pictures`

如果你的截图保存在别处（如 OneDrive 同步目录），请通过环境变量指定。

### Q: 不想用三个模型，可以只用一个吗？

可以。把 `KNOWLEDGE_MODEL_ID` 和 `KNOWLEDGE_MODEL_DEEP_ID` 填成同一个模型即可，效果会有差异但功能正常。如果只用视觉不用知识，知识工具仍会被注册但你可以忽略。

## 限制说明

- **视觉描述是文字，不是"看"**：对截图报错、图表数据、OCR、布局结构等信息型任务足够；对配色/间距/微调等审美型任务有局限（描述过程会丢视觉信息）
- **知识工具有上下文隔离**：知识模型看不到主控的对话历史，只看到 `query` 参数。适合独立于上下文的客观事实查询，不适合依赖对话上下文的分析
- **知识回复经主控转述**：知识模型的回答回到主控手里后，主控可能提炼或加工。如需原样保留，可在客户端指令里要求主控"原样转述知识工具的回复"

## 文件说明

```
multi-helper-mcp/
├── server.py          # MCP server 主程序（三个工具）
├── start.bat          # Windows 启动脚本（需自行设置 ARK_API_KEY）
├── pyproject.toml     # 依赖声明和 entry point
├── uv.lock            # 依赖版本锁
├── .env.example       # 配置模板（复制为 .env 或参考填入客户端 env）
├── .gitignore
├── LICENSE            # MIT
└── README.md          # 本文件
```

## License

MIT
