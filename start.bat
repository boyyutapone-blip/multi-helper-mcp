@echo off
REM ============================================================
REM  Multi-Helper MCP Server 启动脚本
REM  通过环境变量配置 API key 和模型，不硬编码任何密钥
REM ============================================================

REM --- 必填：你的 API key（请通过环境变量 ARK_API_KEY 传入）---
REM 如果尚未设置，请在此处取消注释并填入你的 key，或改用客户端 MCP 配置传入
REM set "ARK_API_KEY=your-api-key-here"

REM --- 网关地址（默认火山云 coding plan，其他网关请改）---
if not defined ARK_BASE_URL set "ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3"

REM --- 模型 ID（默认值适配火山云 coding plan，其他网关请按需修改）---
if not defined VISION_MODEL_ID set "VISION_MODEL_ID=doubao-seed-2.0-lite"
if not defined KNOWLEDGE_MODEL_ID set "KNOWLEDGE_MODEL_ID=deepseek-v4-flash"
if not defined KNOWLEDGE_MODEL_DEEP_ID set "KNOWLEDGE_MODEL_DEEP_ID=deepseek-v4-pro"

cd /d "%~dp0"
uv run server.py
