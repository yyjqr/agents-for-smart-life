#!/bin/bash

echo "🚀 启动 NVIDIA NeMo Agent Toolkit AI智慧城市-道路信息机器人"
echo "=============================================="

# 获取项目根目录和NeMo目录
NEMO_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$NEMO_DIR")

# 设置环境变量
export TAVILY_API_KEY=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

# 激活Python虚拟环境
source .venv/bin/activate

# 启动后端服务
echo "📡 启动后端服务..."
nat serve --config_file configs/hackathon_config.yml --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

# 等待后端启动
echo "⏳ 等待后端服务启动..."
sleep 10

# 启动前端服务
echo "🎨 启动前端服务..."
cd "$PROJECT_ROOT/external/aiqtoolkit-opensource-ui"
npm run dev &
FRONTEND_PID=$!

# 返回NeMo目录
cd "$NEMO_DIR"

echo ""
echo "✅ 系统启动完成！"
echo ""
echo "🌐 访问地址:"
echo "   前端界面: http://localhost:3000"
echo "   API文档:  http://localhost:8001/docs"
echo ""
echo "📝 测试建议:"
echo "   1. 天气查询: '北京今天的天气怎么样，气温是多少？'"
echo "   2. 公司信息: '帮我介绍一下NVIDIA Agent Intelligence Toolkit'"
echo "   3. 时间查询: '现在几点了？'"
echo ""
echo "🛑 停止服务: 按 Ctrl+C 或运行 ./stop.sh"
echo ""

# 保存进程ID
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# 等待用户中断
wait
