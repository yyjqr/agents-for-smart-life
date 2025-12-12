# 路侧场景分析 - 快速入门指南

## 功能概述

本插件为NVIDIA NeMo Agent Toolkit提供完整的路侧场景图片分析解决方案，支持：

✅ **多种图片上传方式**
- 本地文件路径
- 远程URL链接  
- Base64编码数据

✅ **智能图像理解**
- 集成千问(Qwen)视觉模型
- 交通状况分析（拥堵、流量、信号灯等）
- 环境信息识别（建筑物、设施、人流等）
- 天气条件分析（能见度、气象等）

✅ **位置和时间记录**
- 自动记录GPS坐标
- 时间戳记录
- 设备ID标识

✅ **信息存储和共享**
- 本地数据库存储
- 位置查询能力
- 多设备数据汇聚
- 交通热点识别

---

## 安装步骤

### 1. 前置条件

```bash
# Python >= 3.10
python --version

# 已安装nvidia-nat
pip list | grep nvidia-nat
```

### 2. 安装插件

```bash
cd examples/custom_functions/road_scene_analysis
pip install -e .
```

### 3. 配置API密钥

从阿里云DashScope获取API密钥，然后设置环境变量：

```bash
# Linux/Mac
export DASHSCOPE_API_KEY="sk-your-api-key-here"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-your-api-key-here"

# Windows CMD
set DASHSCOPE_API_KEY=sk-your-api-key-here
```

---

## 使用方式

### 方式一：与Agent集成（推荐）

1. **配置Agent**：修改 `configs/hackathon_config.yml`

```yaml
functions:
  road_scene_analyzer:
    _type: road_scene_analyzer
    llm_name: default_llm
    
  traffic_info_storage:
    _type: traffic_info_storage
    storage_path: "./data/traffic_info"
    
  traffic_info_query:
    _type: traffic_info_query
    storage_path: "./data/traffic_info"

llms:
  default_llm:
    _type: openai
    model_name: "qwen-vl-plus"
    api_key: "${DASHSCOPE_API_KEY}"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

2. **启动Agent**

```bash
# 使用NeMo Agent CLI
python -m nat.cli.runner configs/hackathon_config.yml
```

3. **与Agent交互**

```
用户: 分析这张北京路侧图片的交通状况
     图片: https://example.com/road_scene.jpg
     位置: 116.3975,39.9093

Agent回应:
- 使用road_scene_analyzer分析图片
- 识别交通拥堵/正常/流畅
- 使用traffic_info_storage存储结果
- 可选：使用traffic_info_query查询相关位置的历史信息
```

### 方式二：Python脚本调用

```python
import asyncio
from pathlib import Path
import sys

# 添加到路径
sys.path.insert(0, "examples/custom_functions/road_scene_analysis/src")

async def example():
    # 这里可以直接调用工具函数
    # 需要正确初始化Builder等环境
    pass

asyncio.run(example())
```

### 方式三：REST API服务器

```bash
# 启动API服务器（需要安装fastapi）
python examples/custom_functions/road_scene_analysis/api_server.py --port 8000

# 上传图片
curl -X POST "http://localhost:8000/api/v1/upload-image" \
  -F "file=@road_scene.jpg" \
  -F "location=116.3975,39.9093" \
  -F "device_id=device_001"

# 查询交通信息
curl "http://localhost:8000/api/v1/query?location=116.3975,39.9093&radius_km=5&hours=24"
```

---

## 使用案例

### 案例1：单个用户上报交通事件

```
场景：用户在北京发现交通事故
流程：
  1. 拍照并上传图片 + 位置信息
  2. Agent调用road_scene_analyzer分析图片
  3. 识别出"交通事故"
  4. 使用traffic_info_storage存储
  5. 返回"交通事故已记录，已推送给周边用户"

数据结构：
{
  "record_id": "abc12345",
  "location": "116.3975,39.9093",
  "device_id": "user_phone_001",
  "timestamp": "2025-12-05T10:30:00",
  "analysis": {
    "event_type": "accident",
    "severity": "high",
    "description": "两车追尾，占用一条车道"
  }
}
```

### 案例2：多设备协作信息共享

```
场景：多个设备共享交通信息
流程：
  1. 设备A在位置1报告拥堵
  2. 设备B在位置2报告拥堵
  3. 设备C在位置3报告正常
  4. 用户查询位置1周边5km的24小时数据
  5. Agent返回设备A、B的报告
  6. 用户可以评估交通流向和拥堵传播

分析维度：
  - 地理维度：拥堵的空间分布
  - 时间维度：拥堵的时间演变
  - 设备维度：哪些设备贡献了数据
```

### 案例3：交通热点识别

```
场景：系统自动识别交通热点区域
数据来源：多个设备在不同时间的上报
分析内容：
  - 识别重点拥堵路口
  - 估计拥堵持续时间
  - 推荐的绕行路线

输出：
{
  "hotspots": [
    {
      "location": "116.4,39.91",
      "congestion_rate": 0.85,  # 85%的报告显示拥堵
      "avg_severity": "high",
      "affected_devices": 12,
      "recommendation": "建议绕行东三环"
    }
  ]
}
```

---

## 数据存储结构

```
./data/traffic_info/
├── index.jsonl                 # 索引文件，快速查询
├── uploads/                    # 上传的原始图片
│   ├── abc12345_road.jpg
│   └── def67890_road.jpg
├── abc12345.json              # 单条记录
└── def67890.json

# 记录格式
{
  "id": "abc12345",
  "timestamp": "2025-12-05T10:30:00",
  "location": "116.3975,39.9093",
  "device_id": "device_001",
  "image_source": "./data/traffic_info/uploads/abc12345_road.jpg",
  "analysis_result": {
    "scene_description": "高峰期，车流量大",
    "traffic_info": {
      "status": "congestion",
      "vehicle_count": "200+/minute",
      "signal_status": "green"
    },
    "environment_info": {
      "buildings": "商业区",
      "facility": "人行天桥"
    },
    "weather_info": {
      "condition": "sunny",
      "visibility": "excellent"
    }
  }
}
```

---

## API文档

### road_scene_analyzer（图片分析）

**输入**：
```python
{
    "image_source": str,        # 本地路径、URL或Base64
    "location": Optional[str],  # 经度,纬度
    "device_id": Optional[str], # 设备标识
    "analysis_type": str        # traffic|environment|weather|all
}
```

**输出**：
```python
{
    "success": bool,
    "scene_description": str,
    "traffic_info": dict,
    "environment_info": dict,
    "weather_info": dict,
    "timestamp": str,
    "location": Optional[str],
    "device_id": Optional[str]
}
```

### traffic_info_storage（数据存储）

**输入**：
```python
{
    "analysis_result": dict,
    "location": str,           # 经度,纬度
    "timestamp": str,
    "device_id": Optional[str]
}
```

**输出**：
```python
{
    "success": bool,
    "record_id": str,
    "message": str
}
```

### traffic_info_query（数据查询）

**输入**：
```python
{
    "location": Optional[str],     # 查询位置
    "radius_km": float,            # 查询半径（默认5）
    "time_range_hours": int,       # 查询时间范围（默认24）
    "device_id": Optional[str]     # 过滤设备
}
```

**输出**：
```python
{
    "success": bool,
    "total_records": int,
    "records": list,               # 符合条件的记录
    "message": str
}
```

---

## 故障排除

### 问题1：API密钥错误

```
错误: Invalid API key
解决:
  1. 确保设置了DASHSCOPE_API_KEY环境变量
  2. 检查密钥是否过期
  3. 从https://dashscope.console.aliyun.com获取新密钥
```

### 问题2：图片加载失败

```
错误: Unable to load image from source
解决:
  1. 检查文件路径是否正确
  2. 检查URL是否可访问
  3. 检查Base64编码是否正确
  4. 检查文件大小是否超过20MB
```

### 问题3：模型调用超时

```
错误: Request timeout
解决:
  1. 增加timeout_seconds配置
  2. 检查网络连接
  3. 重试请求
```

---

## 进阶功能

### 1. 数据分析

```bash
python examples/custom_functions/road_scene_analysis/advanced_analytics.py
```

输出：
- 交通数据报告
- 拥堵热点识别
- 设备统计信息
- 可视化建议

### 2. REST API服务

```bash
pip install fastapi uvicorn
python examples/custom_functions/road_scene_analysis/api_server.py
```

访问 http://localhost:8000/docs 查看API文档

### 3. 与其他组件集成

- 结合`tavily_search`获取实时交通新闻
- 结合`current_datetime`进行时间相关分析
- 自定义工作流处理交通事件

---

## 最佳实践

1. **位置信息准确性**
   - 使用GPS坐标而非地址描述
   - 定期验证设备的地理定位

2. **数据隐私**
   - 不存储人脸图像
   - 定期清理过期数据
   - 遵守当地数据保护法规

3. **系统可靠性**
   - 为API调用添加重试逻辑
   - 监控存储空间使用
   - 定期备份重要数据

4. **用户体验**
   - 提供清晰的反馈信息
   - 支持图片压缩以加速上传
   - 缓存热门位置的查询结果

---

## 相关资源

- [阿里云DashScope文档](https://dashscope.aliyuncs.com)
- [千问VL模型说明](https://dashscope.aliyuncs.com/docs/api/vl_api)
- [NVIDIA NeMo Agent文档](https://docs.nvidia.com/nemo)
- [FastAPI文档](https://fastapi.tiangolo.com)

---

## 技术支持

如有问题或建议，请提交Issue或联系开发团队。
