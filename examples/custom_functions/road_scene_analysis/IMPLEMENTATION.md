# 路侧场景分析插件 - 完整功能说明

## 项目概述

本项目完善了基于NVIDIA NeMo Agent智能体的路侧场景图片分析功能，支持多设备上传、位置记录、交通信息存储和共享。

### 核心改进

✅ **解决了之前MCP的问题**
- 不再依赖外部MCP服务器，图片分析集成在Agent工具中
- 直接使用千问VL模型进行图像理解
- 更加稳定和易于维护

✅ **完整的图片上传支持**
- 本地文件路径上传
- 远程URL上传
- Base64编码上传
- 自动MIME类型检测

✅ **基于千问的图像理解**
- 集成千问VL Plus模型（支持OpenAI API格式）
- 支持交通、环境、天气等多维度分析
- 可扩展的分析框架

✅ **位置和时间记录**
- GPS坐标自动记录
- UTC时间戳记录
- 设备ID标识

✅ **交通信息存储和共享**
- 本地JSON数据库
- 快速索引查询
- 多设备数据汇聚

---

## 文件结构和功能说明

```
examples/custom_functions/road_scene_analysis/
├── README.md                      # 项目说明文档
├── QUICKSTART.md                  # 快速入门指南
├── pyproject.toml                 # Python项目配置
├── 
├── src/nat_road_scene_analysis/
│   ├── __init__.py               # 模块初始化
│   ├── register.py               # 工具注册和实现
│   └── utils.py                  # 数据模型和工具函数
├── 
├── configs/
│   └── config_example.yml        # 配置文件示例
├── 
├── example_usage.py              # 基础使用示例
├── advanced_analytics.py         # 数据分析工具
├── api_server.py                 # REST API服务器
└── test_road_scene_analysis.py   # 单元测试
```

### 核心模块说明

#### 1. `register.py` - 工具注册和实现

**包含3个主要工具：**

##### a) road_scene_analyzer
```python
# 功能：分析路侧场景图片
# 输入：
#   - image_source: 本地路径、URL或Base64
#   - location: GPS坐标 (可选)
#   - device_id: 设备标识 (可选)
#   - analysis_type: traffic|environment|weather|all

# 输出：
#   - success: 是否成功
#   - scene_description: 场景描述
#   - traffic_info: 交通信息
#   - environment_info: 环境信息
#   - weather_info: 天气信息
#   - timestamp: 分析时间
```

##### b) traffic_info_storage
```python
# 功能：存储分析结果
# 输入：
#   - analysis_result: 分析结果字典
#   - location: 位置坐标
#   - timestamp: 时间戳
#   - device_id: 设备ID (可选)

# 输出：
#   - success: 是否成功
#   - record_id: 记录ID
#   - message: 状态消息
```

##### c) traffic_info_query
```python
# 功能：查询交通信息
# 输入：
#   - location: 查询位置 (可选)
#   - radius_km: 查询半径（默认5）
#   - time_range_hours: 时间范围（默认24）
#   - device_id: 过滤设备 (可选)

# 输出：
#   - success: 是否成功
#   - total_records: 记录数
#   - records: 结果记录列表
#   - message: 状态消息
```

#### 2. `utils.py` - 数据模型和工具

**数据模型：**

- `LocationInfo`: 位置信息（支持坐标和地址描述）
- `TrafficEvent`: 交通事件（事件类型、严重级别、描述）
- `SceneAnalysisResult`: 场景分析结果（包含完整的分析结果和元数据）

**工具类：**

- `TrafficInfoDatabase`: 交通信息数据库
  - `save_analysis()`: 保存分析结果
  - `query_by_location()`: 按位置查询
  - `query_by_device()`: 按设备查询
  - `query_by_time_range()`: 按时间查询
  - `get_heatmap_data()`: 生成热力图数据

#### 3. `example_usage.py` - 使用示例

5个示例演示不同的使用场景：

1. 分析本地图片
2. 多设备多位置场景
3. 查询交通信息
4. Agent工作流集成
5. 交通信息共享

#### 4. `advanced_analytics.py` - 数据分析

提供交通数据分析功能：

- `generate_traffic_report()`: 生成交通数据报告
- `get_congestion_hotspots()`: 识别拥堵热点
- `get_device_statistics()`: 设备统计
- `export_report()`: 导出报告
- 可视化建议（Folium、Matplotlib、Plotly）

#### 5. `api_server.py` - REST API服务

使用FastAPI的REST服务，提供HTTP接口：

- `POST /api/v1/upload-image`: 上传图片
- `POST /api/v1/analyze`: 分析图片
- `GET /api/v1/query`: 查询交通信息
- `GET /api/v1/devices`: 获取设备列表
- `GET /api/v1/report`: 获取交通报告

#### 6. `test_road_scene_analysis.py` - 单元测试

提供单元测试覆盖核心功能：

- 位置信息解析
- 交通事件创建
- 数据保存和查询
- 图片加载
- 插件注册

---

## 配置说明

### hackathon_config.yml 更新

```yaml
functions:
  # 新增：路侧场景分析
  road_scene_analyzer:
    _type: road_scene_analyzer
    llm_name: default_llm
    description: "分析路侧场景图片..."
    max_image_size_mb: 20
    timeout_seconds: 30
  
  # 新增：交通信息存储
  traffic_info_storage:
    _type: traffic_info_storage
    storage_path: "./data/traffic_info"
    description: "存储交通信息..."
  
  # 新增：交通信息查询
  traffic_info_query:
    _type: traffic_info_query
    storage_path: "./data/traffic_info"
    description: "查询交通信息..."

llms:
  default_llm:
    _type: openai
    model_name: "qwen-vl-plus"  # 千问视觉模型
    api_key: "${DASHSCOPE_API_KEY}"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

workflow:
  tool_names:
    - road_scene_analyzer      # 新增工具
    - traffic_info_storage
    - traffic_info_query
  
  specialized_agents:
    - road_scene_agent:        # 新增专有代理
        trigger_condition: "contains_image_or_road_scene"
        tools: ["road_scene_analyzer", "traffic_info_storage"]
    
    - info_sharing_agent:      # 新增信息共享代理
        trigger_condition: "requires_traffic_info_query"
        tools: ["traffic_info_query"]
```

---

## 关键特性对比

### 旧方案（MCP）
❌ 依赖外部MCP服务器
❌ 需要单独部署和维护
❌ 通信复杂度高
❌ 扩展性有限

### 新方案（直接集成）
✅ 集成在Agent工具中，无额外依赖
✅ 部署简单，配置即用
✅ 通信直接高效
✅ 易于扩展和定制

---

## 使用工作流示例

### 单设备场景

```
用户 → Agent
  ↓
  理解用户意图（上传图片、分析交通）
  ↓
  调用road_scene_analyzer
  ├─ 加载图片
  ├─ 调用千问模型分析
  └─ 返回分析结果
  ↓
  调用traffic_info_storage
  ├─ 记录位置信息
  ├─ 记录时间戳
  └─ 保存分析结果
  ↓
  返回给用户
```

### 多设备共享场景

```
设备A上报      设备B上报      设备C上报
  ↓              ↓              ↓
road_scene_analyzer → traffic_info_storage (数据汇聚)
  ↓                          ↓
用户查询 → traffic_info_query (从数据库查询)
  ↓
返回多个设备的数据供参考
```

### 热点识别场景

```
多个时间点的多个设备数据
  ↓
advanced_analytics.get_congestion_hotspots()
  ↓
识别重点拥堵区域
  ↓
生成热力图数据
  ↓
可视化展示
```

---

## 数据持久化

### 存储位置

```
./data/traffic_info/
├── index.jsonl          # 快速查询索引
├── uploads/             # 原始图片文件
│   ├── abc12345_road.jpg
│   └── def67890_road.jpg
├── abc12345.json        # 完整分析记录
└── def67890.json
```

### 索引格式 (index.jsonl)

```json
{
  "id": "abc12345",
  "location": "116.3975,39.9093",
  "timestamp": "2025-12-05T10:30:00",
  "device_id": "device_001",
  "file": "./data/traffic_info/abc12345.json"
}
```

### 完整记录格式

```json
{
  "id": "abc12345",
  "location": "116.3975,39.9093",
  "timestamp": "2025-12-05T10:30:00",
  "device_id": "device_001",
  "image_url": "./data/traffic_info/uploads/abc12345_road.jpg",
  "analysis_result": {
    "scene_description": "高峰期交通拥堵",
    "traffic_info": {
      "status": "congestion",
      "vehicle_count": "200+/minute"
    },
    "environment_info": {...},
    "weather_info": {...}
  },
  "stored_at": "2025-12-05T10:35:00"
}
```

---

## 扩展可能性

### 1. 机器学习增强
- 使用深度学习模型进行更精确的车流量计数
- 交通拥堵预测
- 异常事件检测

### 2. 地图集成
- 集成高德地图API进行路网数据关联
- 实时路况对接
- 导航建议

### 3. 数据同步
- 与云端同步共享数据
- 跨区域数据汇聚
- 实时协作

### 4. 告警系统
- 拥堵告警
- 事故告警
- 天气告警

### 5. 历史分析
- 交通规律分析
- 高峰期预测
- 路网优化建议

---

## 依赖项

```
nvidia-nat          # NVIDIA NeMo Agent Toolkit
aiohttp>=3.8.0      # 异步HTTP客户端
aiofiles>=23.0.0    # 异步文件操作
Pillow>=10.0.0      # 图片处理
httpx>=0.24.0       # HTTP客户端
fastapi>=0.100.0    # REST API框架（可选）
uvicorn>=0.23.0     # ASGI服务器（可选）
```

---

## 安装和部署

### 快速安装

```bash
cd examples/custom_functions/road_scene_analysis
pip install -e .
```

### 配置环境

```bash
export DASHSCOPE_API_KEY="sk-your-key-here"
```

### 启动Agent

```bash
python -m nat.cli.runner configs/hackathon_config.yml
```

### 启动API服务（可选）

```bash
pip install fastapi uvicorn
python examples/custom_functions/road_scene_analysis/api_server.py --port 8000
```

---

## 总结

本项目完善了NVIDIA NeMo Agent的路侧场景分析功能，提供了完整的解决方案：

1. **完整的图片处理流程** - 支持多种上传方式
2. **智能图像分析** - 基于千问VL模型
3. **数据持久化** - 本地数据库存储
4. **信息共享** - 多设备数据汇聚
5. **数据分析** - 热点识别和报告生成
6. **API服务** - REST接口供外部调用

所有功能都经过精心设计，可以直接应用于实际的智能交通管理系统、城市大脑等场景。
