# 路侧场景图片分析插件

本插件为NVIDIA NeMo Agent Toolkit提供路侧场景图片分析功能，支持：

## 功能特性

1. **多种图片上传方式**：
   - 本地文件路径上传
   - URL链接上传
   - Base64编码上传

2. **千问图像理解模型集成**：
   - 使用千问VL模型进行图像分析
   - 支持OpenAI兼容API格式

3. **交通场景分析**：
   - 道路状况识别
   - 交通事件检测
   - 建筑物环境识别
   - 天气条件分析

4. **位置和时间记录**：
   - GPS坐标记录
   - 时间戳记录
   - 设备ID标识

5. **信息共享**：
   - 交通信息存储
   - 位置查询
   - 信息共享给其他用户

## 配置示例

```yaml
functions:
  road_scene_analyzer:
    _type: road_scene_analyzer
    llm_name: qwen_vl_llm
    description: "分析路侧场景图片，识别交通状况和环境信息"
    
  traffic_info_storage:
    _type: traffic_info_storage
    storage_path: "./data/traffic_info"
    description: "存储和共享交通信息"
    
  traffic_info_query:
    _type: traffic_info_query
    storage_path: "./data/traffic_info"
    description: "查询特定位置的交通信息"

llms:
  qwen_vl_llm:
    _type: openai
    model_name: "qwen-vl-plus"
    api_key: "${DASHSCOPE_API_KEY}"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: 0.7
    max_tokens: 2048
```

## 使用方法

### 1. 安装插件

```bash
cd examples/custom_functions/road_scene_analysis
pip install -e .
```

### 2. 配置环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

### 3. 运行Agent

使用配置文件启动Agent，即可使用路侧场景分析功能。

## API说明

### road_scene_analyzer

分析路侧场景图片。

**输入参数**：
- `image_source`: 图片来源，可以是本地路径、URL或base64编码
- `location`: 可选，位置信息（经度,纬度）
- `device_id`: 可选，设备标识
- `analysis_type`: 分析类型，可选值：traffic, environment, weather, all

**返回**：
- 分析结果JSON，包含场景描述、交通信息、环境信息等

### traffic_info_storage

存储交通信息。

**输入参数**：
- `traffic_info`: 交通信息对象
- `location`: 位置坐标
- `timestamp`: 时间戳

### traffic_info_query

查询指定位置的交通信息。

**输入参数**：
- `location`: 查询位置
- `radius`: 查询半径（米）
- `time_range`: 可选，时间范围
