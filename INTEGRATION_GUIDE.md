# 与Hackathon配置集成指南

本指南说明如何在 `configs/hackathon_config.yml` 中集成和使用路侧场景分析工具。

## 已做的更改

### 1. 添加了3个新工具

```yaml
functions:
  road_scene_analyzer:         # 路侧场景分析工具
  traffic_info_storage:        # 交通信息存储工具
  traffic_info_query:          # 交通信息查询工具
```

### 2. 配置了千问VL模型

```yaml
llms:
  default_llm:
    _type: openai
    model_name: "qwen-vl-plus"  # 千问视觉模型
    api_key: "${DASHSCOPE_API_KEY}"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### 3. 添加了专有代理

```yaml
workflow:
  specialized_agents:
    - road_scene_agent:        # 路侧场景分析代理
    - info_sharing_agent:      # 信息共享代理
```

---

## 使用步骤

### 第一步：安装插件

```bash
cd examples/custom_functions/road_scene_analysis
pip install -e .
```

### 第二步：配置API密钥

```bash
# Linux/Mac
export DASHSCOPE_API_KEY="sk-your-actual-key-here"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-your-actual-key-here"

# Windows CMD
set DASHSCOPE_API_KEY=sk-your-actual-key-here
```

> **获取API密钥**：访问 [阿里云DashScope](https://dashscope.console.aliyun.com) 申请

### 第三步：验证配置

检查 `configs/hackathon_config.yml` 已包含新工具：

```bash
# 检查是否有road_scene_analyzer
grep "road_scene_analyzer" configs/hackathon_config.yml

# 检查是否有traffic_info_storage
grep "traffic_info_storage" configs/hackathon_config.yml

# 检查是否有traffic_info_query
grep "traffic_info_query" configs/hackathon_config.yml
```

### 第四步：启动Agent

```bash
python -m nat.cli.runner configs/hackathon_config.yml
```

---

## 交互示例

### 示例1：上报交通事故

```
用户: 我在北京朝阳区发现了一起交通事故，位置是116.3975,39.9093
     这是图片链接: https://example.com/accident.jpg

Agent执行步骤:
1. 理解用户意图（包含图片和位置信息）
2. 触发road_scene_agent
3. 调用road_scene_analyzer分析图片
   - 加载图片
   - 调用千问模型识别"交通事故"
   - 返回分析结果
4. 调用traffic_info_storage存储结果
   - 记录位置: 116.3975,39.9093
   - 记录时间戳
   - 记录设备ID（如果提供）
5. 返回用户: "事故已记录，已推送给周边用户"
```

### 示例2：查询区域交通信息

```
用户: 查询北京朝阳区最近24小时的交通情况

Agent执行步骤:
1. 理解用户意图（查询交通信息）
2. 触发info_sharing_agent
3. 调用traffic_info_query查询
   - 位置: 北京朝阳区
   - 半径: 默认5km
   - 时间范围: 24小时
4. 返回结果:
   - 总共收集了5条交通信息
   - 来自3个不同设备
   - 其中2个显示拥堵，3个显示正常
5. Agent总结并返回给用户
```

### 示例3：多设备协作

```
场景：智能交通系统中多个监测点协作

设备A (116.39, 39.90): 上报"交通拥堵" - 时间10:30
设备B (116.40, 39.90): 上报"交通正常" - 时间10:35
设备C (116.41, 39.90): 上报"交通拥堵" - 时间10:40

用户查询: "最近1小时内朝阳区的交通信息"

Agent返回:
- 设备A/C在116.39-116.41之间检测到拥堵
- 可能形成了一个拥堵热点
- 建议用户绕行
```

---

## 配置细节说明

### road_scene_analyzer 配置

```yaml
road_scene_analyzer:
  _type: road_scene_analyzer        # 工具类型
  llm_name: default_llm             # 使用的LLM模型
  description: "分析路侧场景..."    # 工具描述（Agent会看到）
  max_image_size_mb: 20             # 最大图片大小限制
  timeout_seconds: 30               # 网络请求超时
```

**参数说明**：
- `llm_name`: 必须指向一个支持视觉的模型（如qwen-vl-plus）
- `max_image_size_mb`: 防止过大文件，降低成本
- `timeout_seconds`: 网络不稳定时的保护机制

### traffic_info_storage 配置

```yaml
traffic_info_storage:
  _type: traffic_info_storage
  storage_path: "./data/traffic_info"   # 数据存储位置
  description: "存储交通信息..."
```

**数据存储位置**：
```
./data/traffic_info/
├── index.jsonl          # 索引文件（快速查询）
├── uploads/             # 图片存储
└── [record_id].json     # 完整记录
```

### traffic_info_query 配置

```yaml
traffic_info_query:
  _type: traffic_info_query
  storage_path: "./data/traffic_info"   # 与storage保持一致
  description: "查询交通信息..."
```

**查询能力**：
- 按位置范围查询
- 按时间范围查询
- 按设备ID过滤
- 返回最多20条记录

---

## 进阶配置

### 自定义触发条件

修改 `specialized_agents` 中的 `trigger_condition` 来自定义何时使用这些工具：

```yaml
specialized_agents:
  - road_scene_agent:
      trigger_condition: "contains_image_or_road_scene"
      # 可能的条件值：
      # - "contains_image_or_location"
      # - "contains_image_or_traffic"
      # - "user_request_contains:traffic|road|scene"
      tools: ["road_scene_analyzer", "traffic_info_storage"]
```

### 增加更多工具到工作流

如果需要其他工具与这些新工具协同工作：

```yaml
workflow:
  tool_names:
    - tavily_search              # 网络搜索（获取实时交通新闻）
    - current_datetime           # 时间工具（记录精确时间）
    - road_scene_analyzer        # 新：图片分析
    - traffic_info_storage       # 新：数据存储
    - traffic_info_query         # 新：数据查询
```

### 多LLM配置

如果需要为不同工具使用不同的LLM：

```yaml
llms:
  default_llm:
    _type: openai
    model_name: "qwen-vl-plus"
    api_key: "${DASHSCOPE_API_KEY}"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  
  qwen_vl_large:
    _type: openai
    model_name: "qwen-vl-max"  # 更强大的模型
    api_key: "${DASHSCOPE_API_KEY}"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

functions:
  road_scene_analyzer:
    _type: road_scene_analyzer
    llm_name: qwen_vl_large    # 使用更强大的模型
```

---

## 故障排除

### 问题：工具无法加载

```
错误: Cannot find function road_scene_analyzer
解决:
1. 确保已安装插件：pip install -e examples/custom_functions/road_scene_analysis
2. 检查配置文件中的 _type 是否正确
3. 重启Agent进程
```

### 问题：API调用超时

```
错误: Request timeout
解决:
1. 增加 timeout_seconds: 60
2. 检查网络连接
3. 确保API密钥有效
4. 尝试更小的图片
```

### 问题：没有查询到数据

```
症状: traffic_info_query 返回 total_records: 0
排查:
1. 确保 storage_path 相同
2. 检查是否有数据被存储过
3. 检查位置坐标格式是否正确
4. 扩大查询范围（radius_km）
```

---

## 性能优化建议

### 1. 图片优化

```yaml
road_scene_analyzer:
  max_image_size_mb: 10  # 减小大小限制
  # 建议用户压缩图片到 512x512 以下
```

### 2. 批量处理

对于大量请求，考虑批量处理：

```python
# 伪代码
for device_id in device_list:
    results = await batch_analyze(device_id, image_list)
    await traffic_info_storage.store_batch(results)
```

### 3. 缓存策略

经常查询的热点位置可以缓存结果：

```yaml
# 建议使用Redis存储最近1小时的查询结果
```

### 4. 定期清理

```bash
# 删除超过30天的数据
find ./data/traffic_info -name "*.json" -mtime +30 -delete
```

---

## 监控和日志

### 启用详细日志

```bash
# 运行时添加详细日志
LOGLEVEL=DEBUG python -m nat.cli.runner configs/hackathon_config.yml
```

### 查看工具调用日志

```python
# 在Agent输出中查看类似以下内容：
# [INFO] Calling tool: road_scene_analyzer
# [INFO] Input: {image_source: "...", location: "..."}
# [INFO] Output: {success: true, scene_description: "..."}
```

---

## 下一步

1. **安装插件**：`pip install -e examples/custom_functions/road_scene_analysis`
2. **设置API密钥**：`export DASHSCOPE_API_KEY="your-key"`
3. **启动Agent**：`python -m nat.cli.runner configs/hackathon_config.yml`
4. **开始对话**：描述包含图片和位置的交通场景

有问题？查看 `examples/custom_functions/road_scene_analysis/QUICKSTART.md`
