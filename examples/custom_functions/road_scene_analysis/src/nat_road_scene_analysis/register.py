# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path
from typing import Optional, Union

import aiofiles
import aiohttp
from pydantic import Field, BaseModel, field_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class RoadSceneAnalyzerConfig(FunctionBaseConfig, name="road_scene_analyzer"):
    """
    自定义函数配置：路侧场景图片分析器
    支持本地文件上传、URL上传和Base64编码的图片分析
    """
    llm_name: str = Field(description="LLM名称，应该指向支持视觉的模型如qwen-vl-plus")
    max_image_size_mb: int = Field(default=20, description="最大图片大小（MB）")
    timeout_seconds: int = Field(default=30, description="网络请求超时时间（秒）")


class TrafficInfoStorageConfig(FunctionBaseConfig, name="traffic_info_storage"):
    """
    自定义函数配置：交通信息存储
    存储分析后的交通信息和位置时间数据
    """
    storage_path: str = Field(default="./data/traffic_info", description="数据存储路径")


class TrafficInfoQueryConfig(FunctionBaseConfig, name="traffic_info_query"):
    """
    自定义函数配置：交通信息查询
    查询特定位置和时间范围内的交通信息
    """
    storage_path: str = Field(default="./data/traffic_info", description="数据存储路径")


async def _load_image_data(image_source: str) -> tuple[bytes, str]:
    """
    加载图片数据，支持本地路径、URL和Base64编码
    返回 (image_bytes, mime_type)
    """
    # 检查是否为Base64编码
    if image_source.startswith("data:"):
        # 处理Data URI格式
        parts = image_source.split(";")
        mime_type = parts[0].replace("data:", "")
        if "base64," in parts[1]:
            data = parts[1].split("base64,")[1]
        else:
            data = parts[1]
        return base64.b64decode(data), mime_type
    
    # 检查是否为Base64字符串（无Data URI前缀）
    if not image_source.startswith(("http://", "https://", "/")):
        try:
            return base64.b64decode(image_source), "image/jpeg"
        except Exception:
            pass
    
    # 检查是否为本地文件路径
    if Path(image_source).exists():
        async with aiofiles.open(image_source, "rb") as f:
            data = await f.read()
        
        # 推断MIME类型
        suffix = Path(image_source).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")
        return data, mime_type
    
    # 否则尝试从URL下载
    if image_source.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as session:
            async with session.get(image_source, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    raise ValueError(f"无法下载图片: HTTP {resp.status}")
                data = await resp.read()
                
                # 从Content-Type头获取MIME类型
                content_type = resp.headers.get("Content-Type", "image/jpeg")
                mime_type = content_type.split(";")[0]
                return data, mime_type
    
    raise ValueError(f"无法识别图片来源: {image_source}")


@register_function(config_type=RoadSceneAnalyzerConfig)
async def road_scene_analyzer(config: RoadSceneAnalyzerConfig, builder: Builder):
    """
    路侧场景图片分析函数
    支持本地上传、URL上传，基于千问图像理解模型分析场景
    """
    class RoadSceneAnalysisInput(BaseModel):
        image_source: str = Field(
            description="图片来源：本地路径、URL或Base64编码"
        )
        location: Optional[str] = Field(
            default=None,
            description="位置信息，格式：经度,纬度 或 地址描述"
        )
        device_id: Optional[str] = Field(
            default=None,
            description="设备ID，用于标识上传者"
        )
        analysis_type: str = Field(
            default="all",
            description="分析类型：traffic(交通), environment(环境), weather(天气), all(全部)"
        )
    
    class RoadSceneAnalysisOutput(BaseModel):
        success: bool = Field(description="分析是否成功")
        scene_description: str = Field(description="场景描述")
        traffic_info: dict = Field(description="交通信息")
        environment_info: dict = Field(description="环境信息")
        weather_info: dict = Field(description="天气信息")
        timestamp: str = Field(description="分析时间戳")
        location: Optional[str] = Field(description="记录的位置")
        device_id: Optional[str] = Field(description="设备ID")
    
    async def _analyze_road_scene(input_data: RoadSceneAnalysisInput) -> RoadSceneAnalysisOutput:
        """分析路侧场景"""
        from datetime import datetime
        
        # 获取LLM
        try:
            # 使用LangChain wrapper以便使用ainvoke
            llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type="langchain")
        except Exception as e:
            logger.warning(f"无法获取LLM {config.llm_name}: {e}，将使用直接分析")
            llm = None
        
        try:
            # 加载图片
            image_bytes, mime_type = await _load_image_data(input_data.image_source)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # 构建分析提示词
            base_prompt = """
            请分析这张路侧场景图片，并返回一个严格的JSON格式结果（不要包含markdown代码块标记）。
            JSON结构如下：
            {
                "congestion_level": "评估拥堵等级：畅通、缓行、拥堵、严重拥堵",
                "traffic_analysis": "详细的交通状况分析，包括道路通畅度、车辆流量、交通标志、交通灯状态、事故等",
                "environment_analysis": "详细的环境信息分析，包括建筑物、街道设施、地标、人流、是否有摆摊等",
                "weather_analysis": "详细的天气条件分析，包括能见度、天气状况、光照条件等",
                "vehicle_count": 0, // 估计的机动车总数（整数）
                "vulnerable_count": 0, // 估计的行人、自行车和摩托车总数（整数）
                "is_traffic_event": false, // 是否有交通事故、严重拥堵或施工等事件（布尔值）
                "event_summary": "事件简述，无事件则填None",
                "detections": [ // 仅当检测到交通事件、车辆密集(>20)或人群密集(>20)时返回，否则为空数组
                    {
                        "label": "目标类别(如car, person, accident)",
                        "box_2d": [ymin, xmin, ymax, xmax], // 归一化坐标 [0-1000]
                        "description": "简短描述"
                    }
                ],
                "osd_timestamp": "从图片中识别出的时间戳(YYYY-MM-DD HH:MM:SS)，如果无法识别则为null"
            }
            """
            
            prompt = base_prompt
            
            # 如果有LLM，使用LLM进行分析
            if llm:
                try:
                    from langchain_core.messages import HumanMessage
                    import json
                    import re
                    
                    # 使用OpenAI兼容的API格式调用视觉模型
                    response = await llm.ainvoke(
                        input=[
                            HumanMessage(
                                content=[
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{image_base64}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            )
                        ]
                    )
                    
                    raw_content = response.content if hasattr(response, 'content') else str(response)
                    
                    # 尝试解析JSON
                    try:
                        # 移除可能的markdown标记
                        cleaned_content = re.sub(r'^```json\s*|\s*```$', '', raw_content.strip(), flags=re.MULTILINE)
                        data = json.loads(cleaned_content)
                        
                        congestion = data.get("congestion_level", "未知")
                        traffic_text = data.get("traffic_analysis", "无交通信息")
                        env_text = data.get("environment_analysis", "无环境信息")
                        weather_text = data.get("weather_analysis", "无天气信息")
                        v_count = data.get("vehicle_count", 0)
                        p_count = data.get("vulnerable_count", 0)
                        is_event = data.get("is_traffic_event", False)
                        event_desc = data.get("event_summary", "None")
                        detections = data.get("detections", [])
                        osd_ts = data.get("osd_timestamp")
                        
                        # 确定时间戳
                        final_timestamp = datetime.now().isoformat()
                        if osd_ts and osd_ts != "null":
                            final_timestamp = osd_ts
                        else:
                            # 尝试从文件名解析
                            import re
                            filename_match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', input_data.image_source)
                            if filename_match:
                                y, m, d, H, M, S = filename_match.groups()
                                final_timestamp = f"{y}-{m}-{d} {H}:{M}:{S}"

                        # 构建场景描述 - 优化前端显示效果
                        event_status = "🔴 有" if is_event else "🟢 无"
                        event_detail = f"({event_desc})" if is_event else ""
                        
                        congestion_icon = "🟢"
                        if congestion in ["缓行"]: congestion_icon = "🟡"
                        if congestion in ["拥堵", "严重拥堵"]: congestion_icon = "🔴"
                        
                        scene_desc = f"""### 🚦 交通路况概览
| 指标 | 状态 | 详情 |
| :--- | :--- | :--- |
| **交通事件** | {event_status} | {event_detail} |
| **通行状况** | {congestion_icon} {congestion} | 机动车约 {v_count} 辆 |

### 📝 详细分析
**交通状况**: {traffic_text}

**环境信息**: {env_text}

**天气条件**: {weather_text}
"""
                        
                        # 告警逻辑
                        if is_event or v_count > 20 or p_count > 20:
                            alert_reason = []
                            if is_event: alert_reason.append(f"检测到交通事件: {event_desc}")
                            if v_count > 20: alert_reason.append(f"车辆密集 ({v_count}辆)")
                            if p_count > 20: alert_reason.append(f"人群/非机动车密集 ({p_count}个)")
                            
                            scene_desc += f"\n\n🚨 **注意**: {', '.join(alert_reason)}"
                            
                            # 如果有检测框，添加到描述中供前端解析（暂时以文本形式）
                            if detections:
                                scene_desc += f"\n\n**检测目标**: {len(detections)} 个重点目标已标记。"
                                
                                # 尝试绘制检测框
                                try:
                                    import cv2
                                    import numpy as np
                                    
                                    # 检查是否为本地文件路径
                                    if Path(input_data.image_source).exists():
                                        img = cv2.imread(input_data.image_source)
                                        if img is not None:
                                            h, w = img.shape[:2]
                                            
                                            # 绘制框
                                            for det in detections:
                                                box = det.get("box_2d")
                                                label = det.get("label", "unknown")
                                                if box and len(box) == 4:
                                                    # 归一化坐标 [ymin, xmin, ymax, xmax] -> 像素坐标
                                                    ymin, xmin, ymax, xmax = box
                                                    pt1 = (int(xmin * w / 1000), int(ymin * h / 1000))
                                                    pt2 = (int(xmax * w / 1000), int(ymax * h / 1000))
                                                    
                                                    # 绘制矩形
                                                    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
                                                    
                                                    # 绘制标签
                                                    cv2.putText(img, label, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                            
                                            # 保存标注后的图片
                                            output_path = Path(input_data.image_source).parent / f"annotated_{Path(input_data.image_source).name}"
                                            cv2.imwrite(str(output_path), img)
                                            
                                            # 添加到描述中
                                            # Convert to base64 for frontend display since local path is not accessible
                                            _, buffer = cv2.imencode('.jpg', img)
                                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                                            scene_desc += f"\n\n![Annotated Image](data:image/jpeg;base64,{img_base64})"
                                            
                                except ImportError:
                                    logger.warning("OpenCV not installed, skipping annotation.")
                                except Exception as e:
                                    logger.warning(f"Failed to draw annotations: {e}")

                        # 添加时间戳信息到描述
                        scene_desc += f"\n\n**时间信息**:\n- 图片时间: {final_timestamp}\n- 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        
                        # 提示Agent进行存储
                        scene_desc += "\n\n**SYSTEM NOTE**: Analysis complete. You MUST now call the `traffic_info_storage` tool to save this result."
                        
                        analysis_result = scene_desc
                        
                        # 分离信息
                        traffic_info = {
                            "status": "analyzed", 
                            "details": traffic_text, 
                            "congestion": congestion,
                            "vehicle_count": v_count, 
                            "vulnerable_count": p_count,
                            "event_detected": is_event,
                            "detections": detections
                        }
                        environment_info = {"status": "analyzed", "details": env_text}
                        weather_info = {"status": "analyzed", "details": weather_text}
                        
                    except json.JSONDecodeError:
                        logger.warning(f"JSON解析失败，使用原始文本: {raw_content[:100]}...")
                        analysis_result = raw_content
                        traffic_info = {"status": "analyzed", "details": raw_content}
                        environment_info = {"status": "analyzed", "details": raw_content}
                        weather_info = {"status": "analyzed", "details": raw_content}
                        final_timestamp = datetime.now().isoformat()

                except Exception as e:
                    logger.warning(f"LLM调用失败: {e}，使用默认分析")
                    analysis_result = f"图片分析失败: {str(e)}"
                    traffic_info = {}
                    environment_info = {}
                    weather_info = {}
                    final_timestamp = datetime.now().isoformat()
            else:
                analysis_result = f"已加载图片，大小: {len(image_bytes)} 字节，类型: {mime_type}"
                traffic_info = {}
                environment_info = {}
                weather_info = {}
                final_timestamp = datetime.now().isoformat()
            
            return RoadSceneAnalysisOutput(
                success=True,
                scene_description=analysis_result,
                traffic_info=traffic_info,
                environment_info=environment_info,
                weather_info=weather_info,
                timestamp=final_timestamp,
                location=input_data.location,
                device_id=input_data.device_id,
            )
        
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return RoadSceneAnalysisOutput(
                success=False,
                scene_description=f"分析失败: {str(e)}",
                traffic_info={},
                environment_info={},
                weather_info={},
                timestamp=datetime.now().isoformat(),
                location=input_data.location,
                device_id=input_data.device_id,
            )
    
    yield FunctionInfo.create(
        single_fn=_analyze_road_scene,
        description="分析路侧场景图片，识别交通状况、环境信息和天气条件。支持本地路径、URL和Base64编码的图片输入。",
        input_schema=RoadSceneAnalysisInput,
    )


@register_function(config_type=TrafficInfoStorageConfig)
async def traffic_info_storage(config: TrafficInfoStorageConfig, builder: Builder):
    """
    交通信息存储函数
    将分析的交通信息和位置、时间数据持久化存储
    """
    import json
    from typing import Union, Optional
    from pydantic import BaseModel, Field, field_validator
    
    class TrafficInfoInput(BaseModel):
        analysis_result: Union[dict, str] = Field(description="分析结果")
        location: str = Field(description="位置信息（经度,纬度）")
        timestamp: str = Field(description="时间戳")
        device_id: Optional[str] = Field(default=None, description="设备ID")

        @field_validator('analysis_result', mode='before')
        @classmethod
        def parse_analysis_result(cls, v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return {"raw_content": v}
            return v

    
    class TrafficInfoStorageOutput(BaseModel):
        success: bool = Field(description="存储是否成功")
        record_id: str = Field(description="记录ID")
        message: str = Field(description="状态消息")
    
    # 确保存储目录存在
    storage_dir = Path(config.storage_path)
    
    async def _store_traffic_info(input_data: TrafficInfoInput) -> TrafficInfoStorageOutput:
        """存储交通信息"""
        try:
            from datetime import datetime
            import uuid
            
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成记录ID
            record_id = str(uuid.uuid4())[:8]
            
            # 构建记录数据
            record = {
                "id": record_id,
                "location": input_data.location,
                "timestamp": input_data.timestamp,
                "device_id": input_data.device_id or "unknown",
                "analysis_result": input_data.analysis_result,
                "stored_at": datetime.now().isoformat(),
            }
            
            # 保存到JSON文件
            record_file = storage_dir / f"{record_id}.json"
            async with aiofiles.open(record_file, "w") as f:
                await f.write(json.dumps(record, ensure_ascii=False, indent=2))
            
            logger.info(f"交通信息已存储: {record_id}")
            
            return TrafficInfoStorageOutput(
                success=True,
                record_id=record_id,
                message=f"交通信息已成功存储，记录ID: {record_id}",
            )
        
        except Exception as e:
            logger.error(f"存储失败: {e}")
            return TrafficInfoStorageOutput(
                success=False,
                record_id="",
                message=f"存储失败: {str(e)}",
            )
    
    yield FunctionInfo.create(
        single_fn=_store_traffic_info,
        description="存储分析后的交通信息和位置、时间数据，支持多设备数据汇聚。",
        input_schema=TrafficInfoInput,
    )


@register_function(config_type=TrafficInfoQueryConfig)
async def traffic_info_query(config: TrafficInfoQueryConfig, builder: Builder):
    """
    交通信息查询函数
    查询指定位置和时间范围内的交通信息
    """
    import json
    from datetime import datetime, timedelta
    from pydantic import BaseModel, Field
    
    class TrafficInfoQueryInput(BaseModel):
        location: Optional[str] = Field(
            default=None,
            description="查询位置（经度,纬度）"
        )
        radius_km: float = Field(
            default=5.0,
            description="查询半径（公里）"
        )
        time_range_hours: int = Field(
            default=24,
            description="查询时间范围（小时）"
        )
        device_id: Optional[str] = Field(
            default=None,
            description="可选：指定设备ID"
        )
    
    class TrafficInfoQueryOutput(BaseModel):
        success: bool = Field(description="查询是否成功")
        total_records: int = Field(description="返回的记录数")
        records: list = Field(description="查询结果记录")
        message: str = Field(description="状态消息")
    
    async def _query_traffic_info(input_data: TrafficInfoQueryInput) -> TrafficInfoQueryOutput:
        """查询交通信息"""
        try:
            storage_dir = Path(config.storage_path)
            
            if not storage_dir.exists():
                return TrafficInfoQueryOutput(
                    success=False,
                    total_records=0,
                    records=[],
                    message="数据存储目录不存在",
                )
            
            # 读取所有JSON文件
            records = []
            now = datetime.now()
            cutoff_time = now - timedelta(hours=input_data.time_range_hours)
            
            for json_file in storage_dir.glob("*.json"):
                try:
                    async with aiofiles.open(json_file, "r") as f:
                        content = await f.read()
                        record = json.loads(content)
                    
                    # 过滤时间范围
                    record_time = datetime.fromisoformat(record.get("timestamp", ""))
                    if record_time < cutoff_time:
                        continue
                    
                    # 过滤设备ID
                    if input_data.device_id and record.get("device_id") != input_data.device_id:
                        continue
                    
                    # 过滤位置（如果指定）
                    if input_data.location:
                        record_location = record.get("location", "")
                        if record_location and input_data.location in record_location:
                            records.append(record)
                        elif not record_location:
                            records.append(record)
                    else:
                        records.append(record)
                
                except Exception as e:
                    logger.warning(f"读取记录失败 {json_file}: {e}")
            
            return TrafficInfoQueryOutput(
                success=True,
                total_records=len(records),
                records=records[:20],  # 限制返回数量
                message=f"查询成功，找到 {len(records)} 条记录",
            )
        
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return TrafficInfoQueryOutput(
                success=False,
                total_records=0,
                records=[],
                message=f"查询失败: {str(e)}",
            )
    
    yield FunctionInfo.create(
        single_fn=_query_traffic_info,
        description="查询特定位置和时间范围内的交通信息，支持按设备ID过滤。",
        input_schema=TrafficInfoQueryInput,
    )
