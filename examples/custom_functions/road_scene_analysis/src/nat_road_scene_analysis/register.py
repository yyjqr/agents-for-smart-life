# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from pydantic import Field

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
    from nat.data_models.function import FunctionBaseConfig
    from pydantic import BaseModel, Field
    
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
            analysis_prompts = {
                "traffic": "请分析这张图片中的交通状况。包括：道路通畅度、车辆流量、交通标志、交通灯状态、是否有交通事故等。",
                "environment": "请分析这张图片中的环境信息。包括：建筑物、街道设施、地标、人流等。",
                "weather": "请分析这张图片中的天气条件。包括：能见度、天气状况、光照条件等。",
                "all": "请全面分析这张路侧场景图片。包括：交通状况（道路通畅度、车辆流量、交通标志、交通灯状态、事故等）、环境信息（建筑物、街道设施、地标、人流等）和天气条件（能见度、天气状况等）。",
            }
            
            prompt = analysis_prompts.get(input_data.analysis_type, analysis_prompts["all"])
            
            # 如果有LLM，使用LLM进行分析
            if llm:
                try:
                    from langchain_core.messages import HumanMessage
                    
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
                    analysis_result = response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    logger.warning(f"LLM调用失败: {e}，使用默认分析")
                    analysis_result = f"图片分析失败: {str(e)}"
            else:
                analysis_result = f"已加载图片，大小: {len(image_bytes)} 字节，类型: {mime_type}"
            
            # 解析分析结果
            traffic_info = {"status": "analyzed", "details": analysis_result} if "traffic" in input_data.analysis_type or input_data.analysis_type == "all" else {}
            environment_info = {"status": "analyzed", "details": analysis_result} if "environment" in input_data.analysis_type or input_data.analysis_type == "all" else {}
            weather_info = {"status": "analyzed", "details": analysis_result} if "weather" in input_data.analysis_type or input_data.analysis_type == "all" else {}
            
            return RoadSceneAnalysisOutput(
                success=True,
                scene_description=analysis_result,
                traffic_info=traffic_info,
                environment_info=environment_info,
                weather_info=weather_info,
                timestamp=datetime.now().isoformat(),
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
    from pydantic import BaseModel, Field
    
    class TrafficInfoInput(BaseModel):
        analysis_result: dict = Field(description="分析结果")
        location: str = Field(description="位置信息（经度,纬度）")
        timestamp: str = Field(description="时间戳")
        device_id: Optional[str] = Field(default=None, description="设备ID")
    
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
