# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path
from typing import Optional, Union

import aiofiles
import aiohttp
from pydantic import Field, BaseModel, field_validator, model_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from .core_analyzer import CoreRoadSceneAnalyzer

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
        
        # 获取LLM
        try:
            # 使用LangChain wrapper以便使用ainvoke
            llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type="langchain")
        except Exception as e:
            logger.warning(f"无法获取LLM {config.llm_name}: {e}，将使用直接分析")
            llm = None
        
        analyzer = CoreRoadSceneAnalyzer(llm=llm)
        result = await analyzer.analyze(input_data.image_source, input_data.location)
        
        # 如果成功，添加提示信息
        if result["success"]:
            result["scene_description"] += "\n\n**SYSTEM NOTE**: Analysis complete. You MUST now call the `traffic_info_storage` tool to save this result."

        return RoadSceneAnalysisOutput(
            success=result["success"],
            scene_description=result["scene_description"],
            traffic_info=result["traffic_info"],
            environment_info=result["environment_info"],
            weather_info=result["weather_info"],
            timestamp=result["timestamp"],
            location=result["location"],
            device_id=input_data.device_id,
        )
    
    yield FunctionInfo.create(
        single_fn=_analyze_road_scene,
        description="分析路侧场景图片，识别交通状况、环境信息和天气条件。支持本地路径、URL和Base64编码的图片输入。注意：工具输出包含标注后的图片（Markdown格式），你必须在最终回复中原样包含该图片Markdown代码，不要修改或省略。",
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
    from pydantic import BaseModel, Field, field_validator, model_validator
    
    class TrafficInfoInput(BaseModel):
        analysis_result: Optional[Union[dict, str]] = Field(default=None, description="分析结果")
        location: Optional[str] = Field(default="未知位置", description="位置信息（经度,纬度）")
        timestamp: Optional[str] = Field(default=None, description="时间戳")
        device_id: Optional[str] = Field(default="default_device", description="设备ID")
        image_source: Optional[str] = Field(default=None, description="图片来源路径或URL")

        @model_validator(mode='before')
        @classmethod
        def validate_and_flatten(cls, data):
            if isinstance(data, dict):
                # 处理 Agent 将所有参数包装在 analysis_result 中的情况
                if "analysis_result" in data and len(data) == 1:
                    inner = data["analysis_result"]
                    if isinstance(inner, str):
                        try:
                            import json
                            inner = json.loads(inner)
                        except:
                            pass
                    if isinstance(inner, dict):
                        # 将内部字段提升到顶层
                        new_data = inner.copy()
                        if "analysis_result" not in new_data:
                            new_data["analysis_result"] = inner
                        return new_data
            return data

        @field_validator('analysis_result', mode='before')
        @classmethod
        def parse_analysis_result(cls, v):
            if isinstance(v, str):
                try:
                    import json
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
    
    async def _store_traffic_info(input_data: object) -> TrafficInfoStorageOutput:
        """存储交通信息"""
        try:
            from datetime import datetime
            import uuid
            import json
            import ast
            
            # 手动处理输入转换，解决框架类型转换问题
            if isinstance(input_data, str):
                # 尝试多种方式解析字符串
                parsed = False
                # 1. 尝试标准 JSON 解析
                try:
                    input_data = json.loads(input_data)
                    parsed = True
                except json.JSONDecodeError:
                    pass
                
                # 2. 如果失败，尝试 Python 字面量解析 (处理单引号等情况)
                if not parsed:
                    try:
                        input_data = ast.literal_eval(input_data)
                        parsed = True
                    except (ValueError, SyntaxError):
                        pass
            
            if isinstance(input_data, dict):
                # 如果是字典，转换为 TrafficInfoInput 对象
                input_data = TrafficInfoInput(**input_data)
            
            if not isinstance(input_data, TrafficInfoInput):
                raise ValueError(f"Invalid input type: {type(input_data)}. Expected TrafficInfoInput, dict, or JSON string.")

            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成记录ID
            record_id = str(uuid.uuid4())[:8]
            
            # 确保有时间戳
            final_timestamp = input_data.timestamp or datetime.now().isoformat()
            
            # 构建记录数据
            record = {
                "id": record_id,
                "location": input_data.location,
                "timestamp": final_timestamp,
                "device_id": input_data.device_id or "unknown",
                "image_source": input_data.image_source,
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
