# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LocationInfo(BaseModel):
    """位置信息数据模型"""
    latitude: float = Field(description="纬度")
    longitude: float = Field(description="经度")
    address: Optional[str] = Field(default=None, description="地址描述")
    
    @classmethod
    def from_string(cls, location_str: str):
        """从字符串解析位置信息"""
        if "," in location_str:
            parts = location_str.split(",")
            if len(parts) >= 2:
                try:
                    return cls(longitude=float(parts[0]), latitude=float(parts[1]))
                except ValueError:
                    return cls(longitude=0.0, latitude=0.0, address=location_str)
        return cls(longitude=0.0, latitude=0.0, address=location_str)
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        return f"{self.longitude},{self.latitude}"


class TrafficEvent(BaseModel):
    """交通事件数据模型"""
    event_type: str = Field(description="事件类型: accident, congestion, weather, construction等")
    severity: str = Field(default="normal", description="严重级别: low, normal, high, critical")
    description: str = Field(description="事件描述")
    affected_lanes: Optional[int] = Field(default=None, description="影响的车道数")
    
    
class SceneAnalysisResult(BaseModel):
    """场景分析结果数据模型"""
    record_id: str = Field(description="记录ID")
    timestamp: datetime = Field(description="分析时间")
    location: LocationInfo = Field(description="位置信息")
    device_id: str = Field(description="上传设备ID")
    image_url: str = Field(description="图片来源URL或路径")
    
    # 分析结果
    scene_description: str = Field(description="场景描述")
    
    traffic_info: dict = Field(description="交通信息")
    environment_info: dict = Field(description="环境信息")
    weather_info: dict = Field(description="天气信息")
    traffic_events: list[TrafficEvent] = Field(default_factory=list, description="检测到的交通事件")
    
    confidence: float = Field(default=0.8, description="分析置信度")


class TrafficInfoDatabase:
    """本地交通信息数据库"""
    
    def __init__(self, storage_path: str = "./data/traffic_info"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.jsonl"
        logger.info(f"初始化交通信息数据库: {self.storage_path}")
    
    async def save_analysis(self, result: SceneAnalysisResult) -> str:
        """保存分析结果"""
        import json
        import aiofiles
        
        record_id = result.record_id
        record_file = self.storage_path / f"{record_id}.json"
        
        try:
            # 保存完整记录
            async with aiofiles.open(record_file, "w") as f:
                await f.write(result.model_dump_json(indent=2))
            
            # 更新索引
            async with aiofiles.open(self.index_file, "a") as f:
                index_entry = {
                    "id": record_id,
                    "location": result.location.to_string(),
                    "timestamp": result.timestamp.isoformat(),
                    "device_id": result.device_id,
                    "file": str(record_file),
                }
                await f.write(json.dumps(index_entry) + "\n")
            
            logger.info(f"分析结果已保存: {record_id}")
            return record_id
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            raise
    
    async def query_by_location(self, location: LocationInfo, radius_km: float = 5.0) -> list:
        """按位置查询记录"""
        import json
        import aiofiles
        
        if not self.index_file.exists():
            return []
        
        results = []
        try:
            async with aiofiles.open(self.index_file, "r") as f:
                async for line in f:
                    entry = json.loads(line.strip())
                    # 简单的距离计算（欧几里得距离）
                    if self._calculate_distance(location, entry["location"]) <= radius_km:
                        results.append(entry)
        except Exception as e:
            logger.error(f"查询失败: {e}")
        
        return results
    
    async def query_by_device(self, device_id: str) -> list:
        """按设备ID查询记录"""
        import json
        import aiofiles
        
        if not self.index_file.exists():
            return []
        
        results = []
        try:
            async with aiofiles.open(self.index_file, "r") as f:
                async for line in f:
                    entry = json.loads(line.strip())
                    if entry["device_id"] == device_id:
                        results.append(entry)
        except Exception as e:
            logger.error(f"查询失败: {e}")
        
        return results
    
    async def query_by_time_range(self, start_time: datetime, end_time: datetime) -> list:
        """按时间范围查询记录"""
        import json
        import aiofiles
        
        if not self.index_file.exists():
            return []
        
        results = []
        try:
            async with aiofiles.open(self.index_file, "r") as f:
                async for line in f:
                    entry = json.loads(line.strip())
                    record_time = datetime.fromisoformat(entry["timestamp"])
                    if start_time <= record_time <= end_time:
                        results.append(entry)
        except Exception as e:
            logger.error(f"查询失败: {e}")
        
        return results
    
    @staticmethod
    def _calculate_distance(loc1: LocationInfo, loc2_str: str) -> float:
        """计算两点间的距离（简化版，使用欧几里得距离）"""
        try:
            loc2 = LocationInfo.from_string(loc2_str)
            # 简单的近似计算，实际应用中应使用Haversine公式
            lat_diff = loc1.latitude - loc2.latitude
            lon_diff = loc1.longitude - loc2.longitude
            return (lat_diff ** 2 + lon_diff ** 2) ** 0.5 * 111  # 粗略转换为公里
        except Exception:
            return float("inf")
    
    async def get_heatmap_data(self, grid_size: float = 0.1) -> dict:
        """生成热力图数据，用于可视化"""
        import json
        import aiofiles
        
        if not self.index_file.exists():
            return {}
        
        heatmap = {}
        try:
            async with aiofiles.open(self.index_file, "r") as f:
                async for line in f:
                    entry = json.loads(line.strip())
                    location_str = entry["location"]
                    try:
                        loc = LocationInfo.from_string(location_str)
                        grid_key = (
                            round(loc.latitude / grid_size) * grid_size,
                            round(loc.longitude / grid_size) * grid_size,
                        )
                        if grid_key not in heatmap:
                            heatmap[grid_key] = 0
                        heatmap[grid_key] += 1
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"生成热力图失败: {e}")
        
        return heatmap
