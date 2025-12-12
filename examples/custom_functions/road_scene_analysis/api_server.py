# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
REST API服务器示例
允许多个IoT设备/客户端通过HTTP上传图片和查询交通信息

使用FastAPI框架（可选，如需启用此功能请安装: pip install fastapi uvicorn）
"""

try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


from typing import Optional
import logging
import asyncio
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TrafficAPIServer:
    """交通信息REST API服务器"""
    
    def __init__(self, storage_path: str = "./data/traffic_info", port: int = 8000):
        self.storage_path = storage_path
        self.port = port
        self.app = None
        
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="Traffic Information API")
            self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/api/v1/upload-image")
        async def upload_image(
            file: UploadFile = File(...),
            location: str = Form(...),
            device_id: str = Form(...),
            analysis_type: str = Form(default="all")
        ):
            """
            上传图片进行交通分析
            
            参数:
            - file: 图片文件
            - location: 位置信息 (格式: "经度,纬度")
            - device_id: 设备ID
            - analysis_type: 分析类型 (traffic, environment, weather, all)
            """
            try:
                # 保存临时文件
                import uuid
                temp_dir = Path(self.storage_path) / "uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                file_id = str(uuid.uuid4())[:8]
                file_path = temp_dir / f"{file_id}_{file.filename}"
                
                # 保存文件
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                
                return {
                    "success": True,
                    "file_id": file_id,
                    "message": "图片上传成功",
                    "next_step": f"调用分析工具分析图片: {file_path}",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"上传失败: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/v1/analyze")
        async def analyze_image(request_body: dict):
            """
            分析图片
            
            参数:
            {
                "image_source": "文件路径、URL或Base64",
                "location": "经度,纬度",
                "device_id": "设备ID",
                "analysis_type": "分析类型"
            }
            """
            try:
                # 这里调用road_scene_analyzer工具
                # 实际实现需要集成NeMo Agent的工具调用机制
                
                return {
                    "success": True,
                    "record_id": "mock_id_123",
                    "analysis_result": {
                        "scene_description": "道路畅通，交通流量正常",
                        "traffic_info": {
                            "status": "normal",
                            "vehicle_count": "~50 vehicles/minute"
                        }
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"分析失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/query")
        async def query_traffic_info(
            location: Optional[str] = None,
            radius_km: float = 5.0,
            hours: int = 24,
            device_id: Optional[str] = None
        ):
            """
            查询交通信息
            
            参数:
            - location: 查询位置 (可选)
            - radius_km: 查询半径，单位公里 (默认5)
            - hours: 查询时间范围，单位小时 (默认24)
            - device_id: 过滤设备ID (可选)
            """
            try:
                # 这里调用traffic_info_query工具
                return {
                    "success": True,
                    "location": location,
                    "radius_km": radius_km,
                    "time_range_hours": hours,
                    "total_records": 5,
                    "records": [
                        {
                            "id": "record_001",
                            "location": location,
                            "timestamp": "2025-12-05T10:30:00",
                            "device_id": "device_001",
                            "analysis": {
                                "traffic_status": "congestion",
                                "description": "交通拥堵"
                            }
                        }
                    ]
                }
            except Exception as e:
                logger.error(f"查询失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/devices")
        async def get_device_list():
            """获取已上报的设备列表"""
            try:
                # 从存储中读取所有设备
                devices = set()
                storage_dir = Path(self.storage_path)
                
                if storage_dir.exists():
                    for json_file in storage_dir.glob("*.json"):
                        try:
                            import json
                            with open(json_file, "r") as f:
                                record = json.load(f)
                                device_id = record.get("device_id", "unknown")
                                devices.add(device_id)
                        except Exception:
                            pass
                
                return {
                    "success": True,
                    "total_devices": len(devices),
                    "devices": list(devices),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"获取设备列表失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/report")
        async def get_traffic_report(hours: int = 24):
            """获取交通数据报告"""
            try:
                return {
                    "success": True,
                    "time_range": f"最近{hours}小时",
                    "summary": {
                        "total_reports": 42,
                        "total_devices": 5,
                        "congestion_areas": 3,
                        "critical_events": 1
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"获取报告失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self):
        """启动服务器"""
        if not FASTAPI_AVAILABLE:
            print("错误: FastAPI未安装")
            print("请运行: pip install fastapi uvicorn")
            return
        
        if self.app is None:
            self._setup_routes()
        
        print(f"启动交通信息API服务器 (端口: {self.port})")
        print(f"API文档: http://localhost:{self.port}/docs")
        
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)


# CLI使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动交通信息API服务器")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--storage", type=str, default="./data/traffic_info", help="数据存储路径")
    
    args = parser.parse_args()
    
    server = TrafficAPIServer(storage_path=args.storage, port=args.port)
    server.run()
