# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
REST API服务器示例
允许多个IoT设备/客户端通过HTTP上传图片和查询交通信息

使用FastAPI框架（可选，如需启用此功能请安装: pip install fastapi uvicorn）
"""

import sys
from pathlib import Path
import os

# 添加src目录到Python路径，以便导入模块
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
if src_dir.exists():
    sys.path.append(str(src_dir))

try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from typing import Optional, Dict, Any
import logging
import asyncio
import json
from datetime import datetime

# 尝试导入核心分析器
try:
    from nat_road_scene_analysis.core_analyzer import CoreRoadSceneAnalyzer
except ImportError:
    CoreRoadSceneAnalyzer = None
    print("Warning: Could not import CoreRoadSceneAnalyzer. Analysis will be disabled.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrafficAPIServer:
    """交通信息REST API服务器"""
    
    def __init__(self, storage_path: str = "./data/traffic_info", port: int = 8000, llm_config: Optional[Dict] = None):
        self.storage_path = storage_path
        self.port = port
        self.llm_config = llm_config or {}
        self.app = None
        self.analysis_queue = asyncio.Queue()
        self.analyzer = None
        
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(title="Traffic Information API")
            self._setup_routes()
            self._setup_events()
            
    def _init_analyzer(self):
        """初始化分析器和LLM"""
        if not CoreRoadSceneAnalyzer:
            return

        llm = None
        # 尝试初始化LLM
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = self.llm_config.get("api_key") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = self.llm_config.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model_name = self.llm_config.get("model_name", "qwen-vl-plus")
            
            if api_key:
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.01,
                    max_tokens=2048
                )
                logger.info(f"LLM initialized: {model_name}")
            else:
                logger.warning("No API Key found. LLM analysis will be skipped.")
        except ImportError:
            logger.warning("langchain_openai not installed. LLM analysis will be skipped.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")

        self.analyzer = CoreRoadSceneAnalyzer(llm=llm)

    async def _process_queue(self):
        """后台任务：处理分析队列"""
        logger.info("Starting analysis worker...")
        while True:
            try:
                task = await self.analysis_queue.get()
                file_path = task.get("file_path")
                location = task.get("location")
                device_id = task.get("device_id")
                
                logger.info(f"Processing image: {file_path}")
                
                if self.analyzer:
                    result = await self.analyzer.analyze(str(file_path), location)
                    
                    if result["success"]:
                        logger.info(f"Analysis successful for {file_path}")
                        # 这里可以将结果保存到文件或数据库
                        self._save_result(result, device_id)
                    else:
                        logger.error(f"Analysis failed for {file_path}: {result.get('scene_description')}")
                else:
                    logger.warning("Analyzer not initialized, skipping analysis.")
                
                self.analysis_queue.task_done()
            except Exception as e:
                logger.error(f"Error in worker: {e}")
                await asyncio.sleep(1)

    def _save_result(self, result: Dict, device_id: str):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{device_id}_{timestamp}.json"
            save_path = Path(self.storage_path) / filename
            
            # 确保存储目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换 datetime 对象为字符串以便 JSON 序列化
            def json_serial(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError ("Type not serializable")

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=json_serial)
            
            logger.info(f"Result saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def _setup_events(self):
        """设置启动和关闭事件"""
        @self.app.on_event("startup")
        async def startup_event():
            self._init_analyzer()
            asyncio.create_task(self._process_queue())

    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok", "timestamp": datetime.now().isoformat(), "queue_size": self.analysis_queue.qsize()}
        
        @self.app.post("/api/v1/upload-image")
        async def upload_image(
            file: UploadFile = File(...),
            location: str = Form(...),
            device_id: str = Form(...),
            analysis_type: str = Form(default="all")
        ):
            """
            上传图片进行交通分析
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
                
                # 添加到分析队列
                await self.analysis_queue.put({
                    "file_path": str(file_path),
                    "location": location,
                    "device_id": device_id,
                    "analysis_type": analysis_type
                })
                
                return {
                    "success": True,
                    "file_id": file_id,
                    "message": "图片上传成功，已加入分析队列",
                    "queue_position": self.analysis_queue.qsize(),
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"上传失败: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/v1/results")
        async def get_results(limit: int = 10):
            """获取最近的分析结果"""
            results = []
            try:
                storage_dir = Path(self.storage_path)
                if storage_dir.exists():
                    files = sorted(storage_dir.glob("analysis_*.json"), key=os.path.getmtime, reverse=True)
                    for f in files[:limit]:
                        try:
                            with open(f, "r", encoding="utf-8") as json_file:
                                results.append(json.load(json_file))
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"获取结果失败: {e}")
            
            return {"success": True, "results": results}

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
    import json
    import os
    
    parser = argparse.ArgumentParser(description="启动交通信息API服务器")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--storage", type=str, default="./data/traffic_info", help="数据存储路径")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (JSON)")
    
    args = parser.parse_args()
    
    port = args.port
    storage_path = args.storage
    llm_config = {}
    
    # 如果提供了配置文件，从配置文件加载
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                if "server" in config:
                    server_config = config["server"]
                    if "port" in server_config:
                        port = server_config["port"]
                    if "storage_path" in server_config:
                        storage_path = server_config["storage_path"]
                if "llm" in config:
                    llm_config = config["llm"]
                    
                print(f"已从配置文件 {args.config} 加载配置")
        except Exception as e:
            print(f"加载配置文件失败: {e}，将使用命令行参数或默认值")
    
    server = TrafficAPIServer(storage_path=storage_path, port=port, llm_config=llm_config)
    server.run()
