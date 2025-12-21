# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import json
import re
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None
import aiofiles
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

async def load_image_data(image_source: str) -> Tuple[bytes, str]:
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

class CoreRoadSceneAnalyzer:
    def __init__(self, llm=None):
        self.llm = llm

    async def analyze(self, image_source: str, location: Optional[str] = None) -> Dict[str, Any]:
        """
        执行核心分析逻辑
        """
        try:
            # 加载图片
            image_bytes, mime_type = await load_image_data(image_source)
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
            if self.llm:
                try:
                    # 使用OpenAI兼容的API格式调用视觉模型
                    response = await self.llm.ainvoke(
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
                            filename_match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', image_source)
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
                        
                        # 总是尝试在描述中包含图片（带标注）
                        try:
                            img = None
                            # 检查是否为本地文件路径且存在
                            if Path(image_source).exists():
                                if cv2 is not None:
                                    img = cv2.imread(image_source)
                                else:
                                    raise ImportError("OpenCV not installed")
                            
                            if img is not None:
                                # 如果有检测框，绘制它们
                                if detections:
                                    scene_desc += f"\n\n**检测目标**: {len(detections)} 个重点目标已标记。"
                                    h, w = img.shape[:2]
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
                                
                                # Resize if too large to save bandwidth (max width 600)
                                h, w = img.shape[:2]
                                if w > 600:
                                    scale = 600 / w
                                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                                
                                # Encode with lower quality to reduce size
                                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                                _, buffer = cv2.imencode('.jpg', img, encode_param)
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                scene_desc += f"\n\n![Annotated Image](data:image/jpeg;base64,{img_base64})"
                            
                            else:
                                # 图片对象为空（读取失败或路径不存在）
                                if image_source.startswith(('http://', 'https://')):
                                    scene_desc += f"\n\n![Annotated Image]({image_source})"
                                else:
                                    # 尝试作为普通文件读取
                                    try:
                                        if Path(image_source).exists():
                                            # 对于普通文件读取，我们也尝试压缩（如果PIL可用）
                                            try:
                                                from PIL import Image
                                                import io
                                                with Image.open(image_source) as pil_img:
                                                    # Resize
                                                    if pil_img.width > 600:
                                                        ratio = 600 / pil_img.width
                                                        new_height = int(pil_img.height * ratio)
                                                        pil_img = pil_img.resize((600, new_height), Image.LANCZOS)
                                                    
                                                    # Convert to RGB if necessary
                                                    if pil_img.mode in ('RGBA', 'P'):
                                                        pil_img = pil_img.convert('RGB')
                                                        
                                                    # Save to buffer with compression
                                                    buffer = io.BytesIO()
                                                    pil_img.save(buffer, format="JPEG", quality=60)
                                                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                            except ImportError:
                                                # Fallback to raw read if PIL not available
                                                with open(image_source, "rb") as f:
                                                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                                            
                                            scene_desc += f"\n\n![Annotated Image](data:image/jpeg;base64,{img_base64})"
                                        else:
                                            scene_desc += f"\n\n![Annotated Image]({image_source})"
                                    except Exception:
                                        scene_desc += f"\n\n![Annotated Image]({image_source})"
                                
                        except ImportError:
                            logger.warning("OpenCV not installed, skipping annotation.")
                            # Fallback to base64 encoding for local files
                            try:
                                if Path(image_source).exists():
                                    # 尝试使用PIL压缩
                                    try:
                                        from PIL import Image
                                        import io
                                        with Image.open(image_source) as pil_img:
                                            if pil_img.width > 600:
                                                ratio = 600 / pil_img.width
                                                new_height = int(pil_img.height * ratio)
                                                pil_img = pil_img.resize((600, new_height), Image.LANCZOS)
                                            
                                            if pil_img.mode in ('RGBA', 'P'):
                                                pil_img = pil_img.convert('RGB')
                                                
                                            buffer = io.BytesIO()
                                            pil_img.save(buffer, format="JPEG", quality=60)
                                            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                    except ImportError:
                                        with open(image_source, "rb") as f:
                                            img_base64 = base64.b64encode(f.read()).decode('utf-8')
                                    
                                    scene_desc += f"\n\n![Annotated Image](data:image/jpeg;base64,{img_base64})"
                                else:
                                    scene_desc += f"\n\n![Annotated Image]({image_source})"
                            except Exception:
                                scene_desc += f"\n\n![Annotated Image]({image_source})"
                        except Exception as e:
                            logger.warning(f"Failed to embed image: {e}")
                            # Fallback to base64 encoding for local files
                            try:
                                if Path(image_source).exists():
                                    with open(image_source, "rb") as f:
                                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                                    scene_desc += f"\n\n![Annotated Image](data:image/jpeg;base64,{img_base64})"
                                else:
                                    scene_desc += f"\n\n![Annotated Image]({image_source})"
                            except Exception:
                                scene_desc += f"\n\n![Annotated Image]({image_source})"

                        # 添加时间戳信息到描述
                        scene_desc += f"\n\n**时间信息**:\n- 图片时间: {final_timestamp}\n- 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        
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
                analysis_result = f"已加载图片，大小: {len(image_bytes)} 字节，类型: {mime_type} (无LLM)"
                traffic_info = {}
                environment_info = {}
                weather_info = {}
                final_timestamp = datetime.now().isoformat()
            
            return {
                "success": True,
                "scene_description": analysis_result,
                "traffic_info": traffic_info,
                "environment_info": environment_info,
                "weather_info": weather_info,
                "timestamp": final_timestamp,
                "location": location,
            }
        
        except Exception as e:
            logger.error(f"分析失败: {e}")
            return {
                "success": False,
                "scene_description": f"分析失败: {str(e)}",
                "traffic_info": {},
                "environment_info": {},
                "weather_info": {},
                "timestamp": datetime.now().isoformat(),
                "location": location,
            }
