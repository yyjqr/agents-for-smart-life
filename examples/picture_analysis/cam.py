# 在适当的位置创建自定义工具模块，例如 my_tools.py
from nat.tool.base import BaseTool
from pydantic import Field
from typing import Optional, Dict, Any
import requests
import base64

import aiohttp
import base64
import json




class ImageAnalyzerTool(BaseTool):
    name: str = "image_analyzer_tool"
    description: str = "使用千问百炼大模型分析道路图像，检测交通事件、道路异常和行驶安全问题"
    
    # 千问百炼API配置
    api_key: str = Field(..., description="千问百炼大模型的API密钥")
    api_endpoint: str = Field("https://api.qianwen.com/v1/chat/completions", 
                             description="千问百炼大模型的API端点")
    
    async def execute(self, image_data: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        使用千问百炼大模型分析道路图像
        
        Args:
            image_data: base64编码的图像数据或图像URL
            context: 可选的上下文描述，如位置、时间等信息
        
        Returns:
            分析结果字典，包含交通事件检测和道路异常分析
        """
        try:
            # 准备请求千问百炼大模型的提示词
            prompt = self._build_prompt(context)
            
            # 准备多模态请求内容
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}" if not image_data.startswith('http') else image_data
                            }
                        }
                    ]
                }
            ]
            
            # 调用千问百炼大模型API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "model": "qwen-vl-plus",  # 使用千问视觉语言模型
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.1  # 低温度以获得更确定的回答
                }
                
                async with session.post(self.api_endpoint, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result['choices'][0]['message']['content']
                        
                        # 解析模型返回的文本为结构化数据
                        structured_result = self._parse_analysis_result(analysis_text)
                        return {
                            "status": "success",
                            "analysis": structured_result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "message": f"API请求失败: {response.status} - {error_text}"
                        }
                        
        except Exception as e:
            return {
                "status": "error",
                "message": f"图像分析失败: {str(e)}"
            }
    
    def _build_prompt(self, context: Optional[str] = None) -> str:
        """构建分析提示词"""
        base_prompt = """
        你是一个专业的交通分析系统，请分析提供的道路图像，检测以下内容：
        
        1. 交通事件检测：
           - 是否有交通事故（车辆碰撞、侧翻等）
           - 是否有交通拥堵
           - 是否有异常停车
           - 是否有行人或动物突然穿行
        
        2. 道路异常检测：
           - 道路损坏情况（坑洼、裂缝、塌陷等）
           - 道路积水或积雪
           - 能见度问题（雾、雨、烟等）
           - 障碍物或施工区域
        
        3. 行驶安全评估：
           - 当前道路条件对行驶安全的影响
           - 建议的行驶注意事项或绕行建议
        
        请以JSON格式返回分析结果，包含以下字段：
        - has_traffic_incident: boolean (是否有交通事件)
        - incident_type: string (事件类型，如accident, congestion等)
        - has_road_anomaly: boolean (是否有道路异常)
        - anomaly_type: string (异常类型，如pothole, flooding等)
        - severity: string (严重程度: low, medium, high)
        - safety_impact: string (安全影响评估)
        - recommendations: array (建议措施列表)
        - confidence: float (分析置信度0-1)
        """
        
        if context:
            base_prompt += f"\n\n上下文信息: {context}"
            
        return base_prompt
    
    def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        """解析模型返回的文本为结构化数据"""
        try:
            # 尝试从文本中提取JSON部分
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # 如果找不到JSON，返回默认结构
                return {
                    "has_traffic_incident": True,
                    "incident_type": "unknown",
                    "has_road_anomaly": True,
                    "anomaly_type": "unknown",
                    "severity": "medium",
                    "safety_impact": "需要谨慎驾驶",
                    "recommendations": ["减速慢行", "注意观察路况"],
                    "confidence": 0.7,
                    "raw_analysis": analysis_text  # 保留原始分析文本
                }
        except json.JSONDecodeError:
            # JSON解析失败时返回默认结构
            return {
                "has_traffic_incident": True,
                "incident_type": "unknown",
                "has_road_anomaly": True,
                "anomaly_type": "unknown",
                "severity": "medium",
                "safety_impact": "需要谨慎驾驶",
                "recommendations": ["减速慢行", "注意观察路况"],
                "confidence": 0.7,
                "raw_analysis": analysis_text  # 保留原始分析文本
            }