# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
路侧场景分析示例脚本
演示如何使用路侧场景分析功能进行图片上传、分析和信息共享
"""

import asyncio
import logging
from pathlib import Path

# 假设安装了nat库
# 实际使用时需要：
# 1. pip install -e examples/custom_functions/road_scene_analysis
# 2. 配置DASHSCOPE_API_KEY环境变量
# 3. 将配置写入hackathon_config.yml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def example_1_analyze_local_image():
    """示例1：分析本地图片"""
    print("\n=== 示例1：分析本地图片 ===")
    
    # 这里假设有一张本地图片
    local_image_path = "./road_scene_sample.jpg"
    
    # 如果本地没有图片，可以使用URL或Base64
    # 或者使用一个示例URL
    image_url = "https://example.com/road_scene.jpg"
    
    request = {
        "image_source": image_url,  # 可以是本地路径或URL
        "location": "116.3975,39.9093",  # 北京坐标示例
        "device_id": "device_001",
        "analysis_type": "traffic"
    }
    
    print(f"请求: {request}")
    print("返回: 分析结果包括交通状况、车辆流量等信息")


async def example_2_analyze_multiple_locations():
    """示例2：分析来自不同设备的多个位置的图片"""
    print("\n=== 示例2：多设备多位置场景 ===")
    
    scenes = [
        {
            "image_source": "https://example.com/road_scene_1.jpg",
            "location": "116.3975,39.9093",  # 北京
            "device_id": "device_001",
            "analysis_type": "traffic"
        },
        {
            "image_source": "https://example.com/road_scene_2.jpg",
            "location": "121.4737,31.2304",  # 上海
            "device_id": "device_002",
            "analysis_type": "environment"
        },
        {
            "image_source": "https://example.com/road_scene_3.jpg",
            "location": "113.2644,23.1291",  # 广州
            "device_id": "device_003",
            "analysis_type": "all"
        },
    ]
    
    print(f"分析来自 {len(scenes)} 个设备的图片")
    for scene in scenes:
        print(f"  - 设备 {scene['device_id']}: {scene['location']}")


async def example_3_query_traffic_info():
    """示例3：查询特定位置和时间范围的交通信息"""
    print("\n=== 示例3：查询交通信息 ===")
    
    # 查询北京地区最近24小时的交通信息
    query_request = {
        "location": "116.3975,39.9093",
        "radius_km": 5.0,
        "time_range_hours": 24,
        "device_id": None  # None表示查询所有设备
    }
    
    print(f"查询请求: {query_request}")
    print("查询结果: 返回该区域内最近24小时的所有交通信息")


async def example_4_workflow_integration():
    """示例4：与Agent工作流集成"""
    print("\n=== 示例4：Agent工作流集成 ===")
    
    workflow_request = """
    我需要分析北京地区（116.3975,39.9093）的交通情况。
    请上传一张最近的路侧图片来分析交通状况。
    """
    
    print(f"用户请求: {workflow_request}")
    print("Agent会依次执行:")
    print("  1. 使用road_scene_analyzer工具分析上传的图片")
    print("  2. 使用traffic_info_storage工具存储分析结果")
    print("  3. 使用traffic_info_query工具查询该位置的历史交通信息")
    print("  4. 基于分析结果和历史数据为用户生成报告")


async def example_5_sharing_traffic_info():
    """示例5：共享交通信息给其他用户/设备"""
    print("\n=== 示例5：交通信息共享 ===")
    
    print("场景描述:")
    print("  - 设备A在位置1检测到交通拥堵")
    print("  - 存储该信息到共享数据库")
    print("  - 用户B查询位置1时获得该信息")
    print("  - 用户B可以提前规划路线，避开拥堵区域")
    
    print("\n共享数据结构:")
    shared_data = {
        "record_id": "abc12345",
        "timestamp": "2025-12-05T10:30:00",
        "location": "116.3975,39.9093",
        "device_id": "device_001",
        "analysis_result": {
            "scene_description": "严重交通拥堵",
            "traffic_info": {
                "status": "congestion",
                "congestion_level": "high",
                "vehicle_count": "300+"
            },
            "weather_info": {
                "condition": "rain",
                "visibility": "low"
            }
        }
    }
    print(f"  {shared_data}")


async def main():
    """运行所有示例"""
    print("=" * 60)
    print("NVIDIA NeMo Agent - 路侧场景分析工具使用示例")
    print("=" * 60)
    
    await example_1_analyze_local_image()
    await example_2_analyze_multiple_locations()
    await example_3_query_traffic_info()
    await example_4_workflow_integration()
    await example_5_sharing_traffic_info()
    
    print("\n" + "=" * 60)
    print("部署和配置说明:")
    print("=" * 60)
    print("""
1. 安装插件:
   cd examples/custom_functions/road_scene_analysis
   pip install -e .

2. 配置环境变量:
   export DASHSCOPE_API_KEY="your-api-key"

3. 配置hackathon_config.yml:
   - road_scene_analyzer: 配置千问VL模型
   - traffic_info_storage: 配置数据存储路径
   - traffic_info_query: 配置数据查询

4. 启动Agent:
   python -m nat.cli.runner configs/hackathon_config.yml

5. 使用Agent:
   输入: "分析这张路侧图片的交通状况: [图片URL或本地路径]，位置: 116.3975,39.9093"
   Agent会自动调用相关工具进行分析和存储
    """)


if __name__ == "__main__":
    asyncio.run(main())
