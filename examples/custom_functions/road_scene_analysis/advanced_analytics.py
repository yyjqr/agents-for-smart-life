# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
高级使用示例：交通信息热力图和数据分析
展示如何利用多设备上报的交通数据进行可视化和分析
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# 如果需要可视化，可以使用以下库（可选）：
# import matplotlib.pyplot as plt
# import folium  # 用于地图热力图


class TrafficDataAnalyzer:
    """交通数据分析工具"""
    
    def __init__(self, storage_path: str = "./data/traffic_info"):
        self.storage_path = Path(storage_path)
    
    async def load_all_records(self) -> list:
        """加载所有交通记录"""
        import aiofiles
        
        records = []
        if not self.storage_path.exists():
            return records
        
        # 加载JSON文件
        for json_file in self.storage_path.glob("*.json"):
            try:
                async with aiofiles.open(json_file, "r") as f:
                    content = await f.read()
                    record = json.loads(content)
                    records.append(record)
            except Exception as e:
                print(f"加载文件失败 {json_file}: {e}")
        
        return records
    
    async def generate_traffic_report(self, hours: int = 24) -> dict:
        """生成交通数据报告"""
        records = await self.load_all_records()
        
        if not records:
            return {"status": "no_data", "message": "没有交通数据"}
        
        # 按时间过滤
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_records = []
        
        for record in records:
            try:
                record_time = datetime.fromisoformat(record.get("timestamp", ""))
                if record_time >= cutoff_time:
                    recent_records.append(record)
            except Exception:
                pass
        
        # 分析数据
        report = {
            "time_range": f"最近 {hours} 小时",
            "total_reports": len(recent_records),
            "devices": set(),
            "locations": [],
            "traffic_events": [],
            "severity_stats": {
                "critical": 0,
                "high": 0,
                "normal": 0,
                "low": 0,
            }
        }
        
        for record in recent_records:
            # 统计设备
            device_id = record.get("device_id", "unknown")
            report["devices"].add(device_id)
            
            # 收集位置
            location = record.get("location", "")
            if location:
                report["locations"].append({
                    "location": location,
                    "timestamp": record.get("timestamp"),
                    "device_id": device_id,
                })
            
            # 统计严重级别
            analysis = record.get("analysis_result", {})
            traffic_info = analysis.get("traffic_info", {})
            severity = traffic_info.get("severity", "normal")
            if severity in report["severity_stats"]:
                report["severity_stats"][severity] += 1
        
        report["devices"] = list(report["devices"])
        return report
    
    async def get_congestion_hotspots(self, threshold: int = 2) -> list:
        """识别拥堵热点区域"""
        records = await self.load_all_records()
        
        # 按位置分组
        location_groups = {}
        for record in records:
            location = record.get("location", "")
            if location:
                if location not in location_groups:
                    location_groups[location] = []
                location_groups[location].append(record)
        
        # 找出拥堵位置（多个设备报告相同位置）
        hotspots = []
        for location, group in location_groups.items():
            if len(group) >= threshold:
                # 统计该位置的拥堵情况
                congestion_count = 0
                for record in group:
                    analysis = record.get("analysis_result", {})
                    traffic_info = analysis.get("traffic_info", {})
                    if traffic_info.get("status") == "congestion":
                        congestion_count += 1
                
                if congestion_count > 0:
                    hotspots.append({
                        "location": location,
                        "report_count": len(group),
                        "congestion_reports": congestion_count,
                        "congestion_rate": congestion_count / len(group),
                        "records": group,
                    })
        
        # 按拥堵率排序
        hotspots.sort(key=lambda x: x["congestion_rate"], reverse=True)
        return hotspots
    
    async def get_device_statistics(self) -> dict:
        """获取设备统计信息"""
        records = await self.load_all_records()
        
        device_stats = {}
        for record in records:
            device_id = record.get("device_id", "unknown")
            if device_id not in device_stats:
                device_stats[device_id] = {
                    "reports": 0,
                    "locations": set(),
                    "first_report": None,
                    "last_report": None,
                }
            
            device_stats[device_id]["reports"] += 1
            
            location = record.get("location", "")
            if location:
                device_stats[device_id]["locations"].add(location)
            
            timestamp = record.get("timestamp", "")
            if timestamp:
                if device_stats[device_id]["first_report"] is None:
                    device_stats[device_id]["first_report"] = timestamp
                device_stats[device_id]["last_report"] = timestamp
        
        # 转换set为list以便序列化
        for device_id in device_stats:
            device_stats[device_id]["locations"] = list(device_stats[device_id]["locations"])
        
        return device_stats
    
    async def export_report(self, output_file: str = "traffic_report.json"):
        """导出交通数据报告"""
        report = await self.generate_traffic_report()
        hotspots = await self.get_congestion_hotspots()
        device_stats = await self.get_device_statistics()
        
        full_report = {
            "generated_at": datetime.now().isoformat(),
            "summary": report,
            "congestion_hotspots": hotspots,
            "device_statistics": device_stats,
        }
        
        # 保存报告
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"报告已保存到: {output_path}")
        return full_report


async def example_analysis():
    """演示数据分析"""
    print("=" * 60)
    print("交通数据分析示例")
    print("=" * 60)
    
    analyzer = TrafficDataAnalyzer()
    
    # 生成报告
    print("\n1. 生成交通数据报告...")
    report = await analyzer.generate_traffic_report(hours=24)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    
    # 识别拥堵热点
    print("\n2. 识别拥堵热点区域...")
    hotspots = await analyzer.get_congestion_hotspots()
    if hotspots:
        for hotspot in hotspots[:5]:  # 显示前5个
            print(f"  位置: {hotspot['location']}")
            print(f"    报告数: {hotspot['report_count']}")
            print(f"    拥堵率: {hotspot['congestion_rate']*100:.1f}%")
    else:
        print("  未发现明显拥堵热点")
    
    # 设备统计
    print("\n3. 设备统计信息...")
    device_stats = await analyzer.get_device_statistics()
    for device_id, stats in list(device_stats.items())[:5]:  # 显示前5个设备
        print(f"  {device_id}:")
        print(f"    报告数: {stats['reports']}")
        print(f"    覆盖位置: {len(stats['locations'])}")
        if stats['first_report']:
            print(f"    首次上报: {stats['first_report']}")
        if stats['last_report']:
            print(f"    最后上报: {stats['last_report']}")
    
    # 导出完整报告
    print("\n4. 导出完整报告...")
    full_report = await analyzer.export_report("./traffic_analysis_report.json")
    print(f"  报告包含:")
    print(f"    - 总数据摘要")
    print(f"    - {len(hotspots)} 个拥堵热点")
    print(f"    - {len(device_stats)} 个设备的统计信息")


async def example_visualization():
    """演示可视化建议（伪代码）"""
    print("\n" + "=" * 60)
    print("可视化建议")
    print("=" * 60)
    
    print("""
    如需可视化交通数据，可以使用以下库：
    
    1. Folium - 地图热力图:
       ```python
       import folium
       from folium.plugins import HeatMap
       
       # 创建地图
       m = folium.Map(location=[39.9, 116.4], zoom_start=12)
       
       # 添加热力图数据
       heat_data = [[lat, lon, intensity], ...]
       HeatMap(heat_data).add_to(m)
       
       m.save('traffic_heatmap.html')
       ```
    
    2. Matplotlib - 柱状图和趋势图:
       ```python
       import matplotlib.pyplot as plt
       
       # 设备报告数柱状图
       plt.bar(devices, report_counts)
       plt.title('各设备上报次数')
       plt.savefig('device_reports.png')
       ```
    
    3. Plotly - 交互式仪表板:
       ```python
       import plotly.graph_objects as go
       
       fig = go.Figure(data=[...])
       fig.write_html('traffic_dashboard.html')
       ```
    
    建议的可视化维度：
    - 地理维度：在地图上显示交通事件的空间分布
    - 时间维度：显示交通事件随时间的变化趋势
    - 设备维度：显示各设备的贡献情况
    - 事件维度：分类统计不同类型的交通事件
    """)


async def main():
    await example_analysis()
    await example_visualization()


if __name__ == "__main__":
    asyncio.run(main())
