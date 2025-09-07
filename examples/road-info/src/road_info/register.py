# flake8: noqa

# Import any tools which need to be automatically registered here
# examples/road-info/src/road_info/register.py


# from road_info import road_info_function

# from nat.tool.register import register_function

# # MCP 服务地址
# MCP_URL = "http://127.0.0.1:9901/mcp"
# TOOL_NAME = "image_analyzer_tool"

# # 注册 road_info 工具
# # road_info = MCPToolWrapper(
# #     name="road_info",
# #     mcp_tool_name=TOOL_NAME,
# #     url=MCP_URL,
# #     transport="streamable-http",
# # )

# # 将工具注册到 NAT
# register_function(road_info)

# ======================
# /mcp/call 工具调用
# ======================
import logging
logging.basicConfig(level=logging.DEBUG)

from nat.tool.mcp.mcp_tool import MCPToolConfig,register_function
#from nat.tool.register import register_function
logging.debug("Initializing MCPToolWrapper for road_info...")



# logging.debug("MCPToolWrapper created: %s", road_info)