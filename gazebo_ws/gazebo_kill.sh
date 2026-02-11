#!/bin/bash

# 彩色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# 1. 终止Gazebo图形客户端
echo -e "${GREEN}[1/4] 正在终止Gazebo客户端...${NC}"
pkill -f "gzclient" || echo "未找到gzclient进程"

# 2. 终止Gazebo服务器
echo -e "${GREEN}[2/4] 正在终止Gazebo服务器...${NC}"
pkill -f "gzserver" || echo "未找到gzserver进程"

# 3. 终止ROS节点
echo -e "${GREEN}[3/4] 正在清理ROS节点...${NC}"
if rostopic list &>/dev/null; then
    rosnode kill --all 2>/dev/null
    pkill -f "rosmaster"
    echo "ROS节点已终止"
else
    echo "ROS未运行，跳过清理"
fi

# 4. 强制清理残留
echo -e "${GREEN}[4/4] 强制清理残留进程...${NC}"
pkill -9 -f "gazebo" || echo "无残留进程"
pkill -9 -f "python.*gazebo" || echo "无Python插件进程"

# 结果统计
echo -e "\n${RED}终止结果：${NC}"
pgrep -fla "gazebo|gzserver|gzclient|rosmaster" || echo "无相关进程存活"

# 可选缓存清理
echo -e "\n${GREEN}操作完成！${NC}"