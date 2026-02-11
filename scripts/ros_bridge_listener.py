#!/usr/bin/env python3
from roslibpy import Ros, Topic
import logging
import time
import os

# 设置ROS主节点URI（关键！）
os.environ['ROS_MASTER_URI'] = 'http://192.168.1.109:11311'

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("roslibpy")

def main():
    # 修正连接配置（移除了不支持的参数）
    ros = Ros(
        host='192.168.1.109',
        port=9090,
        is_secure=False  # 仅当使用wss时需要True
    )
    
    # 必须带proto参数的回调
    def on_connected(proto):
        logger.info(f"✅ 连接成功！使用协议: {proto}")
        
        # 示例：订阅chatter话题
        listener = Topic(
            ros,
            name='/start_navigation',
            message_type='std_msgs/String'
        )
        
        def callback(msg):
            logger.info(f"📡 收到消息: {msg['data']}")
        
        listener.subscribe(callback)
        logger.info("已订阅 /start_navigation 话题")
    
    def on_error(error):
        logger.error(f"❌ 连接错误: {error}")
        ros.close()
    
    # 绑定事件处理器
    ros.on('ready', on_connected)
    ros.on('error', on_error)
    
    # 启动连接
    logger.info("正在连接到 rosbridge...")
    ros.run()
    
    try:
        while True:
            time.sleep(1)
            if not ros.is_connected:
                logger.warning("连接断开，请检查服务状态")
                break
    except KeyboardInterrupt:
        logger.info("用户终止操作")
    finally:
        ros.close()

if __name__ == '__main__':
    main()