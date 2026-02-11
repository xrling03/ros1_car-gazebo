from roslibpy import Ros
import logging

logging.basicConfig(level=logging.DEBUG)
ros = Ros('192.168.1.109', 9090)

@ros.on('ready')
def connect():
    print("Connection success!")
    ros.close()

ros.run(timeout=10)