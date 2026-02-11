#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import time

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/cam", Image, self.callback)
        self.save_dir = os.path.expanduser("~/gazebo_images")  # 保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        rospy.loginfo(f"图像将保存到: {self.save_dir}")

    def callback(self, msg):
        try:
            # 转换ROS图像为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"image_{timestamp}.jpg")
            
            # 保存图像
            cv2.imwrite(filename, cv_image)
            rospy.loginfo(f"已保存: {filename}")
            
            # 可选：显示图像
            cv2.imshow("Camera View", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"保存失败: {str(e)}")

if __name__ == "__main__":
    rospy.init_node("image_saver")
    saver = ImageSaver()
    rospy.spin()