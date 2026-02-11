#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import actionlib
import threading
import os
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import queue

# 全局配置
CLASS_NAMES = {
    0: 'apple', 1: 'banana', 2: 'board', 3: 'cake', 4: 'chili', 5: 'cola',
    6: 'greenlight', 7: 'milk', 8: 'potato', 9: 'redlight', 10: 'tomato', 11: 'watermelon'
}

TARGET_POINTS = {
    "start": (0.000, 0.000, 0.000, 1.000),
    "roomA": (3.834, 1.039, 0.698, 0.716),
    "roomA1":(3.853, 1.920, 0.698, 0.716),
    "roomA2":(3.925, 2.387, 0.491, 0.871),
    "roomB": (2.304, 1.640, 0.698, 0.716),
    "roomB2":(2.300, 2.355, 0.702, 0.712),
    "roomC": (0.637, 1.046, 0.698, 0.716),
    "roomC1":(0.566, 1.854, 0.702, 0.712),
    "roomC2":(0.583, 2.394, 0.678, 0.735),
    "end":   (0.000, 0.000, 1.000, 0.000)
}

class ONNXDetector:
    def __init__(self, model_path):
        """初始化ONNX检测器（保留原有识别逻辑）"""
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.bridge = CvBridge()
        rospy.loginfo("ONNX视觉识别模块初始化完成")

    def detect(self, cv_img):
        """执行目标检测（保留原有识别逻辑）"""
        # 预处理
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        input_data = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # 后处理（保留原有逻辑）
        detections = []
        if len(outputs) > 0:
            for i, score in enumerate(outputs[0][0][:, 4]):
                if score > 0.5:  # 置信度阈值
                    class_id = int(outputs[0][0][i, 5])
                    if class_id in CLASS_NAMES:
                        detections.append(CLASS_NAMES[class_id])
        return detections

class NavigationController:
    def __init__(self):
        rospy.init_node('gazebo_navigation_controller')
        
        # 初始化导航客户端
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base.wait_for_server()
        
        # 初始化视觉识别
        model_path = "/home/ros/Desktop/gazebo/gazebo_test_ws/src/rknn_detection/model/best.onnx"
        if not os.path.exists(model_path):
            rospy.logerr(f"模型文件不存在: {model_path}")
            rospy.signal_shutdown("模型文件缺失")
            return
        self.detector = ONNXDetector(model_path)
        self.bridge = CvBridge()
        
        # 结果发布和存储
        self.result_pub = rospy.Publisher('/gazebo_onnx', String, queue_size=10)
        self.detection_results = queue.Queue()  # 存储检测结果
        self.current_room = None
        
        # 订阅启动信号和摄像头
        rospy.Subscriber('/start_navigation', String, self.start_callback)
        rospy.Subscriber('/cam', Image, self.image_callback)
        
        # 状态变量
        self.navigation_active = False
        self.current_image = None
        self.image_lock = threading.Lock()
        
        rospy.loginfo("导航控制器初始化完成，等待启动信号...")

    def start_callback(self, msg):
        if msg.data == "gazebo_start" and not self.navigation_active:
            self.navigation_active = True
            rospy.loginfo("收到启动信号，开始导航任务！")
            threading.Thread(target=self.run_navigation_sequence).start()

    def image_callback(self, msg):
        try:
            with self.image_lock:
                self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"图像转换失败: {str(e)}")

    def send_navigation_goal(self, target_name):
        """发送导航目标"""
        if target_name not in TARGET_POINTS:
            rospy.logerr(f"未知目标点: {target_name}")
            return False
        
        x, y, z, w = TARGET_POINTS[target_name]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = z
        goal.target_pose.pose.orientation.w = w
        
        self.move_base.send_goal(goal)
        rospy.loginfo(f"导航至: {target_name}")
        
        # 更新当前房间
        if target_name.startswith("room"):
            self.current_room = target_name
        return True

    def wait_for_navigation_result(self, timeout=30.0):
        """等待导航完成"""
        finished = self.move_base.wait_for_result(rospy.Duration(timeout))
        
        if not finished:
            self.move_base.cancel_goal()
            rospy.logwarn("导航超时！")
            return False
        
        state = self.move_base.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("到达目标点")
            return True
        else:
            rospy.logwarn(f"导航失败，状态码: {state}")
            return False

    def _clear_detection_results(self):
        """清空检测结果队列"""
        while not self.detection_results.empty():
            try:
                self.detection_results.get_nowait()
            except queue.Empty:
                break

    def perform_detection(self):
        """执行视觉识别并存储结果（每次清空旧结果）"""
        with self.image_lock:
            if self.current_image is None:
                rospy.logwarn("无可用图像")
                return
            
            # 清空之前的结果
            while not self.detection_results.empty():
                self.detection_results.get()
            
            rospy.loginfo("开始视觉识别...")
            detections = self.detector.detect(self.current_image)
            
            if detections and self.current_room:
                for item in detections:
                    result = f"{self.current_room}.{item}"
                    self.detection_results.put(result)
                    rospy.loginfo(f"存储检测结果: {result}")

    def publish_results(self):
        """到达终点后发布所有存储的结果"""
        rospy.loginfo("到达终点，开始发布存储的检测结果...")
        published_count = 0
        while not self.detection_results.empty() and published_count < 3:
            result = self.detection_results.get()
            self.result_pub.publish(String(result))
            rospy.loginfo(f"发布: {result}")
            published_count += 1
            rospy.sleep(1)  # 每次发布间隔1秒

    def run_navigation_sequence(self):
        """执行完整的导航序列"""
        sequence = ["start", "roomA", "roomB", "roomC", "end"]
        
        for target in sequence:
            if not rospy.is_shutdown() and self.navigation_active:
                # 发送导航目标
                if not self.send_navigation_goal(target):
                    continue
                
                # 等待到达
                if self.wait_for_navigation_result():
                    # 在room点位执行识别
                    if target.startswith("room"):
                        self.perform_detection()
                        rospy.sleep(2)  # 停留2秒
                    # 在终点发布结果
                    elif target == "end":
                        self.publish_results()
                else:
                    rospy.logwarn(f"无法到达 {target}，终止任务")
                    break
        
        self.navigation_active = False
        rospy.loginfo("导航任务完成！")

if __name__ == '__main__':
    try:
        controller = NavigationController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点已关闭")