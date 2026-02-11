#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import actionlib
import threading
import os
import math
import cv2
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
    "roomA1": (3.853, 1.920, 0.698, 0.716),
    "roomA2": (3.925, 2.387, 0.491, 0.871),
    "roomB": (2.304, 1.640, 0.698, 0.716),
    "roomB1": (2.300, 2.355, 0.702, 0.712),
    "roomB2": (2.300, 2.355, 0.702, 0.712),
    "roomC": (0.637, 1.046, 0.698, 0.716),
    "roomC1": (0.566, 1.854, 0.702, 0.712),
    "roomC2": (0.583, 2.394, 0.678, 0.735),
    "end": (0.000, 0.000, 1.000, 0.000)
}

class ONNXDetector:
    def __init__(self, model_path):
        """初始化ONNX检测器（基于ros_rknn_detection.py改进）"""
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.bridge = CvBridge()
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        rospy.loginfo(f"ONNX模型加载成功. 输入形状: {self.session.get_inputs()[0].shape}")

    def preprocess_image(self, cv_img, target_size=640):
        """图像预处理"""
        original_h, original_w = cv_img.shape[:2]
        
        # 计算缩放比例
        scale = min(target_size / original_h, target_size / original_w)
        new_h, new_w = int(original_h * scale), int(original_w * scale)
        
        # 缩放图像
        img_resized = cv2.resize(cv_img, (new_w, new_h))
        
        # 创建填充图像
        padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        padded_img[top:top+new_h, left:left+new_w] = img_resized
        
        return padded_img, scale, (top, left), (original_h, original_w)

    def postprocess_detections(self, output, scale, offset, original_size):
        """后处理检测结果"""
        if len(output.shape) == 3:
            output = output[0]
        
        if output.shape[0] == 16:
            output = output.T
        
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # 置信度过滤
        valid_indices = max_scores >= self.conf_threshold
        if not np.any(valid_indices):
            return []
        
        boxes = boxes[valid_indices]
        max_scores = max_scores[valid_indices]
        class_ids = class_ids[valid_indices]
        
        # 转换边界框格式 (center_x, center_y, width, height) -> (x1, y1, x2, y2)
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes = np.column_stack([x1, y1, x2, y2])
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), self.conf_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                x1, y1, x2, y2 = boxes[i]
                conf = max_scores[i]
                class_id = class_ids[i]
                
                # 过滤board类别
                if class_id == 2:
                    continue
                
                if class_id not in CLASS_NAMES:
                    continue
                
                # 坐标转换回原始图像
                top, left = offset
                original_h, original_w = original_size
                
                x1 = (x1 - left) / scale
                y1 = (y1 - top) / scale
                x2 = (x2 - left) / scale
                y2 = (y2 - top) / scale
                
                # 边界检查
                x1 = max(0, min(x1, original_w))
                y1 = max(0, min(y1, original_h))
                x2 = max(0, min(x2, original_w))
                y2 = max(0, min(y2, original_h))
                
                detections.append({
                    'class_id': int(class_id),
                    'class_name': CLASS_NAMES[class_id],
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        return detections

    def detect(self, cv_img):
        """执行目标检测（完整流程）"""
        try:
            # 预处理
            padded_img, scale, offset, original_size = self.preprocess_image(cv_img)
            
            # 转换为NCHW格式并归一化
            img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
            input_data = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            
            # 推理
            outputs = self.session.run(None, {self.input_name: input_data})
            
            # 后处理
            detections = self.postprocess_detections(outputs[0], scale, offset, original_size)
            
            # 返回检测到的类别名称列表
            return [d['class_name'] for d in detections] if detections else []
            
        except Exception as e:
            rospy.logerr(f"检测失败: {str(e)}")
            return []

class NavigationController:
    def __init__(self):
        rospy.init_node('gazebo_navigation_controller')
        
        # 初始化导航客户端
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base.wait_for_server()
        
        # 初始化速度发布器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
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
        self.detection_results = queue.Queue()
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

    def quick_head_swing(self):
        """快速车头摆动确认(左30°→右30°)"""
        rospy.loginfo("执行快速车头摆动确认...")
        swing_speed = 0.6  # 约34.4°/s (比原来的0.35更快)
        swing_duration1 = 0.5  # 约15°摆动
        swing_duration2 = 1  # 约30°摆动
        
        # 左转30°
        twist = Twist()
        twist.angular.z = swing_speed
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(swing_duration1)
        
        # 右转30°
        twist.angular.z = -swing_speed
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(swing_duration2)
        
        # 停止
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def swing_head_with_detection(self, total_duration=7.0):
        """优化后的非对称摆动(左20°→右40°→左20°)，统一转速"""
        rospy.loginfo("开始摆动...")
        
        # 统一摆动速度 (约28.6°/s)
        swing_speed = 0.4  
        
        # 各阶段持续时间计算 (角度/速度)
        swing_sequence = [
            {'angle': 20, 'duration': 20/28.6, 'speed': swing_speed},    # 左转20°
            {'angle': 40, 'duration': 40/28.6, 'speed': -swing_speed},   # 右转40°
            {'angle': 20, 'duration': 20/28.6, 'speed': swing_speed}     # 左转20°
        ]
        
        detected = False
        start_time = rospy.Time.now().to_sec()
        
        for swing in swing_sequence:
            if rospy.is_shutdown() or detected or \
               (rospy.Time.now().to_sec() - start_time) > total_duration:
                break
                
            twist = Twist()
            twist.angular.z = swing['speed']
            self.cmd_vel_pub.publish(twist)
            
            # 实时检测（更高频率）
            stage_start = rospy.Time.now().to_sec()
            while (rospy.Time.now().to_sec() - stage_start) < swing['duration']:
                if rospy.is_shutdown():
                    return False
                
                if (rospy.Time.now().to_sec() - stage_start) % 0.25 < 0.025:
                    detected = self.perform_detection_and_check()
                    if detected:
                        break
                
                rospy.sleep(0.05)  # 更高控制频率
        
        # 停止
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        return detected

    def rotate_360_with_detection(self, duration=6.0):  # 原10秒缩短为6秒
        """增大旋转速度的360度检测"""
        rospy.loginfo("开始快速旋转寻找目标(实时检测)...")
        rate = rospy.Rate(20)  # 提高控制频率到20Hz
        start_time = rospy.Time.now().to_sec()
        detected = False
        
        twist = Twist()
        twist.angular.z = 1.047  # 原0.628，增大到约60度/秒 (6秒完成360度)
        
        while (rospy.Time.now().to_sec() - start_time) < duration and not detected:
            if rospy.is_shutdown():
                return False
            
            self.cmd_vel_pub.publish(twist)
            
            # 保持相同的检测频率但更快的旋转
            if (rospy.Time.now().to_sec() - start_time) % 0.3 < 0.03:  # 约每0.3秒检测一次
                detected = self.perform_detection_and_check()
                if detected:
                    break
                
            rate.sleep()
        
        # 停止
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        return detected

    def perform_detection_and_check(self):
        """执行视觉识别并检查是否检测到目标"""
        detections = self.perform_detection()
        if detections:
            rospy.loginfo(f"在运动过程中检测到目标: {detections}")
            return True
        return False

    def perform_detection(self):
        """执行视觉识别并存储结果"""
        with self.image_lock:
            if self.current_image is None:
                rospy.logwarn("无可用图像")
                return []
            
            detections = self.detector.detect(self.current_image)
            
            if detections and self.current_room:
                for item in detections:
                    result = f"{self.current_room}.{item}"
                    self.detection_results.put(result)
                    rospy.loginfo(f"检测到: {result}")
            
            return detections

    def enhanced_detection(self, room_base):
        """增强型检测流程(现在在运动中实时检测)"""
        # 第一级检测
        if self.perform_detection_and_check():
            self.quick_head_swing()
            return True
        
        # 第二级：摇摆车头(带实时检测)
        if self.swing_head_with_detection():
            return True
        
        # 第三级：导航到room*1
        room_next = f"{room_base}1"
        if room_next in TARGET_POINTS:
            if self.send_navigation_goal(room_next) and self.wait_for_navigation_result():
                # 到达后立即检测
                if self.perform_detection_and_check():
                    return True
                
                # 第四级：摇摆车头(带实时检测)
                if self.swing_head_with_detection():
                    return True
                
                # 第五级：导航到room*2
                room_next2 = f"{room_base}2"
                if room_next2 in TARGET_POINTS:
                    if self.send_navigation_goal(room_next2) and self.wait_for_navigation_result():
                        # 第六级：360度旋转(带实时检测)
                        if self.rotate_360_with_detection():
                            return True
        
        return False

    def publish_results(self):
        """到达终点后发布所有存储的结果"""
        rospy.loginfo("到达终点，开始发布存储的检测结果...")
        published_count = 0
        while not self.detection_results.empty() and published_count < 3:
            result = self.detection_results.get()
            self.result_pub.publish(String(result))
            rospy.loginfo(f"发布: {result}")
            published_count += 1
            rospy.sleep(1)

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
                        # 执行增强型检测流程
                        self.enhanced_detection(target)
                        rospy.sleep(2)
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