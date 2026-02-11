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
from roslibpy import Ros, Topic  # 新增roslibpy相关导入
import logging


# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("roslibpy")

# 全局配置
CLASS_NAMES = {
    0: 'apple', 1: 'banana', 2: 'board', 3: 'cake', 4: 'chili', 5: 'cola',
    6: 'greenlight', 7: 'milk', 8: 'potato', 9: 'redlight', 10: 'tomato', 11: 'watermelon'
}

TASK_GOODS_MAP = {
    "Fruit": ["apple", "banana", "watermelon"],
    "Vegetable": ["chili", "tomato", "potato"],
    "Dessert": ["milk", "cake", "cola"]
}

TARGET_POINTS = {
    "start": (0.200, 0.000, 0.000, 1.000),
    # 原roomA改为roomC
    "roomC": (3.834, 1.039, 0.698, 0.716),
    "roomC1": (3.853, 1.920, 0.698, 0.716),
    "roomC2": (3.925, 2.387, 0.491, 0.871),
    "roomB": (2.304, 1.640, 0.698, 0.716),
    "roomB1": (2.300, 2.355, 0.702, 0.712),
    "roomB2": (2.300, 2.800, 0.702, 0.712),
    # 原roomC改为roomA
    "roomA": (0.637, 1.046, 0.698, 0.716),
    "roomA1": (0.566, 1.854, 0.702, 0.712),
    "roomA2": (0.583, 2.394, 0.678, 0.735),
    "end": (0.030, 0.000, 1.000, 0.000)
}

class ONNXDetector:
    def __init__(self, model_path):
        """初始化ONNX检测器（基于ros_rknn_detection.py改进）"""
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.bridge = CvBridge()
        self.conf_threshold = 0.6
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
                
                # 过滤不需要的类别 (board, greenlight, redlight)
                if class_id in [2, 6, 9]:  # 2:board, 6:greenlight, 9:redlight
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

        # 添加任务相关变量
        self.current_task = None  # 当前任务类型(Fruit/Vegetable/Dessert)
        self.target_goods = []    # 当前任务目标商品列表
        self.detected_goods = {}  # 存储检测到的商品 {room: [goods]}

        # 修改存储结构以保留所有检测结果
        self.all_detections = {}      # 存储所有检测到的商品 {room: [(goods, is_target)]}
        self.task_detections = {}     # 仅存储任务目标商品 {room: goods}
        
        # 初始化视觉识别
        model_path = "/home/ros/Desktop/gazebo/gazebo_test_ws/src/rknn_detection/model/best.onnx"
        if not os.path.exists(model_path):
            rospy.logerr(f"模型文件不存在: {model_path}")
            rospy.signal_shutdown("模型文件缺失")
            return
        self.detector = ONNXDetector(model_path)
        self.bridge = CvBridge()
        
        # # 结果发布和存储
        # self.result_pub = rospy.Publisher('/gazebo_onnx', String, queue_size=10)
        # self.detection_results = queue.Queue()
        # self.current_room = None
        
        # # 订阅启动信号和摄像头
        # rospy.Subscriber('/start_navigation', String, self.start_callback)
        # rospy.Subscriber('/cam', Image, self.image_callback)
        
        # # 状态变量
        # self.navigation_active = False
        # self.current_image = None
        # self.image_lock = threading.Lock()
        
        # rospy.loginfo("导航控制器初始化完成，等待启动信号...")
        # 初始化rosbridge连接
        self.ros = Ros(host='192.168.1.156', port=9090, is_secure=False)
        
        # 使用rosbridge的结果发布器 (替换原有rospy.Publisher)
        self.result_pub = Topic(self.ros, '/gazebo_onnx', 'std_msgs/String')
        
        # 使用rosbridge的订阅器 (替换原有rospy.Subscriber)
        self.start_sub = Topic(self.ros, '/start_navigation', 'std_msgs/String')
        self.start_sub.subscribe(self.rosbridge_start_callback)  # 使用新的回调函数
        
        # 保持原有的摄像头订阅 (仍使用rospy)
        rospy.Subscriber('/cam', Image, self.image_callback)
        
        # 状态变量
        self.navigation_active = False
        self.current_image = None
        self.image_lock = threading.Lock()
        
        # 启动rosbridge连接
        self.ros.run()
        rospy.loginfo("导航控制器初始化完成，等待启动信号...")

    def rosbridge_start_callback(self, msg):
        """rosbridge版本的启动回调"""
        if msg['data'] in TASK_GOODS_MAP and not self.navigation_active:
            self.current_task = msg['data']
            self.target_goods = TASK_GOODS_MAP[msg['data']]
            self.detected_goods = {}
            self.navigation_active = True
            rospy.loginfo(f"收到{msg['data']}任务，目标商品: {self.target_goods}")
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
        """非对称摆动，检测到任何物体立即停止"""
        rospy.loginfo("开始摆动...")
        
        swing_speed = 0.5
        swing_sequence = [
            {'angle': 20, 'duration': 20/28.6, 'speed': swing_speed},
            {'angle': 40, 'duration': 40/28.6, 'speed': -swing_speed},
            {'angle': 20, 'duration': 20/28.6, 'speed': swing_speed}
        ]
        
        detected = False
        start_time = rospy.Time.now().to_sec()
        
        for swing in swing_sequence:
            if rospy.is_shutdown() or detected:
                break
                
            twist = Twist()
            twist.angular.z = swing['speed']
            self.cmd_vel_pub.publish(twist)
            
            # 实时检测循环
            stage_start = rospy.Time.now().to_sec()
            while (rospy.Time.now().to_sec() - stage_start) < swing['duration']:
                if rospy.is_shutdown():
                    self.cmd_vel_pub.publish(Twist())
                    return False
                
                # 检测到任何物体立即停止
                if self.perform_detection_and_check():
                    rospy.loginfo("检测到物体，进行下一步！")
                    self.cmd_vel_pub.publish(Twist())
                    return True
                
                rospy.sleep(0.1)
        
        self.cmd_vel_pub.publish(Twist())
        return detected

    def rotate_360_with_detection(self, duration=6.0):
        """360°旋转，检测到任何物体立即停止"""
        rospy.loginfo("开始快速旋转寻找目标...")
        rate = rospy.Rate(20)
        start_time = rospy.Time.now().to_sec()
        
        twist = Twist()
        twist.angular.z = 1.047
        
        while (rospy.Time.now().to_sec() - start_time) < duration:
            if rospy.is_shutdown():
                self.cmd_vel_pub.publish(Twist())
                return False
            
            self.cmd_vel_pub.publish(twist)
            
            # 检测到任何物体立即停止
            if self.perform_detection_and_check():
                rospy.loginfo("检测到物体，进行下一步！")
                self.cmd_vel_pub.publish(Twist())
                return True
            
            rate.sleep()
        
        self.cmd_vel_pub.publish(Twist())
        return False

    def perform_detection_and_check(self):
        """检查是否检测到任何物体"""
        return bool(self.perform_detection())  # 只要列表不为空就返回True

    def perform_detection(self):
        """执行视觉识别，返回所有检测到的物体列表"""
        with self.image_lock:
            if self.current_image is None:
                rospy.logwarn("无可用图像")
                return []
            
            # 获取所有检测结果
            detections = self.detector.detect(self.current_image)
            if not detections or not self.current_room:
                return []
            
            room_base = self.current_room.rstrip('12')  # 去除房间后缀
            
            # 记录所有检测到的物体（用于运动控制）
            current_detections = []
            for item in detections:
                is_target = self.current_task and item in self.target_goods
                current_detections.append(item)
                
                # 记录任务目标（用于最终发布）
                if is_target and room_base not in self.task_detections:
                    self.task_detections[room_base] = item
                    rospy.loginfo(f"在{room_base}检测到目标商品: {item}")
                
                rospy.loginfo(f"在{room_base}检测到: {item}{' (目标)' if is_target else ''}")
            
            return current_detections  # 返回所有检测到的物体

    def enhanced_detection(self, room_base):
        """增强型检测流程，包含多级回退验证机制"""
        first_detection = None
        
        # 第一级检测（静止状态）
        initial_detections = self.perform_detection()
        if initial_detections:
            first_detection = initial_detections[0]
            
            # 仅对roomA和roomC执行详细确认流程
            if room_base in ["roomA", "roomC"]:
                rospy.loginfo(f"初次检测到物体: {first_detection}，尝试前往{room_base}1确认")
                
                # 尝试导航到room*1（缩短超时时间）
                room_next = f"{room_base}1"
                if room_next in TARGET_POINTS:
                    if self.send_navigation_goal(room_next):
                        if self.wait_for_navigation_result(timeout=15.0):
                            # 在room*1进行360度旋转检测
                            if not self.rotate_360_with_detection():  # 如果旋转未检测到
                                rospy.loginfo(f"在{room_base}1未确认目标，尝试前往{room_base}2")
                                
                                # 导航到room*2
                                room_next2 = f"{room_base}2"
                                if room_next2 in TARGET_POINTS and self.send_navigation_goal(room_next2):
                                    if self.wait_for_navigation_result(timeout=15.0):
                                        if self.rotate_360_with_detection():  # 在room*2再次尝试
                                            latest_detections = self.perform_detection()
                                            if latest_detections:
                                                final_detection = latest_detections[0]
                                                if final_detection != first_detection:
                                                    rospy.loginfo(f"检测结果更新: {first_detection} -> {final_detection}")
                                                return True
                            else:  # 如果在room*1旋转检测成功
                                latest_detections = self.perform_detection()
                                if latest_detections:
                                    final_detection = latest_detections[0]
                                    if final_detection != first_detection:
                                        rospy.loginfo(f"检测结果更新: {first_detection} -> {final_detection}")
                                    return True
            # 对于roomB，直接返回检测结果
            return True
        
        # 原流程（未检测到时的处理）
        # 第二级：摆动检测
        if self.swing_head_with_detection():
            return True
        
        # 第三级：前往room*1
        if room_base in ["roomA", "roomB","roomC"]:
            room_next = f"{room_base}1"
            if room_next in TARGET_POINTS:
                if self.send_navigation_goal(room_next) and self.wait_for_navigation_result(timeout=5.0):
                    if self.perform_detection_and_check():
                        return True
                    
                    if self.swing_head_with_detection():
                        return True
                    
                    # 第五级：前往room*2
                    room_next2 = f"{room_base}2"
                    if room_next2 in TARGET_POINTS:
                        if self.send_navigation_goal(room_next2) and self.wait_for_navigation_result(timeout=5.0):
                            if self.rotate_360_with_detection():
                                return True
        
        return False

    def publish_results(self):
        """修改为使用rosbridge发布结果"""
        if not self.task_detections:
            rospy.loginfo("未检测到任何目标商品")
            return
            
        rospy.loginfo("到达终点，开始发布检测结果...")
        for room, goods in self.task_detections.items():
            # 使用rosbridge发布
            rospy.sleep(0.5)
            self.result_pub.publish({'data': room})
            rospy.loginfo(f"发布房间: {room}")
            rospy.sleep(1)
            self.result_pub.publish({'data': goods})
            rospy.loginfo(f"发布商品: {goods}")
            rospy.sleep(0.5)
            self.result_pub.publish({'data': room})
            rospy.loginfo(f"发布房间: {room}")
            rospy.sleep(1)
            self.result_pub.publish({'data': goods})
            rospy.loginfo(f"发布商品: {goods}")
            rospy.sleep(0.5)
            self.result_pub.publish({'data': room})
            rospy.loginfo(f"发布房间: {room}")
            rospy.sleep(1)
            self.result_pub.publish({'data': goods})
            rospy.loginfo(f"发布商品: {goods}")
            
        # 调试用：打印所有检测结果
        # rospy.loginfo("完整检测记录:")
        # for room, items in self.all_detections.items():
        #     rospy.loginfo(f"{room}: {[x[0] for x in items]}")
            
    def run_navigation_sequence(self):
        """修改后的导航序列：start -> roomC -> roomB -> roomA -> end"""
        sequence = ["start", "roomC", "roomB", "roomA", "end"]
        
        for target in sequence:
            if not rospy.is_shutdown() and self.navigation_active:
                if not self.send_navigation_goal(target):
                    # 如果导航到主房间失败，尝试直接导航到room*1
                    if target.startswith("room"):
                        backup_target = f"{target}1"
                        rospy.logwarn(f"无法到达{target}，尝试备用路径{backup_target}")
                        if not self.send_navigation_goal(backup_target):
                            continue
                    
                if self.wait_for_navigation_result(timeout=30.0):
                    if target.startswith("room"):
                        room_base = target.split('1')[0].split('2')[0]  # 处理可能带后缀的情况
                        self.enhanced_detection(room_base)
                        rospy.sleep(1)
                    elif target == "end":
                        self.publish_results()
                else:
                    # 如果导航超时，尝试下一个目标点
                    rospy.logwarn(f"到达{target}超时，尝试继续后续导航")
                    continue
        
        self.navigation_active = False
        self.current_task = None
        rospy.loginfo("导航任务完成！")

if __name__ == '__main__':
    try:
        controller = NavigationController()
        rospy.spin()
    except rospy.ROSInterruptException:
        controller.ros.close()  # 确保关闭rosbridge连接
        rospy.loginfo("节点已关闭")
        