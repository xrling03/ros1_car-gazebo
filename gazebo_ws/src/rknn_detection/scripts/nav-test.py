#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from time import sleep, time
import rospy
import actionlib
import os
import threading
import cv2
import numpy as np
import warnings
import math
import time
import onnxruntime as ort
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf

warnings.filterwarnings("ignore")

class LaserController:
    def __init__(self):
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.latest_scan = None
        rospy.loginfo("Gazebo激光雷达控制模块初始化完成")

    def scan_callback(self, data):
        self.latest_scan = data

    def get_distance_to_target(self):
        if not self.latest_scan:
            return float('inf')
        
        # 使用Gazebo激光雷达数据（360度扫描）
        ranges = self.latest_scan.ranges
        num_points = len(ranges)
        center_index = num_points // 2
        
        # 取前方30度范围（约15度每侧）
        sample_size = int(num_points * 30 / 360)
        start_idx = max(0, center_index - sample_size // 2)
        end_idx = min(num_points, center_index + sample_size // 2)
        
        # 过滤无效值
        valid_distances = [
            d for d in ranges[start_idx:end_idx] 
            if not math.isinf(d) and not math.isnan(d) and 
               self.latest_scan.range_min <= d <= self.latest_scan.range_max
        ]
        
        if not valid_distances:
            return float('inf')
            
        return np.median(valid_distances)
    
    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def move_forward_slow(self, speed=0.1):
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        return True

    def rotate(self, angular_speed=0.3):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(twist)

class ONNXDetector:
    def __init__(self, model_path, task_type, task_goods_map):
        """初始化ONNX检测器"""
        self.model_path = model_path
        self.task_type = task_type
        self.task_goods_map = task_goods_map
        self.target_goods = task_goods_map.get(task_type, [])
        
        # 初始化ONNX模型
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        rospy.loginfo(f"ONNX模型加载成功. 输入形状: {self.session.get_inputs()[0].shape}")
        
        # 初始化ROS组件
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/cam', Image, self.image_callback)
        self.detection_pub = rospy.Publisher('/detection_results', Image, queue_size=1)
        
        # 检测结果存储
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        
        rospy.loginfo("ONNX视觉识别模块初始化完成")

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

    def postprocess_detections(self, output, scale, offset, original_size, conf_threshold=0.25, nms_threshold=0.45):
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
        valid_indices = max_scores >= conf_threshold
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
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), conf_threshold, nms_threshold)
        
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

    def image_callback(self, msg):
        try:
            # 转换ROS Image为OpenCV格式
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
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
            
            # 更新检测结果
            with self.detection_lock:
                self.latest_detections = detections
            
            # 可视化并发布结果
            if detections:
                vis_img = cv_img.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    color = (0, 255, 0)  # 绿色
                    cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(vis_img, label, (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                self.detection_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"图像处理失败: {str(e)}")

    def get_target_detections(self):
        """获取目标物品的检测结果"""
        with self.detection_lock:
            target_class_ids = [CLASS_NAMES_TO_ID[item] for item in self.target_goods if item in CLASS_NAMES_TO_ID]
            return [det for det in self.latest_detections if det['class_id'] in target_class_ids]

class Navigation:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server(rospy.Duration(5))
        self.laser = LaserController()
        
        # 修改为订阅消息触发
        self.start_nav_sub = rospy.Subscriber('/start_navigation', String, self.start_nav_callback)
        self.navigation_active = False
        rospy.loginfo("导航模块初始化完成，等待启动信号...")

    def start_nav_callback(self, msg):
        """接收到消息后开始导航"""
        if not self.navigation_active:
            self.navigation_active = True
            rospy.loginfo(f"接收到导航启动信号: {msg.data}")
            self.run_navigation(msg.data)

    def run_navigation(self, target_name):
        """执行导航任务"""
        if target_name in TARGET_POINTS:
            x, y, z, w = TARGET_POINTS[target_name]
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.orientation.z = z
            goal.target_pose.pose.orientation.w = w
            
            self.client.send_goal(goal)
            rospy.loginfo(f"开始导航到: {target_name}")
            
            # 等待结果
            self.client.wait_for_result()
            state = self.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo(f"成功到达: {target_name}")
            else:
                rospy.logwarn(f"导航到 {target_name} 失败")
    
    def rotate_in_place(self, angular_speed=0.3, duration=2):
        """原地旋转指定时间"""
        rospy.loginfo("开始原地旋转...")
        start_time = rospy.get_time()
        rate = rospy.Rate(10)
        
        while rospy.get_time() - start_time < duration:
            self.laser.rotate(angular_speed)
            rate.sleep()
            
        self.laser.stop()
        return True

# 全局配置
CLASS_NAMES = {
    0: 'apple', 1: 'banana', 2: 'board', 3: 'cake', 4: 'chili', 5: 'cola',
    6: 'greenlight', 7: 'milk', 8: 'potato', 9: 'redlight', 10: 'tomato', 11: 'watermelon'
}

CLASS_NAMES_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

TARGET_POINTS = {
    "start":    (0.000, 0.000, 0.000, 1.000),
    "room1":    (3.834, 1.039, 0.698, 0.716),
    "room2":    (2.304, 1.640, 0.698, 0.716),
    "room3":    (0.634, 1.461, 0.698, 0.716),
    "end":      (0.000, 0.000, 1.000, 0.000)
    # "pick": (1.10, 3.010, 0.000, 1.000),
    # "wait": (1.10, 3.010, 0.000, 1.000),
    # "crossing": (1.73, 3.920, 0.000, 1.000),
    # "end": (3.52, 0.07, 0.7071, -0.7071)
}

TASK_GOODS_MAP = {
    "Fruit": ["apple", "banana", "watermelon"],
    "Vegetable": ["chili", "tomato", "potato"],
    "Dessert": ["milk", "cake", "cola"]
}

def main():
    rospy.init_node('ucar_nav_modified')
    
    # 初始化导航模块（等待消息触发）
    nav = Navigation()
    
    # 初始化视觉识别模块
    detector = ONNXDetector(
        model_path="/home/ros/Desktop/gazebo/gazebo_test_ws/src/rknn_detection/model/best.onnx",
        task_type="Fruit",  # 默认任务类型
        task_goods_map=TASK_GOODS_MAP
    )
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr("节点中断")
    except Exception as e:
        rospy.logerr(f"程序错误: {e}")