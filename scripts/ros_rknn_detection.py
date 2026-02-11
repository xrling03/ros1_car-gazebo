#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型识别节点 - 从Gazebo仿真小车(/cam)获取图像并识别
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import onnxruntime as ort

# 类别映射
CLASS_NAMES = {
    0: 'apple', 1: 'banana', 2: 'board', 3: 'cake', 4: 'chili', 5: 'cola',
    6: 'greenlight', 7: 'milk', 8: 'potato', 9: 'redlight', 10: 'tomato', 11: 'watermelon'
}

class ONNXDetectorNode:
    def __init__(self):
        rospy.init_node('onnx_detector_node')
        
        # 参数配置
        self.model_path = rospy.get_param('~model_path', '/home/ros/Desktop/gazebo/gazebo_test_ws/src/rknn_detection/model/best.onnx')
        self.camera_topic = rospy.get_param('~camera_topic', '/cam')
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.25)
        self.nms_threshold = rospy.get_param('~nms_threshold', 0.45)
        self.visualization = rospy.get_param('~visualization', True)
        
        # 初始化ONNX模型
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        rospy.loginfo(f"ONNX模型加载成功. 输入形状: {self.session.get_inputs()[0].shape}")
        
        # 初始化ROS组件
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.detection_pub = rospy.Publisher('/detection_results', Image, queue_size=1)
        
        rospy.loginfo("ONNX检测节点已启动，等待图像输入...")

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

    def visualize_detections(self, cv_img, detections):
        """可视化检测结果"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(cv_img, label, (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return cv_img

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
            
            # 打印检测结果
            if detections:
                rospy.loginfo(f"检测到 {len(detections)} 个目标: {[d['class_name'] for d in detections]}")
            
            # 可视化并发布结果
            if self.visualization and detections:
                vis_img = self.visualize_detections(cv_img.copy(), detections)
                self.detection_pub.publish(self.bridge.cv2_to_imgmsg(vis_img, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"图像处理失败: {str(e)}")

if __name__ == '__main__':
    try:
        node = ONNXDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass