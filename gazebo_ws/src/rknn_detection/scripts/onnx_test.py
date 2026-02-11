#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型识别脚本
输入图片通过ONNX模型做识别
"""

import os
import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import onnxruntime as ort

# 类别映射
CLASS_NAMES = {
    0: 'apple', 1: 'banana', 2: 'board', 3: 'cake', 4: 'chili', 5: 'cola',
    6: 'greenlight', 7: 'milk', 8: 'potato', 9: 'redlight', 10: 'tomato', 11: 'watermelon'
}

def preprocess_image(image_path, target_size=640):
    """图像预处理"""
    # 使用numpy读取支持中文路径
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None, None
    
    original_h, original_w = img.shape[:2]
    
    # 计算缩放比例
    scale = min(target_size / original_h, target_size / original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)
    
    # 缩放图像
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # 创建填充图像
    padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded_img[top:top+new_h, left:left+new_w] = img_resized
    
    return padded_img, scale, (top, left), (original_h, original_w)

def postprocess_detections(output, scale, offset, original_size, conf_threshold=0.25, nms_threshold=0.45):
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

def run_onnx_inference(image_path, model_path, conf_threshold=0.25):
    """ONNX模型推理"""
    try:
        print(f"Loading ONNX model: {model_path}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 预处理
        padded_img, scale, offset, original_size = preprocess_image(image_path)
        if padded_img is None:
            return None, "Image preprocessing failed"
        
        # 转换为NCHW格式并归一化
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        input_data = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        print(f"ONNX input shape: {input_data.shape}")
        print(f"ONNX input dtype: {input_data.dtype}")
        print(f"ONNX input range: [{input_data.min():.6f}, {input_data.max():.6f}]")
        
        # 推理
        start_time = time.time()
        outputs = session.run(None, {'images': input_data})
        inference_time = time.time() - start_time
        
        print(f"ONNX inference time: {inference_time*1000:.2f}ms")
        print(f"ONNX output shape: {outputs[0].shape}")
        print(f"ONNX output dtype: {outputs[0].dtype}")
        print(f"ONNX output range: [{outputs[0].min():.6f}, {outputs[0].max():.6f}]")
        
        # 后处理
        detections = postprocess_detections(outputs[0], scale, offset, original_size, conf_threshold)
        
        return detections, f"ONNX inference successful, time: {inference_time*1000:.2f}ms"
        
    except Exception as e:
        import traceback
        error_msg = f"ONNX inference failed: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def visualize_detections(image_path, detections, output_path=None):
    """可视化检测结果"""
    # 使用numpy读取支持中文路径
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(img, label, (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow("Detection Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='ONNX模型识别脚本')
    parser.add_argument('--onnx-model', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--output', type=str, help='输出图片路径(可选)')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    
    args = parser.parse_args()
    
    print("ONNX模型识别")
    print("=" * 80)
    print(f"ONNX模型: {args.onnx_model}")
    print(f"输入图片: {args.image}")
    print(f"置信度阈值: {args.conf_threshold}")
    
    # 检查模型文件
    if not os.path.exists(args.onnx_model):
        print(f"错误: ONNX模型不存在: {args.onnx_model}")
        return
    
    # 检查图片文件
    if not os.path.exists(args.image):
        print(f"错误: 图片不存在: {args.image}")
        return
    
    # 运行推理
    detections, msg = run_onnx_inference(args.image, args.onnx_model, args.conf_threshold)
    print(f"\n识别结果: {msg}")
    
    if detections:
        print(f"检测到 {len(detections)} 个目标:")
        for det in detections:
            print(f"  {det['class_name']}: 置信度={det['confidence']:.4f}, 位置={det['bbox']}")
        
        # 可视化结果
        visualize_detections(args.image, detections, args.output)
        
        # 保存结果到JSON
        if args.output:
            result_json = os.path.splitext(args.output)[0] + ".json"
            with open(result_json, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_path': args.image,
                    'detections': detections,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {result_json}")
    else:
        print("没有检测到任何目标")

if __name__ == '__main__':
    main()