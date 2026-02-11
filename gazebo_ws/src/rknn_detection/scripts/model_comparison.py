#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX和RKNN模型对比脚本
在RK3588板子上运行，对比两个模型的输出结果
"""

import os
import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

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
        import onnxruntime as ort
        
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

def run_rknn_inference(image_path, model_path, conf_threshold=0.25):
    """RKNN模型推理"""
    try:
        from rknn.api import RKNN
        
        print(f"Loading RKNN model: {model_path}")
        rknn = RKNN(verbose=False)
        
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            return None, f"Failed to load RKNN model, ret: {ret}"
        
        ret = rknn.init_runtime()
        if ret != 0:
            return None, f"Failed to init RKNN runtime, ret: {ret}"
        
        # 预处理
        padded_img, scale, offset, original_size = preprocess_image(image_path)
        if padded_img is None:
            return None, "Image preprocessing failed"
        
        # RKNN使用NHWC格式，BGR顺序，uint8类型
        input_data = padded_img.astype(np.uint8)
        input_data = np.expand_dims(input_data, axis=0)
        
        print(f"RKNN input shape: {input_data.shape}")
        print(f"RKNN input dtype: {input_data.dtype}")
        print(f"RKNN input range: [{input_data.min()}, {input_data.max()}]")
        
        # 推理
        start_time = time.time()
        outputs = rknn.inference(inputs=[input_data], data_format=['nhwc'])
        inference_time = time.time() - start_time
        
        print(f"RKNN inference time: {inference_time*1000:.2f}ms")
        print(f"RKNN output shape: {outputs[0].shape}")
        print(f"RKNN output dtype: {outputs[0].dtype}")
        print(f"RKNN output range: [{outputs[0].min():.6f}, {outputs[0].max():.6f}]")
        
        # 后处理
        detections = postprocess_detections(outputs[0], scale, offset, original_size, conf_threshold)
        
        rknn.release()
        
        return detections, f"RKNN inference successful, time: {inference_time*1000:.2f}ms"
        
    except Exception as e:
        import traceback
        error_msg = f"RKNN inference failed: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def compare_detections(onnx_detections, rknn_detections, tolerance=0.1):
    """对比检测结果"""
    comparison = {
        'onnx_count': len(onnx_detections) if onnx_detections else 0,
        'rknn_count': len(rknn_detections) if rknn_detections else 0,
        'matches': [],
        'onnx_only': [],
        'rknn_only': [],
        'differences': []
    }
    
    if not onnx_detections or not rknn_detections:
        return comparison
    
    # 简单匹配：按置信度排序后逐一对比
    onnx_sorted = sorted(onnx_detections, key=lambda x: x['confidence'], reverse=True)
    rknn_sorted = sorted(rknn_detections, key=lambda x: x['confidence'], reverse=True)
    
    matched_rknn = set()
    
    for onnx_det in onnx_sorted:
        best_match = None
        best_score = float('inf')
        best_idx = -1
        
        for i, rknn_det in enumerate(rknn_sorted):
            if i in matched_rknn:
                continue
            
            if onnx_det['class_id'] == rknn_det['class_id']:
                conf_diff = abs(onnx_det['confidence'] - rknn_det['confidence'])
                if conf_diff < best_score:
                    best_score = conf_diff
                    best_match = rknn_det
                    best_idx = i
        
        if best_match and best_score <= tolerance:
            matched_rknn.add(best_idx)
            comparison['matches'].append({
                'onnx': onnx_det,
                'rknn': best_match,
                'conf_diff': best_score
            })
        else:
            comparison['onnx_only'].append(onnx_det)
    
    # 未匹配的RKNN检测
    for i, rknn_det in enumerate(rknn_sorted):
        if i not in matched_rknn:
            comparison['rknn_only'].append(rknn_det)
    
    return comparison

def run_comparison(image_path, onnx_model, rknn_model, conf_threshold=0.25):
    """运行单张图片的对比"""
    print(f"\nProcessing image: {Path(image_path).name}")
    print("=" * 80)
    
    # ONNX推理
    print("\nRunning ONNX inference...")
    print("-" * 40)
    onnx_detections, onnx_msg = run_onnx_inference(image_path, onnx_model, conf_threshold)
    print(f"ONNX result: {onnx_msg}")
    if onnx_detections:
        print(f"ONNX detections: {len(onnx_detections)}")
        for det in onnx_detections:
            print(f"  {det['class_name']}: {det['confidence']:.4f}")
    
    # RKNN推理
    print("\nRunning RKNN inference...")
    print("-" * 40)
    rknn_detections, rknn_msg = run_rknn_inference(image_path, rknn_model, conf_threshold)
    print(f"RKNN result: {rknn_msg}")
    if rknn_detections:
        print(f"RKNN detections: {len(rknn_detections)}")
        for det in rknn_detections:
            print(f"  {det['class_name']}: {det['confidence']:.4f}")
    
    # 对比结果
    print("\nComparison analysis...")
    print("-" * 40)
    comparison = compare_detections(onnx_detections, rknn_detections)
    
    print(f"ONNX detections: {comparison['onnx_count']}")
    print(f"RKNN detections: {comparison['rknn_count']}")
    print(f"Matched pairs: {len(comparison['matches'])}")
    print(f"ONNX only: {len(comparison['onnx_only'])}")
    print(f"RKNN only: {len(comparison['rknn_only'])}")
    
    if comparison['matches']:
        print("\nMatched detections:")
        for match in comparison['matches']:
            onnx_det = match['onnx']
            rknn_det = match['rknn']
            conf_diff = match['conf_diff']
            print(f"  {onnx_det['class_name']}: ONNX={onnx_det['confidence']:.4f}, RKNN={rknn_det['confidence']:.4f}, diff={conf_diff:.4f}")
    
    if comparison['onnx_only']:
        print("\nONNX only detections:")
        for det in comparison['onnx_only']:
            print(f"  {det['class_name']}: {det['confidence']:.4f}")
    
    if comparison['rknn_only']:
        print("\nRKNN only detections:")
        for det in comparison['rknn_only']:
            print(f"  {det['class_name']}: {det['confidence']:.4f}")
    
    return {
        'image_name': Path(image_path).name,
        'onnx_detections': onnx_detections,
        'rknn_detections': rknn_detections,
        'onnx_message': onnx_msg,
        'rknn_message': rknn_msg,
        'comparison': comparison
    }

def main():
    parser = argparse.ArgumentParser(description='ONNX vs RKNN model comparison')
    parser.add_argument('--onnx-model', type=str, default='models/best.onnx', help='ONNX model path')
    parser.add_argument('--rknn-model', type=str, default='output/yolov11_rk3588_fp16_simulation.rknn', help='RKNN model path')
    parser.add_argument('--images', type=str, default='image', help='Images directory')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--max-images', type=int, default=10, help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    print("ONNX vs RKNN Model Comparison")
    print("=" * 80)
    print(f"ONNX model: {args.onnx_model}")
    print(f"RKNN model: {args.rknn_model}")
    print(f"Images directory: {args.images}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Max images: {args.max_images}")
    
    # 检查模型文件
    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model not found: {args.onnx_model}")
        return
    
    if not os.path.exists(args.rknn_model):
        print(f"Error: RKNN model not found: {args.rknn_model}")
        return
    
    # 获取图片列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(args.images).glob(f'*{ext}'))
        image_files.extend(Path(args.images).glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"Error: No images found in {args.images}")
        return
    
    image_files = image_files[:args.max_images]
    print(f"Found {len(image_files)} images to process")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理图片
    results = []
    for image_path in image_files:
        result = run_comparison(str(image_path), args.onnx_model, args.rknn_model, args.conf_threshold)
        results.append(result)
    
    # 保存结果
    output_file = os.path.join(output_dir, 'comparison_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'onnx_model': args.onnx_model,
                'rknn_model': args.rknn_model,
                'conf_threshold': args.conf_threshold,
                'total_images': len(image_files)
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # 统计总结
    total_onnx = sum(len(r['onnx_detections']) if r['onnx_detections'] else 0 for r in results)
    total_rknn = sum(len(r['rknn_detections']) if r['rknn_detections'] else 0 for r in results)
    total_matches = sum(len(r['comparison']['matches']) for r in results)
    
    print(f"\nSummary:")
    print(f"Total ONNX detections: {total_onnx}")
    print(f"Total RKNN detections: {total_rknn}")
    print(f"Total matched pairs: {total_matches}")
    print(f"Match rate: {total_matches/max(total_onnx, total_rknn)*100:.1f}%" if max(total_onnx, total_rknn) > 0 else "N/A")

if __name__ == '__main__':
    main()
