import numpy as np
import pandas as pd
import json

from typing import Dict, List, Union
from pathlib import Path
from datetime import datetime

class ChangeDetectionEvaluator:
    """Комплексная оценка результатов детекции изменений"""
    
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ['precision', 'recall', 'f1', 'iou', 'accuracy']
        
    def compute_metrics(self, 
                       predicted: np.ndarray, 
                       ground_truth: np.ndarray,
                       method_name: str = "") -> Dict:
        """Вычисление всех метрик качества"""
        print(f"ВЫЧИСЛЕНИЕ МЕТРИК:")
        print(f"Predicted: shape={predicted.shape}, "
              f"min={predicted.min()}, max={predicted.max()}, "
              f"изменений={np.sum(predicted > 0)}")
        print(f"Ground truth: shape={ground_truth.shape}, "
              f"min={ground_truth.min()}, max={ground_truth.max()}, "
              f"изменений={np.sum(ground_truth > 0)}")
        
        # Конвертация в бинарный формат
        pred_bin = (predicted > 0).astype(bool)
        gt_bin = (ground_truth > 0).astype(bool)
        
        print(f"После бинаризации:")
        print(f"Predicted: {np.sum(pred_bin)} True, {np.sum(~pred_bin)} False")
        print(f"Ground truth: {np.sum(gt_bin)} True, {np.sum(~gt_bin)} False")
        
        # Матрица ошибок
        tp = np.sum(pred_bin & gt_bin)
        fp = np.sum(pred_bin & ~gt_bin)
        fn = np.sum(~pred_bin & gt_bin)
        tn = np.sum(~pred_bin & ~gt_bin)
        
        print(f"Матрица ошибок:")
        print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        # Основные метрики
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Метрики: Precision={precision:.3f}, "
              f"Recall={recall:.3f}, F1={f1:.3f}")
        
        # IoU (Jaccard Index)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        results = {
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'total_pixels': int(tp + fp + fn + tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou),
            'accuracy': float(accuracy)
        }
        
        return results
    
    def compare_methods(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Сравнение нескольких методов в табличном формате"""
        comparison_data = []
        
        for method_name, metrics in results_dict.items():
            row = {'method': method_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
    
    def generate_report(self, results: Dict, output_path: Union[str, Path] = None) -> Dict:
        """Отчет"""
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'config': results.get('config', {}),
                'metrics': results.get('metrics', {}),
                'detection_info': results.get('detection_info', {}),
                'image_info': results.get('image_info', {})
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Отчет сохранен: {output_path}")
            return report
        
        return results.get('metrics', {})