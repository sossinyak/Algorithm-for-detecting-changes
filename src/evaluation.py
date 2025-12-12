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
        
    def compute_metrics(self, predicted: np.ndarray, ground_truth: np.ndarray, method_name: str = "") -> Dict:
        """Вычисление всех метрик качества"""
        
        # Конвертация в бинарный формат
        pred_bin = (predicted > 0).astype(bool)
        gt_bin = (ground_truth > 0).astype(bool)
        
        # Матрица ошибок
        tp = np.sum(pred_bin & gt_bin)
        fp = np.sum(pred_bin & ~gt_bin)
        fn = np.sum(~pred_bin & gt_bin)
        tn = np.sum(~pred_bin & ~gt_bin)
        
        # Основные метрики
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Метрики: Precision={precision:.5f}, Recall={recall:.5f}, F1={f1:.5f}")
        
        # IoU (Jaccard Index)
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        results = {
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
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
            if metrics and isinstance(metrics, dict):
                row = {'method': method_name}
                for key, value in metrics.items():
                    if isinstance(value, (int, float, str)):
                        row[key] = value
                comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Сортируем по F1-Score, если он есть
        if 'f1_score' in df.columns and not df.empty:
            df = df.sort_values('f1_score', ascending=False)

        return df  
    
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