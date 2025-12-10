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
        
        # Сортировка по F1-score, если он есть
        if 'f1_score' in df.columns:
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
            
            print(f"Report saved to {output_path}")
            return report
        
        return results.get('metrics', {})
    
    @staticmethod
    def calculate_roc_curve(predicted: np.ndarray, ground_truth: np.ndarray):
        """Вычисление ROC кривой"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(
            ground_truth.ravel() > 0,
            predicted.ravel()
        )
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }