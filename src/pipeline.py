import yaml
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

@dataclass
class ChangeDetectionConfig:
    """Контейнер для параметров конфигурации"""
    preprocessing: Dict
    detection: Dict
    postprocessing: Dict
    evaluation: Dict
    
    @classmethod
    def load(cls, config_path: str) -> 'ChangeDetectionConfig':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, config_path: str):
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

class ChangeDetectionPipeline:
    """Основной конвейер для обнаружения изменений"""
    
    def __init__(self, config: Dict):
        self.config = ChangeDetectionConfig(**config)
        self.results = {}
        
    def run(self, image1_path: str, image2_path: str,ground_truth_path: Optional[str] = None) -> Dict:
        """Запуск полного конвейера обработки"""

        print("\nЗАПУСК КОНВЕЙЕРА ОБНАРУЖЕНИЯ ИЗМЕНЕНИЙ")
        
        # Загрузка изображений
        img1 = self._load_image(image1_path)
        img2 = self._load_image(image2_path)
        print(f"\nЗагрузка изображений:")
        print(f"T1: {img1.shape}, диапазон: [{img1.min()}, {img1.max()}]")
        print(f"T2: {img2.shape}, диапазон: [{img2.min()}, {img2.max()}]")
        
        # Предобработка
        from preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor(self.config.preprocessing)

        img1_proc, img2_proc = preprocessor.process_pair(img1, img2)
        print(f"\nПредобработка:")
        print(f"Размер после обработки: {img1_proc.shape}")
        
        # Детекция изменений
        from detection import ChangeDetector
        
        detector = ChangeDetector(
            method=self.config.detection['method'],
            params=self.config.detection
        )
        change_mask, detection_info = detector.detect(img1_proc, img2_proc)
        print(f"\nДетекция изменений:")
        print(f"Метод: {detection_info['method']}")
        print(f"Процент изменений: {np.sum(change_mask > 0) / change_mask.size * 100:.2f}%")
        
        # Постобработка
        final_mask = self._postprocess_mask(change_mask)
        
        # Оценка (если есть ground truth)
        metrics = {}
        if ground_truth_path:
            gt_mask = self._load_ground_truth(ground_truth_path, final_mask.shape)
            
            from evaluation import ChangeDetectionEvaluator

            evaluator = ChangeDetectionEvaluator()
            metrics = evaluator.compute_metrics(final_mask, gt_mask, self.config.detection['method'])
            print(f"\nОценка результатов:")
            print(f"F1-Score: {metrics['f1_score']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
        
        # Сохранение результатов
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'processed_images': {
                'T1_shape': img1_proc.shape,
                'T2_shape': img2_proc.shape
            },
            'change_mask': final_mask,
            'detection_info': detection_info,
            'metrics': metrics
        }
        
        return self.results
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Загрузка изображения"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        return img
    
    def _load_ground_truth(self, gt_path: str, target_shape: Tuple) -> np.ndarray:
        """Загрузка ground truth маски"""
        print(f"   📂 Загрузка ground truth: {gt_path}")
        print(f"   📂 Файл существует: {Path(gt_path).exists()}")
        
        gt = self._load_image(gt_path)
        print(f"   📂 Загружен ground truth: shape={gt.shape}, "
              f"min={gt.min()}, max={gt.max()}, "
              f"изменений={np.sum(gt > 0)} пикселей")
        
        # Приведение к бинарному формату
        if gt.max() > 1:
            print(f"   📂 Конвертация в бинарный формат...")
            gt_binary = (gt > 127).astype(np.uint8) * 255
            print(f"   📂 После бинаризации: min={gt_binary.min()}, "
                  f"max={gt_binary.max()}, "
                  f"изменений={np.sum(gt_binary > 0)} пикселей")
            gt = gt_binary
        
        # Приведение к целевому размеру
        if gt.shape != target_shape:
            print(f"   📂 Изменение размера ground truth: {gt.shape} -> {target_shape}")
            gt = cv2.resize(gt, (target_shape[1], target_shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
            print(f"   📂 После изменения размера: shape={gt.shape}, "
                  f"изменений={np.sum(gt > 0)} пикселей")
        
        return gt
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Постобработка маски изменений"""
        if not self.config.postprocessing:
            return mask
        
        result = mask.copy()
        
        # Морфологические операции
        morph_config = self.config.postprocessing.get('morphological', {})
        if morph_config.get('operations'):
            kernel_size = morph_config.get('kernel_size', 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            for operation in morph_config['operations']:
                if operation == 'open':
                    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                elif operation == 'close':
                    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        # Фильтрация по площади
        filter_config = self.config.postprocessing.get('filtering', {})
        min_area = filter_config.get('min_area_pixels', 50)
        
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                result, connectivity=filter_config.get('connectivity', 8)
            )
            
            # Создаем чистую маску
            cleaned = np.zeros_like(result)
            
            for i in range(1, num_labels):  # Пропускаем фон (0)
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == i] = 255
            
            result = cleaned
        
        return result
    
    def save_results(self, output_dir: str):
        """Сохранение результатов в файлы"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохранение маски
        if 'change_mask' in self.results:
            mask_path = output_path / "change_mask.png"
            cv2.imwrite(str(mask_path), self.results['change_mask'])
            print(f"\nМаска сохранена: {mask_path}")
        
        # Сохранение конфигурации
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        # Сохранение метрик
        if self.results.get('metrics'):
            metrics_path = output_path / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.results['metrics'], f, indent=2)
        
        # Сохранение полного отчета
        report_path = output_path / "report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nРезультаты сохранены в: {output_path}")