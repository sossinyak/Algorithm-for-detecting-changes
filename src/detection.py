import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_yen
from typing import Tuple, Dict

class ChangeDetector:
    """Класс для детекции изменений различными методами"""
    
    def __init__(self, method: str = 'cva', params: Dict = None):
        self.method = method
        self.params = params or {}
        
    def detect(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Основной метод детекции изменений"""
        if self.method == 'differencing':
            return self._difference_method(img1, img2)
        elif self.method == 'ratio':
            return self._ratio_method(img1, img2)
        elif self.method == 'cva':
            return self._cva_method(img1, img2)
        elif self.method == 'ndbi_diff':
            return self._ndbi_method(img1, img2)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
    
    def _difference_method(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Метод абсолютной разности"""
        diff = np.abs(img2.astype(np.float32) - img1.astype(np.float32))
        threshold = threshold_otsu(diff)
        mask = (diff > threshold).astype(np.uint8) * 255
        
        return mask, {
            'method': 'absolute_difference',
            'threshold': float(threshold),
            'diff_range': (float(diff.min()), float(diff.max()))
        }
    
    def _ratio_method(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Метод отношения изображений"""
        epsilon = 1e-10
        ratio = (img2.astype(np.float32) + epsilon) / (img1.astype(np.float32) + epsilon)
        ratio_norm = np.abs(ratio - 1.0)
        
        threshold = self.params.get('ratio_threshold', 0.3)
        mask = (ratio_norm > threshold).astype(np.uint8) * 255
        
        return mask, {
            'method': 'ratio_image',
            'threshold': float(threshold),
            'ratio_range': (float(ratio.min()), float(ratio.max()))
        }
    
    def _cva_method(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Change Vector Analysis (CVA)"""
        # Нормализация
        img1_norm = img1.astype(np.float32) / 255.0 if img1.max() > 1 else img1.astype(np.float32)
        img2_norm = img2.astype(np.float32) / 255.0 if img2.max() > 1 else img2.astype(np.float32)
        
        # Вектор изменений
        change_vector = img2_norm - img1_norm
        
        # Модуль вектора
        if change_vector.ndim == 3:
            magnitude = np.sqrt(np.sum(change_vector**2, axis=2))
        else:
            magnitude = np.abs(change_vector)
        
        # Нормализация
        magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)
        
        # Выбор порога
        threshold_method = self.params.get('threshold_method', 'otsu')
        
        if threshold_method == 'otsu':
            threshold = threshold_otsu(magnitude_norm * 255) / 255.0
        elif threshold_method == 'yen':
            threshold = threshold_yen(magnitude_norm * 255) / 255.0
        else:
            threshold = self.params.get('manual_threshold', 0.3)
        
        # Бинаризация
        mask = (magnitude_norm > threshold).astype(np.uint8) * 255
        
        return mask, {
            'method': 'cva',
            'threshold': float(threshold),
            'magnitude_range': (float(magnitude.min()), float(magnitude.max()))
        }
    
    def _ndbi_method(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Метод на основе разности индексов застройки"""
        # Для градаций серого используем упрощенный подход
        if img1.ndim == 2:
            ndbi1 = img1.astype(float) / 255.0 if img1.max() > 1 else img1.astype(float)
            ndbi2 = img2.astype(float) / 255.0 if img2.max() > 1 else img2.astype(float)
        else:
            # Для цветных изображений
            ndbi1 = (img1[:, :, 2].astype(float) - img1[:, :, 1].astype(float)) / \
                   (img1[:, :, 2].astype(float) + img1[:, :, 1].astype(float) + 1e-10)
            ndbi2 = (img2[:, :, 2].astype(float) - img2[:, :, 1].astype(float)) / \
                   (img2[:, :, 2].astype(float) + img2[:, :, 1].astype(float) + 1e-10)
        
        diff_ndbi = np.abs(ndbi2 - ndbi1)
        threshold = threshold_otsu(diff_ndbi * 255) / 255.0
        mask = (diff_ndbi > threshold).astype(np.uint8) * 255
        
        return mask, {
            'method': 'ndbi_difference',
            'threshold': float(threshold),
            'ndbi_range_t1': (float(ndbi1.min()), float(ndbi1.max())),
            'ndbi_range_t2': (float(ndbi2.min()), float(ndbi2.max()))
        }