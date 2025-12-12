import cv2
import numpy as np
from typing import Tuple, Dict, Optional

class ImagePreprocessor:
    """Модуль предобработки изображений"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def process_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основной метод обработки пары изображений"""
        # Выравнивание размеров
        img1_aligned, img2_aligned = self.align_sizes(img1, img2)
        
        # Нормализация
        img1_norm, img2_norm = self.normalize_images(img1_aligned, img2_aligned)
        
        # Гистограммное выравнивание
        if self.config.get('normalization') == 'histogram_matching':
            from skimage.exposure import match_histograms
            img2_norm = match_histograms(img2_norm, img1_norm)
        
        return img1_norm, img2_norm
    
    def align_sizes(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Выравнивание размеров изображений"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        method = self.config.get('resize_method', 'crop')
        
        if method == 'crop':
            # Обрезание до минимального размера
            h_min = min(h1, h2)
            w_min = min(w1, w2)
            img1 = img1[:h_min, :w_min]
            img2 = img2[:h_min, :w_min]
        elif method == 'resize':
            # Изменение размера до среднего
            h_avg = (h1 + h2) // 2
            w_avg = (w1 + w2) // 2
            img1 = cv2.resize(img1, (w_avg, h_avg))
            img2 = cv2.resize(img2, (w_avg, h_avg))
        
        return img1, img2
    
    def normalize_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Нормализация изображений к диапазону [0, 1]"""
        # Конвертация в float32
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # Нормализация
        if img1.max() > 1:
            img1_f = img1_f / 255.0 if img1.dtype == np.uint8 else img1_f / 65535.0
        
        if img2.max() > 1:
            img2_f = img2_f / 255.0 if img2.dtype == np.uint8 else img2_f / 65535.0
        
        return img1_f, img2_f
    
    @staticmethod
    def load_image(image_path: str, grayscale: bool = True) -> np.ndarray:
        """Загрузка изображения"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        return img