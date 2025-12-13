import cv2
import numpy as np
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
        elif self.method == 'pca':
            return self._pca_method(img1, img2)
        elif self.method == 'cascade':
            return self._cascade_method(img1, img2)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
    
    def _difference_method(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Метод разности изображений"""
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
        """Метод на основе разности индексов"""
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
    
    def _pca_method(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """PCA-based change detection"""
        from sklearn.decomposition import PCA
        
        # Объединяем изображения
        h, w = img1.shape
        X = np.column_stack([img1.ravel(), img2.ravel()])
        
        # Применяем PCA с 2 компонентами
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Вторая компонента часто содержит изменения
        changes = X_pca[:, 1].reshape(h, w)
        
        # Нормализация
        changes_normalized = (changes - changes.min()) / (changes.max() - changes.min())
        
        # Пороговая обработка
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(changes_normalized * 255) / 255.0
        
        mask = (changes_normalized > threshold).astype(np.uint8) * 255
        
        return mask, {
            'method': 'pca',
            'threshold': float(threshold),
            'explained_variance': pca.explained_variance_ratio_.tolist()
        }
    
    # Добавим в detection.py новый метод _cascade_method
    def _cascade_method(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Каскадный метод с CLAHE, многоуровневой фильтрацией и контурным анализом"""

        # 1. CLAHE для адаптивного контрастирования
        clahe = cv2.createCLAHE(
            clipLimit=self.params.get('clahe_clip_limit', 2.0),
            tileGridSize=self.params.get('clahe_grid_size', (8, 8))
        )

        img1_clahe = clahe.apply((img1 * 255).astype(np.uint8) if img1.max() <= 1 else img1.astype(np.uint8))
        img2_clahe = clahe.apply((img2 * 255).astype(np.uint8) if img2.max() <= 1 else img2.astype(np.uint8))

        # 2. Многоуровневая фильтрация (Гаусс + медианная)
        # Гауссово размытие для уменьшения шума
        gauss_kernel = self.params.get('gauss_kernel', (5, 5))
        img1_gauss = cv2.GaussianBlur(img1_clahe, gauss_kernel, 0)
        img2_gauss = cv2.GaussianBlur(img2_clahe, gauss_kernel, 0)

        # Медианная фильтрация для сохранения границ
        median_kernel = self.params.get('median_kernel', 3)
        img1_filtered = cv2.medianBlur(img1_gauss, median_kernel)
        img2_filtered = cv2.medianBlur(img2_gauss, median_kernel)

        # 3. Разностное изображение с адаптивной нормализацией
        diff = cv2.absdiff(img2_filtered, img1_filtered)

        # 4. Контурный анализ (оператор Кэнни)
        # Находим контуры на обоих изображениях
        edges1 = cv2.Canny(img1_filtered, 
                          threshold1=self.params.get('canny_low', 50),
                          threshold2=self.params.get('canny_high', 150))
        edges2 = cv2.Canny(img2_filtered, 
                          threshold1=self.params.get('canny_low', 50),
                          threshold2=self.params.get('canny_high', 150))

        # Объединение контуров
        combined_edges = cv2.bitwise_or(edges1, edges2)

        # 5. Комбинирование разностного изображения с контурами
        # Нормализация разности
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        # Усиление областей с контурами
        edge_weight = self.params.get('edge_weight', 1.5)
        enhanced_diff = diff_norm.copy().astype(np.float32)
        enhanced_diff[combined_edges > 0] *= edge_weight

        # 6. Адаптивная пороговая обработка
        # Используем метод Отсу для автоматического выбора порога
        enhanced_diff_8bit = enhanced_diff.astype(np.uint8)

        # Применяем несколько пороговых методов и комбинируем результаты
        otsu_thresh, otsu_mask = cv2.threshold(enhanced_diff_8bit, 0, 255, 
                                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Адаптивный порог для разных областей изображения
        adaptive_mask = cv2.adaptiveThreshold(enhanced_diff_8bit, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 
                                             self.params.get('adaptive_block_size', 11),
                                             self.params.get('adaptive_c', 2))

        # 7. Комбинирование масок
        combined_mask = cv2.bitwise_and(otsu_mask, adaptive_mask)

        # 8. Морфологическая постобработка
        # Открытие для удаления мелкого шума
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (self.params.get('open_kernel', 3), 
                                               self.params.get('open_kernel', 3)))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)

        # Закрытие для заполнения небольших разрывов
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (self.params.get('close_kernel', 5), 
                                                self.params.get('close_kernel', 5)))
        final_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)

        # 9. Фильтрация по минимальной площади
        min_area = self.params.get('min_area', 50)
        if min_area > 0:
            # Находим компоненты связности
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                final_mask, connectivity=8
            )

            # Создаем чистую маску с учетом минимальной площади
            filtered_mask = np.zeros_like(final_mask)
            for i in range(1, num_labels):  # Пропускаем фон (0)
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered_mask[labels == i] = 255
            
            final_mask = filtered_mask
        
        return final_mask, {
            'method': 'cascade_clahe_filter_canny',
            'parameters': {
                'clahe_clip_limit': self.params.get('clahe_clip_limit', 2.0),
                'clahe_grid_size': self.params.get('clahe_grid_size', (8, 8)),
                'gauss_kernel': gauss_kernel,
                'median_kernel': median_kernel,
                'canny_thresholds': (self.params.get('canny_low', 50), 
                                    self.params.get('canny_high', 150)),
                'edge_weight': edge_weight,
                'adaptive_block_size': self.params.get('adaptive_block_size', 11),
                'otsu_threshold': float(otsu_thresh)
            },
            'change_percentage': np.sum(final_mask > 0) / final_mask.size * 100
        }
    