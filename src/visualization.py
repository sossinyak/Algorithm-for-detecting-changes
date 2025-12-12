import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd

class ResultVisualizer:
    """Класс для визуализации результатов"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def create_comparison_figure(self, img1: np.ndarray, img2: np.ndarray, change_mask: np.ndarray,
                                ground_truth: Optional[np.ndarray] = None, diff_map: Optional[np.ndarray] = None,
                                title: str = "Результаты обнаружения изменений") -> plt.Figure:
        """Результаты"""
        if diff_map is None:
            diff_map = np.abs(img2.astype(float) - img1.astype(float))
        
        n_rows = 2 if ground_truth is None else 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        
        # Исходные изображения и разность
        self._plot_image(axes[0, 0], img1, "Изображение T1", cmap='gray')
        self._plot_image(axes[0, 1], img2, "Изображение T2", cmap='gray')
        self._plot_image(axes[0, 2], diff_map, "Разностная карта", cmap='hot')
        
        # Результаты
        self._plot_image(axes[1, 0], change_mask, "Маска изменений", cmap='gray')
        overlay = self._create_overlay(img1, change_mask)
        self._plot_image(axes[1, 1], overlay, "Наложение маски на T1")
        
        # Статистика изменений
        change_percentage = np.sum(change_mask > 0) / change_mask.size * 100
        axes[1, 2].text(0.1, 0.5, 
                       f"Изменения: {change_percentage:.2f}%\n"
                       f"Пикселей: {np.sum(change_mask > 0):,}\n"
                       f"Размер: {change_mask.shape}",
                       fontsize=12, verticalalignment='center')
        axes[1, 2].axis('off')
        axes[1, 2].set_title("Статистика изменений")
        
        # Ground truth и оценка (если есть)
        if ground_truth is not None:
            self._plot_image(axes[2, 0], ground_truth, "Ground Truth", cmap='gray')
            
            # Матрица ошибок
            from sklearn.metrics import confusion_matrix
            pred_binary = (change_mask > 127).astype(int).ravel()
            gt_binary = (ground_truth > 127).astype(int).ravel()
            cm = confusion_matrix(gt_binary, pred_binary)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 1],
                       xticklabels=['Фон', 'Изменение'],
                       yticklabels=['Фон', 'Изменение'])
            axes[2, 1].set_title('Матрица ошибок')
            axes[2, 1].set_ylabel('Истинный класс')
            axes[2, 1].set_xlabel('Предсказанный класс')
            
            # ROC кривая
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(gt_binary, pred_binary)
            roc_auc = auc(fpr, tpr)
            
            axes[2, 2].plot(fpr, tpr, color='darkorange', lw=2,
                          label=f'ROC (AUC = {roc_auc:.2f})')
            axes[2, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[2, 2].set_xlim([0.0, 1.0])
            axes[2, 2].set_ylim([0.0, 1.05])
            axes[2, 2].set_xlabel('False Positive Rate')
            axes[2, 2].set_ylabel('True Positive Rate')
            axes[2, 2].set_title('ROC кривая')
            axes[2, 2].legend(loc="lower right")
            axes[2, 2].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_image(self, ax, image, title, cmap=None):
        """Вспомогательная функция для отображения изображения"""
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    
    def _create_overlay(self, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Создание наложения маски на изображение"""
        if len(background.shape) == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
        
        if len(mask.shape) == 2:
            mask_rgb = np.zeros_like(background)
            mask_rgb[:, :, 0] = mask  # Красный канал
        else:
            mask_rgb = mask
        
        # Наложение с прозрачностью
        overlay = cv2.addWeighted(background, 0.7, mask_rgb, 0.3, 0)
        
        return overlay
    
    def save_visualization(self, fig: plt.Figure, output_path: str, dpi: int = 150):
        """Сохранение визуализации в файл"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Визуализация сохранена: {output_path}")
    
    @staticmethod
    def create_comparison_plot(results_df: pd.DataFrame, output_path: str):
        """Создание графика сравнения методов"""
        # Проверяем, есть ли нужные колонки
        required_columns = ['f1_score', 'precision', 'recall', 'iou']
        
        # Проверяем, какие колонки есть в DataFrame
        available_columns = []
        for col in required_columns:
            if col in results_df.columns:
                available_columns.append(col)
        
        if not available_columns:
            print(f"В DataFrame нет ни одной из требуемых колонок: {required_columns}")
            print(f"Доступные колонки: {list(results_df.columns)}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Барплот метрик
        methods = results_df['method'].tolist() if 'method' in results_df.columns else []
        if not methods:
            # Если нет колонки 'method', используем индексы
            methods = [f"Method {i}" for i in range(len(results_df))]
        
        x = np.arange(len(methods))
        width = 0.8 / len(available_columns)
        
        for i, metric in enumerate(available_columns):
            if metric in results_df.columns:
                values = results_df[metric].tolist()
                axes[0].bar(x + i*width - width*(len(available_columns)-1)/2, 
                           values, width, label=metric)
        
        axes[0].set_xlabel('Метод')
        axes[0].set_ylabel('Значение')
        axes[0].set_title('Сравнение метрик по методам')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Тепловая карта (если есть данные и методы)
        if len(methods) > 0 and len(available_columns) > 0:
            # Создаем DataFrame для тепловой карты
            heatmap_data = []
            for method in methods:
                row = []
                for metric in available_columns:
                    # Находим значение для этого метода и метрики
                    if 'method' in results_df.columns:
                        value = results_df.loc[results_df['method'] == method, metric].values
                    else:
                        value = results_df.iloc[methods.index(method)][metric]
                    
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        row.append(float(value[0]))
                    else:
                        row.append(float(value))
                heatmap_data.append(row)
            
            heatmap_df = pd.DataFrame(heatmap_data, 
                                     index=methods, 
                                     columns=available_columns)
            
            sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[1], cbar_kws={'label': 'Значение'})
            axes[1].set_title('Тепловая карта метрик')
            axes[1].set_ylabel('Метод')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График сравнения сохранен: {output_path}")