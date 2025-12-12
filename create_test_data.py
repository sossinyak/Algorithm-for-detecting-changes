import cv2
import numpy as np
from pathlib import Path

def create_test_images():
    """Создание тестовых данных"""
    try:
        # Создаем простые тестовые изображения
        data_dir = Path("data/satellite_images")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем тестовое изображение T1
        img1 = np.zeros((512, 512), dtype=np.uint8)
        img1[200:300, 200:300] = 200  # Квадрат
        img1[100:150, 400:450] = 180  # Маленький квадрат
        img1[350:400, 100:180] = 160  # Прямоугольник
        
        # Создаем изображение T2 с изменениями
        img2 = img1.copy()
        img2[250:350, 250:350] = 200  # Сдвинутый квадрат
        img2[100:150, 100:150] = 200   # Новый квадрат
        img2[350:400, 100:180] = 0     # Удаленный прямоугольник
        img2[50:80, 300:350] = 150     # Новый маленький объект
        
        # Добавляем немного шума
        noise = np.random.normal(0, 10, img2.shape).astype(np.int16)
        img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Сохраняем
        cv2.imwrite(str(data_dir / "satellite_T1.png"), img1)
        cv2.imwrite(str(data_dir / "satellite_T2.png"), img2)
        
        print(f"Тестовые данные созданы в {data_dir}/")
        print(f"  satellite_T1.png: {img1.shape}")
        print(f"  satellite_T2.png: {img2.shape}")
        
        # Создаем простой ground truth
        gt = np.zeros((512, 512), dtype=np.uint8)
        gt[250:350, 250:350] = 255  # Сдвинутый квадрат
        gt[100:150, 100:150] = 255   # Новый квадрат
        gt[50:80, 300:350] = 255     # Новый маленький объект
        cv2.imwrite(str(data_dir / "ground_truth.png"), gt)
        print(f"  ground_truth.png: Создан для тестирования")
        
        return img1, img2
        
    except Exception as e:
        print(f"Ошибка при создании тестовых данных: {e}")
        return None, None

if __name__ == "__main__":
    create_test_images()