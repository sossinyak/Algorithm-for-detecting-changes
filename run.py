import argparse
import yaml
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Добавляем путь к текущей директории и src
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))


from pipeline import ChangeDetectionPipeline
from visualization import ResultVisualizer
from evaluation import ChangeDetectionEvaluator


def create_test_data():
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

def main():
    parser = argparse.ArgumentParser(
        description='Система обнаружения изменений на спутниковых снимках',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" Примеры использования:
                python run.py                               - Запуск с настройками по умолчанию
                python run.py --create-test-data           - Создать тестовые данные
                python run.py --compare                    - Сравнить все методы
             """
    )
    
    parser.add_argument('--config', type=str, default='config/params.yaml',
                       help='Файл конфигурации (по умолчанию: config/params.yaml)')
    parser.add_argument('--image1', type=str, default='data/satellite_images/satellite_T1.png',
                       help='Путь к изображению T1')
    parser.add_argument('--image2', type=str, default='data/satellite_images/satellite_T2.png',
                       help='Путь к изображению T2')
    parser.add_argument('--ground-truth', type=str,
                       help='Путь к ground truth маске')
    parser.add_argument('--output', type=str, default='results',
                       help='Выходная директория')
    parser.add_argument('--compare', action='store_true',
                       help='Сравнить несколько методов')
    parser.add_argument('--create-test-data', action='store_true',
                       help='Создать тестовые данные')
    parser.add_argument('--method', type=str,
                       choices=['differencing', 'cva', 'ratio', 'ndbi_diff'],
                       help='Конкретный метод для использования')
    
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Неизвестные аргументы: {unknown}")
        parser.print_help()
        return
    
    # Создание необходимых директорий
    Path("config").mkdir(exist_ok=True)
    Path("data/satellite_images").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Создание тестовых данных
    if args.create_test_data:
        create_test_data()
        return
    
    # Проверка файла конфигурации
    config_path = Path(args.config)
    if not config_path.exists():
        default_config = {
            'preprocessing': {
                'resize_method': 'crop',
                'normalization': 'histogram_matching'
            },
            'detection': {
                'method': 'cva',
                'cva': {
                    'magnitude_threshold': 'otsu',
                    'manual_threshold': 0.3
                },
                'ratio_threshold': 0.3
            },
            'postprocessing': {
                'morphological': {
                    'kernel_size': 3,
                    'operations': ['open', 'close']
                },
                'filtering': {
                    'min_area_pixels': 50,
                    'connectivity': 8
                }
            },
            'evaluation': {
                'metrics': ['precision', 'recall', 'f1', 'iou'],
                'save_report': True
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Конфигурация создана: {config_path}")
    
    # Загрузка конфигурации
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Конфигурация загружена: {config_path}")

    # Проверка изображений
    for img_path in [args.image1, args.image2]:
        if not Path(img_path).exists():
            print(f"Изображение не найдено: {img_path}")
            return
    
    # Создание выходной директории
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("--------------------------------")
    
    if args.compare:
        compare_methods(args, config, output_dir)
    elif args.method:
        run_specific_method(args, config, output_dir)
    else:
        run_pipeline(args, config, output_dir)

def run_pipeline(args, config, output_dir):
    """Запуск конвейера"""
    print(f"Запуск детекции изменений...")
    print(f"Изображение 1: {args.image1}")
    print(f"Изображение 2: {args.image2}")
    if args.ground_truth:
        print(f"Ground truth: {args.ground_truth}")
    print(f"Метод: {config['detection']['method']}")
    print(f"Выходная директория: {output_dir}")
    print("--------------------------------")
    
    try:
        pipeline = ChangeDetectionPipeline(config)
        results = pipeline.run(args.image1, args.image2, args.ground_truth)
        
        timestamp = results['timestamp'].replace(':', '').replace('-', '')[:14]
        experiment_dir = output_dir / f"exp_{timestamp}"
        
        pipeline.save_results(experiment_dir)
        
        # Визуализация
        if 'change_mask' in results:
            try:
                visualizer = ResultVisualizer()
                
                img1 = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(args.image2, cv2.IMREAD_GRAYSCALE)
                
                gt = None
                if args.ground_truth and Path(args.ground_truth).exists():
                    gt = cv2.imread(args.ground_truth, cv2.IMREAD_GRAYSCALE)
                    print(f"Ground truth загружен: {gt.shape}")
                
                fig = visualizer.create_comparison_figure(
                    img1=img1,
                    img2=img2,
                    change_mask=results['change_mask'],
                    ground_truth=gt
                )
                
                visualizer.save_visualization(fig, str(experiment_dir / "visualization.png"))
                print(f"Визуализация сохранена")
                
            except Exception as e:
                print(f"Ошибка при визуализации: {e}")
        
        print(f"ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
        print(f"Результаты сохранены в: {experiment_dir}")
        
        if results.get('metrics'):
            print("МЕТРИКИ КАЧЕСТВА:")
            metrics = results['metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key in ['precision', 'recall', 'f1_score', 'iou', 'accuracy']:
                        print(f"  {key:20} {value:.3f}")
                    elif key in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']:
                        print(f"  {key:20} {value:,}")
        
    except Exception as e:
        print(f"ОШИБКА ПРИ ЗАПУСКЕ КОНВЕЙЕРА: {e}")
        import traceback
        traceback.print_exc()

def compare_methods(args, config, output_dir):
    """Сравнение методов"""
    methods = ['differencing', 'cva', 'ratio', 'ndbi_diff']
    all_results = {}
    
    print(f"СРАВНЕНИЕ {len(methods)} МЕТОДОВ")
    print("--------------------------------")
    
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    for method in methods:
        print(f"Тестирование метода: {method}")
        
        config_copy = config.copy()
        config_copy['detection'] = config['detection'].copy()
        config_copy['detection']['method'] = method
        
        pipeline = ChangeDetectionPipeline(config_copy)
        
        try:
            results = pipeline.run(args.image1, args.image2, args.ground_truth)
            all_results[method] = results
            
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"  F1: {metrics.get('f1_score', 0):.3f} | "
                      f"Precision: {metrics.get('precision', 0):.3f} | "
                      f"Recall: {metrics.get('recall', 0):.3f}")
            
            method_dir = comparison_dir / f"method_{method}"
            pipeline.save_results(method_dir)
            
            print(f"  Результаты сохранены в: {method_dir}")
            
        except Exception as e:
            print(f"  Ошибка: {e}")
            all_results[method] = {'error': str(e)}
    
    if all_results:
        try:
            evaluator = ChangeDetectionEvaluator()
            
            metrics_dict = {}
            for method, results in all_results.items():
                if 'metrics' in results and results['metrics']:
                    metrics = results['metrics'].copy()
                    
                    for metric in ['f1_score', 'precision', 'recall', 'iou']:
                        if metric not in metrics:
                            metrics[metric] = 0.0
                    
                    metrics['method'] = method
                    metrics_dict[method] = metrics
            
            if metrics_dict:
                comparison_df = evaluator.compare_methods(metrics_dict)
                comparison_path = comparison_dir / "methods_comparison.csv"
                comparison_df.to_csv(comparison_path, index=False)
                
                print(f"ТАБЛИЦА СРАВНЕНИЯ:")
                print(f"Сохранена: {comparison_path}")
                
                print(f"Колонки в таблице: {list(comparison_df.columns)}")
                
                if not comparison_df.empty:
                    display_cols = ['method', 'f1_score', 'precision', 'recall', 'iou']
                    display_cols = [col for col in display_cols if col in comparison_df.columns]
                    
                    if display_cols:
                        print(comparison_df[display_cols].to_string(index=False))
                
                try:
                    visualizer = ResultVisualizer()
                    visualizer.create_comparison_plot(
                        comparison_df,
                        str(comparison_dir / "methods_comparison.png")
                    )
                    print(f"График сравнения сохранен: {comparison_dir}/methods_comparison.png")
                    
                except Exception as e:
                    print(f"Ошибка при создании графика сравнения: {e}")
            
            print(f"СРАВНЕНИЕ ЗАВЕРШЕНО")
            print(f"Все результаты сохранены в: {comparison_dir}")
            
        except Exception as e:
            print(f"Ошибка при сравнении методов: {e}")
            import traceback
            traceback.print_exc()

def run_specific_method(args, config, output_dir):
    """Запуск конкретного метода"""
    config['detection']['method'] = args.method
    print(f"Запуск метода: {args.method}")
    run_pipeline(args, config, output_dir)

if __name__ == "__main__":
    main()