import argparse
import yaml
import sys
from pathlib import Path

from experiment import ExperimentRunner
# Добавляем пути для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))
from pipeline import ChangeDetectionPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Система обнаружения изменений на спутниковых снимках'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда запуска одного эксперимента
    run_parser = subparsers.add_parser('run', help='Запустить один эксперимент')
    run_parser.add_argument('--config', type=str, 
                           default='config/params.yaml',
                           help='Файл конфигурации')
    run_parser.add_argument('--image1', type=str, 
                           default='data/satellite_images/satellite_T1.png',
                           help='Путь к изображению T1')
    run_parser.add_argument('--image2', type=str, 
                           default='data/satellite_images/satellite_T2.png',
                           help='Путь к изображению T2')
    run_parser.add_argument('--ground-truth', type=str,
                           help='Путь к ground truth маске')
    run_parser.add_argument('--output', type=str, 
                           default='results/single',
                           help='Выходная директория')
    run_parser.add_argument('--method', type=str,
                           choices=['differencing', 'cva', 'ratio', 
                                    'ndbi_diff', 'pca'],
                           help='Метод для использования')
    
    # Команда сравнения методов
    compare_parser = subparsers.add_parser('compare', 
                                          help='Сравнить методы с разными параметрами')
    compare_parser.add_argument('--config', type=str,
                               default='config/params.yaml',
                               help='Базовый файл конфигурации')
    compare_parser.add_argument('--image1', type=str,
                               default='data/satellite_images/satellite_T1.png',
                               help='Путь к изображению T1')
    compare_parser.add_argument('--image2', type=str,
                               default='data/satellite_images/satellite_T2.png',
                               help='Путь к изображению T2')
    compare_parser.add_argument('--ground-truth', type=str,
                               default='data/satellite_images/ground_truth.png',
                               help='Путь к ground truth маске')
    compare_parser.add_argument('--output', type=str,
                               default='experiments',
                               help='Директория для результатов')
    
    args = parser.parse_args()
    
    # Создание необходимых директорий
    Path("results").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    
    if args.command == 'run':
        # Запуск одного эксперимента
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.method:
            config['detection']['method'] = args.method
        
        pipeline = ChangeDetectionPipeline(config)
        results = pipeline.run(args.image1, args.image2, args.ground_truth)
        pipeline.save_results(args.output)

    elif args.command == 'compare':
        # Запуск сравнительного анализа

        runner = ExperimentRunner(args.config)
        
        # Определяем параметры для тестирования
        param_grid = {
            'detection.method': ['differencing', 'cva', 'ratio', 
                                'ndbi_diff', 'pca'],
            'detection.cva.manual_threshold': [0.2, 0.3, 0.4],
            'detection.ratio_threshold': [0.2, 0.3, 0.4],
            'postprocessing.filtering.min_area_pixels': [0, 50, 100]
        }
        
        runner.define_experiments(param_grid)
        runner.run_experiments(
            image1_path=args.image1,
            image2_path=args.image2,
            ground_truth_path=args.ground_truth,
            output_dir=args.output
        )
        
    else:
        # Если команда не указана, показываем справку
        parser.print_help()

if __name__ == "__main__":
    main()