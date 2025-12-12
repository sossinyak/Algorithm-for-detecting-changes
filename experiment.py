import yaml
import json
import sys
import itertools
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import numpy as np

# Добавляем пути для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))
from pipeline import ChangeDetectionPipeline

class ExperimentRunner:
    """Класс для проведения экспериментов с разными параметрами"""
    
    def __init__(self, base_config_path: str = "config/params.yaml"):
        """Инициализация с базовой конфигурацией"""
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.experiments = []
        self.results = []
        
    def define_experiments(self, param_grid: Dict[str, List[Any]]):
        """Определение сетки параметров для экспериментов"""
        print("="*70)
        print("ОПРЕДЕЛЕНИЕ ЭКСПЕРИМЕНТОВ")
        print("="*70)
        
        # Преобразуем сетку параметров в список всех комбинаций
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Генерируем все комбинации
        combinations = list(itertools.product(*param_values))
        
        # Создаем конфигурации для каждой комбинации
        for combo in combinations:
            config = self._deep_copy_config(self.base_config)
            
            # Устанавливаем параметры
            for param_name, param_value in zip(param_names, combo):
                self._set_nested_param(config, param_name, param_value)
            
            # Добавляем уникальный ID эксперимента
            exp_id = f"exp_{len(self.experiments):03d}"
            experiment = {
                'id': exp_id,
                'config': config,
                'params': dict(zip(param_names, combo))
            }
            self.experiments.append(experiment)
    
    def run_experiments(self, image1_path: str, image2_path: str, 
                       ground_truth_path: str = None,
                       output_dir: str = "experiments"):
        """Запуск всех экспериментов"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, experiment in enumerate(self.experiments):
            try:
                # Создаем и запускаем пайплайн
                pipeline = ChangeDetectionPipeline(experiment['config'])
                results = pipeline.run(
                    image1_path=image1_path,
                    image2_path=image2_path,
                    ground_truth_path=ground_truth_path
                )
                
                exp_dir = output_path / experiment['id']
        
                # Добавляем результаты в общий список
                exp_result = {
                    'experiment_id': experiment['id'],
                    'params': experiment['params'],
                    'metrics': results.get('metrics', {}),
                    'detection_info': results.get('detection_info', {}),
                    'timestamp': datetime.now().isoformat(),
                    'output_dir': str(exp_dir)
                }
                
                self.results.append(exp_result)
                
            except Exception as e:
                print(f"Ошибка в эксперименте {experiment['id']}: {e}")
                print("-"*50)
        
        # Сохраняем сводные результаты
        self.save_summary(output_path)
    
    def save_summary(self, output_path: Path):
        """Сохранение сводного отчета по всем экспериментам"""
        if not self.results:
            print("Нет результатов для сохранения")
            return
        
        # Создаем DataFrame с результатами
        summary_data = []
        
        for result in self.results:
            row = {
                'experiment_id': result['experiment_id'],
                **result['params']
            }
            
            # Добавляем метрики, если они есть
            if result['metrics']:
                for metric_name, metric_value in result['metrics'].items():
                    if isinstance(metric_value, (int, float)):
                        row[metric_name] = metric_value
            
            # Добавляем информацию о детекции
            if result['detection_info']:
                row['detection_method'] = result['detection_info'].get('method', '')
                row['threshold'] = result['detection_info'].get('threshold', 0)
            
            row['timestamp'] = result['timestamp']
            row['output_dir'] = result['output_dir']
            
            summary_data.append(row)
        
        # Создаем и сохраняем DataFrame
        df = pd.DataFrame(summary_data)
        
        # Сортируем по F1-Score (если есть)
        if 'f1_score' in df.columns:
            df = df.sort_values('f1_score', ascending=False)

        # Создаем сводный отчет
        self._create_summary_report(df, output_path)
    
    def _create_summary_report(self, df: pd.DataFrame, output_path: Path):
        """Создание текстового отчета"""
        report_path = output_path / "summary_report.txt"
        original_stdout = sys.stdout
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                sys.stdout = f
                
                print("ОТЧЕТ ПО ЭКСПЕРИМЕНТАМ\n")
                print(f"Всего экспериментов: {len(df)}\n")
                
                # Лучшие результаты
                if 'f1_score' in df.columns and not df.empty:
                    print("ТОП-5 ЛУЧШИХ РЕЗУЛЬТАТОВ (по F1-Score)")
                    print("-"*70)
                    
                    top5 = df.head(5)
                    for i, (_, row) in enumerate(top5.iterrows(), 1):
                        print(f"{i}. {row['experiment_id']}:")
                        print(f"   Метод: {row.get('detection_method', 'N/A')}")
                        print(f"   F1-Score: {row.get('f1_score', 0):.4f}")
                        print(f"   Precision: {row.get('precision', 0):.4f}")
                        print(f"   Recall: {row.get('recall', 0):.4f}")
                        print(f"   IoU: {row.get('iou', 0):.4f}")
                        
                        # Выводим только измененные параметры
                        params_summary = {}
                        for key, value in row.items():
                            if key not in ['experiment_id', 'timestamp', 'output_dir', 
                                         'f1_score', 'precision', 'recall', 'iou', 
                                         'accuracy', 'detection_method', 'threshold',
                                         'TP', 'FP', 'FN', 'TN', 'total_pixels']:
                                if key in self.base_config:
                                    base_val = self._get_nested_param(self.base_config, key)
                                    if value != base_val:
                                        params_summary[key] = value
                                else:
                                    params_summary[key] = value
                        
                        if params_summary:
                            print(f"   Параметры: {params_summary}")
                        print()
                
                # Статистика по методам
                if 'detection_method' in df.columns:
                    print("СТАТИСТИКА ПО МЕТОДАМ")
                    print("-"*70)
                    
                    method_stats = df.groupby('detection_method').agg({
                        'f1_score': ['mean', 'std', 'min', 'max', 'count'],
                        'precision': ['mean', 'std'],
                        'recall': ['mean', 'std']
                    }).round(4)
                    
                    print(method_stats)
                    print()
                
                # Корреляция параметров с F1-Score
                print("КОРРЕЛЯЦИЯ ПАРАМЕТРОВ С F1-SCORE")
                print("-"*70)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                param_cols = [col for col in numeric_cols if col not in 
                            ['f1_score', 'precision', 'recall', 'iou', 'accuracy',
                             'TP', 'FP', 'FN', 'TN', 'total_pixels']]
                
                if param_cols and 'f1_score' in df.columns:
                    correlations = []
                    for col in param_cols:
                        corr = df[col].corr(df['f1_score'])
                        correlations.append((col, corr))
                    
                    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    for col, corr in correlations[:10]:  # Топ-10 корреляций
                        print(f"{col:40} : {corr:7.4f}")
                
        finally:
            sys.stdout = original_stdout
        
        print(f"Текстовый отчет: {report_path}")
    
    def _deep_copy_config(self, config: Dict) -> Dict:
        """Глубокое копирование конфигурации"""
        import copy
        return copy.deepcopy(config)
    
    def _set_nested_param(self, config: Dict, param_path: str, value: Any):
        """ Установка вложенного параметра в конфигурации"""
        parts = param_path.split('.')
        current = config
        
        # Проходим по всем частям пути, кроме последней
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Устанавливаем значение
        current[parts[-1]] = value
    
    def _get_nested_param(self, config: Dict, param_path: str) -> Any:
        """Получение вложенного параметра из конфигурации"""
        parts = param_path.split('.')
        current = config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return None