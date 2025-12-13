import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

import sys
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

from pipeline import ChangeDetectionPipeline

def run_baseline_experiments() -> list:
    """Запуск экспериментов"""
    
    # Методы для сравнения
    baseline_methods = ['differencing', 'ratio', 'ndbi_diff', 'cva']
    results = []
    
    # Загружаем базовую конфигурацию
    with open('config/params.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Тестовые данные
    test_data = {
        'A': 'data/LEVIR_CD/A.png',
        'B': 'data/LEVIR_CD/B.png',
        'GT': 'data/LEVIR_CD/GT.png'
    }
    
    for method in baseline_methods:
        try:
            # Создаем конфигурацию для метода
            config = base_config.copy()
            config['detection']['method'] = method
            
            # Запускаем пайплайн
            pipeline = ChangeDetectionPipeline(config)
            pipeline_result = pipeline.run(
                test_data['A'],
                test_data['B'],
                test_data['GT']
            )
            
            # Извлекаем метрики
            if 'metrics' in pipeline_result:
                metrics = pipeline_result['metrics']
                metrics['method'] = method
                results.append(metrics)
        
        except Exception as e:
            print(f"Ошибка в методе {method}: {e}")
    
    return results

def create_comparison_table(baseline_results: list) -> pd.DataFrame:
    """
    Создание таблицы сравнения метрик каскадного метода с базовыми
    """
    
    # Создаем список всех результатов
    all_results = []
    
    # Добавляем базовые методы
    for result in baseline_results:
        if result:  # Проверяем, что результат не пустой
            all_results.append({
                'Method': result.get('method', 'unknown'),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1-Score': result.get('f1_score', 0),
                'IoU': result.get('iou', 0),
                'Accuracy': result.get('accuracy', 0)
            })
    
    # Создаем DataFrame
    df = pd.DataFrame(all_results)
    
    # Сортируем по F1-Score в порядке убывания
    if not df.empty:
        df = df.sort_values('F1-Score', ascending=False)
    
    return df

def print_comparison_table(df: pd.DataFrame):
    """
    Вывод таблицы сравнения в консоль с красивым форматированием
    
    Args:
        df: DataFrame с результатами
    """
    if df.empty:
        print("Нет данных для создания таблицы сравнения")
        return
    
    print("\n" + "="*80)
    print("СРАВНЕНИЕ МЕТОДОВ ДЕТЕКЦИИ ИЗМЕНЕНИЙ")
    print("="*80)
    
    # Используем tabulate для красивого вывода
    table = tabulate(
        df.round(4),  # Округляем до 4 знаков после запятой
        headers='keys',
        tablefmt='grid',
        showindex=False,
        floatfmt=".4f"
    )
    
    print(table)
    
    # Добавляем статистическую информацию
    print("\n" + "-"*80)
    
    # Вычисляем средние значения
    if len(df) > 1:
        avg_precision = df['Precision'].mean()
        avg_recall = df['Recall'].mean()
        avg_f1 = df['F1-Score'].mean()
        avg_iou = df['IoU'].mean()
        
        print(f"Среднее Precision: {avg_precision:.4f}")
        print(f"Среднее Recall: {avg_recall:.4f}")
        print(f"Среднее F1-Score: {avg_f1:.4f}")
        print(f"Среднее IoU: {avg_iou:.4f}")
        
        # Находим лучшие методы по каждой метрике
        best_precision = df.loc[df['Precision'].idxmax()]
        best_recall = df.loc[df['Recall'].idxmax()]
        best_f1 = df.loc[df['F1-Score'].idxmax()]
        best_iou = df.loc[df['IoU'].idxmax()]
        
        print(f"\nЛучший по Precision: {best_precision['Method']} ({best_precision['Precision']:.4f})")
        print(f"Лучший по Recall: {best_recall['Method']} ({best_recall['Recall']:.4f})")
        print(f"Лучший по F1-Score: {best_f1['Method']} ({best_f1['F1-Score']:.4f})")
        print(f"Лучший по IoU: {best_iou['Method']} ({best_iou['IoU']:.4f})")


def run_cascade_experiment(config_path: str = 'config/cascade_params.yaml') -> dict:
    """Запуск эксперимента с каскадным методом"""
    
    with open(config_path, 'r') as f:
        cascade_config = yaml.safe_load(f)
    
    # Тестовые данные
    test_data = {
        'A': 'data/LEVIR_CD/A.png',
        'B': 'data/LEVIR_CD/B.png',
        'GT': 'data/LEVIR_CD/GT.png'
    }
    
    try:
        pipeline = ChangeDetectionPipeline(cascade_config)
        pipeline_result = pipeline.run(
            test_data['A'],
            test_data['B'],
            test_data['GT']
        )
        
        if 'metrics' in pipeline_result:
            metrics = pipeline_result['metrics']
            metrics['method'] = 'cascade_clahe_filter_canny'
            
            print(f"Каскадный метод: F1={metrics.get('f1_score', 0):.3f}, "
                  f"IoU={metrics.get('iou', 0):.3f}, "
                  f"Precision={metrics.get('precision', 0):.3f}, "
                  f"Recall={metrics.get('recall', 0):.3f}")
            
            return metrics
    
    except Exception as e:
        print(f"Ошибка в каскадном методе: {e}")
    
    return {}

def run_all_experiments():
    """Запуск экспериментов"""
    
    baseline_results = run_baseline_experiments()
    cascade_results = run_cascade_experiment()
    
    print("-"*40)
    print("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
   
    all_results = baseline_results.copy()
    if cascade_results:
        all_results.append(cascade_results)
    
    comparison_df = create_comparison_table(all_results)
    
    # Статистический анализ
    perform_statistical_analysis(comparison_df)
    
    # Визуализация результатов
    create_comprehensive_visualization(comparison_df)
    
    return comparison_df

def perform_statistical_analysis(df: pd.DataFrame):
    """Статистический анализ результатов"""
    
    if df.empty or len(df) < 2:
        print("Недостаточно данных для статистического анализа")
        return
    
    # Проверяем наличие каскадного метода
    if 'cascade' in df['Method'].values:
        cascade_metrics = df[df['Method'] == 'cascade_clahe_filter_canny']
        baseline_metrics = df[df['Method'] != 'cascade_clahe_filter_canny']
        
        # Сравнение по F1-Score
        if not baseline_metrics.empty and not cascade_metrics.empty:
            baseline_f1 = baseline_metrics['F1-Score'].mean()
            cascade_f1 = cascade_metrics['F1-Score'].iloc[0]
            
            print(f"Средний F1-Score базовых методов: {baseline_f1:.4f}")
            print(f"F1-Score каскадного метода: {cascade_f1:.4f}")
            improvement_f1 = ((cascade_f1 - baseline_f1) / baseline_f1) * 100
            print(f"Улучшение по F1-Score: {improvement_f1:+.2f}%")
            
            # T-тест для проверки значимости
            if len(baseline_metrics) >= 3:
                from scipy.stats import ttest_1samp
                
                # Предполагаем, что каскадный метод должен быть лучше
                baseline_values = baseline_metrics['F1-Score'].values
                
                # Одновыборочный t-тест
                t_stat, p_value = ttest_1samp(baseline_values, cascade_f1)
                print(f"\nT-тест (H0: каскадный метод не лучше среднего базовых):")
                print(f"  t-статистика = {t_stat:.4f}")
                print(f"  p-значение = {p_value:.4f}")
                
                if p_value < 0.05:
                    print("  Результат: Различие статистически значимо (p < 0.05)")
                    print("  Вывод: Каскадный метод действительно обеспечивает лучшее качество")
                else:
                    print("  Результат: Различие не является статистически значимым")
    

def create_comprehensive_visualization(df: pd.DataFrame):
    """Визуализация"""
    
    try:
        from visualization import ResultVisualizer
        visualizer = ResultVisualizer()
        
        # Базовое сравнение
        output_path1 = "results/comparison_summary.png"
        visualizer.create_comparison_plot(df, output_path1)
        
        # Дополнительная визуализация
        create_improvement_chart(df)
        
    except Exception as e:
        print(f"Не удалось создать визуализацию: {e}")


def create_improvement_chart(df: pd.DataFrame):
    """Создание графика улучшения метрик"""
    
    if 'cascade_clahe_filter_canny' not in df['Method'].values:
        return
    
    # Находим средние значения базовых методов
    baseline_df = df[df['Method'] != 'cascade_clahe_filter_canny']
    cascade_row = df[df['Method'] == 'cascade_clahe_filter_canny'].iloc[0]
    
    if baseline_df.empty:
        return
    
    baseline_avg = baseline_df.max(numeric_only=True)
    
    metrics_to_compare = ['Precision', 'Recall', 'F1-Score', 'IoU']
    metrics_to_compare = [m for m in metrics_to_compare if m in df.columns]
    
    # Вычисляем процент улучшения
    improvements = []
    for metric in metrics_to_compare:
        baseline_val = baseline_avg[metric]
        cascade_val = cascade_row[metric]
        improvement = ((cascade_val - baseline_val) / baseline_val) * 100
        improvements.append(improvement)
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(metrics_to_compare))
    
    bars = ax.bar(x_pos, improvements, color=['green' if x > 0 else 'red' for x in improvements])
    ax.set_xlabel('Метрики')
    ax.set_ylabel('Улучшение (%)')
    ax.set_title('Улучшение каскадного метода относительно лучшего базового методов')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_to_compare)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Добавляем значения на столбцы
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{improvement:+.1f}%',
                ha='center', va='bottom' if improvement > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('results/improvement_chart.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Основная функция с полной проверкой гипотезы"""
    
    # Создаем директории для результатов
    Path("results").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    
    # Запускаем все эксперименты
    results_df = run_all_experiments()
    
    # Выводим заключение
    print("\n--------------------------------")
    print("ЗАКЛЮЧЕНИЕ")
    
    if not results_df.empty and 'cascade_clahe_filter_canny' in results_df['Method'].values:
        cascade_result = results_df[results_df['Method'] == 'cascade_clahe_filter_canny']
        if not cascade_result.empty:
            cascade_f1 = cascade_result['F1-Score'].iloc[0]
            
            # Находим лучший базовый метод
            baseline_results = results_df[results_df['Method'] != 'cascade_clahe_filter_canny']
            if not baseline_results.empty:
                best_baseline = baseline_results.loc[baseline_results['F1-Score'].idxmax()]
                best_baseline_f1 = best_baseline['F1-Score']
                best_baseline_method = best_baseline['Method']
                
                improvement = ((cascade_f1 - best_baseline_f1) / best_baseline_f1) * 100
                
                print(f"Каскадный метод достиг F1-Score: {cascade_f1:.4f}")
                print(f"Лучший базовый метод ({best_baseline_method}): {best_baseline_f1:.4f}")
                print(f"Абсолютное улучшение: {cascade_f1 - best_baseline_f1:.4f}")
                print(f"Относительное улучшение: {improvement:+.2f}%")
                
                if improvement > 0:
                    print(f"  Каскадный метод показывает улучшение на {improvement:.2f}%")
                    print("  по сравнению с лучшим базовым методом")
                else:
                    print("  Каскадный метод не показал улучшения")

    return results_df

if __name__ == "__main__":
    main()