#!/usr/bin/env python3
"""
Анализатор результатов экспериментов
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import sys

# Добавляем пути для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

try:
    from visualization import ResultVisualizer
except ImportError:
    print("Не удалось импортировать модули. Убедитесь, что структура проекта правильная.")
    sys.exit(1)

def analyze_results(results_dir: str = "results", output_dir: str = "analysis"):
    """Анализ результатов экспериментов"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Директория {results_dir} не найдена!")
        print("Сначала запустите эксперименты: python run.py")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Поиск файлов с метриками
    json_files = list(results_path.rglob("metrics.json"))
    csv_files = list(results_path.rglob("*.csv"))
    
    print(f"Найдено файлов: {len(json_files)} JSON, {len(csv_files)} CSV")
    
    # Загрузка данных из JSON файлов
    all_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Добавляем имя эксперимента
            exp_name = json_file.parent.name
            row = {'experiment': exp_name}
            row.update(data)
            all_data.append(row)
            
        except Exception as e:
            print(f"Ошибка загрузки {json_file}: {e}")
    
    # Загрузка данных из CSV файлов
    for csv_file in csv_files:
        try:
            df_csv = pd.read_csv(csv_file)
            if not df_csv.empty:
                # Добавляем имя файла как эксперимент
                df_csv['experiment'] = csv_file.stem
                all_data.extend(df_csv.to_dict('records'))
        except Exception as e:
            print(f"Ошибка загрузки {csv_file}: {e}")
    
    if not all_data:
        print("Нет данных для анализа")
        return
    
    # Создание DataFrame
    df = pd.DataFrame(all_data)
    print(f"\nЗагружено {len(df)} записей")
    
    # Сохранение объединенных данных
    combined_path = output_path / "all_results.csv"
    df.to_csv(combined_path, index=False)
    print(f"Объединенные данные сохранены: {combined_path}")
    
    # Статистический анализ
    print("\n" + "=" * 60)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("=" * 60)
    
    if 'method' in df.columns and 'f1_score' in df.columns:
        # Группировка по методам
        stats = df.groupby('method').agg({
            'f1_score': ['mean', 'std', 'min', 'max', 'count'],
            'precision': ['mean', 'std', 'min', 'max'],
            'recall': ['mean', 'std', 'min', 'max'],
            'iou': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        print("\nСтатистика по методам:")
        print(stats.to_string())
        
        # Сохранение статистики
        stats.to_csv(output_path / "statistics_by_method.csv")
    
    # Визуализация
    create_visualizations(df, output_path)
    
    print(f"\n" + "=" * 60)
    print(f"АНАЛИЗ ЗАВЕРШЕН")
    print(f"Результаты сохранены в: {output_path}")
    print("=" * 60)

def create_visualizations(df: pd.DataFrame, output_path: Path):
    """Создание визуализаций"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Сравнение методов (если есть данные)
    if 'method' in df.columns and 'f1_score' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Боксплоты F1-Score
        sns.boxplot(data=df, x='method', y='f1_score', ax=axes[0, 0])
        axes[0, 0].set_title('Распределение F1-Score по методам')
        axes[0, 0].set_xlabel('Метод')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Scatter plot Precision-Recall
        if 'precision' in df.columns and 'recall' in df.columns:
            for method in df['method'].unique():
                method_data = df[df['method'] == method]
                axes[0, 1].scatter(method_data['precision'], method_data['recall'], 
                                  label=method, s=100, alpha=0.7)
            axes[0, 1].set_xlabel('Precision')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].set_title('Precision-Recall по методам')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_ylim(0, 1)
        
        # Средние значения метрик
        if 'method' in df.columns:
            metrics_to_plot = ['f1_score', 'precision', 'recall', 'iou']
            method_means = df.groupby('method')[metrics_to_plot].mean()
            
            sns.heatmap(method_means, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=axes[1, 0], cbar_kws={'label': 'Среднее значение'})
            axes[1, 0].set_title('Средние значения метрик по методам')
        
        # Лучшие результаты
        if 'f1_score' in df.columns:
            top_results = df.nlargest(5, 'f1_score')
            bars = axes[1, 1].bar(range(len(top_results)), top_results['f1_score'])
            axes[1, 1].set_xticks(range(len(top_results)))
            axes[1, 1].set_xticklabels(top_results['method'], rotation=45)
            axes[1, 1].set_title('Топ-5 результатов по F1-Score')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_ylim(0, 1)
            
            for bar, value in zip(bars, top_results['f1_score']):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Анализ результатов обнаружения изменений', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "analysis_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Корреляционная матрица
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, square=True, linewidths=.5, cbar_kws={'shrink': 0.8})
        ax.set_title('Корреляция метрик качества')
        plt.tight_layout()
        plt.savefig(output_path / "correlation_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Корреляционная матрица создана")
    
    # 3. Рекомендации
    if 'method' in df.columns and 'f1_score' in df.columns:
        print("\n" + "-" * 40)
        print("РЕКОМЕНДАЦИИ")
        print("-" * 40)
        
        # Лучший метод
        best_method = df.loc[df['f1_score'].idxmax(), 'method']
        best_f1 = df['f1_score'].max()
        print(f"🏆 Лучший метод: {best_method} (F1-Score: {best_f1:.3f})")
        
        # Стабильность методов
        if 'method' in df.columns:
            stability = df.groupby('method')['f1_score'].std()
            if not stability.empty:
                most_stable = stability.idxmin()
                print(f"📊 Наиболее стабильный метод: {most_stable} (std: {stability.min():.3f})")
                
        # Precision-Recall баланс
        if 'precision' in df.columns and 'recall' in df.columns:
            avg_precision = df['precision'].mean()
            avg_recall = df['recall'].mean()
            if avg_precision > avg_recall:
                print(f"⚖️  Совет: Сфокусируйтесь на улучшении Recall")
                print(f"   Precision ({avg_precision:.3f}) > Recall ({avg_recall:.3f})")

def main():
    parser = argparse.ArgumentParser(
        description='Анализатор результатов экспериментов по обнаружению изменений',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python analyze.py                          # Анализ всех результатов в results/
  python analyze.py --results-dir my_results # Анализ конкретной директории
  python analyze.py --output-dir my_analysis # Сохранить анализ в другую директорию
        """
    )
    
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Директория с результатами (по умолчанию: results)')
    parser.add_argument('--output-dir', type=str, default='analysis',
                       help='Директория для сохранения анализа (по умолчанию: analysis)')
    
    args = parser.parse_args()
    analyze_results(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()