Проект на тему: Алгоритм обнаружения изменений на разновременных спутниковых снимках с применением классических алгоритмов обработки изображений

ИНСТРУКЦИЯ К ЗАПУСКУ КОДА

# Установите необходимые библиотеки
pip install -r requirements.txt

# Создайте синтетические спутниковые снимки для тестирования
python run.py --create-test-data

Результат:
Создаст папку data/satellite_images/
Сгенерирует 3 файла:
satellite_T1.png - снимок "до"
satellite_T2.png - снимок "после"
ground_truth.png - эталонная маска для оценки

# Сравнение 4 методов детекции
python run.py --compare
results/comparison/
|- method_differencing/     # Результаты метода абсолютной разности
|- method_cva/              # Результаты метода CVA  
|- method_ratio/            # Результаты метода отношения изображений
|- method_ndbi_diff/        # Результаты метода на основе разности индексов
|- methods_comparison.csv   # Таблица сравнения
|- methods_comparison.png   # График сравнения

# Запуск детекции изменений с методом по умолчанию (CVA)
python run.py

results/exp_дата и время/
|- change_mask.png          # Бинарная маска изменений
|- config.json              # Конфигурация эксперимента
|- metrics.json             # Метрики качества (если был ground truth)
|- report.json              # Полный отчёт
|- visualization.png        # Визуализация результатов

# Запуск только метода абсолютной разности / отношения изображений / на основе спектральных индексов
python run.py --method differencing / ratio / ndbi_diff

# Укажите пути к своим изображениям
python run.py --image1 path/to/your_image1.png --image2 path/to/your_image2.png

# Анализ всех экспериментов в папке results
python analyze.py

analysis/
|- all_results.csv          # Объединённые данные всех экспериментов
|- statistics_by_method.csv # Статистика по методам
|- analysis_summary.png     # Графики сравнения
|- correlation_matrix.png   # Корреляция метрик

