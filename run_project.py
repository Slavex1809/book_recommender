"""
Основной скрипт для запуска всего проекта рекомендательной системы
(Упрощенная версия без библиотеки surprise)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
from pathlib import Path
import sys

print('=' * 80)
print('ЗАПУСК ПРОЕКТА: РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА ДЛЯ КНИГ')
print('(Упрощенная версия для Python 3.10)')
print('=' * 80)

def check_environment():
    """Проверка окружения"""
    print(f'\nВерсия Python: {sys.version}')
    
    # Проверяем базовые импорты
    try:
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        print('✓ Все основные библиотеки доступны')
        return True
    except ImportError as e:
        print(f'✗ Ошибка импорта: {e}')
        return False

def main():
    """Основная функция запуска проекта"""
    start_time = time.time()
    
    try:
        # Проверка окружения
        if not check_environment():
            print('\nПроблемы с окружением. Попробуйте установить зависимости вручную:')
            print('pip install pandas numpy scikit-learn matplotlib seaborn tqdm')
            return
        
        print('\n1. Импорт модулей проекта...')
        try:
            import config
            from data.loader import DataLoader
            from data.preprocessor import DataPreprocessor
            print('✓ Модули проекта загружены')
            
            # Проверяем конфигурацию
            print('\nПроверка конфигурации...')
            config_ok = config.print_config()
            if not config_ok:
                print('\nПожалуйста, скачайте и разместите датасет согласно инструкции выше.')
                return
                
        except ImportError as e:
            print(f'✗ Ошибка импорта модулей: {e}')
            print('Убедитесь, что все файлы проекта созданы')
            return
        
        print('\n2. Загрузка данных...')
        loader = DataLoader()
        
        try:
            ratings, books, tags, book_tags = loader.load_all_data()
        except Exception as e:
            print(f'Ошибка загрузки данных: {e}')
            print('\nУбедитесь, что датасет находится в правильной папке.')
            print('Скачайте с: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k')
            print('Распакуйте в папку: C:\\Users\\Arina\\goodbooks-10k\\')
            return
        
        print(f'\nЗагружено:')
        print(f'  - {len(ratings):,} оценок от {ratings["user_id"].nunique():,} пользователей')
        print(f'  - {len(books):,} книг')
        print(f'  - {len(tags):,} тегов')
        print(f'  - {len(book_tags):,} связей книг-тегов')
        
        print('\n3. Предобработка данных...')
        preprocessor = DataPreprocessor()
        
        # Создаем папки для результатов
        os.makedirs('figures', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Сохраняем информацию о данных
        with open('results/data_info.txt', 'w', encoding='utf-8') as f:
            f.write(f'Датасет: Goodbooks-10k\n')
            f.write(f'Общее количество оценок: {len(ratings):,}\n')
            f.write(f'Уникальных пользователей: {ratings["user_id"].nunique():,}\n')
            f.write(f'Уникальных книг: {ratings["book_id"].nunique():,}\n')
            f.write(f'Средняя оценка: {ratings["rating"].mean():.2f}\n')
        
        # Визуализация
        print('\nСоздание графиков EDA...')
        
        try:
            # Распределение оценок
            fig = preprocessor.plot_rating_distribution(ratings)
            plt.savefig('figures/rating_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Активность пользователей
            user_stats = preprocessor.analyze_user_activity(ratings)
            fig = preprocessor.plot_user_activity(user_stats)
            plt.savefig('figures/user_activity.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Популярность книг
            book_stats = preprocessor.analyze_book_popularity(ratings)
            fig = preprocessor.plot_book_popularity(book_stats)
            plt.savefig('figures/book_popularity.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f'Ошибка при создании графиков: {e}')
            print('Продолжаем без визуализации...')
        
        # Анализ разреженности
        print('\n4. Анализ данных...')
        sparsity = preprocessor.calculate_sparsity(ratings)
        
        # Выявление проблем
        cold_start_users = preprocessor.identify_cold_start_users(ratings, threshold=5)
        cold_start_books = preprocessor.identify_cold_start_books(ratings, threshold=10)
        
        print(f'\nПроблемы данных:')
        print(f'  - Разреженность матрицы: {sparsity:.2%}')
        print(f'  - Пользователей с холодным стартом: {len(cold_start_users):,}')
        print(f'  - Книг с холодным стартом: {len(cold_start_books):,}')
        
        # Разделение данных
        print('\n5. Разделение данных...')
        train_ratings, test_ratings = preprocessor.train_test_split_temporal(
            ratings, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        print(f'  Train: {len(train_ratings):,} записей')
        print(f'  Test: {len(test_ratings):,} записей')
        
        # Подготовка данных для контентной модели
        print('\n6. Подготовка данных для моделей...')
        books_with_tags = preprocessor.prepare_book_content_data(books, book_tags, tags, top_n_tags=10)
        print(f'  Подготовлено {len(books_with_tags)} книг с тегами')
        
        print('\n7. Обучение моделей...')
        
        # Модель популярности
        print('\n  a) Модель популярности...')
        try:
            from models.popularity_model import PopularityRecommender
            pop_model = PopularityRecommender(min_ratings=config.MIN_RATINGS)
            pop_model.fit(train_ratings)
            print('✓ Модель популярности обучена')
        except Exception as e:
            print(f'✗ Ошибка обучения модели популярности: {e}')
            pop_model = None
        
        # Контентная модель
        print('\n  b) Контентная модель...')
        try:
            from models.content_model import ContentBasedRecommender
            content_model = ContentBasedRecommender(max_features=config.TFIDF_MAX_FEATURES)
            content_model.fit(books_with_tags)
            print('✓ Контентная модель обучена')
        except Exception as e:
            print(f'✗ Ошибка обучения контентной модели: {e}')
            content_model = None
        
        # Упрощенная SVD модель
        print('\n  c) Упрощенная SVD модель...')
        try:
            from models.simple_svd_model import SimpleSVDRecommender
            svd_model = SimpleSVDRecommender(n_factors=config.SVD_N_FACTORS, n_epochs=10)
            # Обучаем на небольшом подмножестве для скорости
            sample_size = min(10000, len(train_ratings))
            sample_train = train_ratings.sample(sample_size, random_state=config.RANDOM_STATE)
            svd_model.fit(sample_train)
            print('✓ SVD модель обучена')
        except Exception as e:
            print(f'✗ Ошибка обучения SVD модели: {e}')
            print('  (Может потребоваться установка scipy: pip install scipy)')
            svd_model = None
        
        print('\nОбучение моделей завершено!')
        
        # Демонстрация рекомендаций
        print('\n' + '=' * 80)
        print('ДЕМОНСТРАЦИЯ РЕКОМЕНДАЦИЙ')
        print('=' * 80)
        
        # Выбираем несколько пользователей для демонстрации
        test_users = test_ratings['user_id'].unique()[:3]
        
        for user_id in test_users:
            print(f'\n=== Пользователь {user_id} ===')
            
            # Что пользователю понравилось
            liked_books = test_ratings[
                (test_ratings['user_id'] == user_id) & 
                (test_ratings['rating'] >= config.RATING_THRESHOLD)
            ].merge(books[['book_id', 'title']], on='book_id')
            
            if not liked_books.empty:
                print('Понравившиеся книги:')
                for idx, row in liked_books.head(2).iterrows():
                    title = row['title']
                    rating = row['rating']
                    print(f'  - {title} (оценка: {rating})')
            
            # Популярные рекомендации
            if pop_model:
                print('\nПопулярные рекомендации:')
                pop_recs = pop_model.recommend(user_id, n=config.TOP_N)
                for i, book_id in enumerate(pop_recs[:3], 1):
                    title = books[books['book_id'] == book_id]['title'].values
                    if len(title) > 0:
                        print(f'  {i}. {title[0]}')
            
            # Контентные рекомендации (если есть понравившиеся книги)
            if content_model and not liked_books.empty:
                print('\nКонтентные рекомендации:')
                seed_book = liked_books.iloc[0]['book_id']
                seed_title = liked_books.iloc[0]['title']
                print(f'  (похожие на "{seed_title}"):')
                
                content_recs = content_model.recommend_similar_books(seed_book, n=3)
                for i, book_id in enumerate(content_recs, 1):
                    title = books[books['book_id'] == book_id]['title'].values
                    if len(title) > 0:
                        print(f'  {i}. {title[0]}')
            
            # SVD рекомендации
            if svd_model:
                print('\nПерсональные рекомендации (SVD):')
                svd_recs = svd_model.recommend(user_id, n=3)
                for i, book_id in enumerate(svd_recs, 1):
                    title = books[books['book_id'] == book_id]['title'].values
                    if len(title) > 0:
                        print(f'  {i}. {title[0]}')
        
        # Анализ и выводы
        print('\n' + '=' * 80)
        print('АНАЛИЗ РЕЗУЛЬТАТОВ')
        print('=' * 80)
        
        print('\n1. Реализованные модели:')
        print('   - Модель популярности: рекомендации самых популярных книг')
        print('   - Контентная модель: рекомендации по схожести тегов')
        print('   - Упрощенная SVD: матричные разложения для персонализации')
        
        print('\n2. Проблемы данных:')
        print(f'   - Высокая разреженность: {sparsity:.2%}')
        print('   - Смещение в сторону популярных книг')
        print('   - Проблема холодного старта')
        
        print('\n3. Выводы:')
        print('   - Для новых пользователей: использовать популярные книги')
        print('   - Для новых книг: использовать контентные рекомендации')
        print('   - Для активных пользователей: использовать SVD')
        print('   - Лучший подход: гибридная система')
        
        print('\n4. Метрики оценки:')
        print('   - Precision@K: точность рекомендаций')
        print('   - Recall@K: полнота рекомендаций')
        print('   - nDCG@K: учет порядка рекомендаций')
        
        # Сохраняем финальный отчет
        with open('results/final_report.md', 'w', encoding='utf-8') as f:
            f.write('# Отчет по проекту рекомендательной системы\n\n')
            f.write('## Реализованные модели\n\n')
            f.write('1. **Модель популярности** - рекомендации топ-N самых популярных книг\n')
            f.write('2. **Контентная модель** - рекомендации на основе TF-IDF векторизации тегов\n')
            f.write('3. **Упрощенная SVD** - матричные разложения для персонализации\n\n')
            
            f.write('## Анализ данных\n\n')
            f.write(f'- Общее количество оценок: {len(ratings):,}\n')
            f.write(f'- Уникальных пользователей: {ratings["user_id"].nunique():,}\n')
            f.write(f'- Уникальных книг: {ratings["book_id"].nunique():,}\n')
            f.write(f'- Разреженность матрицы: {sparsity:.2%}\n\n')
            
            f.write('## Проблемы и решения\n\n')
            f.write('### Проблемы:\n')
            f.write('1. **Холодный старт** - новые пользователи и книги\n')
            f.write('2. **Разреженность данных** - мало взаимодействий\n')
            f.write('3. **Смещение популярности** - популярные книги доминируют\n\n')
            
            f.write('### Решения:\n')
            f.write('1. **Для холодного старта**: использовать контентные рекомендации\n')
            f.write('2. **Для разреженности**: использовать матричные разложения (SVD)\n')
            f.write('3. **Для разнообразия**: добавлять менее популярные книги\n\n')
            
            f.write('## Рекомендации по улучшению\n\n')
            f.write('1. Реализовать гибридный подход\n')
            f.write('2. Добавить временные факторы\n')
            f.write('3. Использовать нейросетевые модели\n')
            f.write('4. Внедрить A/B тестирование\n')
        
        execution_time = time.time() - start_time
        print(f'\nПроект успешно выполнен за {execution_time:.2f} секунд!')
        print('\nРезультаты сохранены в папках:')
        print('  - figures/: графики анализа данных')
        print('  - results/: отчеты и информация')
        print('\nДля детального анализа запустите Jupyter notebook:')
        print('  jupyter notebook notebooks/analysis.ipynb')
        
    except Exception as e:
        print(f'\nОшибка при выполнении проекта: {str(e)}')
        import traceback
        traceback.print_exc()
        
        print('\nВозможные решения:')
        print('1. Убедитесь, что датасет скачан и находится в правильной папке')
        print('2. Установите зависимости: pip install pandas numpy scikit-learn matplotlib seaborn tqdm scipy')
        print('3. Проверьте, что все файлы проекта созданы')
        print('4. Если scipy не устанавливается, удалите строку import scipy из models/simple_svd_model.py')

if __name__ == '__main__':
    main()
