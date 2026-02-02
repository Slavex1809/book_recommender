"""
Конфигурационные параметры проекта
"""

import os
from pathlib import Path

# Пути к данным
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'goodbooks-10k'

# Параметры моделей
RATING_THRESHOLD = 4  # Порог для определения релевантности
TOP_N = 10  # Количество рекомендаций
MIN_RATINGS = 50  # Минимальное количество оценок для популярности
TFIDF_MAX_FEATURES = 5000  # Максимальное количество фичей для TF-IDF
CF_K_NEIGHBORS = 50  # Количество соседей для Item-based CF
SVD_N_FACTORS = 50  # Количество факторов для SVD (уменьшено для скорости)
TEST_SIZE = 0.2  # Размер тестовой выборки
RANDOM_STATE = 42  # Seed для воспроизводимости

# Параметры оценки
METRICS_K_VALUES = [5, 10, 20]  # Значения K для метрик

# Вывод информации о конфигурации
def print_config():
    print('=' * 60)
    print('КОНФИГУРАЦИЯ ПРОЕКТА')
    print('=' * 60)
    print(f'Путь к данным: {DATA_DIR}')
    
    if not DATA_DIR.exists():
        print(f'\nВНИМАНИЕ: Папка с данными не найдена!')
        print('Скачайте датасет Goodbooks-10k с Kaggle:')
        print('https://www.kaggle.com/datasets/zygmunt/goodbooks-10k')
        print('\nРаспакуйте в папку рядом с проектом:')
        print('Ваша структура должна быть:')
        print('C:\\Users\\Arina\\')
        print('├── book_recommender\\   (этот проект)')
        print('└── goodbooks-10k\\      (датасет)')
        print('    ├── ratings.csv')
        print('    ├── books.csv')
        print('    ├── tags.csv')
        print('    └── book_tags.csv')
        return False
    else:
        print(f'\n✓ Папка с данными найдена')
        data_files = ['ratings.csv', 'books.csv', 'tags.csv', 'book_tags.csv']
        all_files_exist = True
        for file in data_files:
            if (DATA_DIR / file).exists():
                print(f'✓ {file}')
            else:
                print(f'✗ {file} - не найден')
                all_files_exist = False
        
        return all_files_exist

# Если запускаем config.py напрямую, показываем информацию
if __name__ == '__main__':
    print_config()
