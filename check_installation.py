"""
Проверка установки проекта
"""

print('Проверка установки проекта...\n')

# Проверяем Python версию
import sys
print(f'Python версия: {sys.version}')

# Проверяем основные библиотеки
required_packages = [
    'pandas',
    'numpy', 
    'sklearn',
    'scipy',
    'matplotlib',
    'seaborn',
    'surprise',  # scikit-surprise
    'tqdm',
    'jupyter'
]

print('\nПроверка зависимостей:')
for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        print(f'✗ {package} - не установлен')

# Проверяем структуру проекта
import os

print('\nПроверка структуры проекта:')
required_folders = ['data', 'models', 'notebooks', 'results', 'figures']
for folder in required_folders:
    if os.path.exists(folder):
        print(f'✓ Папка "{folder}" существует')
    else:
        print(f'✗ Папка "{folder}" не найдена')

required_files = [
    'requirements.txt',
    'run_project.py',
    'config.py',
    'data/loader.py',
    'data/preprocessor.py',
    'models/popularity_model.py',
    'models/content_model.py',
    'models/svd_model.py',
    'models/evaluator.py',
    'notebooks/analysis.ipynb'
]

print('\nПроверка файлов:')
for file in required_files:
    if os.path.exists(file):
        print(f'✓ {file}')
    else:
        print(f'✗ {file} - не найден')

# Проверяем данные
print('\nПроверка данных:')
data_dir = '..\\goodbooks-10k'
if os.path.exists(data_dir):
    data_files = ['ratings.csv', 'books.csv', 'tags.csv', 'book_tags.csv']
    for file in data_files:
        if os.path.exists(os.path.join(data_dir, file)):
            print(f'✓ {file}')
        else:
            print(f'✗ {file} - не найден')
else:
    print(f'✗ Папка с данными не найдена: {data_dir}')
    print('Скачайте датасет с https://www.kaggle.com/datasets/zygmunt/goodbooks-10k')
    print(f'Распакуйте в папку: {data_dir}')

print('\n' + '=' * 50)
print('РЕКОМЕНДАЦИИ:')
print('1. Установите недостающие пакеты: pip install -r requirements.txt')
print('2. Если scikit-surprise не устанавливается, попробуйте:')
print('   pip install scikit-surprise')
print('3. Запустите проект: python run_project.py')
print('=' * 50)
