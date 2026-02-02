"""
Скрипт для установки всех зависимостей
"""

import sys
import subprocess
import os

def install_packages():
    """Установка необходимых пакетов"""
    
    # Базовые пакеты, которые хорошо работают на Windows
    packages = [
        'pandas==1.5.3',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'tqdm==4.65.0',
        'jupyter==1.0.0',
        'notebook==6.5.4',
        'scipy==1.10.1'  # Добавляем отдельно
    ]
    
    print('Установка зависимостей...')
    print('=' * 50)
    
    for package in packages:
        print(f'\nУстановка {package}...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f'✓ {package} установлен')
        except subprocess.CalledProcessError as e:
            print(f'✗ Ошибка установки {package}: {e}')
    
    print('\n' + '=' * 50)
    print('Все зависимости установлены!')
    
    # Проверяем установку
    print('\nПроверка установки...')
    test_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    
    for package in test_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f'✓ {package} импортируется успешно')
        except ImportError:
            print(f'✗ {package} не может быть импортирован')

if __name__ == '__main__':
    print('Установщик зависимостей для проекта рекомендательной системы')
    print(f'Python версия: {sys.version}')
    print()
    
    response = input('Установить зависимости? (y/n): ')
    if response.lower() == 'y':
        install_packages()
        
        print('\n' + '=' * 50)
        print('Следующие шаги:')
        print('1. Скачайте датасет с https://www.kaggle.com/datasets/zygmunt/goodbooks-10k')
        print('2. Распакуйте в папку: C:\\Users\\Arina\\goodbooks-10k\\')
        print('3. Запустите проект: python run_project.py')
    else:
        print('Установка отменена.')
