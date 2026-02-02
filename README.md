# Book Recommendation System

Прототип рекомендательной системы для книг на основе датасета Goodbooks-10k.

## Описание проекта

В этом проекте реализованы различные подходы рекомендательных систем:
- Неперсонализированные рекомендации (Top-N популярных книг)
- Контентные рекомендации (TF-IDF + косинусная схожесть)
- Коллаборативная фильтрация (Item-based CF)
- Матричные разложения (SVD)
- Гибридный подход
        
## Структура проекта
book_recommender/
├── data/ # Загрузка и предобработка данных
├── models/ # Реализация моделей рекомендаций
├── notebooks/ # Jupyter notebook с анализом
├── results/ # Результаты и метрики
├── figures/ # Графики и визуализации
├── cache/ # Кэшированные данные
├── requirements.txt # Зависимости
├── config.py # Конфигурация проекта
├── run_project.py # Основной скрипт
└── README.md # Документация

## Установка

1. Установите зависимости:
`ash
pip install -r requirements.txt
Скачайте датасет Goodbooks-10k с Kaggle:
https://www.kaggle.com/datasets/zygmunt/goodbooks-10k

Распакуйте датасет в папку goodbooks-10k рядом с проектом:
your-project-folder/
├── book_recommender/
└── goodbooks-10k/
    ├── ratings.csv
    ├── books.csv
    ├── tags.csv
    └── book_tags.csv
Запуск
Запустите основной скрипт:
python run_project.py
jupyter notebook notebooks/analysis.ipynb
Метрики оценки
Реализованные метрики:

Precision

Recall

nDCG

RMSE (для регрессионных моделей)

Автор
Рекомендательная система для курса по ML/AI.
