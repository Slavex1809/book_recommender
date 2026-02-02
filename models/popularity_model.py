"""
Модель рекомендаций на основе популярности
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class PopularityRecommender:
    """Рекомендательная система на основе популярности книг"""
    
    def __init__(self, min_ratings=50):
        """
        Инициализация модели
        
        Args:
            min_ratings: минимальное количество оценок для учета книги
        """
        self.min_ratings = min_ratings
        self.popular_books = None
        self.book_stats = None
        
    def fit(self, ratings):
        """
        Обучение модели на исторических данных
        
        Args:
            ratings: DataFrame с колонками ['user_id', 'book_id', 'rating']
        """
        print('Обучение модели популярности...')
        
        # Вычисляем статистики по книгам
        self.book_stats = ratings.groupby('book_id').agg({
            'rating': ['count', 'mean']
        }).round(3)
        
        self.book_stats.columns = ['num_ratings', 'avg_rating']
        
        # Фильтруем книги с достаточным количеством оценок
        valid_books = self.book_stats[self.book_stats['num_ratings'] >= self.min_ratings]
        
        # Вычисляем взвешенный рейтинг (как на IMDb)
        C = self.book_stats['avg_rating'].mean()
        m = self.min_ratings
        
        valid_books['weighted_score'] = (
            (valid_books['num_ratings'] / (valid_books['num_ratings'] + m)) * valid_books['avg_rating'] +
            (m / (valid_books['num_ratings'] + m)) * C
        )
        
        # Сортируем по взвешенному скорингу
        self.popular_books = valid_books.sort_values('weighted_score', ascending=False).index.tolist()
        
        print(f'Найдено {len(self.popular_books)} популярных книг (с >= {self.min_ratings} оценками)')
        
        return self
    
    def recommend(self, user_id, n=10):
        """
        Рекомендации для пользователя
        
        Args:
            user_id: ID пользователя
            n: количество рекомендаций
            
        Returns:
            Список ID рекомендованных книг
        """
        if self.popular_books is None:
            raise ValueError('Модель не обучена. Сначала вызовите fit().')
        
        return self.popular_books[:n]
    
    def get_top_books(self, n=10):
        """Получить топ-N популярных книг"""
        if self.popular_books is None:
            raise ValueError('Модель не обучена.')
        
        return self.popular_books[:n]
    
    def get_book_stats(self, book_id):
        """Получить статистику по книге"""
        if self.book_stats is None:
            raise ValueError('Модель не обучена.')
        
        if book_id in self.book_stats.index:
            return self.book_stats.loc[book_id]
        else:
            return None
