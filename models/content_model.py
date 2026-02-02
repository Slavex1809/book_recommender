"""
Контентная модель рекомендаций (TF-IDF + косинусная схожесть)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    """Контентная рекомендательная система"""
    
    def __init__(self, max_features=5000, min_df=2, max_df=0.8):
        """
        Инициализация модели
        
        Args:
            max_features: максимальное количество фичей TF-IDF
            min_df: минимальная частота термина
            max_df: максимальная частота термина
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.tfidf_matrix = None
        self.book_ids = None
        self.book_id_to_idx = None
        self.idx_to_book_id = None
        
    def fit(self, books_data):
        """
        Обучение модели на данных о книгах
        
        Args:
            books_data: DataFrame с колонками ['book_id', 'content']
        """
        print('Обучение контентной модели...')
        
        # Сохраняем mapping индексов
        self.book_ids = books_data['book_id'].values
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(self.book_ids)}
        self.idx_to_book_id = {idx: book_id for idx, book_id in enumerate(self.book_ids)}
        
        # Создаем TF-IDF векторайзер
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Обучаем TF-IDF и преобразуем тексты
        self.tfidf_matrix = self.vectorizer.fit_transform(books_data['content'])
        
        print(f'Создана матрица TF-IDF: {self.tfidf_matrix.shape}')
        print(f'Количество фичей: {len(self.vectorizer.get_feature_names_out())}')
        
        return self
    
    def get_similar_books(self, book_id, n=10):
        """
        Найти похожие книги
        
        Args:
            book_id: ID книги
            n: количество похожих книг
            
        Returns:
            Список кортежей (book_id, similarity_score)
        """
        if self.tfidf_matrix is None:
            raise ValueError('Модель не обучена. Сначала вызовите fit().')
        
        if book_id not in self.book_id_to_idx:
            print(f'Книга {book_id} не найдена в данных')
            return []
        
        # Получаем индекс книги
        book_idx = self.book_id_to_idx[book_id]
        
        # Вычисляем косинусную схожесть
        book_vector = self.tfidf_matrix[book_idx]
        similarities = cosine_similarity(book_vector, self.tfidf_matrix).flatten()
        
        # Получаем индексы самых похожих книг (исключая саму книгу)
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        # Формируем результат
        similar_books = []
        for idx in similar_indices:
            similar_book_id = self.idx_to_book_id[idx]
            similarity_score = similarities[idx]
            similar_books.append((similar_book_id, similarity_score))
        
        return similar_books
    
    def recommend_similar_books(self, book_id, n=10):
        """
        Рекомендовать похожие книги (упрощенный интерфейс)
        
        Args:
            book_id: ID книги
            n: количество рекомендаций
            
        Returns:
            Список ID рекомендованных книг
        """
        similar_books = self.get_similar_books(book_id, n)
        return [book_id for book_id, _ in similar_books]
    
    def get_feature_names(self):
        """Получить названия фичей"""
        if self.vectorizer is None:
            raise ValueError('Модель не обучена.')
        
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features_for_book(self, book_id, n=10):
        """
        Получить топ-N важных слов для книги
        
        Args:
            book_id: ID книги
            n: количество слов
            
        Returns:
            Список кортежей (слово, TF-IDF score)
        """
        if self.tfidf_matrix is None:
            raise ValueError('Модель не обучена.')
        
        if book_id not in self.book_id_to_idx:
            return []
        
        book_idx = self.book_id_to_idx[book_id]
        feature_names = self.get_feature_names()
        tfidf_scores = self.tfidf_matrix[book_idx].toarray().flatten()
        
        # Получаем индексы топ-N фичей
        top_indices = np.argsort(tfidf_scores)[::-1][:n]
        
        top_features = []
        for idx in top_indices:
            if tfidf_scores[idx] > 0:
                top_features.append((feature_names[idx], tfidf_scores[idx]))
        
        return top_features
