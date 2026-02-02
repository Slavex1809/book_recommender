"""
Item-based Collaborative Filtering модель
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ItemBasedCF:
    """Item-based Collaborative Filtering рекомендательная система"""
    
    def __init__(self, k_neighbors=50, min_similarity=0.1):
        self.k_neighbors = k_neighbors
        self.min_similarity = min_similarity
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.book_ids = None
        self.user_ids = None
        
    def fit(self, ratings):
        print('Обучение Item-based CF...')
        
        # Создаем user-item матрицу
        self.user_ids = ratings['user_id'].unique()
        self.book_ids = ratings['book_id'].unique()
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        book_to_idx = {book_id: idx for idx, book_id in enumerate(self.book_ids)}
        
        # Создаем разреженную матрицу
        rows = [user_to_idx[row['user_id']] for _, row in ratings.iterrows()]
        cols = [book_to_idx[row['book_id']] for _, row in ratings.iterrows()]
        values = ratings['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.book_ids))
        )
        
        # Вычисляем схожесть между книгами (item-item)
        print('Вычисление матрицы схожести...')
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        # Оставляем только k ближайших соседей для каждой книги
        self.similarity_matrix = np.zeros_like(item_similarity)
        
        for i in tqdm(range(len(self.book_ids)), desc='Поиск соседей'):
            similarities = item_similarity[i]
            top_indices = np.argsort(similarities)[::-1][1:self.k_neighbors+1]
            top_similarities = similarities[top_indices]
            
            # Фильтруем по минимальной схожести
            mask = top_similarities > self.min_similarity
            if mask.any():
                self.similarity_matrix[i, top_indices[mask]] = top_similarities[mask]
        
        print(f'Матрица схожести создана: {self.similarity_matrix.shape}')
        return self
    
    def predict_rating(self, user_id, book_id):
        if user_id not in self.user_ids or book_id not in self.book_ids:
            return None
        
        user_idx = np.where(self.user_ids == user_id)[0][0]
        book_idx = np.where(self.book_ids == book_id)[0][0]
        
        # Получаем оценки пользователя
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Получаем схожести для книги
        book_similarities = self.similarity_matrix[book_idx]
        
        # Находим похожие книги, которые пользователь оценил
        similar_books_mask = book_similarities > 0
        rated_books_mask = user_ratings > 0
        
        mask = similar_books_mask & rated_books_mask
        
        if not mask.any():
            return None
        
        # Взвешенное среднее
        similarities = book_similarities[mask]
        ratings = user_ratings[mask]
        
        if similarities.sum() == 0:
            return None
        
        predicted = np.dot(ratings, similarities) / similarities.sum()
        return predicted
    
    def recommend(self, user_id, n=10):
        user_ratings = self.get_user_ratings(user_id)
        if user_ratings is None:
            return []
        
        # Вычисляем предсказанные оценки для всех книг
        predictions = []
        for book_id in self.book_ids:
            if book_id not in user_ratings:
                pred = self.predict_rating(user_id, book_id)
                if pred is not None:
                    predictions.append((book_id, pred))
        
        # Сортируем по убыванию предсказанной оценки
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [book_id for book_id, _ in predictions[:n]]
    
    def get_user_ratings(self, user_id):
        if user_id not in self.user_ids:
            return None
        
        user_idx = np.where(self.user_ids == user_id)[0][0]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        rated_books = {}
        for book_idx, rating in enumerate(user_ratings):
            if rating > 0:
                book_id = self.book_ids[book_idx]
                rated_books[book_id] = rating
        
        return rated_books
