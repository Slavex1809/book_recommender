"""
Модуль для предобработки данных
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Класс для предобработки данных рекомендательной системы"""
    
    def __init__(self):
        pass
    
    def plot_rating_distribution(self, ratings):
        """Визуализация распределения оценок"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Гистограмма
        ax = axes[0]
        ratings['rating'].value_counts().sort_index().plot(kind='bar', ax=ax, edgecolor='black')
        ax.set_title('Распределение оценок')
        ax.set_xlabel('Оценка')
        ax.set_ylabel('Количество')
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1]
        ratings['rating'].plot(kind='box', ax=ax)
        ax.set_title('Box plot оценок')
        ax.set_ylabel('Оценка')
        ax.grid(True, alpha=0.3)
        
        # Pie chart
        ax = axes[2]
        rating_counts = ratings['rating'].value_counts().sort_index()
        ax.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Процентное распределение')
        
        plt.tight_layout()
        return fig
    
    def analyze_user_activity(self, ratings):
        """Анализ активности пользователей"""
        user_stats = ratings.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        user_stats.columns = ['num_ratings', 'avg_rating', 'rating_std']
        return user_stats
    
    def plot_user_activity(self, user_stats):
        """Визуализация активности пользователей"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        ax = axes[0]
        user_stats['num_ratings'].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_title('Распределение количества оценок на пользователя')
        ax.set_xlabel('Количество оценок')
        ax.set_ylabel('Количество пользователей')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        user_stats['num_ratings'].describe().drop(['count', '50%']).plot(kind='bar', ax=ax, edgecolor='black')
        ax.set_title('Статистики по оценкам пользователей')
        ax.set_ylabel('Количество оценок')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_book_popularity(self, ratings):
        """Анализ популярности книг"""
        book_stats = ratings.groupby('book_id').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        book_stats.columns = ['num_ratings', 'avg_rating', 'rating_std']
        return book_stats
    
    def plot_book_popularity(self, book_stats):
        """Визуализация популярности книг"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        ax = axes[0]
        book_stats['num_ratings'].hist(bins=50, ax=ax, edgecolor='black', log=True)
        ax.set_title('Распределение популярности книг (log scale)')
        ax.set_xlabel('Количество оценок')
        ax.set_ylabel('Количество книг (log)')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        top_books = book_stats.nlargest(20, 'num_ratings')
        ax.barh(range(len(top_books)), top_books['num_ratings'])
        ax.set_yticks(range(len(top_books)))
        ax.set_yticklabels([f'Книга {i}' for i in top_books.index])
        ax.set_title('Топ-20 самых популярных книг')
        ax.set_xlabel('Количество оценок')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def prepare_book_content_data(self, books, book_tags, tags, top_n_tags=20):
        """Подготовка данных для контентной модели"""
        # Объединяем книги с тегами
        book_tags_merged = book_tags.merge(tags, on='tag_id')
        
        # Для каждой книги берем топ-N тегов
        book_content = []
        for book_id in books['book_id']:
            book_tag_data = book_tags_merged[book_tags_merged['goodreads_book_id'] == book_id]
            top_tags = book_tag_data.nlargest(top_n_tags, 'count')['tag_name'].tolist()
            
            # Получаем информацию о книге
            book_info = books[books['book_id'] == book_id]
            if not book_info.empty:
                title = book_info['original_title'].values[0]
                content = f'{title} ' + ' '.join(top_tags)
                book_content.append({
                    'book_id': book_id,
                    'title': title,
                    'content': content,
                    'tags': ' '.join(top_tags)
                })
        
        return pd.DataFrame(book_content)
    
    def train_test_split_temporal(self, ratings, test_size=0.2, random_state=42):
        """Разделение данных по времени"""
        # Сортируем по user_id и создаем случайное разделение
        train_data, test_data = train_test_split(
            ratings, 
            test_size=test_size, 
            random_state=random_state,
            stratify=ratings['user_id']
        )
        return train_data, test_data
    
    def calculate_sparsity(self, ratings):
        """Вычисление разреженности матрицы"""
        n_users = ratings['user_id'].nunique()
        n_books = ratings['book_id'].nunique()
        n_ratings = len(ratings)
        
        total_cells = n_users * n_books
        sparsity = 1 - (n_ratings / total_cells)
        
        print(f'Разреженность матрицы: {sparsity:.4%}')
        print(f'Пользователей: {n_users:,}')
        print(f'Книг: {n_books:,}')
        print(f'Оценок: {n_ratings:,}')
        print(f'Всего ячеек: {total_cells:,}')
        
        return sparsity
    
    def identify_cold_start_users(self, ratings, threshold=5):
        """Выявление пользователей с холодным стартом"""
        user_rating_counts = ratings.groupby('user_id')['rating'].count()
        cold_start_users = user_rating_counts[user_rating_counts < threshold].index.tolist()
        return cold_start_users
    
    def identify_cold_start_books(self, ratings, threshold=10):
        """Выявление книг с холодным стартом"""
        book_rating_counts = ratings.groupby('book_id')['rating'].count()
        cold_start_books = book_rating_counts[book_rating_counts < threshold].index.tolist()
        return cold_start_books
