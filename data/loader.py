"""
Модуль для загрузки данных
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Класс для загрузки и первичной обработки данных"""
    
    def __init__(self, data_dir='../goodbooks-10k'):
        self.data_dir = Path(data_dir)
        
    def load_ratings(self):
        """Загрузка рейтингов"""
        ratings_path = self.data_dir / 'ratings.csv'
        ratings = pd.read_csv(ratings_path)
        print(f'Загружено оценок: {len(ratings):,}')
        return ratings
    
    def load_books(self):
        """Загрузка информации о книгах"""
        books_path = self.data_dir / 'books.csv'
        books = pd.read_csv(books_path)
        books['original_title'] = books['original_title'].fillna(books['title'])
        print(f'Загружено книг: {len(books):,}')
        return books
    
    def load_tags(self):
        """Загрузка тегов"""
        tags_path = self.data_dir / 'tags.csv'
        tags = pd.read_csv(tags_path)
        print(f'Загружено тегов: {len(tags):,}')
        return tags
    
    def load_book_tags(self):
        """Загрузка связей книг и тегов"""
        book_tags_path = self.data_dir / 'book_tags.csv'
        book_tags = pd.read_csv(book_tags_path)
        print(f'Загружено связей книг-тегов: {len(book_tags):,}')
        return book_tags
    
    def load_all_data(self):
        """Загрузка всех данных"""
        print('Загрузка данных Goodbooks-10k...')
        
        ratings = self.load_ratings()
        books = self.load_books()
        tags = self.load_tags()
        book_tags = self.load_book_tags()
        
        return ratings, books, tags, book_tags
