"""
SVD (Matrix Factorization) модель рекомендаций
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SVDRecommender:
    """SVD рекомендательная система"""
    
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model = None
        self.trainset = None
        
    def fit(self, ratings):
        print('Обучение SVD модели...')
        
        # Подготовка данных для surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
        
        # Разделение на train/test
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
        self.trainset = trainset
        
        # Обучение модели
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42
        )
        
        self.model.fit(trainset)
        print(f'SVD обучена: {self.n_factors} факторов, {self.n_epochs} эпох')
        return self
    
    def predict(self, user_id, book_id):
        if self.model is None:
            raise ValueError('Модель не обучена.')
        
        # Проверяем, есть ли пользователь и книга в trainset
        try:
            pred = self.model.predict(user_id, book_id)
            return pred.est
        except:
            return None
    
    def recommend(self, user_id, n=10):
        if self.model is None:
            raise ValueError('Модель не обучена.')
        
        # Получаем все книги, которые пользователь еще не оценивал
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
            user_rated_items = set([iid for (iid, _) in self.trainset.ur[inner_user_id]])
            all_items = set(range(self.trainset.n_items))
            unrated_items = all_items - user_rated_items
            
            # Предсказываем оценки для непросмотренных книг
            predictions = []
            for inner_item_id in tqdm(unrated_items, desc='Предсказание оценок', leave=False):
                item_id = self.trainset.to_raw_iid(inner_item_id)
                pred = self.predict(user_id, item_id)
                if pred is not None:
                    predictions.append((item_id, pred))
            
            # Сортируем по убыванию предсказанной оценки
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return [item_id for item_id, _ in predictions[:n]]
        except Exception as e:
            # Если пользователь не найден в trainset, возвращаем популярные
            print(f"Пользователь {user_id} не найден: {e}")
            return []
    
    def get_user_factors(self, user_id):
        if self.model is None:
            raise ValueError('Модель не обучена.')
        
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
            return self.model.pu[inner_user_id]
        except:
            return None
    
    def get_item_factors(self, book_id):
        if self.model is None:
            raise ValueError('Модель не обучена.')
        
        try:
            inner_item_id = self.trainset.to_inner_iid(book_id)
            return self.model.qi[inner_item_id]
        except:
            return None
