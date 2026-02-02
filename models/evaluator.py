"""
Модуль для оценки моделей рекомендаций
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ModelEvaluator:
    """Класс для оценки рекомендательных моделей"""
    
    def __init__(self, test_ratings, rating_threshold=4):
        self.test_ratings = test_ratings
        self.rating_threshold = rating_threshold
        
    def precision_at_k(self, recommended_items, relevant_items, k):
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / k if k > 0 else 0
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / len(relevant_items) if len(relevant_items) > 0 else 0
    
    def dcg_at_k(self, recommended_items, relevant_items, k):
        relevant_set = set(relevant_items)
        dcg = 0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_set:
                dcg += 1 / np.log2(i + 2)
        return dcg
    
    def ndcg_at_k(self, recommended_items, relevant_items, k):
        dcg = self.dcg_at_k(recommended_items, relevant_items, k)
        ideal_recommendations = relevant_items[:k]
        idcg = self.dcg_at_k(ideal_recommendations, relevant_items, k)
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate_model(self, predictions, model_name, k_values=[5, 10, 20]):
        results = {}
        
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            
            for pred in predictions:
                if pred['recommendations'] and pred['relevant_items']:
                    prec = self.precision_at_k(pred['recommendations'], pred['relevant_items'], k)
                    rec = self.recall_at_k(pred['recommendations'], pred['relevant_items'], k)
                    ndcg = self.ndcg_at_k(pred['recommendations'], pred['relevant_items'], k)
                    
                    precisions.append(prec)
                    recalls.append(rec)
                    ndcgs.append(ndcg)
            
            if precisions:
                results[f'precision@{k}'] = np.mean(precisions)
                results[f'recall@{k}'] = np.mean(recalls)
                results[f'ndcg@{k}'] = np.mean(ndcgs)
            else:
                results[f'precision@{k}'] = 0
                results[f'recall@{k}'] = 0
                results[f'ndcg@{k}'] = 0
        
        print(f'\nРезультаты для {model_name}:')
        for k in k_values:
            print(f'  K={k}: Precision={results[f"precision@{k}"]:.3f}, '
                  f'Recall={results[f"recall@{k}"]:.3f}, '
                  f'nDCG={results[f"ndcg@{k}"]:.3f}')
        
        return results
    
    def create_summary_dataframe(self, results_dict):
        rows = []
        for model_name, metrics in results_dict.items():
            row = {'Model': model_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def plot_metrics_comparison(self, results_dict):
        metrics = ['precision@5', 'recall@5', 'ndcg@5', 
                   'precision@10', 'recall@10', 'ndcg@10']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            model_names = []
            metric_values = []
            
            for model_name, metrics_dict in results_dict.items():
                if metric in metrics_dict:
                    model_names.append(model_name)
                    metric_values.append(metrics_dict[metric])
            
            bars = ax.bar(model_names, metric_values)
            ax.set_title(metric.replace('@', ' @'))
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
