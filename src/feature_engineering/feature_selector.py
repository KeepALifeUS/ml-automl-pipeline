"""
Advanced Feature Selection for AutoML Pipeline
Implements enterprise patterns for robust feature selection
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    f_regression, f_classif, mutual_info_regression, mutual_info_classif,
    chi2, RFE, RFECV
)
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from loguru import logger
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config_manager import AutoMLConfig


class SelectionMethod(Enum):
    """Методы отбора признаков"""
    STATISTICAL = "statistical"
    MODEL_BASED = "model_based"
    UNIVARIATE = "univariate"
    RECURSIVE = "recursive"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    VARIANCE = "variance"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


@dataclass
class FeatureSelectionResult:
    """Результат отбора признаков"""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_metadata: Dict[str, Any]
    eliminated_features: List[str]
    selection_time: float
    method_used: str


class BaseFeatureSelector(ABC):
    """Базовый класс для селекторов признаков -  pattern"""
    
    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> FeatureSelectionResult:
        """Выбрать признаки"""
        pass
    
    @abstractmethod
    def get_selection_params(self) -> Dict[str, Any]:
        """Получить параметры селекции"""
        pass


class StatisticalFeatureSelector(BaseFeatureSelector):
    """Статистический селектор признаков"""
    
    def __init__(self, method: str = 'f_regression', k: int = 50, percentile: float = 50):
        self.method = method
        self.k = k
        self.percentile = percentile
        self.selector = None
        
        # Выбор статистической функции
        self.stat_functions = {
            'f_regression': f_regression,
            'f_classif': f_classif,
            'mutual_info_regression': mutual_info_regression,
            'mutual_info_classif': mutual_info_classif,
            'chi2': chi2
        }
        
    def select(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> FeatureSelectionResult:
        """Статистический отбор признаков"""
        import time
        start_time = time.time()
        
        logger.info(f"📊 Статистический отбор признаков методом {self.method}")
        
        try:
            # Выбор функции скоринга
            score_func = self.stat_functions.get(self.method, f_regression)
            
            # Определение стратегии отбора
            if self.k > 0:
                self.selector = SelectKBest(score_func=score_func, k=min(self.k, X.shape[1]))
            else:
                self.selector = SelectPercentile(score_func=score_func, percentile=self.percentile)
            
            # Очистка данных
            X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
            y_clean = y.fillna(y.mean()) if y.isna().any() else y
            
            # Отбор признаков
            X_selected = self.selector.fit_transform(X_clean, y_clean)
            
            # Получение выбранных признаков
            selected_mask = self.selector.get_support()
            selected_features = X.columns[selected_mask].tolist()
            eliminated_features = X.columns[~selected_mask].tolist()
            
            # Получение скоров
            scores = self.selector.scores_
            feature_scores = dict(zip(X.columns, scores))
            
            processing_time = time.time() - start_time
            
            result = FeatureSelectionResult(
                selected_features=selected_features,
                feature_scores=feature_scores,
                selection_metadata={
                    'method': self.method,
                    'k_features': len(selected_features),
                    'original_features': X.shape[1],
                    'reduction_ratio': 1 - len(selected_features) / X.shape[1]
                },
                eliminated_features=eliminated_features,
                selection_time=processing_time,
                method_used=f"statistical_{self.method}"
            )
            
            logger.info(f"✅ Отобрано {len(selected_features)} из {X.shape[1]} признаков")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка статистического отбора: {e}")
            return FeatureSelectionResult(
                selected_features=list(X.columns),
                feature_scores={col: 0.0 for col in X.columns},
                selection_metadata={'error': str(e)},
                eliminated_features=[],
                selection_time=time.time() - start_time,
                method_used=f"statistical_{self.method}_failed"
            )
    
    def get_selection_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'k': self.k,
            'percentile': self.percentile
        }


class ModelBasedFeatureSelector(BaseFeatureSelector):
    """Модельный селектор признаков"""
    
    def __init__(self, model_type: str = 'random_forest', max_features: int = 100):
        self.model_type = model_type
        self.max_features = max_features
        self.model = None
        
    def _get_model(self, task_type: str = 'regression'):
        """Получить модель для отбора признаков"""
        if self.model_type == 'random_forest':
            if task_type == 'regression':
                return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                return RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        elif self.model_type == 'xgboost':
            if task_type == 'regression':
                return xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                return xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            # По умолчанию Random Forest
            return RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    def select(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'regression') -> FeatureSelectionResult:
        """Модельный отбор признаков"""
        import time
        start_time = time.time()
        
        logger.info(f"🤖 Модельный отбор признаков с {self.model_type}")
        
        try:
            # Подготовка данных
            X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
            y_clean = y.fillna(y.mean()) if y.isna().any() else y
            
            # Получение модели
            self.model = self._get_model(task_type)
            
            # Обучение модели
            self.model.fit(X_clean, y_clean)
            
            # Получение важности признаков
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
            else:
                # Fallback: используем корреляцию
                importances = np.abs(X_clean.corrwith(y_clean).fillna(0).values)
            
            # Создание словаря важности
            feature_scores = dict(zip(X.columns, importances))
            
            # Отбор топ признаков
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:self.max_features]]
            eliminated_features = [f[0] for f in sorted_features[self.max_features:]]
            
            processing_time = time.time() - start_time
            
            result = FeatureSelectionResult(
                selected_features=selected_features,
                feature_scores=feature_scores,
                selection_metadata={
                    'model_type': self.model_type,
                    'task_type': task_type,
                    'max_features': self.max_features,
                    'original_features': X.shape[1],
                    'mean_importance': np.mean(importances),
                    'std_importance': np.std(importances)
                },
                eliminated_features=eliminated_features,
                selection_time=processing_time,
                method_used=f"model_{self.model_type}"
            )
            
            logger.info(f"✅ Отобрано {len(selected_features)} топ признаков")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка модельного отбора: {e}")
            return FeatureSelectionResult(
                selected_features=list(X.columns)[:self.max_features],
                feature_scores={col: 0.0 for col in X.columns},
                selection_metadata={'error': str(e)},
                eliminated_features=[],
                selection_time=time.time() - start_time,
                method_used=f"model_{self.model_type}_failed"
            )
    
    def get_selection_params(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'max_features': self.max_features
        }


class CorrelationFeatureSelector(BaseFeatureSelector):
    """Селектор на основе корреляции"""
    
    def __init__(self, correlation_threshold: float = 0.95, target_correlation_min: float = 0.01):
        self.correlation_threshold = correlation_threshold
        self.target_correlation_min = target_correlation_min
        
    def select(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> FeatureSelectionResult:
        """Отбор признаков по корреляции"""
        import time
        start_time = time.time()
        
        logger.info("🔗 Корреляционный отбор признаков")
        
        try:
            # Подготовка данных
            X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
            y_clean = y.fillna(y.mean()) if y.isna().any() else y
            
            # Удаление признаков с низкой корреляцией с целевой переменной
            target_correlations = X_clean.corrwith(y_clean).abs()
            high_target_corr_features = target_correlations[
                target_correlations >= self.target_correlation_min
            ].index.tolist()
            
            if not high_target_corr_features:
                logger.warning("⚠️ Нет признаков с достаточной корреляцией с целевой переменной")
                high_target_corr_features = list(X.columns)
            
            X_filtered = X_clean[high_target_corr_features]
            
            # Удаление высоко коррелирующих между собой признаков
            correlation_matrix = X_filtered.corr().abs()
            
            # Поиск пар с высокой корреляцией
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] >= self.correlation_threshold:
                        col_i = correlation_matrix.columns[i]
                        col_j = correlation_matrix.columns[j]
                        
                        # Оставляем признак с большей корреляцией с целевой переменной
                        target_corr_i = abs(target_correlations[col_i])
                        target_corr_j = abs(target_correlations[col_j])
                        
                        if target_corr_i >= target_corr_j:
                            high_corr_pairs.append(col_j)
                        else:
                            high_corr_pairs.append(col_i)
            
            # Удаление дубликатов
            features_to_remove = list(set(high_corr_pairs))
            selected_features = [f for f in high_target_corr_features if f not in features_to_remove]
            
            # Создание скоров (корреляция с целевой переменной)
            feature_scores = target_correlations.to_dict()
            
            processing_time = time.time() - start_time
            
            result = FeatureSelectionResult(
                selected_features=selected_features,
                feature_scores=feature_scores,
                selection_metadata={
                    'correlation_threshold': self.correlation_threshold,
                    'target_correlation_min': self.target_correlation_min,
                    'removed_high_corr': len(features_to_remove),
                    'removed_low_target_corr': len(X.columns) - len(high_target_corr_features)
                },
                eliminated_features=[f for f in X.columns if f not in selected_features],
                selection_time=processing_time,
                method_used="correlation"
            )
            
            logger.info(f"✅ Отобрано {len(selected_features)} признаков после корреляционной фильтрации")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка корреляционного отбора: {e}")
            return FeatureSelectionResult(
                selected_features=list(X.columns),
                feature_scores={col: 0.0 for col in X.columns},
                selection_metadata={'error': str(e)},
                eliminated_features=[],
                selection_time=time.time() - start_time,
                method_used="correlation_failed"
            )
    
    def get_selection_params(self) -> Dict[str, Any]:
        return {
            'correlation_threshold': self.correlation_threshold,
            'target_correlation_min': self.target_correlation_min
        }


class VarianceFeatureSelector(BaseFeatureSelector):
    """Селектор на основе дисперсии"""
    
    def __init__(self, variance_threshold: float = 0.0):
        self.variance_threshold = variance_threshold
        
    def select(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> FeatureSelectionResult:
        """Отбор признаков по дисперсии"""
        import time
        start_time = time.time()
        
        logger.info("📈 Отбор признаков по дисперсии")
        
        try:
            # Подготовка данных
            X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Вычисление дисперсий
            variances = X_clean.var()
            
            # Отбор признаков с дисперсией выше порога
            high_var_features = variances[variances > self.variance_threshold].index.tolist()
            
            feature_scores = variances.to_dict()
            eliminated_features = [f for f in X.columns if f not in high_var_features]
            
            processing_time = time.time() - start_time
            
            result = FeatureSelectionResult(
                selected_features=high_var_features,
                feature_scores=feature_scores,
                selection_metadata={
                    'variance_threshold': self.variance_threshold,
                    'mean_variance': variances.mean(),
                    'removed_low_variance': len(eliminated_features)
                },
                eliminated_features=eliminated_features,
                selection_time=processing_time,
                method_used="variance"
            )
            
            logger.info(f"✅ Отобрано {len(high_var_features)} признаков с высокой дисперсией")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка отбора по дисперсии: {e}")
            return FeatureSelectionResult(
                selected_features=list(X.columns),
                feature_scores={col: 0.0 for col in X.columns},
                selection_metadata={'error': str(e)},
                eliminated_features=[],
                selection_time=time.time() - start_time,
                method_used="variance_failed"
            )
    
    def get_selection_params(self) -> Dict[str, Any]:
        return {'variance_threshold': self.variance_threshold}


class AdvancedFeatureSelector:
    """
    Продвинутый селектор признаков с множественными методами
    Реализует enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.selectors: Dict[str, BaseFeatureSelector] = {}
        self._setup_selectors()
        
    def _setup_selectors(self):
        """Настройка селекторов"""
        logger.info("🔧 Настройка селекторов признаков...")
        
        selection_config = self.config.feature_selection
        
        # Статистический селектор
        self.selectors['statistical'] = StatisticalFeatureSelector(
            method=selection_config.get('statistical_method', 'f_regression'),
            k=selection_config.get('statistical_k', 50),
            percentile=selection_config.get('statistical_percentile', 50)
        )
        
        # Модельный селектор
        self.selectors['model'] = ModelBasedFeatureSelector(
            model_type=selection_config.get('model_type', 'random_forest'),
            max_features=selection_config.get('model_max_features', 100)
        )
        
        # Корреляционный селектор
        self.selectors['correlation'] = CorrelationFeatureSelector(
            correlation_threshold=selection_config.get('correlation_threshold', 0.95),
            target_correlation_min=selection_config.get('target_correlation_min', 0.01)
        )
        
        # Селектор по дисперсии
        self.selectors['variance'] = VarianceFeatureSelector(
            variance_threshold=selection_config.get('variance_threshold', 0.0)
        )
        
        logger.info(f"✅ Настроено {len(self.selectors)} селекторов")
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: Optional[List[str]] = None,
        task_type: str = 'regression',
        ensemble_selection: bool = True
    ) -> FeatureSelectionResult:
        """
        Основной метод отбора признаков
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            methods: Методы для использования
            task_type: Тип задачи (regression/classification)
            ensemble_selection: Использовать ансамбль методов
        """
        logger.info("🎯 Запуск продвинутого отбора признаков...")
        
        if methods is None:
            methods = list(self.selectors.keys())
        
        results = {}
        
        # Применение каждого метода
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        ) as progress:
            task = progress.add_task("Отбор признаков...", total=len(methods))
            
            for method in methods:
                if method not in self.selectors:
                    continue
                    
                try:
                    progress.update(task, description=f"Метод: {method}")
                    
                    if method == 'model':
                        result = self.selectors[method].select(X, y, task_type=task_type)
                    else:
                        result = self.selectors[method].select(X, y)
                    
                    results[method] = result
                    progress.advance(task)
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка в методе {method}: {e}")
                    progress.advance(task)
        
        if not results:
            logger.error("❌ Ни один метод отбора не сработал")
            return FeatureSelectionResult(
                selected_features=list(X.columns),
                feature_scores={col: 0.0 for col in X.columns},
                selection_metadata={'error': 'All methods failed'},
                eliminated_features=[],
                selection_time=0.0,
                method_used="failed"
            )
        
        if ensemble_selection and len(results) > 1:
            return self._ensemble_selection(X, y, results)
        else:
            # Используем лучший метод (с наибольшим количеством признаков)
            best_method = max(results.keys(), key=lambda m: len(results[m].selected_features))
            return results[best_method]
    
    def _ensemble_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        results: Dict[str, FeatureSelectionResult]
    ) -> FeatureSelectionResult:
        """Ансамблевый отбор признаков"""
        import time
        start_time = time.time()
        
        logger.info("🤝 Ансамблевый отбор признаков...")
        
        # Подсчет голосов за каждый признак
        feature_votes = {}
        all_scores = {}
        
        for method, result in results.items():
            for feature in result.selected_features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
                if feature in result.feature_scores:
                    if feature not in all_scores:
                        all_scores[feature] = []
                    all_scores[feature].append(result.feature_scores[feature])
        
        # Вычисление средних скоров
        average_scores = {}
        for feature, scores in all_scores.items():
            average_scores[feature] = np.mean(scores)
        
        # Определение порога голосов (минимум 2 голоса из 3+ методов)
        min_votes = max(2, len(results) // 2)
        selected_features = [
            feature for feature, votes in feature_votes.items()
            if votes >= min_votes
        ]
        
        # Если слишком мало признаков, добавляем топ по скорам
        if len(selected_features) < 10:
            sorted_by_score = sorted(
                average_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, _ in sorted_by_score:
                if feature not in selected_features:
                    selected_features.append(feature)
                    if len(selected_features) >= 20:  # Максимум 20 признаков
                        break
        
        eliminated_features = [f for f in X.columns if f not in selected_features]
        processing_time = time.time() - start_time
        
        ensemble_result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=average_scores,
            selection_metadata={
                'ensemble_methods': list(results.keys()),
                'min_votes_threshold': min_votes,
                'feature_votes': feature_votes,
                'total_original_features': X.shape[1]
            },
            eliminated_features=eliminated_features,
            selection_time=processing_time,
            method_used="ensemble"
        )
        
        logger.info(f"✅ Ансамбль отобрал {len(selected_features)} признаков")
        return ensemble_result
    
    def plot_feature_importance(
        self,
        result: FeatureSelectionResult,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """Визуализация важности признаков"""
        try:
            # Топ N признаков по важности
            top_features = sorted(
                result.feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            features, scores = zip(*top_features)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(scores), y=list(features), palette='viridis')
            plt.title(f'Топ {top_n} признаков по важности ({result.method_used})')
            plt.xlabel('Важность признака')
            plt.ylabel('Признаки')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 График сохранен: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика: {e}")
    
    def get_selection_report(self, result: FeatureSelectionResult) -> str:
        """Создание отчета по отбору признаков"""
        report = f"""
=== ОТЧЕТ ПО ОТБОРУ ПРИЗНАКОВ ===

Метод: {result.method_used}
Время выполнения: {result.selection_time:.2f}с

Статистика:
- Исходное количество признаков: {len(result.selected_features) + len(result.eliminated_features)}
- Отобранных признаков: {len(result.selected_features)}
- Исключенных признаков: {len(result.eliminated_features)}
- Коэффициент сжатия: {len(result.eliminated_features) / (len(result.selected_features) + len(result.eliminated_features)):.2%}

Топ-10 признаков по важности:
"""
        
        top_features = sorted(
            result.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (feature, score) in enumerate(top_features, 1):
            report += f"{i:2d}. {feature}: {score:.4f}\n"
        
        report += f"\nМетаданные: {result.selection_metadata}"
        
        return report


if __name__ == "__main__":
    # Пример использования
    from ..utils.config_manager import AutoMLConfig
    
    # Создание тестовых данных
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Создание синтетической целевой переменной
    # Первые 10 признаков важные, остальные шум
    important_features = X.iloc[:, :10].values
    y = pd.Series(
        np.sum(important_features * np.random.randn(10), axis=1) + 
        0.1 * np.random.randn(n_samples)
    )
    
    # Создание селектора
    config = AutoMLConfig()
    selector = AdvancedFeatureSelector(config)
    
    # Отбор признаков
    result = selector.select_features(X, y, ensemble_selection=True)
    
    print("=== РЕЗУЛЬТАТЫ ОТБОРА ПРИЗНАКОВ ===")
    print(f"Отобрано признаков: {len(result.selected_features)}")
    print(f"Время отбора: {result.selection_time:.2f}с")
    print(f"Метод: {result.method_used}")
    
    # Отчет
    print(selector.get_selection_report(result))