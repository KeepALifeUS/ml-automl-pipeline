"""
Advanced Model Selection for Crypto Trading AutoML
Implements enterprise patterns for robust model selection
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
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDRegressor, SGDClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
import seaborn as sns
import time

from ..utils.config_manager import AutoMLConfig


class TaskType(Enum):
    """Типы задач машинного обучения"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


@dataclass
class ModelSelectionResult:
    """Результат отбора моделей"""
    model_scores: Dict[str, float]
    best_models: List[str]
    evaluation_metadata: Dict[str, Any]
    task_type: str
    selection_time: float
    cv_results: Dict[str, List[float]]


class BaseModelProvider(ABC):
    """Базовый класс для провайдеров моделей -  pattern"""
    
    @abstractmethod
    def get_models(self, task_type: TaskType) -> Dict[str, Any]:
        """Получить модели для задачи"""
        pass
    
    @abstractmethod
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Получить параметры по умолчанию для модели"""
        pass


class SklearnModelProvider(BaseModelProvider):
    """Провайдер моделей scikit-learn"""
    
    def get_models(self, task_type: TaskType) -> Dict[str, Any]:
        """Получить модели scikit-learn"""
        if task_type == TaskType.REGRESSION:
            return {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(random_state=42),
                'lasso': Lasso(random_state=42, max_iter=2000),
                'elasticnet': ElasticNet(random_state=42, max_iter=2000),
                'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'extra_trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'knn': KNeighborsRegressor(n_jobs=-1),
                'svr': SVR(),
                'mlp': MLPRegressor(random_state=42, max_iter=500)
            }
        else:  # Classification
            return {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=2000),
                'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'extra_trees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'knn': KNeighborsClassifier(n_jobs=-1),
                'svc': SVC(random_state=42, probability=True),
                'mlp': MLPClassifier(random_state=42, max_iter=500)
            }
    
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Параметры по умолчанию для sklearn моделей"""
        default_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'gradient_boosting': {'n_estimators': 100, 'max_depth': 6},
            'extra_trees': {'n_estimators': 100, 'max_depth': 10},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'elasticnet': {'alpha': 1.0, 'l1_ratio': 0.5},
            'knn': {'n_neighbors': 5},
            'mlp': {'hidden_layer_sizes': (100,), 'alpha': 0.001}
        }
        return default_params.get(model_name, {})


class GradientBoostingModelProvider(BaseModelProvider):
    """Провайдер моделей градиентного бустинга"""
    
    def get_models(self, task_type: TaskType) -> Dict[str, Any]:
        """Получить модели градиентного бустинга"""
        if task_type == TaskType.REGRESSION:
            return {
                'xgboost': xgb.XGBRegressor(
                    random_state=42, n_jobs=-1, verbosity=0
                ),
                'lightgbm': lgb.LGBMRegressor(
                    random_state=42, n_jobs=-1, verbose=-1
                ),
                'catboost': cb.CatBoostRegressor(
                    random_state=42, verbose=False
                )
            }
        else:  # Classification
            return {
                'xgboost': xgb.XGBClassifier(
                    random_state=42, n_jobs=-1, verbosity=0
                ),
                'lightgbm': lgb.LGBMClassifier(
                    random_state=42, n_jobs=-1, verbose=-1
                ),
                'catboost': cb.CatBoostClassifier(
                    random_state=42, verbose=False
                )
            }
    
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Параметры по умолчанию для градиентного бустинга"""
        default_params = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'catboost': {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1
            }
        }
        return default_params.get(model_name, {})


class ModelSelector:
    """
    Продвинутый селектор моделей для криптотрейдинга
    Реализует enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.model_providers: Dict[str, BaseModelProvider] = {}
        self.console = Console()
        self._setup_providers()
        
    def _setup_providers(self):
        """Настройка провайдеров моделей"""
        logger.info("🔧 Настройка провайдеров моделей...")
        
        self.model_providers['sklearn'] = SklearnModelProvider()
        self.model_providers['gradient_boosting'] = GradientBoostingModelProvider()
        
        logger.info(f"✅ Настроено {len(self.model_providers)} провайдеров")
    
    def _detect_task_type(self, y: pd.Series) -> TaskType:
        """Определение типа задачи"""
        unique_values = y.nunique()
        
        if y.dtype in ['float64', 'float32'] or unique_values > 20:
            return TaskType.REGRESSION
        elif unique_values == 2:
            return TaskType.BINARY_CLASSIFICATION
        else:
            return TaskType.MULTICLASS_CLASSIFICATION
    
    def _get_scoring_metric(self, task_type: TaskType) -> str:
        """Получение метрики для скоринга"""
        if task_type == TaskType.REGRESSION:
            return 'neg_mean_squared_error'
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            return 'f1'
        else:  # Multiclass
            return 'f1_macro'
    
    def _get_all_models(self, task_type: TaskType, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Получение всех доступных моделей"""
        all_models = {}
        
        for provider_name, provider in self.model_providers.items():
            provider_models = provider.get_models(task_type)
            
            # Фильтрация моделей если указаны конкретные
            if models:
                provider_models = {
                    name: model for name, model in provider_models.items()
                    if name in models
                }
            
            all_models.update(provider_models)
        
        return all_models
    
    def select_best_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Optional[List[str]] = None,
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        time_series_split: bool = True,
        top_k: int = 5
    ) -> ModelSelectionResult:
        """
        Отбор лучших моделей
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            models: Список моделей для тестирования
            cv_folds: Количество фолдов для кросс-валидации
            scoring: Метрика скоринга
            time_series_split: Использовать временное разделение
            top_k: Количество лучших моделей для возврата
        """
        start_time = time.time()
        
        logger.info("🤖 Запуск отбора моделей...")
        
        # Определение типа задачи
        task_type = self._detect_task_type(y)
        logger.info(f"🎯 Тип задачи: {task_type.value}")
        
        # Получение метрики скоринга
        if scoring is None:
            scoring = self._get_scoring_metric(task_type)
        
        # Получение моделей для тестирования
        all_models = self._get_all_models(task_type, models)
        
        if not all_models:
            logger.error("❌ Нет моделей для тестирования")
            return ModelSelectionResult(
                model_scores={},
                best_models=[],
                evaluation_metadata={'error': 'No models available'},
                task_type=task_type.value,
                selection_time=0.0,
                cv_results={}
            )
        
        logger.info(f"📊 Тестирование {len(all_models)} моделей...")
        
        # Настройка кросс-валидации
        if time_series_split and task_type == TaskType.REGRESSION:
            cv = TimeSeriesSplit(n_splits=cv_folds)
            cv_name = f"TimeSeriesSplit({cv_folds})"
        elif task_type != TaskType.REGRESSION:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_name = f"StratifiedKFold({cv_folds})"
        else:
            cv = cv_folds
            cv_name = f"KFold({cv_folds})"
        
        # Оценка моделей
        model_scores = {}
        cv_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task("Оценка моделей...", total=len(all_models))
            
            for model_name, model in all_models.items():
                progress.update(task, description=f"Модель: {model_name}")
                
                try:
                    # Подготовка данных
                    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
                    y_clean = y.fillna(y.mean()) if y.isna().any() else y
                    
                    # Кросс-валидация
                    scores = cross_val_score(
                        model, X_clean, y_clean,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1,
                        error_score='raise'
                    )
                    
                    mean_score = np.mean(scores)
                    model_scores[model_name] = mean_score
                    cv_results[model_name] = scores.tolist()
                    
                    logger.debug(f"✅ {model_name}: {mean_score:.4f} ± {np.std(scores):.4f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка в модели {model_name}: {e}")
                    model_scores[model_name] = float('-inf')  # Плохой скор
                    cv_results[model_name] = []
                
                progress.advance(task)
        
        # Сортировка моделей по скору
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1],
            reverse=True  # Для большинства метрик больше = лучше
        )
        
        # Коррекция для метрик где меньше = лучше (например, MSE)
        if scoring.startswith('neg_'):
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: -x[1],  # Инвертируем для neg_ метрик
                reverse=True
            )
        
        best_models = [model[0] for model in sorted_models[:top_k]]
        
        selection_time = time.time() - start_time
        
        result = ModelSelectionResult(
            model_scores=model_scores,
            best_models=best_models,
            evaluation_metadata={
                'task_type': task_type.value,
                'scoring_metric': scoring,
                'cv_strategy': cv_name,
                'models_tested': len(all_models),
                'successful_models': len([s for s in model_scores.values() if s != float('-inf')])
            },
            task_type=task_type.value,
            selection_time=selection_time,
            cv_results=cv_results
        )
        
        # Вывод результатов
        self._print_results(result)
        
        logger.info(f"✅ Отбор моделей завершен за {selection_time:.2f}с")
        
        return result
    
    def _print_results(self, result: ModelSelectionResult):
        """Вывод результатов отбора моделей"""
        
        # Создание таблицы с результатами
        table = Table(title="🏆 РЕЗУЛЬТАТЫ ОТБОРА МОДЕЛЕЙ")
        
        table.add_column("Ранг", style="cyan", no_wrap=True)
        table.add_column("Модель", style="magenta")
        table.add_column("Скор", style="green")
        table.add_column("Std", style="yellow")
        
        # Сортировка по скору
        sorted_models = sorted(
            result.model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (model_name, score) in enumerate(sorted_models[:10], 1):
            if model_name in result.cv_results and result.cv_results[model_name]:
                std_score = np.std(result.cv_results[model_name])
                std_str = f"±{std_score:.4f}"
            else:
                std_str = "N/A"
            
            table.add_row(
                str(i),
                model_name,
                f"{score:.4f}",
                std_str
            )
        
        self.console.print(table)
    
    def plot_model_comparison(
        self,
        result: ModelSelectionResult,
        top_n: int = 10,
        save_path: Optional[str] = None
    ):
        """Визуализация сравнения моделей"""
        try:
            # Топ N моделей
            sorted_models = sorted(
                result.model_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            models, scores = zip(*sorted_models)
            
            # График скоров
            plt.figure(figsize=(12, 8))
            
            # Основной график
            plt.subplot(2, 1, 1)
            bars = plt.barh(models, scores, color='skyblue', alpha=0.7)
            plt.xlabel('Скор модели')
            plt.title(f'Сравнение топ {top_n} моделей')
            plt.grid(True, alpha=0.3)
            
            # Добавление значений на столбцы
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center')
            
            # Box plot для топ-5 моделей с CV результатами
            plt.subplot(2, 1, 2)
            top_5_models = [m for m in models[:5] if m in result.cv_results and result.cv_results[m]]
            
            if top_5_models:
                cv_data = [result.cv_results[model] for model in top_5_models]
                plt.boxplot(cv_data, labels=top_5_models)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('CV Скор')
                plt.title('Распределение CV скоров для топ-5 моделей')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 График сохранен: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика: {e}")
    
    def get_model_recommendations(
        self,
        result: ModelSelectionResult,
        data_size: int,
        feature_count: int
    ) -> Dict[str, str]:
        """Получение рекомендаций по выбору моделей"""
        
        recommendations = {}
        
        # Рекомендации на основе размера данных
        if data_size < 1000:
            recommendations['data_size'] = "Небольшой датасет: рекомендуются простые модели (Linear, Ridge, Lasso)"
        elif data_size < 10000:
            recommendations['data_size'] = "Средний датасет: подходят Random Forest, Gradient Boosting"
        else:
            recommendations['data_size'] = "Большой датасет: эффективны XGBoost, LightGBM, CatBoost"
        
        # Рекомендации на основе количества признаков
        if feature_count < 10:
            recommendations['features'] = "Мало признаков: простые модели могут быть эффективнее"
        elif feature_count < 100:
            recommendations['features'] = "Умеренное количество признаков: подходят ансамбли"
        else:
            recommendations['features'] = "Много признаков: рекомендуется регуляризация (Ridge, Lasso)"
        
        # Рекомендации на основе типа задачи
        task_type = result.task_type
        if task_type == 'regression':
            recommendations['task'] = "Регрессия: обратите внимание на MSE и R² метрики"
        else:
            recommendations['task'] = "Классификация: важны precision, recall и F1-score"
        
        # Топ модель
        if result.best_models:
            best_model = result.best_models[0]
            best_score = result.model_scores[best_model]
            recommendations['best_model'] = f"Лучшая модель: {best_model} (скор: {best_score:.4f})"
        
        return recommendations
    
    def get_selection_report(self, result: ModelSelectionResult) -> str:
        """Создание отчета по отбору моделей"""
        
        report = f"""
=== ОТЧЕТ ПО ОТБОРУ МОДЕЛЕЙ ===

Тип задачи: {result.task_type}
Время отбора: {result.selection_time:.2f}с
Протестировано моделей: {len(result.model_scores)}
Успешных моделей: {result.evaluation_metadata.get('successful_models', 0)}

Топ-5 моделей:
"""
        
        sorted_models = sorted(
            result.model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (model_name, score) in enumerate(sorted_models[:5], 1):
            if model_name in result.cv_results and result.cv_results[model_name]:
                std_score = np.std(result.cv_results[model_name])
                report += f"{i}. {model_name}: {score:.4f} ± {std_score:.4f}\n"
            else:
                report += f"{i}. {model_name}: {score:.4f}\n"
        
        report += f"\nМетаданные: {result.evaluation_metadata}"
        
        return report


if __name__ == "__main__":
    # Пример использования
    from ..utils.config_manager import AutoMLConfig
    
    # Создание тестовых данных
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Регрессия
    y_reg = pd.Series(
        X.iloc[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)
    )
    
    # Классификация
    y_clf = pd.Series((y_reg > y_reg.median()).astype(int))
    
    # Создание селектора
    config = AutoMLConfig()
    selector = ModelSelector(config)
    
    # Тестирование регрессии
    print("=== ТЕСТИРОВАНИЕ РЕГРЕССИИ ===")
    result_reg = selector.select_best_models(
        X, y_reg,
        models=['linear_regression', 'ridge', 'random_forest', 'xgboost'],
        cv_folds=3,
        top_k=3
    )
    
    print(selector.get_selection_report(result_reg))
    
    # Тестирование классификации
    print("\n=== ТЕСТИРОВАНИЕ КЛАССИФИКАЦИИ ===")
    result_clf = selector.select_best_models(
        X, y_clf,
        models=['logistic_regression', 'random_forest', 'xgboost'],
        cv_folds=3,
        time_series_split=False
    )
    
    print(selector.get_selection_report(result_clf))
    
    # Рекомендации
    print("\n=== РЕКОМЕНДАЦИИ ===")
    recommendations = selector.get_model_recommendations(
        result_reg, data_size=len(X), feature_count=len(X.columns)
    )
    
    for key, rec in recommendations.items():
        print(f"{key}: {rec}")