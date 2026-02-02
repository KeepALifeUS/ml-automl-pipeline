"""
Bayesian Hyperparameter Optimization for Crypto Trading AutoML
Implements enterprise patterns for robust optimization
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
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
import optuna
from loguru import logger
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from ..utils.config_manager import AutoMLConfig


class OptimizationMethod(Enum):
    """Методы оптимизации"""
    GAUSSIAN_PROCESS = "gaussian_process"
    RANDOM_FOREST = "random_forest" 
    GRADIENT_BOOSTING = "gradient_boosting"
    OPTUNA_TPE = "optuna_tpe"
    OPTUNA_RANDOM = "optuna_random"


@dataclass
class OptimizationResult:
    """Результат оптимизации гиперпараметров"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_data: Dict[str, Any]
    optimization_time: float
    method_used: str
    model_name: str


class BaseOptimizer(ABC):
    """Базовый класс для оптимизаторов -  pattern"""
    
    @abstractmethod
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        n_calls: int = 100
    ) -> OptimizationResult:
        """Оптимизировать гиперпараметры"""
        pass


class SkoptBayesianOptimizer(BaseOptimizer):
    """Байесовский оптимизатор на основе scikit-optimize"""
    
    def __init__(self, method: OptimizationMethod = OptimizationMethod.GAUSSIAN_PROCESS):
        self.method = method
        self.optimization_history = []
        
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        n_calls: int = 100,
        random_state: int = 42
    ) -> OptimizationResult:
        """Байесовская оптимизация с scikit-optimize"""
        start_time = time.time()
        
        logger.info(f"🎯 Запуск байесовской оптимизации методом {self.method.value}")
        
        try:
            # Преобразование пространства поиска
            dimensions = self._convert_search_space(search_space)
            
            # Оборачиваем целевую функцию для отслеживания истории
            @use_named_args(dimensions)
            def wrapped_objective(**params):
                score = objective_function(params)
                self.optimization_history.append({'params': params.copy(), 'score': score})
                return score  # scikit-optimize минимизирует, поэтому возвращаем как есть
            
            # Выбор алгоритма оптимизации
            if self.method == OptimizationMethod.GAUSSIAN_PROCESS:
                result = gp_minimize(
                    func=wrapped_objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    random_state=random_state,
                    acq_func='EI'  # Expected Improvement
                )
            elif self.method == OptimizationMethod.RANDOM_FOREST:
                result = forest_minimize(
                    func=wrapped_objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    random_state=random_state
                )
            else:  # GRADIENT_BOOSTING
                result = gbrt_minimize(
                    func=wrapped_objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    random_state=random_state
                )
            
            # Извлечение лучших параметров
            best_params = {}
            for i, dim in enumerate(dimensions):
                best_params[dim.name] = result.x[i]
            
            optimization_time = time.time() - start_time
            
            optimization_result = OptimizationResult(
                best_params=best_params,
                best_score=result.fun,
                optimization_history=self.optimization_history,
                convergence_data={
                    'func_vals': result.func_vals.tolist(),
                    'x_iters': [x.tolist() if isinstance(x, np.ndarray) else x for x in result.x_iters],
                    'n_calls': n_calls,
                    'convergence_rate': self._calculate_convergence_rate(result.func_vals)
                },
                optimization_time=optimization_time,
                method_used=self.method.value,
                model_name="unknown"
            )
            
            logger.info(f"✅ Оптимизация завершена: лучший скор {result.fun:.4f}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка байесовской оптимизации: {e}")
            return OptimizationResult(
                best_params={},
                best_score=float('inf'),
                optimization_history=[],
                convergence_data={},
                optimization_time=time.time() - start_time,
                method_used=f"{self.method.value}_failed",
                model_name="unknown"
            )
    
    def _convert_search_space(self, search_space: Dict[str, Any]) -> List:
        """Конвертация пространства поиска в формат scikit-optimize"""
        dimensions = []
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'real':
                dimensions.append(Real(
                    low=param_config['low'],
                    high=param_config['high'],
                    prior=param_config.get('prior', 'uniform'),
                    name=param_name
                ))
            elif param_config['type'] == 'integer':
                dimensions.append(Integer(
                    low=param_config['low'],
                    high=param_config['high'],
                    name=param_name
                ))
            elif param_config['type'] == 'categorical':
                dimensions.append(Categorical(
                    categories=param_config['categories'],
                    name=param_name
                ))
        
        return dimensions
    
    def _calculate_convergence_rate(self, func_vals: np.ndarray) -> float:
        """Вычисление скорости сходимости"""
        if len(func_vals) < 2:
            return 0.0
        
        # Вычисляем относительное улучшение
        improvements = []
        best_so_far = func_vals[0]
        
        for val in func_vals[1:]:
            if val < best_so_far:
                improvement = (best_so_far - val) / abs(best_so_far) if best_so_far != 0 else 0
                improvements.append(improvement)
                best_so_far = val
            else:
                improvements.append(0.0)
        
        return np.mean(improvements) if improvements else 0.0


class OptunaBayesianOptimizer(BaseOptimizer):
    """Оптимизатор на основе Optuna"""
    
    def __init__(self, method: OptimizationMethod = OptimizationMethod.OPTUNA_TPE):
        self.method = method
        self.study = None
        
    def optimize(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        n_calls: int = 100,
        random_state: int = 42
    ) -> OptimizationResult:
        """Оптимизация с Optuna"""
        start_time = time.time()
        
        logger.info(f"🔥 Запуск оптимизации Optuna методом {self.method.value}")
        
        try:
            # Создание исследования
            if self.method == OptimizationMethod.OPTUNA_TPE:
                sampler = optuna.samplers.TPESampler(seed=random_state)
            else:  # OPTUNA_RANDOM
                sampler = optuna.samplers.RandomSampler(seed=random_state)
            
            self.study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                study_name=f"automl_optimization_{int(time.time())}"
            )
            
            # Определение целевой функции для Optuna
            def optuna_objective(trial):
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'real':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'integer':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['categories']
                        )
                
                return objective_function(params)
            
            # Запуск оптимизации
            self.study.optimize(optuna_objective, n_trials=n_calls, show_progress_bar=True)
            
            # Сбор истории оптимизации
            optimization_history = []
            func_vals = []
            
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    optimization_history.append({
                        'params': trial.params.copy(),
                        'score': trial.value,
                        'trial_number': trial.number,
                        'duration': trial.duration.total_seconds() if trial.duration else 0
                    })
                    func_vals.append(trial.value)
            
            optimization_time = time.time() - start_time
            
            optimization_result = OptimizationResult(
                best_params=self.study.best_params.copy(),
                best_score=self.study.best_value,
                optimization_history=optimization_history,
                convergence_data={
                    'func_vals': func_vals,
                    'n_calls': len(self.study.trials),
                    'n_complete_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    'convergence_rate': self._calculate_optuna_convergence_rate(func_vals)
                },
                optimization_time=optimization_time,
                method_used=self.method.value,
                model_name="unknown"
            )
            
            logger.info(f"✅ Optuna оптимизация завершена: лучший скор {self.study.best_value:.4f}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка Optuna оптимизации: {e}")
            return OptimizationResult(
                best_params={},
                best_score=float('inf'),
                optimization_history=[],
                convergence_data={},
                optimization_time=time.time() - start_time,
                method_used=f"{self.method.value}_failed",
                model_name="unknown"
            )
    
    def _calculate_optuna_convergence_rate(self, func_vals: List[float]) -> float:
        """Вычисление скорости сходимости для Optuna"""
        if len(func_vals) < 2:
            return 0.0
        
        improvements = []
        best_so_far = func_vals[0]
        
        for val in func_vals[1:]:
            if val < best_so_far:
                improvement = (best_so_far - val) / abs(best_so_far) if best_so_far != 0 else 0
                improvements.append(improvement)
                best_so_far = val
            else:
                improvements.append(0.0)
        
        return np.mean(improvements) if improvements else 0.0


class CryptoMLHyperparameterOptimizer:
    """
    Главный класс для оптимизации гиперпараметров в криптотрейдинге
    Реализует enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.optimizers: Dict[str, BaseOptimizer] = {}
        self.model_search_spaces = {}
        self._setup_optimizers()
        self._setup_search_spaces()
        
    def _setup_optimizers(self):
        """Настройка оптимизаторов"""
        logger.info("🔧 Настройка оптимизаторов...")
        
        # Scikit-optimize оптимизаторы
        self.optimizers['gaussian_process'] = SkoptBayesianOptimizer(
            OptimizationMethod.GAUSSIAN_PROCESS
        )
        self.optimizers['random_forest'] = SkoptBayesianOptimizer(
            OptimizationMethod.RANDOM_FOREST
        )
        self.optimizers['gradient_boosting'] = SkoptBayesianOptimizer(
            OptimizationMethod.GRADIENT_BOOSTING
        )
        
        # Optuna оптимизаторы
        self.optimizers['optuna_tpe'] = OptunaBayesianOptimizer(
            OptimizationMethod.OPTUNA_TPE
        )
        self.optimizers['optuna_random'] = OptunaBayesianOptimizer(
            OptimizationMethod.OPTUNA_RANDOM
        )
        
        logger.info(f"✅ Настроено {len(self.optimizers)} оптимизаторов")
    
    def _setup_search_spaces(self):
        """Настройка пространств поиска для различных моделей"""
        logger.info("🌐 Настройка пространств поиска...")
        
        # Random Forest
        self.model_search_spaces['random_forest'] = {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 500},
            'max_depth': {'type': 'integer', 'low': 3, 'high': 30},
            'min_samples_split': {'type': 'integer', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'integer', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'categories': ['auto', 'sqrt', 'log2']},
        }
        
        # XGBoost
        self.model_search_spaces['xgboost'] = {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 500},
            'max_depth': {'type': 'integer', 'low': 3, 'high': 15},
            'learning_rate': {'type': 'real', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'real', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'real', 'low': 0.5, 'high': 1.0},
            'reg_alpha': {'type': 'real', 'low': 0.0, 'high': 10.0},
            'reg_lambda': {'type': 'real', 'low': 0.0, 'high': 10.0},
        }
        
        # LightGBM
        self.model_search_spaces['lightgbm'] = {
            'n_estimators': {'type': 'integer', 'low': 50, 'high': 500},
            'max_depth': {'type': 'integer', 'low': 3, 'high': 15},
            'learning_rate': {'type': 'real', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'real', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'real', 'low': 0.5, 'high': 1.0},
            'reg_alpha': {'type': 'real', 'low': 0.0, 'high': 10.0},
            'reg_lambda': {'type': 'real', 'low': 0.0, 'high': 10.0},
            'num_leaves': {'type': 'integer', 'low': 20, 'high': 200},
        }
        
        # Ridge Regression
        self.model_search_spaces['ridge'] = {
            'alpha': {'type': 'real', 'low': 0.001, 'high': 100.0, 'log': True},
        }
        
        # Lasso Regression
        self.model_search_spaces['lasso'] = {
            'alpha': {'type': 'real', 'low': 0.001, 'high': 100.0, 'log': True},
        }
        
        # ElasticNet
        self.model_search_spaces['elasticnet'] = {
            'alpha': {'type': 'real', 'low': 0.001, 'high': 100.0, 'log': True},
            'l1_ratio': {'type': 'real', 'low': 0.0, 'high': 1.0},
        }
        
        # SVR
        self.model_search_spaces['svr'] = {
            'C': {'type': 'real', 'low': 0.1, 'high': 1000.0, 'log': True},
            'gamma': {'type': 'categorical', 'categories': ['scale', 'auto']},
            'epsilon': {'type': 'real', 'low': 0.01, 'high': 1.0},
        }
        
        logger.info(f"✅ Настроено пространство поиска для {len(self.model_search_spaces)} моделей")
    
    def _get_model(self, model_name: str, params: Dict[str, Any]):
        """Создание модели с заданными параметрами"""
        if model_name == 'random_forest':
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        elif model_name == 'xgboost':
            return xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
        elif model_name == 'lightgbm':
            return lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        elif model_name == 'ridge':
            return Ridge(**params)
        elif model_name == 'lasso':
            return Lasso(**params, max_iter=2000)
        elif model_name == 'elasticnet':
            return ElasticNet(**params, max_iter=2000)
        elif model_name == 'svr':
            return SVR(**params)
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")
    
    def optimize_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        optimizer_method: str = 'optuna_tpe',
        n_calls: int = 100,
        cv_folds: int = 5,
        scoring: str = 'neg_mean_squared_error',
        time_series_split: bool = True
    ) -> OptimizationResult:
        """
        Оптимизация гиперпараметров для конкретной модели
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            model_name: Название модели для оптимизации
            optimizer_method: Метод оптимизации
            n_calls: Количество итераций оптимизации
            cv_folds: Количество фолдов для кросс-валидации
            scoring: Метрика для оптимизации
            time_series_split: Использовать TimeSeriesSplit
        """
        logger.info(f"🎯 Запуск оптимизации модели {model_name}")
        
        if model_name not in self.model_search_spaces:
            raise ValueError(f"Модель {model_name} не поддерживается")
        
        if optimizer_method not in self.optimizers:
            raise ValueError(f"Оптимизатор {optimizer_method} не найден")
        
        # Настройка кросс-валидации
        if time_series_split:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = cv_folds
        
        # Целевая функция для оптимизации
        def objective_function(params: Dict[str, Any]) -> float:
            try:
                # Создание модели с параметрами
                model = self._get_model(model_name, params)
                
                # Кросс-валидация
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                # Возвращаем отрицательное значение для минимизации
                return -np.mean(scores)
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка в целевой функции: {e}")
                return float('inf')  # Плохой скор для неудачных параметров
        
        # Запуск оптимизации
        search_space = self.model_search_spaces[model_name]
        optimizer = self.optimizers[optimizer_method]
        
        result = optimizer.optimize(
            objective_function=objective_function,
            search_space=search_space,
            n_calls=n_calls
        )
        
        # Добавление информации о модели
        result.model_name = model_name
        
        logger.info(f"✅ Оптимизация {model_name} завершена: лучший скор {-result.best_score:.4f}")
        
        return result
    
    def optimize_multiple_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: List[str],
        optimizer_method: str = 'optuna_tpe',
        n_calls: int = 50,
        parallel: bool = False
    ) -> Dict[str, OptimizationResult]:
        """Оптимизация нескольких моделей"""
        logger.info(f"🚀 Оптимизация {len(models)} моделей...")
        
        results = {}
        
        if parallel:
            # Параллельная оптимизация (может потребовать много памяти)
            from joblib import Parallel, delayed
            
            def optimize_single_model(model_name):
                return model_name, self.optimize_model(
                    X, y, model_name, optimizer_method, n_calls
                )
            
            parallel_results = Parallel(n_jobs=-1, verbose=1)(
                delayed(optimize_single_model)(model) for model in models
            )
            
            for model_name, result in parallel_results:
                results[model_name] = result
        else:
            # Последовательная оптимизация
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
            ) as progress:
                task = progress.add_task("Оптимизация моделей...", total=len(models))
                
                for model_name in models:
                    progress.update(task, description=f"Модель: {model_name}")
                    
                    try:
                        result = self.optimize_model(
                            X, y, model_name, optimizer_method, n_calls
                        )
                        results[model_name] = result
                    except Exception as e:
                        logger.error(f"❌ Ошибка оптимизации {model_name}: {e}")
                    
                    progress.advance(task)
        
        logger.info(f"✅ Завершена оптимизация {len(results)} моделей")
        
        return results
    
    def plot_optimization_history(
        self,
        result: OptimizationResult,
        save_path: Optional[str] = None
    ):
        """Визуализация истории оптимизации"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # График сходимости
            if 'func_vals' in result.convergence_data:
                func_vals = result.convergence_data['func_vals']
                axes[0, 0].plot(func_vals)
                axes[0, 0].set_title('Сходимость оптимизации')
                axes[0, 0].set_xlabel('Итерация')
                axes[0, 0].set_ylabel('Значение целевой функции')
                axes[0, 0].grid(True)
            
            # Распределение скоров
            if result.optimization_history:
                scores = [h['score'] for h in result.optimization_history]
                axes[0, 1].hist(scores, bins=20, alpha=0.7)
                axes[0, 1].set_title('Распределение скоров')
                axes[0, 1].set_xlabel('Скор')
                axes[0, 1].set_ylabel('Частота')
                axes[0, 1].grid(True)
            
            # Улучшения со временем
            if result.optimization_history:
                scores = [h['score'] for h in result.optimization_history]
                best_scores = []
                best_so_far = float('inf')
                
                for score in scores:
                    if score < best_so_far:
                        best_so_far = score
                    best_scores.append(best_so_far)
                
                axes[1, 0].plot(best_scores)
                axes[1, 0].set_title('Лучший скор со временем')
                axes[1, 0].set_xlabel('Итерация')
                axes[1, 0].set_ylabel('Лучший скор')
                axes[1, 0].grid(True)
            
            # Статистика оптимизации
            stats_text = f"""
            Модель: {result.model_name}
            Метод: {result.method_used}
            Время: {result.optimization_time:.2f}с
            Лучший скор: {result.best_score:.4f}
            Итераций: {len(result.optimization_history)}
            """
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Статистика оптимизации')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 График оптимизации сохранен: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Ошибка создания графика оптимизации: {e}")
    
    def get_optimization_report(self, results: Dict[str, OptimizationResult]) -> str:
        """Создание отчета по оптимизации"""
        report = "=== ОТЧЕТ ПО ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ ===\n\n"
        
        # Сортировка моделей по лучшему скору
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].best_score
        )
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            report += f"{i}. {model_name.upper()}\n"
            report += f"   Лучший скор: {result.best_score:.4f}\n"
            report += f"   Время оптимизации: {result.optimization_time:.2f}с\n"
            report += f"   Метод: {result.method_used}\n"
            report += f"   Лучшие параметры:\n"
            
            for param, value in result.best_params.items():
                report += f"     {param}: {value}\n"
            
            report += "\n"
        
        # Общая статистика
        total_time = sum(r.optimization_time for r in results.values())
        best_overall = min(results.values(), key=lambda x: x.best_score)
        
        report += f"Общее время оптимизации: {total_time:.2f}с\n"
        report += f"Лучшая модель: {best_overall.model_name} (скор: {best_overall.best_score:.4f})\n"
        
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
    
    # Целевая переменная с некоторой нелинейностью
    y = pd.Series(
        X.iloc[:, :5].sum(axis=1) + 
        0.5 * X['feature_0'] * X['feature_1'] + 
        0.1 * np.random.randn(n_samples)
    )
    
    # Создание оптимизатора
    config = AutoMLConfig()
    optimizer = CryptoMLHyperparameterOptimizer(config)
    
    # Оптимизация одной модели
    result = optimizer.optimize_model(
        X, y, 
        model_name='xgboost',
        optimizer_method='optuna_tpe',
        n_calls=20  # Мало итераций для примера
    )
    
    print("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===")
    print(f"Модель: {result.model_name}")
    print(f"Лучший скор: {result.best_score:.4f}")
    print(f"Время оптимизации: {result.optimization_time:.2f}с")
    print(f"Лучшие параметры: {result.best_params}")
    
    # Оптимизация нескольких моделей
    models = ['random_forest', 'xgboost']
    results = optimizer.optimize_multiple_models(X, y, models, n_calls=10)
    
    print("\n" + optimizer.get_optimization_report(results))