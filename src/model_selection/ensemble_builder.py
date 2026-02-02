"""
Advanced Ensemble Builder for Crypto Trading AutoML
Implements enterprise patterns for robust ensemble construction
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
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from ..utils.config_manager import AutoMLConfig


class EnsembleMethod(Enum):
    """Methods ансамблирования"""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    BAGGING = "bagging"
    DYNAMIC_WEIGHTING = "dynamic_weighting"


@dataclass
class EnsembleResult:
    """Result построения ансамбля"""
    ensembles: Dict[str, Any]
    ensemble_scores: Dict[str, float]
    best_ensemble_method: str
    best_ensemble_score: float
    base_model_scores: Dict[str, float]
    ensemble_weights: Dict[str, Dict[str, float]]
    ensemble_metadata: Dict[str, Any]
    build_time: float


class BaseEnsembleBuilder(ABC):
    """Base класс for построителей ensembles -  pattern"""
    
    @abstractmethod
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Построить ensemble"""
        pass


class VotingEnsembleBuilder(BaseEnsembleBuilder):
    """Строитель голосующего ансамбля"""
    
    def __init__(self, voting_type: str = 'soft', weights: Optional[List[float]] = None):
        self.voting_type = voting_type
        self.weights = weights
        
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str = 'regression',
        **kwargs
    ) -> Any:
        """Построить голосующий ensemble"""
        logger.info(f"🗳️ Build голосующего ансамбля ({self.voting_type})")
        
        # Подготовка models for ансамбля
        estimators = [(name, model) for name, model in models.items()]
        
        try:
            if task_type == 'regression':
                ensemble = VotingRegressor(
                    estimators=estimators,
                    weights=self.weights
                )
            else:
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=self.voting_type,
                    weights=self.weights
                )
            
            # Training ансамбля
            ensemble.fit(X, y)
            
            logger.info(f"✅ Голосующий ensemble построен with {len(models)} моделями")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Error построения голосующего ансамбля: {e}")
            return None


class StackingEnsembleBuilder(BaseEnsembleBuilder):
    """Строитель стекинг ансамбля"""
    
    def __init__(
        self,
        meta_learner: Optional[Any] = None,
        cv_folds: int = 5,
        use_features_in_secondary: bool = True
    ):
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.use_features_in_secondary = use_features_in_secondary
        
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str = 'regression',
        **kwargs
    ) -> Any:
        """Построить стекинг ensemble"""
        logger.info("🥞 Build стекинг ансамбля")
        
        try:
            # Configure meta-learning алгоритма by default
            if self.meta_learner is None:
                if task_type == 'regression':
                    self.meta_learner = Ridge(alpha=1.0)
                else:
                    self.meta_learner = LogisticRegression(max_iter=1000)
            
            # Create стекинг ансамбля
            from sklearn.ensemble import StackingRegressor, StackingClassifier
            
            estimators = [(name, model) for name, model in models.items()]
            
            if task_type == 'regression':
                ensemble = StackingRegressor(
                    estimators=estimators,
                    final_estimator=self.meta_learner,
                    cv=self.cv_folds,
                    passthrough=self.use_features_in_secondary,
                    n_jobs=-1
                )
            else:
                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=self.meta_learner,
                    cv=self.cv_folds,
                    passthrough=self.use_features_in_secondary,
                    n_jobs=-1
                )
            
            # Training ансамбля
            ensemble.fit(X, y)
            
            logger.info(f"✅ Стекинг ensemble построен with {len(models)} базовыми моделями")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Error построения стекинг ансамбля: {e}")
            return None


class BlendingEnsembleBuilder(BaseEnsembleBuilder):
    """Строитель блендинг ансамбля"""
    
    def __init__(
        self,
        holdout_size: float = 0.2,
        meta_learner: Optional[Any] = None
    ):
        self.holdout_size = holdout_size
        self.meta_learner = meta_learner
        self.base_models = None
        self.blending_predictions = None
        
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str = 'regression',
        **kwargs
    ) -> Any:
        """Построить блендинг ensemble"""
        logger.info("🔀 Build блендинг ансамбля")
        
        try:
            from sklearn.model_selection import train_test_split
            
            # Split on training base models and блендинг
            X_base, X_blend, y_base, y_blend = train_test_split(
                X, y,
                test_size=self.holdout_size,
                random_state=42
            )
            
            # Training base models
            trained_models = {}
            blend_predictions = []
            
            for name, model in models.items():
                # Create копии model for независимого training
                if hasattr(model, 'copy'):
                    trained_model = model.copy()
                else:
                    from sklearn.base import clone
                    trained_model = clone(model)
                
                # Training on базовом наборе
                trained_model.fit(X_base, y_base)
                trained_models[name] = trained_model
                
                # Predictions on блендинг наборе
                predictions = trained_model.predict(X_blend)
                blend_predictions.append(predictions)
            
            # Подготовка data for meta-learning
            blend_features = np.column_stack(blend_predictions)
            
            # Configure meta-algorithm by default
            if self.meta_learner is None:
                if task_type == 'regression':
                    self.meta_learner = Ridge(alpha=1.0)
                else:
                    self.meta_learner = LogisticRegression(max_iter=1000)
            
            # Training meta-algorithm
            self.meta_learner.fit(blend_features, y_blend)
            
            # Create финального ансамбля
            ensemble = BlendingEnsemble(
                base_models=trained_models,
                meta_learner=self.meta_learner
            )
            
            logger.info(f"✅ Блендинг ensemble построен with {len(models)} базовыми моделями")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Error построения блендинг ансамбля: {e}")
            return None


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """Кастомный блендинг ensemble"""
    
    def __init__(self, base_models: Dict[str, Any], meta_learner: Any):
        self.base_models = base_models
        self.meta_learner = meta_learner
        
    def fit(self, X, y):
        # Model already обучены in BlendingEnsembleBuilder
        return self
        
    def predict(self, X):
        # Get predictions from base models
        base_predictions = []
        for name, model in self.base_models.items():
            predictions = model.predict(X)
            base_predictions.append(predictions)
        
        # Стекинг predictions
        stacked_predictions = np.column_stack(base_predictions)
        
        # Финальное prediction meta-algorithm
        final_predictions = self.meta_learner.predict(stacked_predictions)
        
        return final_predictions


class DynamicWeightingEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble with динамическими weights"""
    
    def __init__(self, models: Dict[str, Any], window_size: int = 100):
        self.models = models
        self.window_size = window_size
        self.weights_history = []
        self.performance_history = {name: [] for name in models.keys()}
        
    def fit(self, X, y):
        # Training всех base models
        for name, model in self.models.items():
            model.fit(X, y)
        
        return self
        
    def predict(self, X):
        # Get predictions from всех models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # If no истории, use equal weights
        if not self.weights_history:
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        else:
            weights = self._calculate_dynamic_weights()
        
        # Взвешенное усреднение predictions
        final_predictions = np.zeros(len(X))
        for name, weight in weights.items():
            final_predictions += weight * predictions[name]
        
        return final_predictions
    
    def _calculate_dynamic_weights(self):
        """Computation динамических weights on основе недавней performance"""
        # Упрощенная implementation - equal weights
        return {name: 1.0 / len(self.models) for name in self.models.keys()}


class EnsembleBuilder:
    """
    Главный класс for построения ensembles
    Implements enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.ensemble_config = self.config.ensemble
        self.console = Console()
        
        # Строители ensembles
        self.ensemble_builders: Dict[str, BaseEnsembleBuilder] = {}
        self._setup_builders()
        
    def _setup_builders(self):
        """Configure строителей ensembles"""
        logger.info("🔧 Configure строителей ensembles...")
        
        if self.ensemble_config.enable_voting:
            self.ensemble_builders['voting'] = VotingEnsembleBuilder(
                voting_type='soft',
                weights=self.ensemble_config.voting_weights
            )
        
        if self.ensemble_config.enable_stacking:
            self.ensemble_builders['stacking'] = StackingEnsembleBuilder(
                cv_folds=self.ensemble_config.stacking_cv_folds,
                use_features_in_secondary=self.ensemble_config.stacking_use_features_in_secondary
            )
        
        if self.ensemble_config.enable_blending:
            self.ensemble_builders['blending'] = BlendingEnsembleBuilder(
                holdout_size=self.ensemble_config.blending_holdout_size
            )
        
        logger.info(f"✅ Настроено {len(self.ensemble_builders)} строителей ensembles")
    
    def build_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        ensemble_methods: Optional[List[str]] = None,
        task_type: str = 'regression'
    ) -> EnsembleResult:
        """
        Main method построения ensembles
        
        Args:
            X: Matrix features
            y: Target variable
            models: Dictionary base models
            ensemble_methods: Methods ансамблирования for use
            task_type: Тип tasks (regression/classification)
        """
        start_time = time.time()
        
        logger.info(f"🤝 Launch построения ensembles with {len(models)} базовыми моделями")
        
        if ensemble_methods is None:
            ensemble_methods = list(self.ensemble_builders.keys())
        
        # Ограничение number models in ансамбле
        if len(models) > self.ensemble_config.ensemble_size_limit:
            # Sort models by performance and selection best
            sorted_models = self._rank_models_by_performance(X, y, models, task_type)
            models = dict(list(sorted_models.items())[:self.ensemble_config.ensemble_size_limit])
            logger.info(f"📝 Ограничены up to {len(models)} best models for ансамбля")
        
        ensembles = {}
        ensemble_scores = {}
        ensemble_weights = {}
        base_model_scores = {}
        
        # Evaluate base models
        base_model_scores = self._evaluate_base_models(X, y, models, task_type)
        
        # Build ensembles
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        ) as progress:
            
            task = progress.add_task("Build ensembles...", total=len(ensemble_methods))
            
            for method in ensemble_methods:
                progress.update(task, description=f"Method: {method}")
                
                if method not in self.ensemble_builders:
                    logger.warning(f"⚠️ Неизвестный method ансамблирования: {method}")
                    continue
                
                try:
                    # Build ансамбля
                    ensemble = self.ensemble_builders[method].build(
                        X, y, models, task_type=task_type
                    )
                    
                    if ensemble is not None:
                        ensembles[method] = ensemble
                        
                        # Evaluate ансамбля
                        score = self._evaluate_ensemble(X, y, ensemble, task_type)
                        ensemble_scores[method] = score
                        
                        # Get weights (if применимо)
                        weights = self._extract_ensemble_weights(ensemble, method)
                        if weights:
                            ensemble_weights[method] = weights
                        
                        logger.info(f"✅ {method} ensemble: score {score:.4f}")
                
                except Exception as e:
                    logger.error(f"❌ Error построения {method} ансамбля: {e}")
                
                progress.advance(task)
        
        # Determine лучшего ансамбля
        if ensemble_scores:
            best_method = max(ensemble_scores.keys(), key=lambda k: ensemble_scores[k])
            best_score = ensemble_scores[best_method]
        else:
            best_method = "none"
            best_score = 0.0
            logger.warning("⚠️ Ни one ensemble not был успешно построен")
        
        build_time = time.time() - start_time
        
        result = EnsembleResult(
            ensembles=ensembles,
            ensemble_scores=ensemble_scores,
            best_ensemble_method=best_method,
            best_ensemble_score=best_score,
            base_model_scores=base_model_scores,
            ensemble_weights=ensemble_weights,
            ensemble_metadata={
                'task_type': task_type,
                'base_models_count': len(models),
                'ensemble_methods_tried': len(ensemble_methods),
                'successful_ensembles': len(ensembles)
            },
            build_time=build_time
        )
        
        # Вывод results
        self._print_ensemble_results(result)
        
        logger.info(f"✅ Build ensembles завершено for {build_time:.2f}with")
        
        return result
    
    def _rank_models_by_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """Ranking models by performance"""
        logger.info("📊 Ranking models by performance...")
        
        model_scores = {}
        
        for name, model in models.items():
            try:
                # Fast evaluation with 3-fold CV
                if task_type == 'regression':
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=-1)
                else:
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
                
                model_scores[name] = np.mean(scores)
                
            except Exception as e:
                logger.warning(f"⚠️ Error оценки model {name}: {e}")
                model_scores[name] = 0.0
        
        # Sort by descending score
        ranked_models = dict(
            sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {name: models[name] for name in ranked_models.keys()}
    
    def _evaluate_base_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str
    ) -> Dict[str, float]:
        """Evaluate base models"""
        logger.info("📏 Evaluate base models...")
        
        base_scores = {}
        
        for name, model in models.items():
            try:
                if task_type == 'regression':
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=-1)
                else:
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
                
                base_scores[name] = np.mean(scores)
                
            except Exception as e:
                logger.warning(f"⚠️ Error оценки base model {name}: {e}")
                base_scores[name] = 0.0
        
        return base_scores
    
    def _evaluate_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ensemble: Any,
        task_type: str
    ) -> float:
        """Evaluate ансамбля"""
        try:
            if task_type == 'regression':
                scores = cross_val_score(ensemble, X, y, cv=3, scoring='r2', n_jobs=-1)
            else:
                scores = cross_val_score(ensemble, X, y, cv=3, scoring='accuracy', n_jobs=-1)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"❌ Error оценки ансамбля: {e}")
            return 0.0
    
    def _extract_ensemble_weights(self, ensemble: Any, method: str) -> Optional[Dict[str, float]]:
        """Извлечение weights ансамбля"""
        try:
            if method == 'voting' and hasattr(ensemble, 'estimators_'):
                if hasattr(ensemble, 'weights') and ensemble.weights is not None:
                    estimator_names = [name for name, _ in ensemble.estimators]
                    return dict(zip(estimator_names, ensemble.weights))
            
            elif method == 'stacking' and hasattr(ensemble, 'final_estimator_'):
                if hasattr(ensemble.final_estimator_, 'coef_'):
                    estimator_names = [name for name, _ in ensemble.estimators]
                    weights = ensemble.final_estimator_.coef_
                    if len(weights) >= len(estimator_names):
                        return dict(zip(estimator_names, weights[:len(estimator_names)]))
            
            return None
            
        except Exception as e:
            logger.debug(f"Not удалось извлечь weights for {method}: {e}")
            return None
    
    def _print_ensemble_results(self, result: EnsembleResult):
        """Вывод results ансамблирования"""
        
        # Таблица with результатами ensembles
        table = Table(title="🤝 Results АНСАМБЛИРОВАНИЯ")
        
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Score", style="green")
        table.add_column("Улучшение", style="magenta")
        
        # Best base score for сравнения
        best_base_score = max(result.base_model_scores.values()) if result.base_model_scores else 0.0
        
        for method, score in sorted(result.ensemble_scores.items(), key=lambda x: x[1], reverse=True):
            improvement = ((score - best_base_score) / best_base_score * 100) if best_base_score > 0 else 0.0
            table.add_row(
                method,
                f"{score:.4f}",
                f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            )
        
        self.console.print(table)
        
        # Информация о лучшем ансамбле
        if result.best_ensemble_method != "none":
            best_info = f"""
🏆 Best ensemble: {result.best_ensemble_method}
📊 Score: {result.best_ensemble_score:.4f}
⏱️ Время построения: {result.build_time:.2f}with
🔢 Base models: {result.ensemble_metadata['base_models_count']}
"""
            self.console.print(best_info)
    
    def plot_ensemble_comparison(
        self,
        result: EnsembleResult,
        save_path: Optional[str] = None
    ):
        """Visualization сравнения ensembles"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # График 1: Сравнение scores
            all_scores = {**result.base_model_scores, **result.ensemble_scores}
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            
            methods, scores = zip(*sorted_scores)
            colors = ['red' if method in result.ensemble_scores else 'blue' for method in methods]
            
            axes[0].barh(methods, scores, color=colors, alpha=0.7)
            axes[0].set_xlabel('Score')
            axes[0].set_title('Сравнение base models and ensembles')
            axes[0].grid(True, alpha=0.3)
            
            # Легенда
            axes[0].axvline(x=0, color='blue', alpha=0.7, label='Base model')
            axes[0].axvline(x=0, color='red', alpha=0.7, label='Ensembles')
            axes[0].legend()
            
            # График 2: Улучшения from ансамблирования
            if result.ensemble_scores and result.base_model_scores:
                best_base_score = max(result.base_model_scores.values())
                
                improvements = {}
                for method, score in result.ensemble_scores.items():
                    improvement = ((score - best_base_score) / best_base_score * 100) if best_base_score > 0 else 0.0
                    improvements[method] = improvement
                
                if improvements:
                    methods, improve_values = zip(*improvements.items())
                    colors = ['green' if imp > 0 else 'orange' for imp in improve_values]
                    
                    axes[1].bar(methods, improve_values, color=colors, alpha=0.7)
                    axes[1].set_ylabel('Улучшение (%)')
                    axes[1].set_title('Улучшение from ансамблирования')
                    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 График ensembles сохранен: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Error creation графика ensembles: {e}")
    
    def get_ensemble_report(self, result: EnsembleResult) -> str:
        """Create отчета by ансамблированию"""
        
        report = f"""
=== ОТЧЕТ By АНСАМБЛИРОВАНИЮ ===

Base models: {len(result.base_model_scores)}
Methods ансамблирования: {len(result.ensemble_scores)}
Время построения: {result.build_time:.2f}with

Best ensemble: {result.best_ensemble_method}
Best score: {result.best_ensemble_score:.4f}

Results ensembles:
"""
        
        for method, score in sorted(result.ensemble_scores.items(), key=lambda x: x[1], reverse=True):
            report += f"  {method}: {score:.4f}\n"
        
        # Weights ensembles
        if result.ensemble_weights:
            report += "\nWeights in ансамблях:\n"
            for method, weights in result.ensemble_weights.items():
                report += f"  {method}:\n"
                for model, weight in weights.items():
                    report += f"    {model}: {weight:.3f}\n"
        
        report += f"\nМетаданные: {result.ensemble_metadata}"
        
        return report


if __name__ == "__main__":
    # Пример use EnsembleBuilder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import xgboost as xgb
    
    # Create test data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y = pd.Series(
        X.iloc[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)
    )
    
    # Create base models
    models = {
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    # Create строителя ensembles
    config = AutoMLConfig()
    builder = EnsembleBuilder(config)
    
    # Build ensembles
    result = builder.build_ensemble(
        X, y, models,
        ensemble_methods=['voting', 'stacking'],
        task_type='regression'
    )
    
    print("=== Results АНСАМБЛИРОВАНИЯ ===")
    print(f"Best ensemble: {result.best_ensemble_method}")
    print(f"Best score: {result.best_ensemble_score:.4f}")
    print(f"Время построения: {result.build_time:.2f}with")
    
    # Отчет
    print(builder.get_ensemble_report(result))