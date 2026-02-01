"""
Main AutoML Pipeline for Crypto Trading Bot v5.0
Orchestrates the complete machine learning workflow with Context7 enterprise patterns
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import pickle
import json
import time
from pathlib import Path
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import seaborn as sns

from ..feature_engineering.auto_feature_generator import AutoFeatureGenerator, FeatureGenerationResult
from ..feature_engineering.feature_selector import AdvancedFeatureSelector, FeatureSelectionResult
from ..optimization.bayesian_optimizer import CryptoMLHyperparameterOptimizer, OptimizationResult
from ..model_selection.model_selector import ModelSelector, ModelSelectionResult
from ..model_selection.ensemble_builder import EnsembleBuilder, EnsembleResult
from ..evaluation.model_evaluator import ModelEvaluator, EvaluationResult
from ..utils.config_manager import AutoMLConfig
from ..utils.data_preprocessor import DataPreprocessor


class PipelineStage(Enum):
    """Этапы AutoML пайплайна"""
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_GENERATION = "feature_generation"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE_BUILDING = "ensemble_building"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"


@dataclass
class PipelineResult:
    """Результат выполнения AutoML пайплайна"""
    best_model: Any
    best_model_name: str
    best_score: float
    feature_generation_result: Optional[FeatureGenerationResult]
    feature_selection_result: Optional[FeatureSelectionResult]
    optimization_results: Dict[str, OptimizationResult]
    ensemble_result: Optional[EnsembleResult]
    evaluation_result: EvaluationResult
    pipeline_metadata: Dict[str, Any]
    total_time: float
    stages_completed: List[str]


class AutoMLPipeline:
    """
    Главный класс AutoML пайплайна для криптотрейдинга
    Реализует Context7 enterprise patterns для scalable ML systems
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None, output_dir: Optional[str] = None):
        self.config = config or AutoMLConfig()
        self.output_dir = Path(output_dir or "automl_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализация компонентов
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_generator = AutoFeatureGenerator(self.config)
        self.feature_selector = AdvancedFeatureSelector(self.config)
        self.hyperparameter_optimizer = CryptoMLHyperparameterOptimizer(self.config)
        self.model_selector = ModelSelector(self.config)
        self.ensemble_builder = EnsembleBuilder(self.config)
        self.evaluator = ModelEvaluator(self.config)
        
        # Состояние пайплайна
        self.pipeline_state = {}
        self.console = Console()
        
        logger.info("🚀 AutoML Pipeline инициализирован")
    
    def run(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        time_series_split: bool = True,
        stages: Optional[List[str]] = None
    ) -> PipelineResult:
        """
        Запуск полного AutoML пайплайна
        
        Args:
            data: Исходные данные
            target_column: Название колонки с целевой переменной
            test_size: Размер тестовой выборки
            validation_size: Размер валидационной выборки
            time_series_split: Использовать временные разбиения
            stages: Список этапов для выполнения (по умолчанию все)
        """
        start_time = time.time()
        
        self.console.print(
            Panel.fit(
                "🤖 [bold blue]CRYPTO TRADING AUTOML PIPELINE v5.0[/bold blue] 🚀\n"
                f"📊 Данных: {len(data)} записей, {len(data.columns)} признаков\n"
                f"🎯 Целевая переменная: {target_column}",
                title="Запуск AutoML Pipeline"
            )
        )
        
        if stages is None:
            stages = [stage.value for stage in PipelineStage]
        
        stages_completed = []
        pipeline_metadata = {
            'start_time': start_time,
            'data_shape': data.shape,
            'target_column': target_column,
            'config': self.config.dict() if hasattr(self.config, 'dict') else str(self.config)
        }
        
        try:
            # === ЭТАП 1: ПРЕДОБРАБОТКА ДАННЫХ ===
            if PipelineStage.DATA_PREPROCESSING.value in stages:
                logger.info("🔧 Этап 1: Предобработка данных")
                
                X, y = self._preprocess_data(data, target_column)
                X_train, X_test, y_train, y_test = self._split_data(
                    X, y, test_size, validation_size, time_series_split
                )
                
                stages_completed.append(PipelineStage.DATA_PREPROCESSING.value)
                
                self.console.print("✅ [green]Предобработка данных завершена[/green]")
            else:
                logger.info("⏭️ Пропуск предобработки данных")
                X = data.drop(columns=[target_column])
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            # === ЭТАП 2: ГЕНЕРАЦИЯ ПРИЗНАКОВ ===
            feature_generation_result = None
            if PipelineStage.FEATURE_GENERATION.value in stages:
                logger.info("🎨 Этап 2: Генерация признаков")
                
                feature_generation_result = self._generate_features(X_train)
                
                if feature_generation_result and not feature_generation_result.features.empty:
                    # Применение сгенерированных признаков к обучающей выборке
                    X_train_enhanced = pd.concat([X_train, feature_generation_result.features], axis=1)
                    
                    # Применение к тестовой выборке
                    test_features = self.feature_generator.generate_features(X_test)
                    X_test_enhanced = pd.concat([X_test, test_features.features], axis=1)
                    
                    X_train = X_train_enhanced
                    X_test = X_test_enhanced
                
                stages_completed.append(PipelineStage.FEATURE_GENERATION.value)
                self.console.print("✅ [green]Генерация признаков завершена[/green]")
            
            # === ЭТАП 3: ОТБОР ПРИЗНАКОВ ===
            feature_selection_result = None
            if PipelineStage.FEATURE_SELECTION.value in stages:
                logger.info("🎯 Этап 3: Отбор признаков")
                
                feature_selection_result = self._select_features(X_train, y_train)
                
                if feature_selection_result and feature_selection_result.selected_features:
                    X_train = X_train[feature_selection_result.selected_features]
                    X_test = X_test[feature_selection_result.selected_features]
                
                stages_completed.append(PipelineStage.FEATURE_SELECTION.value)
                self.console.print("✅ [green]Отбор признаков завершен[/green]")
            
            # === ЭТАП 4: ОТБОР МОДЕЛЕЙ ===
            model_selection_result = None
            if PipelineStage.MODEL_SELECTION.value in stages:
                logger.info("🤖 Этап 4: Отбор моделей")
                
                model_selection_result = self._select_models(X_train, y_train)
                
                stages_completed.append(PipelineStage.MODEL_SELECTION.value)
                self.console.print("✅ [green]Отбор моделей завершен[/green]")
            
            # === ЭТАП 5: ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ ===
            optimization_results = {}
            if PipelineStage.HYPERPARAMETER_OPTIMIZATION.value in stages:
                logger.info("⚙️ Этап 5: Оптимизация гиперпараметров")
                
                # Определение моделей для оптимизации
                models_to_optimize = []
                if model_selection_result:
                    # Берем топ-3 модели
                    sorted_models = sorted(
                        model_selection_result.model_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    models_to_optimize = [model[0] for model in sorted_models[:3]]
                else:
                    # По умолчанию оптимизируем базовые модели
                    models_to_optimize = ['xgboost', 'random_forest', 'lightgbm']
                
                optimization_results = self._optimize_hyperparameters(
                    X_train, y_train, models_to_optimize
                )
                
                stages_completed.append(PipelineStage.HYPERPARAMETER_OPTIMIZATION.value)
                self.console.print("✅ [green]Оптимизация гиперпараметров завершена[/green]")
            
            # === ЭТАП 6: ПОСТРОЕНИЕ АНСАМБЛЯ ===
            ensemble_result = None
            if PipelineStage.ENSEMBLE_BUILDING.value in stages:
                logger.info("🤝 Этап 6: Построение ансамбля")
                
                ensemble_result = self._build_ensemble(
                    X_train, y_train, optimization_results
                )
                
                stages_completed.append(PipelineStage.ENSEMBLE_BUILDING.value)
                self.console.print("✅ [green]Построение ансамбля завершено[/green]")
            
            # === ЭТАП 7: ОЦЕНКА МОДЕЛЕЙ ===
            evaluation_result = None
            if PipelineStage.MODEL_EVALUATION.value in stages:
                logger.info("📊 Этап 7: Оценка моделей")
                
                # Определение лучшей модели
                best_model, best_model_name, best_score = self._select_best_model(
                    optimization_results, ensemble_result
                )
                
                evaluation_result = self._evaluate_models(
                    X_train, y_train, X_test, y_test,
                    best_model, best_model_name
                )
                
                stages_completed.append(PipelineStage.MODEL_EVALUATION.value)
                self.console.print("✅ [green]Оценка моделей завершена[/green]")
            
            # === СОЗДАНИЕ ИТОГОВОГО РЕЗУЛЬТАТА ===
            total_time = time.time() - start_time
            pipeline_metadata.update({
                'end_time': time.time(),
                'stages_completed': stages_completed,
                'final_features_count': X_train.shape[1] if 'X_train' in locals() else 0
            })
            
            result = PipelineResult(
                best_model=best_model if 'best_model' in locals() else None,
                best_model_name=best_model_name if 'best_model_name' in locals() else "unknown",
                best_score=best_score if 'best_score' in locals() else 0.0,
                feature_generation_result=feature_generation_result,
                feature_selection_result=feature_selection_result,
                optimization_results=optimization_results,
                ensemble_result=ensemble_result,
                evaluation_result=evaluation_result,
                pipeline_metadata=pipeline_metadata,
                total_time=total_time,
                stages_completed=stages_completed
            )
            
            # Сохранение результатов
            self._save_pipeline_results(result)
            
            # Финальный отчет
            self._print_final_report(result)
            
            logger.info(f"🎉 AutoML Pipeline завершен за {total_time:.2f}с")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка в AutoML Pipeline: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Предобработка данных"""
        logger.info("🔧 Предобработка данных...")
        
        # Разделение на признаки и целевую переменную
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Предобработка с помощью DataPreprocessor
        X_processed = self.preprocessor.preprocess(X)
        y_processed = self.preprocessor.preprocess_target(y)
        
        logger.info(f"✅ Данные обработаны: {X_processed.shape[0]} записей, {X_processed.shape[1]} признаков")
        
        return X_processed, y_processed
    
    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        validation_size: float,
        time_series_split: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Разделение данных на обучающую и тестовую выборки"""
        logger.info("✂️ Разделение данных...")
        
        if time_series_split:
            # Временное разделение (без перемешивания)
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
        else:
            # Случайное разделение
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"✅ Данные разделены: обучение={len(X_train)}, тест={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _generate_features(self, X: pd.DataFrame) -> Optional[FeatureGenerationResult]:
        """Генерация признаков"""
        logger.info("🎨 Генерация признаков...")
        
        try:
            result = self.feature_generator.generate_features(X, parallel=True)
            
            logger.info(f"✅ Сгенерировано {len(result.feature_names)} признаков")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации признаков: {e}")
            return None
    
    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Optional[FeatureSelectionResult]:
        """Отбор признаков"""
        logger.info("🎯 Отбор признаков...")
        
        try:
            result = self.feature_selector.select_features(
                X, y,
                ensemble_selection=True
            )
            
            logger.info(f"✅ Отобрано {len(result.selected_features)} признаков")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка отбора признаков: {e}")
            return None
    
    def _select_models(self, X: pd.DataFrame, y: pd.Series) -> Optional[ModelSelectionResult]:
        """Отбор моделей"""
        logger.info("🤖 Отбор моделей...")
        
        try:
            result = self.model_selector.select_best_models(
                X, y,
                models=['xgboost', 'random_forest', 'lightgbm', 'ridge', 'elasticnet'],
                cv_folds=3  # Меньше фолдов для скорости
            )
            
            logger.info(f"✅ Протестировано {len(result.model_scores)} моделей")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка отбора моделей: {e}")
            return None
    
    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: List[str]
    ) -> Dict[str, OptimizationResult]:
        """Оптимизация гиперпараметров"""
        logger.info(f"⚙️ Оптимизация {len(models)} моделей...")
        
        try:
            results = self.hyperparameter_optimizer.optimize_multiple_models(
                X, y,
                models=models,
                optimizer_method='optuna_tpe',
                n_calls=50  # Меньше итераций для скорости
            )
            
            logger.info(f"✅ Оптимизированы гиперпараметры для {len(results)} моделей")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка оптимизации: {e}")
            return {}
    
    def _build_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimization_results: Dict[str, OptimizationResult]
    ) -> Optional[EnsembleResult]:
        """Построение ансамбля"""
        logger.info("🤝 Построение ансамбля...")
        
        try:
            # Создание моделей с оптимальными параметрами
            models = {}
            for model_name, result in optimization_results.items():
                model = self.hyperparameter_optimizer._get_model(model_name, result.best_params)
                models[model_name] = model
            
            if not models:
                logger.warning("⚠️ Нет моделей для ансамбля")
                return None
            
            result = self.ensemble_builder.build_ensemble(
                X, y,
                models,
                ensemble_methods=['voting', 'stacking']
            )
            
            logger.info(f"✅ Ансамбль построен с {len(models)} моделями")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка построения ансамбля: {e}")
            return None
    
    def _select_best_model(
        self,
        optimization_results: Dict[str, OptimizationResult],
        ensemble_result: Optional[EnsembleResult]
    ) -> Tuple[Any, str, float]:
        """Выбор лучшей модели"""
        logger.info("🏆 Выбор лучшей модели...")
        
        best_model = None
        best_model_name = "unknown"
        best_score = float('inf')
        
        # Проверка оптимизированных моделей
        for model_name, result in optimization_results.items():
            if result.best_score < best_score:
                best_score = result.best_score
                best_model_name = model_name
                best_model = self.hyperparameter_optimizer._get_model(
                    model_name, result.best_params
                )
        
        # Проверка ансамбля
        if ensemble_result and ensemble_result.best_ensemble_score < best_score:
            best_score = ensemble_result.best_ensemble_score
            best_model_name = f"ensemble_{ensemble_result.best_ensemble_method}"
            best_model = ensemble_result.ensembles[ensemble_result.best_ensemble_method]
        
        logger.info(f"✅ Лучшая модель: {best_model_name} (скор: {abs(best_score):.4f})")
        
        return best_model, best_model_name, best_score
    
    def _evaluate_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        best_model: Any,
        best_model_name: str
    ) -> EvaluationResult:
        """Оценка моделей"""
        logger.info("📊 Оценка лучшей модели...")
        
        try:
            # Обучение лучшей модели
            best_model.fit(X_train, y_train)
            
            # Предсказания
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            result = self.evaluator.evaluate_model(
                best_model,
                X_train, y_train,
                X_test, y_test,
                model_name=best_model_name
            )
            
            logger.info(f"✅ Модель оценена: R² = {result.test_r2:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка оценки модели: {e}")
            # Возвращаем пустой результат
            return EvaluationResult(
                model_name=best_model_name,
                train_mse=0.0, train_mae=0.0, train_r2=0.0,
                test_mse=0.0, test_mae=0.0, test_r2=0.0,
                cross_val_scores=[], feature_importance={},
                evaluation_metadata={}, evaluation_time=0.0
            )
    
    def _save_pipeline_results(self, result: PipelineResult):
        """Сохранение результатов пайплайна"""
        logger.info("💾 Сохранение результатов...")
        
        try:
            # Сохранение лучшей модели
            if result.best_model:
                model_path = self.output_dir / "best_model.pkl"
                joblib.dump(result.best_model, model_path)
                logger.info(f"💾 Модель сохранена: {model_path}")
            
            # Сохранение метаданных
            metadata_path = self.output_dir / "pipeline_metadata.json"
            with open(metadata_path, 'w') as f:
                # Конвертируем результаты в serializable формат
                serializable_metadata = {
                    'best_model_name': result.best_model_name,
                    'best_score': result.best_score,
                    'total_time': result.total_time,
                    'stages_completed': result.stages_completed,
                    'pipeline_metadata': result.pipeline_metadata
                }
                json.dump(serializable_metadata, f, indent=2)
            
            logger.info(f"✅ Метаданные сохранены: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения результатов: {e}")
    
    def _print_final_report(self, result: PipelineResult):
        """Вывод итогового отчета"""
        
        # Создание таблицы с результатами
        table = Table(title="🎯 РЕЗУЛЬТАТЫ AUTOML PIPELINE")
        
        table.add_column("Метрика", style="cyan", no_wrap=True)
        table.add_column("Значение", style="magenta")
        
        table.add_row("🏆 Лучшая модель", result.best_model_name)
        table.add_row("📊 Лучший скор", f"{abs(result.best_score):.4f}")
        table.add_row("⏱️ Время выполнения", f"{result.total_time:.2f}с")
        table.add_row("🎯 Этапов завершено", f"{len(result.stages_completed)}/8")
        
        if result.feature_generation_result:
            table.add_row("🎨 Признаков сгенерировано", str(len(result.feature_generation_result.feature_names)))
        
        if result.feature_selection_result:
            table.add_row("🔍 Признаков отобрано", str(len(result.feature_selection_result.selected_features)))
        
        if result.optimization_results:
            table.add_row("⚙️ Моделей оптимизировано", str(len(result.optimization_results)))
        
        if result.evaluation_result:
            table.add_row("📈 R² на тесте", f"{result.evaluation_result.test_r2:.4f}")
            table.add_row("📉 MSE на тесте", f"{result.evaluation_result.test_mse:.4f}")
        
        self.console.print(table)
        
        # Детали этапов
        stages_panel = Panel(
            " → ".join(result.stages_completed),
            title="🔄 Выполненные этапы"
        )
        self.console.print(stages_panel)


if __name__ == "__main__":
    # Пример использования AutoML Pipeline
    
    # Создание тестовых данных (имитация криптовалютных данных)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    
    # Базовые OHLCV данные
    data = pd.DataFrame({
        'open': np.random.randn(2000).cumsum() + 50000,
        'high': np.random.randn(2000).cumsum() + 50100,
        'low': np.random.randn(2000).cumsum() + 49900,
        'close': np.random.randn(2000).cumsum() + 50000,
        'volume': np.random.exponential(1000, 2000)
    }, index=dates)
    
    # Целевая переменная (будущая доходность)
    data['future_return'] = data['close'].shift(-1) / data['close'] - 1
    data = data.dropna()
    
    # Создание и запуск пайплайна
    config = AutoMLConfig()
    pipeline = AutoMLPipeline(config, output_dir="test_automl_output")
    
    result = pipeline.run(
        data=data,
        target_column='future_return',
        test_size=0.2,
        time_series_split=True
    )
    
    print(f"\n🎉 AutoML Pipeline завершен!")
    print(f"🏆 Лучшая модель: {result.best_model_name}")
    print(f"📊 Лучший скор: {abs(result.best_score):.4f}")
    print(f"⏱️ Время выполнения: {result.total_time:.2f}с")