"""
Configuration Manager for AutoML Pipeline
Implements enterprise patterns for configuration management
"""

import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import os
import json
import yaml
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from loguru import logger


@dataclass
class FeatureGenerationConfig:
    """Конфигурация генерации признаков"""
    enable_technical_indicators: bool = True
    enable_statistical_features: bool = True
    enable_polynomial_features: bool = True
    enable_tsfresh_features: bool = True
    
    # Параметры технических индикаторов
    technical_indicators_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # Параметры статистических признаков
    statistical_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Параметры полиномиальных признаков
    polynomial_degree: int = 2
    polynomial_max_features: int = 50
    polynomial_interaction_only: bool = True
    
    # Параметры TSFresh
    tsfresh_max_features: int = 30
    tsfresh_default_fc_parameters: str = "efficient"
    
    # Общие параметры
    parallel_generation: bool = True
    max_features_per_generator: int = 100


@dataclass
class FeatureSelectionConfig:
    """Конфигурация отбора признаков"""
    enable_statistical_selection: bool = True
    enable_model_based_selection: bool = True
    enable_correlation_selection: bool = True
    enable_variance_selection: bool = True
    
    # Параметры статистического отбора
    statistical_method: str = "f_regression"  # f_regression, mutual_info_regression
    statistical_k: int = 50
    statistical_percentile: float = 50.0
    
    # Параметры модельного отбора
    model_type: str = "random_forest"  # random_forest, xgboost
    model_max_features: int = 100
    
    # Параметры корреляционного отбора
    correlation_threshold: float = 0.95
    target_correlation_min: float = 0.01
    
    # Параметры отбора по дисперсии
    variance_threshold: float = 0.0
    
    # Ансамблевый отбор
    ensemble_selection: bool = True
    min_votes_threshold: int = 2


@dataclass
class HyperparameterOptimizationConfig:
    """Конфигурация оптимизации гиперпараметров"""
    default_optimizer: str = "optuna_tpe"  # optuna_tpe, optuna_random, gaussian_process
    n_trials: int = 100
    n_jobs: int = -1
    random_state: int = 42
    
    # Параметры Optuna
    optuna_study_name_prefix: str = "automl_optimization"
    optuna_sampler_startup_trials: int = 10
    optuna_sampler_n_ei_candidates: int = 24
    
    # Параметры scikit-optimize
    skopt_n_initial_points: int = 10
    skopt_acq_func: str = "EI"  # EI, PI, LCB
    
    # Общие параметры
    cv_folds: int = 5
    scoring_metric: Optional[str] = None  # Автоматическое определение
    timeout_per_trial: int = 300  # секунд
    
    # Early stopping
    enable_pruning: bool = True
    pruning_min_trials: int = 20


@dataclass
class ModelSelectionConfig:
    """Конфигурация отбора моделей"""
    enable_sklearn_models: bool = True
    enable_xgboost: bool = True
    enable_lightgbm: bool = True
    enable_catboost: bool = True
    
    # Модели для тестирования
    sklearn_models: List[str] = field(default_factory=lambda: [
        'linear_regression', 'ridge', 'lasso', 'elasticnet',
        'random_forest', 'gradient_boosting', 'extra_trees'
    ])
    
    gradient_boosting_models: List[str] = field(default_factory=lambda: [
        'xgboost', 'lightgbm', 'catboost'
    ])
    
    # Параметры кросс-валидации
    cv_folds: int = 5
    time_series_split: bool = True
    shuffle_split: bool = False
    
    # Критерии отбора
    scoring_metric: Optional[str] = None
    top_k_models: int = 5
    
    # Фильтрация моделей
    max_training_time_per_model: int = 600  # секунд
    min_score_threshold: Optional[float] = None


@dataclass
class EnsembleConfig:
    """Конфигурация ансамблей"""
    enable_voting: bool = True
    enable_stacking: bool = True
    enable_blending: bool = True
    enable_bagging: bool = False
    
    # Параметры голосующего ансамбля
    voting_estimators_limit: int = 10
    voting_weights: Optional[List[float]] = None
    
    # Параметры стекинга
    stacking_cv_folds: int = 5
    stacking_meta_learner: str = "ridge"  # ridge, linear_regression
    stacking_use_features_in_secondary: bool = True
    
    # Параметры блендинга
    blending_holdout_size: float = 0.2
    
    # Общие параметры
    ensemble_size_limit: int = 5
    min_ensemble_diversity: float = 0.1


@dataclass
class DataPreprocessingConfig:
    """Конфигурация предобработки данных"""
    # Обработка пропущенных значений
    missing_value_strategy: str = "median"  # mean, median, mode, drop, forward_fill
    missing_value_threshold: float = 0.5  # Порог для удаления колонок/строк
    
    # Обработка выбросов
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    outlier_handling: str = "clip"  # clip, remove, transform
    
    # Масштабирование
    scaling_method: str = "standard"  # standard, robust, minmax, quantile
    scale_target: bool = False
    
    # Кодирование категориальных признаков
    categorical_encoding: str = "onehot"  # onehot, label, target, binary
    max_categories_onehot: int = 10
    
    # Обработка временных рядов
    handle_seasonality: bool = True
    detrend_method: Optional[str] = None  # linear, polynomial
    
    # Общие параметры
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ModelEvaluationConfig:
    """Конфигурация оценки моделей"""
    # Метрики для регрессии
    regression_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'r2', 'mape', 'rmse'
    ])
    
    # Метрики для классификации
    classification_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'auc'
    ])
    
    # Кросс-валидация
    cv_folds: int = 5
    cv_scoring: Optional[str] = None
    
    # Важность признаков
    calculate_feature_importance: bool = True
    feature_importance_method: str = "permutation"  # permutation, shap, built_in
    
    # Визуализация
    generate_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    plot_dpi: int = 300
    
    # Отчеты
    generate_report: bool = True
    report_format: str = "html"  # html, pdf, markdown


class AutoMLConfig(BaseSettings):
    """
    Главная конфигурация AutoML Pipeline
    Реализует enterprise patterns для configuration management
    """
    
    # Основные параметры
    project_name: str = Field(default="crypto_trading_automl", env="AUTOML_PROJECT_NAME")
    version: str = "1.0.0"
    random_state: int = Field(default=42, env="AUTOML_RANDOM_STATE")
    n_jobs: int = Field(default=-1, env="AUTOML_N_JOBS")
    
    # Пути
    output_dir: str = Field(default="automl_output", env="AUTOML_OUTPUT_DIR")
    cache_dir: str = Field(default="automl_cache", env="AUTOML_CACHE_DIR")
    models_dir: str = Field(default="automl_models", env="AUTOML_MODELS_DIR")
    logs_dir: str = Field(default="automl_logs", env="AUTOML_LOGS_DIR")
    
    # Режимы работы
    debug_mode: bool = Field(default=False, env="AUTOML_DEBUG")
    verbose: bool = Field(default=True, env="AUTOML_VERBOSE")
    enable_caching: bool = Field(default=True, env="AUTOML_CACHE")
    
    # Лимиты ресурсов
    max_memory_gb: float = Field(default=8.0, env="AUTOML_MAX_MEMORY")
    max_training_time: int = Field(default=3600, env="AUTOML_MAX_TIME")  # секунд
    max_models_to_try: int = Field(default=50, env="AUTOML_MAX_MODELS")
    
    # Конфигурации компонентов
    feature_generation: FeatureGenerationConfig = field(default_factory=FeatureGenerationConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    hyperparameter_optimization: HyperparameterOptimizationConfig = field(
        default_factory=HyperparameterOptimizationConfig
    )
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    data_preprocessing: DataPreprocessingConfig = field(default_factory=DataPreprocessingConfig)
    model_evaluation: ModelEvaluationConfig = field(default_factory=ModelEvaluationConfig)
    
    # Специфичные для криптотрейдинга параметры
    crypto_specific: Dict[str, Any] = field(default_factory=lambda: {
        'enable_technical_indicators': True,
        'enable_market_regime_detection': True,
        'enable_volatility_features': True,
        'enable_momentum_features': True,
        'lookback_periods': [5, 10, 20, 50],
        'prediction_horizon': 1,  # Горизонт предсказания (периоды)
        'risk_adjusted_metrics': True,
        'walk_forward_validation': True
    })
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
    
    @validator('output_dir', 'cache_dir', 'models_dir', 'logs_dir')
    def create_directories(cls, v):
        """Создание директорий если не существуют"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('max_memory_gb')
    def validate_memory(cls, v):
        """Валидация лимита памяти"""
        if v <= 0:
            raise ValueError("max_memory_gb должно быть положительным числом")
        return v
    
    @validator('n_jobs')
    def validate_n_jobs(cls, v):
        """Валидация количества процессов"""
        if v == 0:
            raise ValueError("n_jobs не может быть 0")
        return v
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Сохранение конфигурации в файл"""
        filepath = Path(filepath)
        
        config_dict = self.dict()
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("Поддерживаются только форматы .json, .yml, .yaml")
        
        logger.info(f"💾 Конфигурация сохранена: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]):
        """Загрузка конфигурации из файла"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {filepath}")
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Поддерживаются только форматы .json, .yml, .yaml")
        
        logger.info(f"📂 Конфигурация загружена: {filepath}")
        
        return cls(**config_dict)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Получение конфигурации для конкретной модели"""
        model_configs = {
            'xgboost': {
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbosity': 0 if not self.verbose else 1
            },
            'lightgbm': {
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1 if not self.verbose else 1
            },
            'catboost': {
                'random_state': self.random_state,
                'verbose': self.verbose
            },
            'sklearn': {
                'random_state': self.random_state,
                'n_jobs': self.n_jobs if model_name in [
                    'random_forest', 'extra_trees', 'knn'
                ] else None
            }
        }
        
        # Базовая конфигурация для sklearn моделей
        base_config = model_configs.get('sklearn', {})
        
        # Специфичные конфигурации
        if model_name.startswith('xgb') or model_name == 'xgboost':
            return {**base_config, **model_configs['xgboost']}
        elif model_name.startswith('lgb') or model_name == 'lightgbm':
            return {**base_config, **model_configs['lightgbm']}
        elif model_name.startswith('cat') or model_name == 'catboost':
            return {**base_config, **model_configs['catboost']}
        else:
            return base_config
    
    def get_crypto_features_config(self) -> Dict[str, Any]:
        """Получение конфигурации для криптовалютных признаков"""
        return {
            **self.crypto_specific,
            'technical_windows': self.feature_generation.technical_indicators_windows,
            'statistical_windows': self.feature_generation.statistical_windows,
            'enable_technical': self.feature_generation.enable_technical_indicators,
            'enable_statistical': self.feature_generation.enable_statistical_features
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Получение конфигурации для валидации"""
        return {
            'cv_folds': self.model_selection.cv_folds,
            'time_series_split': self.model_selection.time_series_split,
            'walk_forward_validation': self.crypto_specific.get('walk_forward_validation', True),
            'random_state': self.random_state
        }
    
    def __str__(self) -> str:
        """Строковое представление конфигурации"""
        return f"AutoMLConfig(project='{self.project_name}', version='{self.version}')"


# Предустановленные конфигурации
class PresetConfigs:
    """Предустановленные конфигурации для разных сценариев"""
    
    @staticmethod
    def fast_prototype() -> AutoMLConfig:
        """Быстрая конфигурация для прототипирования"""
        config = AutoMLConfig()
        
        # Уменьшаем количество итераций
        config.hyperparameter_optimization.n_trials = 20
        config.model_selection.cv_folds = 3
        config.model_evaluation.cv_folds = 3
        
        # Отключаем сложные генераторы признаков
        config.feature_generation.enable_tsfresh_features = False
        config.feature_generation.enable_polynomial_features = False
        
        # Ограничиваем модели
        config.model_selection.sklearn_models = ['ridge', 'random_forest']
        config.model_selection.gradient_boosting_models = ['xgboost']
        
        return config
    
    @staticmethod
    def production_ready() -> AutoMLConfig:
        """Конфигурация для продакшена"""
        config = AutoMLConfig()
        
        # Увеличиваем количество итераций
        config.hyperparameter_optimization.n_trials = 200
        config.model_selection.cv_folds = 10
        config.model_evaluation.cv_folds = 10
        
        # Включаем все функции
        config.feature_generation.enable_tsfresh_features = True
        config.feature_generation.enable_polynomial_features = True
        
        # Включаем ансамбли
        config.ensemble.enable_stacking = True
        config.ensemble.enable_voting = True
        
        # Включаем подробную оценку
        config.model_evaluation.calculate_feature_importance = True
        config.model_evaluation.generate_plots = True
        config.model_evaluation.generate_report = True
        
        return config
    
    @staticmethod
    def crypto_trading() -> AutoMLConfig:
        """Специализированная конфигурация для криптотрейдинга"""
        config = AutoMLConfig()
        
        # Настройка под временные ряды
        config.model_selection.time_series_split = True
        config.data_preprocessing.handle_seasonality = True
        
        # Криптовалютные признаки
        config.feature_generation.enable_technical_indicators = True
        config.feature_generation.technical_indicators_windows = [5, 10, 20, 50, 100]
        
        # Специфичные параметры
        config.crypto_specific.update({
            'enable_volatility_features': True,
            'enable_momentum_features': True,
            'enable_market_regime_detection': True,
            'lookback_periods': [1, 3, 5, 10, 20],
            'prediction_horizon': 1
        })
        
        # Модели подходящие для временных рядов
        config.model_selection.sklearn_models = [
            'ridge', 'lasso', 'elasticnet', 'random_forest', 'gradient_boosting'
        ]
        config.model_selection.gradient_boosting_models = ['xgboost', 'lightgbm']
        
        return config
    
    @staticmethod
    def high_frequency_trading() -> AutoMLConfig:
        """Конфигурация для высокочастотного трейдинга"""
        config = PresetConfigs.crypto_trading()
        
        # Сокращаем время обучения
        config.max_training_time = 1800  # 30 минут
        config.hyperparameter_optimization.n_trials = 50
        config.hyperparameter_optimization.timeout_per_trial = 60
        
        # Быстрые модели
        config.model_selection.sklearn_models = ['ridge', 'lasso']
        config.model_selection.gradient_boosting_models = ['lightgbm']  # Самый быстрый
        
        # Отключаем сложную генерацию признаков
        config.feature_generation.enable_tsfresh_features = False
        config.feature_generation.polynomial_max_features = 20
        
        # Специфичные параметры для HFT
        config.crypto_specific.update({
            'lookback_periods': [1, 2, 3, 5],  # Короткие периоды
            'prediction_horizon': 1,  # Только следующий тик
            'enable_microstructure_features': True,
            'enable_order_book_features': True
        })
        
        return config


if __name__ == "__main__":
    # Пример использования
    
    # Создание базовой конфигурации
    config = AutoMLConfig()
    print(f"Базовая конфигурация: {config}")
    
    # Сохранение в файл
    config.save_to_file("automl_config.json")
    
    # Загрузка из файла
    loaded_config = AutoMLConfig.load_from_file("automl_config.json")
    print(f"Загруженная конфигурация: {loaded_config}")
    
    # Предустановленные конфигурации
    fast_config = PresetConfigs.fast_prototype()
    print(f"Быстрая конфигурация: {fast_config}")
    
    crypto_config = PresetConfigs.crypto_trading()
    print(f"Конфигурация для криптотрейдинга: {crypto_config}")
    
    # Получение конфигурации модели
    xgb_config = config.get_model_config('xgboost')
    print(f"Конфигурация XGBoost: {xgb_config}")
    
    # Конфигурация валидации
    validation_config = config.get_validation_config()
    print(f"Конфигурация валидации: {validation_config}")