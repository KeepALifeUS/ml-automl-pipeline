"""
Automated Feature Generator for Crypto Trading AutoML Pipeline
Implements enterprise patterns for scalable feature generation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import ta
import pandas_ta as pta
from tsfresh import extract_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from featuretools import dfs
import feature_engine.creation as fec
from loguru import logger
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn
from joblib import Parallel, delayed

from ..utils.config_manager import AutoMLConfig
from ..utils.data_preprocessor import DataPreprocessor


@dataclass
class FeatureGenerationResult:
    """Результат генерации признаков"""
    features: pd.DataFrame
    feature_names: List[str]
    feature_importance: Dict[str, float]
    generation_metadata: Dict[str, Any]
    processing_time: float


class BaseFeatureGenerator(ABC):
    """Базовый класс для генераторов признаков -  pattern"""
    
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерировать признаки"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Получить имена сгенерированных признаков"""
        pass


class TechnicalIndicatorGenerator(BaseFeatureGenerator):
    """Генератор технических индикаторов для криптовалют"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_names = []
        
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерация технических индикаторов"""
        logger.info("🔧 Генерация технических индикаторов...")
        
        features = pd.DataFrame(index=data.index)
        
        # Обязательные колонки для технического анализа
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.warning("⚠️ Недостаточно данных для технического анализа")
            return features
        
        try:
            # Основные индикаторы тренда
            features['sma_10'] = ta.trend.sma_indicator(data['close'], window=10)
            features['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
            features['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
            features['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
            features['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
            
            # MACD
            features['macd'] = ta.trend.macd(data['close'])
            features['macd_signal'] = ta.trend.macd_signal(data['close'])
            features['macd_histogram'] = ta.trend.macd_diff(data['close'])
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(data['close'])
            bb_low = ta.volatility.bollinger_lband(data['close'])
            features['bb_high'] = bb_high
            features['bb_low'] = bb_low
            features['bb_width'] = bb_high - bb_low
            features['bb_position'] = (data['close'] - bb_low) / (bb_high - bb_low)
            
            # RSI
            features['rsi'] = ta.momentum.rsi(data['close'])
            features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
            features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
            
            # Stochastic
            features['stoch_k'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
            features['stoch_d'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
            
            # Volume indicators
            features['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume'])
            features['vwap'] = ta.volume.volume_weighted_average_price(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            # ATR - Average True Range
            features['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
            
            # Криптовалютные специфичные индикаторы
            features['price_change_1h'] = data['close'].pct_change(periods=1)
            features['price_change_4h'] = data['close'].pct_change(periods=4)
            features['price_change_24h'] = data['close'].pct_change(periods=24)
            
            # Волатильность
            features['volatility_10'] = data['close'].rolling(10).std()
            features['volatility_20'] = data['close'].rolling(20).std()
            
            # Momentum features
            features['momentum_5'] = ta.momentum.roc(data['close'], window=5)
            features['momentum_10'] = ta.momentum.roc(data['close'], window=10)
            
            self.feature_names = list(features.columns)
            logger.info(f"✅ Сгенерировано {len(self.feature_names)} технических индикаторов")
            
            return features.fillna(0)  # Заполнить NaN нулями
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации технических индикаторов: {e}")
            return features
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class StatisticalFeatureGenerator(BaseFeatureGenerator):
    """Генератор статистических признаков"""
    
    def __init__(self, windows: List[int] = [5, 10, 20, 50]):
        self.windows = windows
        self.feature_names = []
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерация статистических признаков"""
        logger.info("📊 Генерация статистических признаков...")
        
        features = pd.DataFrame(index=data.index)
        
        if 'close' not in data.columns:
            logger.warning("⚠️ Колонка 'close' не найдена")
            return features
        
        try:
            close = data['close']
            
            for window in self.windows:
                # Скользящие статистики
                features[f'mean_{window}'] = close.rolling(window).mean()
                features[f'std_{window}'] = close.rolling(window).std()
                features[f'min_{window}'] = close.rolling(window).min()
                features[f'max_{window}'] = close.rolling(window).max()
                features[f'median_{window}'] = close.rolling(window).median()
                
                # Квантили
                features[f'q25_{window}'] = close.rolling(window).quantile(0.25)
                features[f'q75_{window}'] = close.rolling(window).quantile(0.75)
                
                # Асимметрия и эксцесс
                features[f'skew_{window}'] = close.rolling(window).skew()
                features[f'kurtosis_{window}'] = close.rolling(window).kurt()
                
                # Z-score
                rolling_mean = close.rolling(window).mean()
                rolling_std = close.rolling(window).std()
                features[f'zscore_{window}'] = (close - rolling_mean) / rolling_std
                
                # Относительная позиция в диапазоне
                rolling_min = close.rolling(window).min()
                rolling_max = close.rolling(window).max()
                features[f'position_{window}'] = (close - rolling_min) / (rolling_max - rolling_min)
            
            # Лаговые признаки
            for lag in [1, 2, 3, 5, 10]:
                features[f'lag_{lag}'] = close.shift(lag)
                features[f'diff_{lag}'] = close.diff(lag)
                features[f'pct_change_{lag}'] = close.pct_change(lag)
            
            self.feature_names = list(features.columns)
            logger.info(f"✅ Сгенерировано {len(self.feature_names)} статистических признаков")
            
            return features.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации статистических признаков: {e}")
            return features
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class PolynomialFeatureGenerator(BaseFeatureGenerator):
    """Генератор полиномиальных признаков"""
    
    def __init__(self, degree: int = 2, interaction_only: bool = True, max_features: int = 100):
        self.degree = degree
        self.interaction_only = interaction_only
        self.max_features = max_features
        self.poly_transformer = None
        self.feature_names = []
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерация полиномиальных признаков"""
        logger.info("🔢 Генерация полиномиальных признаков...")
        
        if data.empty:
            return pd.DataFrame(index=data.index)
        
        try:
            # Отбираем только числовые колонки
            numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
            
            if numeric_data.shape[1] > 20:
                # Если слишком много колонок, выбираем топ-20 по важности
                correlations = numeric_data.corrwith(numeric_data.iloc[:, 0]).abs()
                top_features = correlations.nlargest(20).index
                numeric_data = numeric_data[top_features]
            
            self.poly_transformer = PolynomialFeatures(
                degree=self.degree,
                interaction_only=self.interaction_only,
                include_bias=False
            )
            
            poly_features = self.poly_transformer.fit_transform(numeric_data)
            
            # Ограничиваем количество признаков
            if poly_features.shape[1] > self.max_features:
                # Используем SelectKBest для отбора лучших признаков
                if len(numeric_data) > 1:
                    target = numeric_data.iloc[:, 0]  # Первая колонка как целевая
                    selector = SelectKBest(f_regression, k=self.max_features)
                    poly_features = selector.fit_transform(poly_features, target)
                else:
                    poly_features = poly_features[:, :self.max_features]
            
            feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
            features = pd.DataFrame(poly_features, index=data.index, columns=feature_names)
            
            self.feature_names = feature_names
            logger.info(f"✅ Сгенерировано {len(feature_names)} полиномиальных признаков")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации полиномиальных признаков: {e}")
            return pd.DataFrame(index=data.index)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class TSFreshFeatureGenerator(BaseFeatureGenerator):
    """Генератор временных признаков с TSFresh"""
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.feature_names = []
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерация временных признаков"""
        logger.info("⏰ Генерация временных признаков TSFresh...")
        
        if data.empty or len(data) < 10:
            return pd.DataFrame(index=data.index)
        
        try:
            # Подготовка данных для TSFresh
            time_series_data = data.copy()
            time_series_data['id'] = 1
            time_series_data['time'] = range(len(data))
            
            # Выбираем числовые колонки
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return pd.DataFrame(index=data.index)
            
            # Берем первую числовую колонку для генерации признаков
            value_col = numeric_cols[0]
            ts_data = time_series_data[['id', 'time', value_col]].copy()
            ts_data.columns = ['id', 'time', 'value']
            
            # Извлечение признаков
            extracted_features = extract_features(
                ts_data,
                column_id='id',
                column_sort='time',
                column_value='value',
                n_jobs=1,
                disable_progressbar=True
            )
            
            # Импутация пропущенных значений
            imputed_features = impute(extracted_features)
            
            # Ограничиваем количество признаков
            if imputed_features.shape[1] > self.max_features:
                # Отбираем топ признаки по дисперсии
                feature_vars = imputed_features.var()
                top_features = feature_vars.nlargest(self.max_features).index
                imputed_features = imputed_features[top_features]
            
            # Транслируем признаки на весь временной ряд
            features = pd.DataFrame(index=data.index)
            for col in imputed_features.columns:
                features[f'tsfresh_{col}'] = imputed_features[col].iloc[0]
            
            self.feature_names = list(features.columns)
            logger.info(f"✅ Сгенерировано {len(self.feature_names)} TSFresh признаков")
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации TSFresh признаков: {e}")
            return pd.DataFrame(index=data.index)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class AutoFeatureGenerator:
    """
    Главный класс для автоматической генерации признаков
    Реализует enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.generators: Dict[str, BaseFeatureGenerator] = {}
        self.feature_metadata: Dict[str, Any] = {}
        self._setup_generators()
        
    def _setup_generators(self):
        """Настройка генераторов признаков"""
        logger.info("🔧 Настройка генераторов признаков...")
        
        # Технические индикаторы
        self.generators['technical'] = TechnicalIndicatorGenerator(
            self.config.feature_generation.get('technical', {})
        )
        
        # Статистические признаки
        self.generators['statistical'] = StatisticalFeatureGenerator(
            windows=self.config.feature_generation.get('statistical_windows', [5, 10, 20])
        )
        
        # Полиномиальные признаки
        if self.config.feature_generation.get('enable_polynomial', True):
            self.generators['polynomial'] = PolynomialFeatureGenerator(
                degree=self.config.feature_generation.get('polynomial_degree', 2),
                max_features=self.config.feature_generation.get('polynomial_max_features', 50)
            )
        
        # TSFresh признаки
        if self.config.feature_generation.get('enable_tsfresh', True):
            self.generators['tsfresh'] = TSFreshFeatureGenerator(
                max_features=self.config.feature_generation.get('tsfresh_max_features', 30)
            )
        
        logger.info(f"✅ Настроено {len(self.generators)} генераторов")
    
    def generate_features(
        self,
        data: pd.DataFrame,
        generators: Optional[List[str]] = None,
        parallel: bool = True
    ) -> FeatureGenerationResult:
        """
        Основной метод генерации признаков
        
        Args:
            data: Исходные данные
            generators: Список генераторов для использования
            parallel: Использовать параллельную генерацию
            
        Returns:
            FeatureGenerationResult: Результат генерации
        """
        logger.info("🚀 Запуск автоматической генерации признаков...")
        
        import time
        start_time = time.time()
        
        if generators is None:
            generators = list(self.generators.keys())
        
        all_features = []
        all_feature_names = []
        generation_metadata = {}
        
        if parallel and len(generators) > 1:
            # Параллельная генерация
            with ThreadPoolExecutor(max_workers=min(len(generators), 4)) as executor:
                future_to_generator = {
                    executor.submit(self.generators[gen_name].generate, data): gen_name
                    for gen_name in generators if gen_name in self.generators
                }
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                ) as progress:
                    task = progress.add_task("Генерация признаков...", total=len(future_to_generator))
                    
                    for future in as_completed(future_to_generator):
                        gen_name = future_to_generator[future]
                        try:
                            features = future.result()
                            if not features.empty:
                                all_features.append(features)
                                feature_names = self.generators[gen_name].get_feature_names()
                                all_feature_names.extend(feature_names)
                                generation_metadata[gen_name] = {
                                    'feature_count': len(feature_names),
                                    'feature_names': feature_names
                                }
                            progress.advance(task)
                        except Exception as e:
                            logger.error(f"❌ Ошибка в генераторе {gen_name}: {e}")
                            progress.advance(task)
        else:
            # Последовательная генерация
            for gen_name in generators:
                if gen_name not in self.generators:
                    continue
                    
                try:
                    features = self.generators[gen_name].generate(data)
                    if not features.empty:
                        all_features.append(features)
                        feature_names = self.generators[gen_name].get_feature_names()
                        all_feature_names.extend(feature_names)
                        generation_metadata[gen_name] = {
                            'feature_count': len(feature_names),
                            'feature_names': feature_names
                        }
                except Exception as e:
                    logger.error(f"❌ Ошибка в генераторе {gen_name}: {e}")
        
        # Объединение всех признаков
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            # Удаление дубликатов колонок
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        else:
            combined_features = pd.DataFrame(index=data.index)
            
        # Вычисление важности признаков (простая корреляция с первой колонкой)
        feature_importance = {}
        if not combined_features.empty and len(combined_features.columns) > 1:
            try:
                if 'close' in data.columns:
                    target = data['close']
                else:
                    target = data.iloc[:, 0] if not data.empty else combined_features.iloc[:, 0]
                
                correlations = combined_features.corrwith(target).abs()
                feature_importance = correlations.fillna(0).to_dict()
            except:
                feature_importance = {col: 0.0 for col in combined_features.columns}
        
        processing_time = time.time() - start_time
        
        result = FeatureGenerationResult(
            features=combined_features,
            feature_names=list(combined_features.columns),
            feature_importance=feature_importance,
            generation_metadata=generation_metadata,
            processing_time=processing_time
        )
        
        logger.info(f"✅ Генерация завершена: {len(result.feature_names)} признаков за {processing_time:.2f}с")
        
        return result
    
    def get_feature_importance_ranking(self, result: FeatureGenerationResult) -> List[Tuple[str, float]]:
        """Получить ранжирование признаков по важности"""
        return sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )


if __name__ == "__main__":
    # Пример использования
    from ..utils.config_manager import AutoMLConfig
    
    # Создание тестовых данных
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    test_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 50000,
        'high': np.random.randn(1000).cumsum() + 50100,
        'low': np.random.randn(1000).cumsum() + 49900,
        'close': np.random.randn(1000).cumsum() + 50000,
        'volume': np.random.exponential(1000, 1000)
    }, index=dates)
    
    # Создание генератора
    config = AutoMLConfig()
    generator = AutoFeatureGenerator(config)
    
    # Генерация признаков
    result = generator.generate_features(test_data)
    
    print(f"Сгенерировано признаков: {len(result.feature_names)}")
    print(f"Время обработки: {result.processing_time:.2f}с")
    print(f"Метаданные: {result.generation_metadata}")
    
    # Топ-10 важных признаков
    top_features = generator.get_feature_importance_ranking(result)[:10]
    print("\nТоп-10 важных признаков:")
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")