"""
Advanced Data Preprocessor for Crypto Trading AutoML
Implements enterprise patterns for robust data preprocessing
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileUniformTransformer,
    LabelEncoder, OneHotEncoder, TargetEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import pandas_ta as pta
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn
import joblib
from pathlib import Path

from .config_manager import AutoMLConfig, DataPreprocessingConfig


@dataclass
class PreprocessingResult:
    """Результат предобработки данных"""
    processed_data: pd.DataFrame
    preprocessing_metadata: Dict[str, Any]
    transformers: Dict[str, Any]
    processing_time: float
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]


class DataPreprocessor:
    """
    Продвинутый препроцессор данных для криптотрейдинга
    Реализует enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.preprocessing_config = self.config.data_preprocessing
        
        # Сохранение трансформеров для повторного использования
        self.fitted_transformers = {}
        self.preprocessing_pipeline = None
        self.is_fitted = False
        
        # Метаданные
        self.preprocessing_metadata = {}
        
        logger.info("🔧 DataPreprocessor инициализирован")
    
    def preprocess(
        self,
        data: pd.DataFrame,
        fit: bool = True,
        preserve_index: bool = True
    ) -> pd.DataFrame:
        """
        Основной метод предобработки данных
        
        Args:
            data: Исходные данные
            fit: Обучать трансформеры (True для обучающей выборки)
            preserve_index: Сохранять индекс
        """
        import time
        start_time = time.time()
        
        logger.info(f"🔄 Начало предобработки: {data.shape}")
        
        original_shape = data.shape
        processed_data = data.copy()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                
                # Этап 1: Базовая очистка
                task = progress.add_task("Базовая очистка данных...", total=None)
                processed_data = self._basic_cleaning(processed_data)
                
                # Этап 2: Обработка пропущенных значений
                progress.update(task, description="Обработка пропущенных значений...")
                processed_data = self._handle_missing_values(processed_data, fit)
                
                # Этап 3: Обработка выбросов
                progress.update(task, description="Обработка выбросов...")
                processed_data = self._handle_outliers(processed_data, fit)
                
                # Этап 4: Кодирование категориальных признаков
                progress.update(task, description="Кодирование категориальных признаков...")
                processed_data = self._encode_categorical(processed_data, fit)
                
                # Этап 5: Масштабирование числовых признаков
                progress.update(task, description="Масштабирование признаков...")
                processed_data = self._scale_features(processed_data, fit)
                
                # Этап 6: Удаление признаков с низкой дисперсией
                progress.update(task, description="Удаление признаков с низкой дисперсией...")
                processed_data = self._remove_low_variance_features(processed_data, fit)
                
                # Этап 7: Финальная очистка
                progress.update(task, description="Финальная очистка...")
                processed_data = self._final_cleaning(processed_data)
                
                progress.update(task, description="✅ Предобработка завершена", completed=True)
        
            # Сохранение метаданных
            processing_time = time.time() - start_time
            final_shape = processed_data.shape
            
            self.preprocessing_metadata = {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'processing_time': processing_time,
                'rows_removed': original_shape[0] - final_shape[0],
                'columns_removed': original_shape[1] - final_shape[1],
                'missing_values_handled': True,
                'outliers_handled': True,
                'categorical_encoded': True,
                'features_scaled': True
            }
            
            if fit:
                self.is_fitted = True
            
            logger.info(f"✅ Предобработка завершена: {original_shape} → {final_shape} за {processing_time:.2f}с")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Ошибка предобработки: {e}")
            return data  # Возвращаем исходные данные в случае ошибки
    
    def preprocess_target(self, target: pd.Series, fit: bool = True) -> pd.Series:
        """Предобработка целевой переменной"""
        logger.info("🎯 Предобработка целевой переменной...")
        
        processed_target = target.copy()
        
        try:
            # Обработка пропущенных значений
            if processed_target.isna().any():
                if self.preprocessing_config.missing_value_strategy == 'drop':
                    processed_target = processed_target.dropna()
                else:
                    fill_value = processed_target.mean()
                    processed_target = processed_target.fillna(fill_value)
                    logger.info(f"📝 Заполнено {target.isna().sum()} пропущенных значений целевой переменной")
            
            # Обработка выбросов в целевой переменной
            if self.preprocessing_config.outlier_handling != 'none':
                processed_target = self._handle_target_outliers(processed_target, fit)
            
            # Масштабирование целевой переменной (если необходимо)
            if self.preprocessing_config.scale_target:
                processed_target = self._scale_target(processed_target, fit)
            
            logger.info(f"✅ Целевая переменная обработана: {len(target)} → {len(processed_target)}")
            
            return processed_target
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки целевой переменной: {e}")
            return target
    
    def _basic_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Базовая очистка данных"""
        logger.info("🧹 Базовая очистка данных...")
        
        cleaned_data = data.copy()
        
        # Удаление полностью пустых строк и колонок
        initial_shape = cleaned_data.shape
        cleaned_data = cleaned_data.dropna(how='all', axis=0)  # Строки
        cleaned_data = cleaned_data.dropna(how='all', axis=1)  # Колонки
        
        if cleaned_data.shape != initial_shape:
            logger.info(f"📝 Удалены пустые строки/колонки: {initial_shape} → {cleaned_data.shape}")
        
        # Удаление дублирующихся строк
        duplicates = cleaned_data.duplicated().sum()
        if duplicates > 0:
            cleaned_data = cleaned_data.drop_duplicates()
            logger.info(f"📝 Удалено {duplicates} дублирующихся строк")
        
        # Удаление констант (колонки с одним уникальным значением)
        constant_columns = []
        for col in cleaned_data.columns:
            if cleaned_data[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            cleaned_data = cleaned_data.drop(columns=constant_columns)
            logger.info(f"📝 Удалены константные колонки: {constant_columns}")
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        logger.info("🕳️ Обработка пропущенных значений...")
        
        if not data.isna().any().any():
            logger.info("📝 Пропущенные значения не обнаружены")
            return data
        
        strategy = self.preprocessing_config.missing_value_strategy
        threshold = self.preprocessing_config.missing_value_threshold
        
        # Удаление колонок с большим количеством пропусков
        missing_ratios = data.isna().sum() / len(data)
        columns_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
        
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
            logger.info(f"📝 Удалены колонки с >({threshold*100}%) пропусков: {columns_to_drop}")
        
        # Разделение на числовые и категориальные колонки
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Обработка числовых колонок
        if numeric_columns:
            if strategy == 'mean':
                imputer_numeric = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer_numeric = SimpleImputer(strategy='median')
            elif strategy == 'forward_fill':
                data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
                imputer_numeric = None
            else:  # KNN imputation
                imputer_numeric = KNNImputer(n_neighbors=5)
            
            if imputer_numeric and fit:
                data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])
                self.fitted_transformers['numeric_imputer'] = imputer_numeric
            elif imputer_numeric and not fit and 'numeric_imputer' in self.fitted_transformers:
                data[numeric_columns] = self.fitted_transformers['numeric_imputer'].transform(data[numeric_columns])
        
        # Обработка категориальных колонок
        if categorical_columns:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            
            if fit:
                data[categorical_columns] = imputer_categorical.fit_transform(data[categorical_columns])
                self.fitted_transformers['categorical_imputer'] = imputer_categorical
            elif 'categorical_imputer' in self.fitted_transformers:
                data[categorical_columns] = self.fitted_transformers['categorical_imputer'].transform(data[categorical_columns])
        
        remaining_missing = data.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"⚠️ Остались пропущенные значения: {remaining_missing}")
            # Финальная очистка - заполнение нулями
            data = data.fillna(0)
        else:
            logger.info("✅ Все пропущенные значения обработаны")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Обработка выбросов"""
        logger.info("📊 Обработка выбросов...")
        
        method = self.preprocessing_config.outlier_detection_method
        handling = self.preprocessing_config.outlier_handling
        threshold = self.preprocessing_config.outlier_threshold
        
        if handling == 'none':
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return data
        
        outliers_detected = 0
        
        for col in numeric_columns:
            try:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data[col], nan_policy='omit'))
                    outliers_mask = z_scores > threshold
                    
                elif method == 'isolation_forest':
                    if fit:
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers_pred = iso_forest.fit_predict(data[col].values.reshape(-1, 1))
                        self.fitted_transformers[f'isolation_forest_{col}'] = iso_forest
                    else:
                        if f'isolation_forest_{col}' in self.fitted_transformers:
                            iso_forest = self.fitted_transformers[f'isolation_forest_{col}']
                            outliers_pred = iso_forest.predict(data[col].values.reshape(-1, 1))
                        else:
                            continue
                    
                    outliers_mask = outliers_pred == -1
                
                outliers_count = outliers_mask.sum()
                if outliers_count > 0:
                    outliers_detected += outliers_count
                    
                    if handling == 'remove':
                        data = data[~outliers_mask]
                    elif handling == 'clip':
                        if method != 'isolation_forest':
                            data.loc[outliers_mask, col] = data[col].clip(lower_bound, upper_bound)
                        else:
                            # Для isolation forest используем квантили
                            lower_clip = data[col].quantile(0.01)
                            upper_clip = data[col].quantile(0.99)
                            data.loc[outliers_mask, col] = data[col].clip(lower_clip, upper_clip)
                    elif handling == 'transform':
                        # Логарифмическое преобразование для положительных значений
                        if data[col].min() > 0:
                            data.loc[outliers_mask, col] = np.log1p(data.loc[outliers_mask, col])
            
            except Exception as e:
                logger.warning(f"⚠️ Ошибка обработки выбросов в колонке {col}: {e}")
                continue
        
        if outliers_detected > 0:
            logger.info(f"📝 Обработано {outliers_detected} выбросов методом {method}")
        else:
            logger.info("📝 Выбросы не обнаружены")
        
        return data
    
    def _encode_categorical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Кодирование категориальных признаков"""
        logger.info("🔤 Кодирование категориальных признаков...")
        
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not categorical_columns:
            logger.info("📝 Категориальные признаки не обнаружены")
            return data
        
        encoding_method = self.preprocessing_config.categorical_encoding
        max_categories = self.preprocessing_config.max_categories_onehot
        
        encoded_data = data.copy()
        
        for col in categorical_columns:
            try:
                unique_count = data[col].nunique()
                
                if encoding_method == 'onehot' and unique_count <= max_categories:
                    # One-Hot Encoding
                    if fit:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded_features = encoder.fit_transform(data[[col]])
                        self.fitted_transformers[f'onehot_{col}'] = encoder
                    else:
                        if f'onehot_{col}' in self.fitted_transformers:
                            encoder = self.fitted_transformers[f'onehot_{col}']
                            encoded_features = encoder.transform(data[[col]])
                        else:
                            continue
                    
                    # Создание имен признаков
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)
                    
                    # Замена исходного признака
                    encoded_data = encoded_data.drop(columns=[col])
                    encoded_data = pd.concat([encoded_data, encoded_df], axis=1)
                    
                elif encoding_method == 'label' or unique_count > max_categories:
                    # Label Encoding
                    if fit:
                        encoder = LabelEncoder()
                        encoded_data[col] = encoder.fit_transform(data[col].astype(str))
                        self.fitted_transformers[f'label_{col}'] = encoder
                    else:
                        if f'label_{col}' in self.fitted_transformers:
                            encoder = self.fitted_transformers[f'label_{col}']
                            # Обработка неизвестных категорий
                            try:
                                encoded_data[col] = encoder.transform(data[col].astype(str))
                            except ValueError:
                                # Для неизвестных категорий присваиваем -1
                                encoded_values = []
                                for value in data[col].astype(str):
                                    if value in encoder.classes_:
                                        encoded_values.append(encoder.transform([value])[0])
                                    else:
                                        encoded_values.append(-1)
                                encoded_data[col] = encoded_values
            
            except Exception as e:
                logger.warning(f"⚠️ Ошибка кодирования признака {col}: {e}")
                continue
        
        logger.info(f"✅ Закодировано {len(categorical_columns)} категориальных признаков")
        
        return encoded_data
    
    def _scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Масштабирование числовых признаков"""
        logger.info("⚖️ Масштабирование числовых признаков...")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            logger.info("📝 Числовые признаки для масштабирования не найдены")
            return data
        
        scaling_method = self.preprocessing_config.scaling_method
        scaled_data = data.copy()
        
        try:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'quantile':
                scaler = QuantileUniformTransformer()
            else:
                logger.warning(f"⚠️ Неизвестный метод масштабирования: {scaling_method}")
                return data
            
            if fit:
                scaled_data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                self.fitted_transformers['feature_scaler'] = scaler
            else:
                if 'feature_scaler' in self.fitted_transformers:
                    scaler = self.fitted_transformers['feature_scaler']
                    scaled_data[numeric_columns] = scaler.transform(data[numeric_columns])
                else:
                    logger.warning("⚠️ Скейлер не найден, пропуск масштабирования")
            
            logger.info(f"✅ Масштабированы {len(numeric_columns)} числовых признаков методом {scaling_method}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка масштабирования: {e}")
            return data
        
        return scaled_data
    
    def _remove_low_variance_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Удаление признаков с низкой дисперсией"""
        logger.info("📉 Удаление признаков с низкой дисперсией...")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return data
        
        threshold = self.preprocessing_config.variance_threshold
        
        try:
            if fit:
                variance_selector = VarianceThreshold(threshold=threshold)
                selected_features = variance_selector.fit_transform(data[numeric_columns])
                
                # Получение индексов отобранных признаков
                selected_mask = variance_selector.get_support()
                selected_columns = [col for col, mask in zip(numeric_columns, selected_mask) if mask]
                removed_columns = [col for col, mask in zip(numeric_columns, selected_mask) if not mask]
                
                self.fitted_transformers['variance_selector'] = variance_selector
                self.fitted_transformers['selected_numeric_columns'] = selected_columns
            else:
                if 'selected_numeric_columns' in self.fitted_transformers:
                    selected_columns = self.fitted_transformers['selected_numeric_columns']
                    removed_columns = [col for col in numeric_columns if col not in selected_columns]
                else:
                    return data
            
            # Удаление признаков с низкой дисперсией
            filtered_data = data.copy()
            if removed_columns:
                filtered_data = filtered_data.drop(columns=removed_columns)
                logger.info(f"📝 Удалено {len(removed_columns)} признаков с низкой дисперсией")
            else:
                logger.info("📝 Все признаки имеют достаточную дисперсию")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Ошибка фильтрации по дисперсии: {e}")
            return data
    
    def _final_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Финальная очистка данных"""
        logger.info("🏁 Финальная очистка данных...")
        
        cleaned_data = data.copy()
        
        # Удаление бесконечных значений
        infinite_mask = np.isinf(cleaned_data.select_dtypes(include=[np.number]))
        if infinite_mask.any().any():
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
            cleaned_data = cleaned_data.fillna(0)
            logger.info("📝 Обработаны бесконечные значения")
        
        # Финальная проверка на NaN
        nan_count = cleaned_data.isna().sum().sum()
        if nan_count > 0:
            cleaned_data = cleaned_data.fillna(0)
            logger.info(f"📝 Заполнено {nan_count} оставшихся NaN значений")
        
        return cleaned_data
    
    def _handle_target_outliers(self, target: pd.Series, fit: bool = True) -> pd.Series:
        """Обработка выбросов в целевой переменной"""
        method = self.preprocessing_config.outlier_detection_method
        threshold = self.preprocessing_config.outlier_threshold
        
        if method == 'iqr':
            Q1 = target.quantile(0.25)
            Q3 = target.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (target < lower_bound) | (target > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(target, nan_policy='omit'))
            outliers_mask = z_scores > threshold
        
        else:
            return target
        
        outliers_count = outliers_mask.sum()
        if outliers_count > 0:
            # Обрезаем выбросы
            target_clipped = target.clip(target.quantile(0.01), target.quantile(0.99))
            logger.info(f"📝 Обработано {outliers_count} выбросов в целевой переменной")
            return target_clipped
        
        return target
    
    def _scale_target(self, target: pd.Series, fit: bool = True) -> pd.Series:
        """Масштабирование целевой переменной"""
        try:
            if fit:
                scaler = StandardScaler()
                scaled_target = scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
                self.fitted_transformers['target_scaler'] = scaler
            else:
                if 'target_scaler' in self.fitted_transformers:
                    scaler = self.fitted_transformers['target_scaler']
                    scaled_target = scaler.transform(target.values.reshape(-1, 1)).flatten()
                else:
                    return target
            
            return pd.Series(scaled_target, index=target.index)
            
        except Exception as e:
            logger.error(f"❌ Ошибка масштабирования целевой переменной: {e}")
            return target
    
    def inverse_transform_target(self, scaled_target: pd.Series) -> pd.Series:
        """Обратное преобразование целевой переменной"""
        if 'target_scaler' not in self.fitted_transformers:
            return scaled_target
        
        try:
            scaler = self.fitted_transformers['target_scaler']
            original_target = scaler.inverse_transform(scaled_target.values.reshape(-1, 1)).flatten()
            return pd.Series(original_target, index=scaled_target.index)
        except Exception as e:
            logger.error(f"❌ Ошибка обратного преобразования: {e}")
            return scaled_target
    
    def save_transformers(self, filepath: Union[str, Path]):
        """Сохранение обученных трансформеров"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        transformers_data = {
            'fitted_transformers': self.fitted_transformers,
            'preprocessing_metadata': self.preprocessing_metadata,
            'is_fitted': self.is_fitted,
            'config': self.preprocessing_config.__dict__ if hasattr(self.preprocessing_config, '__dict__') else str(self.preprocessing_config)
        }
        
        joblib.dump(transformers_data, filepath)
        logger.info(f"💾 Трансформеры сохранены: {filepath}")
    
    def load_transformers(self, filepath: Union[str, Path]):
        """Загрузка обученных трансформеров"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл трансформеров не найден: {filepath}")
        
        transformers_data = joblib.load(filepath)
        
        self.fitted_transformers = transformers_data['fitted_transformers']
        self.preprocessing_metadata = transformers_data['preprocessing_metadata']
        self.is_fitted = transformers_data['is_fitted']
        
        logger.info(f"📂 Трансформеры загружены: {filepath}")
    
    def get_preprocessing_report(self) -> str:
        """Создание отчета по предобработке"""
        if not self.preprocessing_metadata:
            return "Предобработка еще не выполнена"
        
        metadata = self.preprocessing_metadata
        
        report = f"""
=== ОТЧЕТ ПО ПРЕДОБРАБОТКЕ ДАННЫХ ===

Исходные данные: {metadata.get('original_shape', 'N/A')}
Обработанные данные: {metadata.get('final_shape', 'N/A')}
Время обработки: {metadata.get('processing_time', 0):.2f}с

Изменения:
- Удалено строк: {metadata.get('rows_removed', 0)}
- Удалено колонок: {metadata.get('columns_removed', 0)}

Выполненные этапы:
- Пропущенные значения: {'✅' if metadata.get('missing_values_handled') else '❌'}
- Обработка выбросов: {'✅' if metadata.get('outliers_handled') else '❌'}
- Кодирование категориальных: {'✅' if metadata.get('categorical_encoded') else '❌'}
- Масштабирование признаков: {'✅' if metadata.get('features_scaled') else '❌'}

Обученные трансформеры: {len(self.fitted_transformers)}
"""
        
        return report


if __name__ == "__main__":
    # Пример использования DataPreprocessor
    
    # Создание тестовых данных
    np.random.seed(42)
    n_samples = 1000
    
    # Создание данных с различными проблемами
    data = pd.DataFrame({
        'numeric_normal': np.random.randn(n_samples),
        'numeric_with_outliers': np.concatenate([
            np.random.randn(n_samples - 50),
            np.random.randn(50) * 10  # Выбросы
        ]),
        'numeric_with_missing': np.random.randn(n_samples),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'binary': np.random.choice([0, 1], n_samples),
        'constant': [1] * n_samples,  # Константный признак
    })
    
    # Добавление пропущенных значений
    missing_indices = np.random.choice(n_samples, size=100, replace=False)
    data.loc[missing_indices, 'numeric_with_missing'] = np.nan
    
    # Целевая переменная
    target = pd.Series(
        data['numeric_normal'] * 2 + 
        data['binary'] * 3 + 
        np.random.randn(n_samples) * 0.5
    )
    
    print("=== ИСХОДНЫЕ ДАННЫЕ ===")
    print(f"Форма данных: {data.shape}")
    print(f"Пропущенные значения: {data.isna().sum().sum()}")
    print(f"Типы данных:\n{data.dtypes}")
    
    # Создание и использование препроцессора
    config = AutoMLConfig()
    preprocessor = DataPreprocessor(config)
    
    # Предобработка обучающих данных
    processed_data = preprocessor.preprocess(data, fit=True)
    processed_target = preprocessor.preprocess_target(target)
    
    print("\n=== ОБРАБОТАННЫЕ ДАННЫЕ ===")
    print(f"Форма данных: {processed_data.shape}")
    print(f"Пропущенные значения: {processed_data.isna().sum().sum()}")
    print(f"Колонки: {list(processed_data.columns)}")
    
    # Отчет по предобработке
    print(preprocessor.get_preprocessing_report())
    
    # Тестирование на новых данных (без обучения трансформеров)
    test_data = data.iloc[-100:].copy()
    processed_test_data = preprocessor.preprocess(test_data, fit=False)
    
    print(f"\n=== ТЕСТОВЫЕ ДАННЫЕ ===")
    print(f"Исходная форма: {test_data.shape}")
    print(f"Обработанная форма: {processed_test_data.shape}")