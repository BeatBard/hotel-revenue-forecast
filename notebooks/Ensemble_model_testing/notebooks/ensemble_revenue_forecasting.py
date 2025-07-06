#!/usr/bin/env python3
"""
Hotel Revenue Forecasting Ensemble Model
=========================================

Comprehensive ensemble model for hotel revenue forecasting using Revenue Center 1 data.
This script includes:
- Data preprocessing with leakage prevention
- Feature engineering without data leakage
- Multiple base models (Ridge, RandomForest, XGBoost, LightGBM, GradientBoosting)
- Ensemble strategies (Simple, Weighted, Top-3, Median)
- Hyperparameter tuning with cross-validation
- Comprehensive evaluation and visualization

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import warnings
from typing import Dict, List, Tuple, Any
import json

# Core ML libraries
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

class HotelRevenueEnsemble:
    """
    Comprehensive ensemble model for hotel revenue forecasting
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_weights = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def load_and_explore_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and perform comprehensive exploration of the revenue data
        """
        print("🎯 LOADING AND EXPLORING REVENUE CENTER 1 DATA")
        print("=" * 80)
        
        # Load data
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Dataset Shape: {df.shape}")
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Total Days: {(df['Date'].max() - df['Date'].min()).days}")
        print(f"Revenue Range: ${df['CheckTotal'].min():.2f} - ${df['CheckTotal'].max():.2f}")
        print(f"Average Revenue: ${df['CheckTotal'].mean():.2f}")
        
        # Data quality checks
        print(f"\n🔍 DATA QUALITY CHECKS:")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicates: {df.duplicated(['Date', 'MealPeriod']).sum()}")
        print(f"Zero revenue records: {(df['CheckTotal'] == 0).sum()}")
        
        # Revenue by meal period
        print(f"\n📊 REVENUE BY MEAL PERIOD:")
        revenue_stats = df.groupby('MealPeriod')['CheckTotal'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        print(revenue_stats)
        
        return df
    
    def create_temporal_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create temporal train/validation/test splits to prevent data leakage
        """
        print("\n📅 CREATING TEMPORAL SPLITS")
        print("-" * 50)
        
        # Sort chronologically - CRITICAL for preventing leakage
        df_sorted = df.sort_values(['Date', 'MealPeriod']).reset_index(drop=True)
        
        # Define split boundaries
        total_records = len(df_sorted)
        train_end_idx = int(total_records * 0.6)
        val_end_idx = int(total_records * 0.8)
        
        train_data = df_sorted.iloc[:train_end_idx].copy()
        val_data = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        test_data = df_sorted.iloc[val_end_idx:].copy()
        
        print(f"Training: {len(train_data)} samples ({train_data['Date'].min()} to {train_data['Date'].max()})")
        print(f"Validation: {len(val_data)} samples ({val_data['Date'].min()} to {val_data['Date'].max()})")
        print(f"Test: {len(test_data)} samples ({test_data['Date'].min()} to {test_data['Date'].max()})")
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def engineer_features(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Create comprehensive features while preventing data leakage
        """
        print("\n🔧 FEATURE ENGINEERING (LEAKAGE-FREE)")
        print("-" * 50)
        
        def safe_feature_engineering(data: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
            """Create features without data leakage"""
            df = data.copy()
            df = df.sort_values(['Date', 'MealPeriod']).reset_index(drop=True)
            
            # Basic temporal features
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['day_of_year'] = df['Date'].dt.dayofyear
            df['week_of_year'] = df['Date'].dt.isocalendar().week
            df['quarter'] = df['Date'].dt.quarter
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
            
            # Meal period encoding
            meal_encoder = LabelEncoder()
            df['meal_period_encoded'] = meal_encoder.fit_transform(df['MealPeriod'])
            
            # Meal hour (approximate)
            meal_hour_map = {'Breakfast': 8, 'Lunch': 13, 'Dinner': 19}
            df['meal_hour'] = df['MealPeriod'].map(meal_hour_map)
            
            # Cyclical encoding for temporal features
            for col, max_val in [('month', 12), ('day_of_week', 7), ('meal_hour', 24), ('quarter', 4)]:
                df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
                df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
            
            # Safe lag features (only past values)
            lag_periods = [1, 2, 3, 7, 14, 21, 30]
            for lag in lag_periods:
                df[f'CheckTotal_lag_{lag}'] = df.groupby('meal_period_encoded')['CheckTotal'].shift(lag)
            
            # Safe rolling features (historical only)
            for window in [3, 7, 14, 21, 30]:
                df[f'CheckTotal_roll_{window}d_mean'] = (
                    df.groupby('meal_period_encoded')['CheckTotal']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .shift(1)  # Critical: shift to avoid current value
                    .reset_index(0, drop=True)
                )
            
            # Event interaction features (only numeric columns)
            event_cols = [col for col in df.columns if col.startswith('Is') and col != 'IsWeekend']
            # Filter for numeric columns only
            numeric_event_cols = []
            for col in event_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    numeric_event_cols.append(col)
            
            if numeric_event_cols:
                df['total_events'] = df[numeric_event_cols].sum(axis=1)
                df['has_any_event'] = (df['total_events'] > 0).astype(int)
            
            # Revenue center and meal interactions
            df['meal_dow_interaction'] = df['meal_period_encoded'] * df['day_of_week']
            df['meal_month_interaction'] = df['meal_period_encoded'] * df['month']
            
            # Forward fill missing values to maintain temporal consistency
            lag_cols = [col for col in df.columns if 'lag_' in col or 'roll_' in col]
            for col in lag_cols:
                df[col] = df.groupby('meal_period_encoded')[col].fillna(method='ffill')
                df[col] = df[col].fillna(df[col].median())  # Final fallback
            
            return df
        
        # Apply feature engineering to each split
        print("Creating features for training data...")
        train_features = safe_feature_engineering(splits['train'], is_training=True)
        
        print("Creating features for validation data...")
        val_features = safe_feature_engineering(splits['validation'])
        
        print("Creating features for test data...")
        test_features = safe_feature_engineering(splits['test'])
        
        # Identify feature columns (exclude target and metadata)
        exclude_cols = ['CheckTotal', 'Date', 'MealPeriod', 'RevenueCenterName', 'is_zero']
        feature_cols = [col for col in train_features.columns if col not in exclude_cols]
        
        # Filter for numeric columns only
        numeric_feature_cols = []
        for col in feature_cols:
            if col in train_features.columns and pd.api.types.is_numeric_dtype(train_features[col]):
                numeric_feature_cols.append(col)
        
        # Ensure consistent feature sets across all datasets
        common_features = list(set(numeric_feature_cols) & set(val_features.columns) & set(test_features.columns))
        
        # Final filter to ensure all are numeric in all datasets
        final_features = []
        for col in common_features:
            if (pd.api.types.is_numeric_dtype(train_features[col]) and 
                pd.api.types.is_numeric_dtype(val_features[col]) and 
                pd.api.types.is_numeric_dtype(test_features[col])):
                final_features.append(col)
        
        common_features = final_features
        
        # Prepare final datasets
        X_train = train_features[common_features].fillna(0)
        y_train = train_features['CheckTotal']
        
        X_val = val_features[common_features].fillna(0)
        y_val = val_features['CheckTotal']
        
        X_test = test_features[common_features].fillna(0)
        y_test = test_features['CheckTotal']
        
        print(f"✅ Feature engineering completed:")
        print(f"  Features created: {len(common_features)}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        self.feature_names = common_features
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': common_features
        }
    
    def remove_leakage_sources(self, dataset: Dict) -> Dict:
        """
        Remove potential data leakage sources
        """
        print("\n🛡️ REMOVING DATA LEAKAGE SOURCES")
        print("-" * 50)
        
        X_train, X_val, X_test = dataset['X_train'].copy(), dataset['X_val'].copy(), dataset['X_test'].copy()
        y_train = dataset['y_train']
        
        # Identify potentially leaky features
        leaky_patterns = [
            'High_Revenue', 'Highest_Meal', 'Daily_Meal_Rank', 'SameDay_Cumulative',
            'PreviousMeal_Revenue', 'Revenue_Consistency', 'Revenue_Volatility',
            'Revenue_Percentile', 'Revenue_relative_to'
        ]
        
        leaky_features = []
        for pattern in leaky_patterns:
            matching = [f for f in self.feature_names if pattern in f]
            leaky_features.extend(matching)
        
        # Check for high correlations (potential leakage) - only on numeric features
        high_corr_features = []
        for col in self.feature_names:
            if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col]):
                try:
                    corr = abs(np.corrcoef(X_train[col].fillna(0), y_train)[0,1])
                    if corr > 0.75:  # Suspiciously high correlation
                        high_corr_features.append((col, corr))
                except:
                    pass
        
        # Ensure all features are numeric before variance threshold
        numeric_features_only = []
        for col in self.feature_names:
            if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col]):
                numeric_features_only.append(col)
        
        # Remove variance threshold features (only on numeric features)
        if numeric_features_only:
            X_train_numeric = X_train[numeric_features_only]
            var_threshold = VarianceThreshold(threshold=0.01)
            X_train_var = var_threshold.fit_transform(X_train_numeric)
            low_var_mask = var_threshold.get_support()
            low_var_features = [f for f, keep in zip(numeric_features_only, low_var_mask) if not keep]
        else:
            low_var_features = []
        
        # Combine all features to remove
        all_features_to_remove = set(leaky_features + [f[0] for f in high_corr_features] + low_var_features)
        clean_features = [f for f in self.feature_names if f not in all_features_to_remove]
        
        print(f"Removed {len(all_features_to_remove)} potentially leaky/low-variance features")
        print(f"Remaining clean features: {len(clean_features)}")
        
        # Create clean datasets
        X_train_clean = X_train[clean_features]
        X_val_clean = X_val[clean_features]
        X_test_clean = X_test[clean_features]
        
        self.feature_names = clean_features
        
        return {
            'X_train': X_train_clean,
            'y_train': y_train,
            'X_val': X_val_clean,
            'y_val': dataset['y_val'],
            'X_test': X_test_clean,
            'y_test': dataset['y_test'],
            'feature_names': clean_features
        }
    
    def prepare_scaled_datasets(self, dataset: Dict) -> Dict[str, Dict]:
        """
        Prepare scaled and unscaled versions for different models
        """
        print("\n🔄 PREPARING SCALED DATASETS")
        print("-" * 50)
        
        X_train, X_val, X_test = dataset['X_train'], dataset['X_val'], dataset['X_test']
        
        # Unscaled (for tree-based models)
        datasets = {
            'unscaled': {
                'X_train': X_train.copy(),
                'X_val': X_val.copy(),
                'X_test': X_test.copy(),
                'scaler': None
            }
        }
        
        # StandardScaler
        standard_scaler = StandardScaler()
        X_train_std = pd.DataFrame(
            standard_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_std = pd.DataFrame(
            standard_scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_std = pd.DataFrame(
            standard_scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        datasets['standard'] = {
            'X_train': X_train_std,
            'X_val': X_val_std,
            'X_test': X_test_std,
            'scaler': standard_scaler
        }
        
        # RobustScaler (better for outliers)
        robust_scaler = RobustScaler()
        X_train_rob = pd.DataFrame(
            robust_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_rob = pd.DataFrame(
            robust_scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_rob = pd.DataFrame(
            robust_scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        datasets['robust'] = {
            'X_train': X_train_rob,
            'X_val': X_val_rob,
            'X_test': X_test_rob,
            'scaler': robust_scaler
        }
        
        self.scalers = {k: v['scaler'] for k, v in datasets.items()}
        
        print(f"✅ Prepared 3 scaling variants: unscaled, standard, robust")
        
        return datasets
    
    def define_model_configurations(self) -> Dict[str, Dict]:
        """
        Define base models with conservative hyperparameters to prevent overfitting
        """
        return {
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
                },
                'scaling': 'standard',
                'n_iter': 20
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [4, 6, 8, 10],
                    'min_samples_split': [10, 20, 50],
                    'min_samples_leaf': [5, 10, 20],
                    'max_features': ['sqrt', 'log2', 0.5]
                },
                'scaling': 'unscaled',
                'n_iter': 30
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=self.random_state, verbosity=0),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'reg_alpha': [0, 0.1, 1.0],
                    'reg_lambda': [1, 2, 5]
                },
                'scaling': 'unscaled',
                'n_iter': 50
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=self.random_state, verbosity=-1),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'reg_alpha': [0, 0.1, 1.0],
                    'reg_lambda': [1, 2, 5],
                    'num_leaves': [15, 31, 63]
                },
                'scaling': 'unscaled',
                'n_iter': 50
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'max_features': ['sqrt', 'log2', 0.5]
                },
                'scaling': 'unscaled',
                'n_iter': 30
            }
        }
    
    def train_base_models(self, scaled_datasets: Dict, target_data: Dict) -> Dict[str, Any]:
        """
        Train all base models with hyperparameter tuning
        """
        print("\n🚀 TRAINING BASE MODELS WITH HYPERPARAMETER TUNING")
        print("=" * 80)
        
        model_configs = self.define_model_configurations()
        y_train, y_val = target_data['y_train'], target_data['y_val']
        
        trained_models = {}
        validation_scores = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model_name, config in model_configs.items():
            print(f"\n🔧 Training {model_name}...")
            
            # Get appropriate scaled dataset
            scaling_type = config['scaling']
            dataset = scaled_datasets[scaling_type]
            X_train, X_val = dataset['X_train'], dataset['X_val']
            
            # Hyperparameter tuning
            random_search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['param_grid'],
                n_iter=config['n_iter'],
                cv=tscv,
                scoring='neg_mean_absolute_error',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            # Fit the model
            random_search.fit(X_train, y_train)
            
            # Best model
            best_model = random_search.best_estimator_
            
            # Validation predictions
            val_pred = best_model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            # Store results
            trained_models[model_name] = {
                'model': best_model,
                'best_params': random_search.best_params_,
                'best_cv_score': -random_search.best_score_,
                'scaling_type': scaling_type
            }
            
            validation_scores[model_name] = {
                'MAE': val_mae,
                'R2': val_r2
            }
            
            print(f"  ✅ Best CV MAE: ${-random_search.best_score_:.2f}")
            print(f"  ✅ Validation MAE: ${val_mae:.2f}")
            print(f"  ✅ Validation R²: {val_r2:.4f}")
            print(f"  📊 Best params: {random_search.best_params_}")
        
        self.models = trained_models
        
        return trained_models, validation_scores
    
    def create_ensemble_predictions(self, scaled_datasets: Dict, target_data: Dict) -> Dict[str, np.ndarray]:
        """
        Create ensemble predictions using multiple strategies
        """
        print("\n🎯 CREATING ENSEMBLE PREDICTIONS")
        print("-" * 50)
        
        y_val = target_data['y_val']
        
        # Collect predictions from all models
        val_predictions = {}
        test_predictions = {}
        
        for model_name, model_info in self.models.items():
            scaling_type = model_info['scaling_type']
            model = model_info['model']
            
            # Get appropriate datasets
            X_val = scaled_datasets[scaling_type]['X_val']
            X_test = scaled_datasets[scaling_type]['X_test']
            
            # Make predictions
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            val_predictions[model_name] = val_pred
            test_predictions[model_name] = test_pred
        
        # Ensemble strategies
        ensemble_methods = {}
        
        # 1. Simple Average
        ensemble_methods['Simple_Average'] = {
            'val': np.mean(list(val_predictions.values()), axis=0),
            'test': np.mean(list(test_predictions.values()), axis=0)
        }
        
        # 2. Weighted Average (based on validation R²)
        weights = []
        for model_name in val_predictions.keys():
            val_pred = val_predictions[model_name]
            r2 = max(0, r2_score(y_val, val_pred))  # Ensure non-negative
            weights.append(r2)
        
        if sum(weights) > 0:
            weights = np.array(weights) / sum(weights)
            ensemble_methods['Weighted_Average'] = {
                'val': np.average(list(val_predictions.values()), axis=0, weights=weights),
                'test': np.average(list(test_predictions.values()), axis=0, weights=weights)
            }
            self.ensemble_weights = dict(zip(val_predictions.keys(), weights))
        
        # 3. Top 3 Models (best validation performance)
        model_scores = [(name, r2_score(y_val, val_predictions[name])) for name in val_predictions.keys()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_3_models = [name for name, _ in model_scores[:3]]
        
        top_3_val_preds = [val_predictions[name] for name in top_3_models]
        top_3_test_preds = [test_predictions[name] for name in top_3_models]
        
        ensemble_methods['Top3_Average'] = {
            'val': np.mean(top_3_val_preds, axis=0),
            'test': np.mean(top_3_test_preds, axis=0)
        }
        
        # 4. Median Ensemble (robust to outliers)
        ensemble_methods['Median_Ensemble'] = {
            'val': np.median(list(val_predictions.values()), axis=0),
            'test': np.median(list(test_predictions.values()), axis=0)
        }
        
        print(f"✅ Created {len(ensemble_methods)} ensemble strategies")
        print(f"   Models used: {list(val_predictions.keys())}")
        print(f"   Top 3 models: {top_3_models}")
        
        return {
            'individual': {'val': val_predictions, 'test': test_predictions},
            'ensemble': ensemble_methods
        }
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoiding division by zero)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')
        
        # Directional accuracy
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
        else:
            directional_accuracy = 0
        
        return {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
    
    def evaluate_all_models(self, predictions: Dict, target_data: Dict) -> pd.DataFrame:
        """
        Evaluate all individual models and ensemble methods
        """
        print("\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)
        
        y_val, y_test = target_data['y_val'], target_data['y_test']
        
        results = []
        
        # Evaluate individual models
        print("\n🔍 INDIVIDUAL MODEL PERFORMANCE:")
        for model_name, val_pred in predictions['individual']['val'].items():
            test_pred = predictions['individual']['test'][model_name]
            
            val_metrics = self.calculate_comprehensive_metrics(y_val, val_pred, f"{model_name}_Val")
            test_metrics = self.calculate_comprehensive_metrics(y_test, test_pred, f"{model_name}_Test")
            
            results.append(val_metrics)
            results.append(test_metrics)
            
            print(f"\n  {model_name}:")
            print(f"    Validation - MAE: ${val_metrics['MAE']:.2f}, R²: {val_metrics['R²']:.4f}")
            print(f"    Test       - MAE: ${test_metrics['MAE']:.2f}, R²: {test_metrics['R²']:.4f}")
        
        # Evaluate ensemble methods
        print("\n🎯 ENSEMBLE PERFORMANCE:")
        for ensemble_name, ensemble_preds in predictions['ensemble'].items():
            val_pred = ensemble_preds['val']
            test_pred = ensemble_preds['test']
            
            val_metrics = self.calculate_comprehensive_metrics(y_val, val_pred, f"{ensemble_name}_Val")
            test_metrics = self.calculate_comprehensive_metrics(y_test, test_pred, f"{ensemble_name}_Test")
            
            results.append(val_metrics)
            results.append(test_metrics)
            
            print(f"\n  {ensemble_name}:")
            print(f"    Validation - MAE: ${val_metrics['MAE']:.2f}, R²: {val_metrics['R²']:.4f}")
            print(f"    Test       - MAE: ${test_metrics['MAE']:.2f}, R²: {test_metrics['R²']:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best models
        val_results = results_df[results_df['Model'].str.contains('Val')]
        test_results = results_df[results_df['Model'].str.contains('Test')]
        
        best_val_r2 = val_results.loc[val_results['R²'].idxmax()]
        best_test_r2 = test_results.loc[test_results['R²'].idxmax()]
        
        print(f"\n🏆 BEST MODELS:")
        print(f"  Best Validation R²: {best_val_r2['Model']} ({best_val_r2['R²']:.4f})")
        print(f"  Best Test R²: {best_test_r2['Model']} ({best_test_r2['R²']:.4f})")
        
        self.results['evaluation'] = results_df
        
        return results_df
    
    def create_visualizations(self, predictions: Dict, target_data: Dict, save_plots: bool = True):
        """
        Create comprehensive visualizations
        """
        print("\n📈 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("-" * 50)
        
        y_val, y_test = target_data['y_val'], target_data['y_test']
        
        # 1. Model Performance Comparison
        plt.figure(figsize=(15, 10))
        
        # Individual models validation performance
        plt.subplot(2, 3, 1)
        model_names = list(predictions['individual']['val'].keys())
        val_r2_scores = [r2_score(y_val, predictions['individual']['val'][name]) for name in model_names]
        test_r2_scores = [r2_score(y_test, predictions['individual']['test'][name]) for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, val_r2_scores, width, label='Validation', alpha=0.7)
        plt.bar(x + width/2, test_r2_scores, width, label='Test', alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Individual Model R² Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ensemble performance
        plt.subplot(2, 3, 2)
        ensemble_names = list(predictions['ensemble'].keys())
        ensemble_val_r2 = [r2_score(y_val, predictions['ensemble'][name]['val']) for name in ensemble_names]
        ensemble_test_r2 = [r2_score(y_test, predictions['ensemble'][name]['test']) for name in ensemble_names]
        
        x = np.arange(len(ensemble_names))
        plt.bar(x - width/2, ensemble_val_r2, width, label='Validation', alpha=0.7)
        plt.bar(x + width/2, ensemble_test_r2, width, label='Test', alpha=0.7)
        plt.xlabel('Ensemble Methods')
        plt.ylabel('R² Score')
        plt.title('Ensemble Method R² Comparison')
        plt.xticks(x, ensemble_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Best ensemble predictions vs actual
        best_ensemble = max(ensemble_names, key=lambda x: r2_score(y_test, predictions['ensemble'][x]['test']))
        best_test_pred = predictions['ensemble'][best_ensemble]['test']
        
        plt.subplot(2, 3, 3)
        plt.scatter(y_test, best_test_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Revenue ($)')
        plt.ylabel('Predicted Revenue ($)')
        plt.title(f'Best Ensemble: {best_ensemble}\nTest Set Predictions')
        plt.grid(True, alpha=0.3)
        
        # Residuals analysis
        plt.subplot(2, 3, 4)
        residuals = best_test_pred - y_test
        plt.scatter(best_test_pred, residuals, alpha=0.6, color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Predicted Revenue ($)')
        plt.ylabel('Residuals ($)')
        plt.title('Residuals Analysis')
        plt.grid(True, alpha=0.3)
        
        # Time series comparison
        plt.subplot(2, 3, 5)
        plt.plot(range(len(y_test)), y_test.values, label='Actual', alpha=0.7)
        plt.plot(range(len(best_test_pred)), best_test_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Time Index')
        plt.ylabel('Revenue ($)')
        plt.title('Time Series: Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature importance (for tree-based models)
        plt.subplot(2, 3, 6)
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']['model']
            feature_importance = rf_model.feature_importances_
            top_10_idx = np.argsort(feature_importance)[-10:]
            top_10_features = [self.feature_names[i] for i in top_10_idx]
            top_10_importance = feature_importance[top_10_idx]
            
            plt.barh(range(len(top_10_features)), top_10_importance)
            plt.yticks(range(len(top_10_features)), top_10_features)
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance (RandomForest)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('ensemble_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Error Distribution Analysis
        plt.figure(figsize=(12, 8))
        
        for i, (name, pred_dict) in enumerate(predictions['ensemble'].items()):
            test_pred = pred_dict['test']
            residuals = test_pred - y_test
            
            plt.subplot(2, 2, i+1)
            plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Residuals ($)')
            plt.ylabel('Frequency')
            plt.title(f'{name} - Residuals Distribution')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            plt.text(0.05, 0.95, f'Mean: ${mean_residual:.2f}\nStd: ${std_residual:.2f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('ensemble_residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved")
        
        # Create consolidated predictions vs actual visualization and CSV
        self.create_predictions_vs_actual_summary(predictions, target_data, save_plots)
    
    def create_predictions_vs_actual_summary(self, predictions: Dict, target_data: Dict, save_plots: bool = True):
        """
        Create consolidated predictions vs actual visualization and export to CSV
        """
        print("\n📈 CREATING PREDICTIONS VS ACTUAL SUMMARY")
        print("-" * 50)
        
        # Prepare data for visualization and CSV export
        y_val = target_data['y_val']
        y_test = target_data['y_test']
        
        # Create comprehensive predictions DataFrame
        predictions_df = pd.DataFrame()
        
        # Add actual values
        predictions_df['Actual_Validation'] = y_val.values
        predictions_df['Actual_Test'] = y_test.values
        
        # Add all model predictions
        # Individual models
        if 'individual' in predictions:
            # Validation predictions
            for model_name, preds_val in predictions['individual']['val'].items():
                predictions_df[f'{model_name}_Validation'] = preds_val
            # Test predictions
            for model_name, preds_test in predictions['individual']['test'].items():
                predictions_df[f'{model_name}_Test'] = preds_test
        
        # Ensemble methods
        if 'ensemble' in predictions:
            for ens_name, ens_dict in predictions['ensemble'].items():
                predictions_df[f'{ens_name}_Validation'] = ens_dict['val']
                predictions_df[f'{ens_name}_Test'] = ens_dict['test']
        
        # Add date information for better tracking
        val_dates = target_data.get('val_dates', range(len(y_val)))
        test_dates = target_data.get('test_dates', range(len(y_test)))
        
        predictions_df['Date_Validation'] = val_dates if hasattr(val_dates, '__iter__') else range(len(y_val))
        predictions_df['Date_Test'] = test_dates if hasattr(test_dates, '__iter__') else range(len(y_test))
        
        # Save to CSV
        csv_path = 'predictions_vs_actual_complete.csv'
        predictions_df.to_csv(csv_path, index=False)
        print(f"✅ Predictions exported to: {csv_path}")
        
        if save_plots:
            # Create consolidated visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Predictions vs Actual Values - All Models', fontsize=16, fontweight='bold')
            
            # Get model names (excluding ensemble methods for cleaner visualization)
            base_models = ['Ridge', 'RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting']
            ensemble_models = ['Top3_Average']  # Show best ensemble
            
            # Plot individual models
            for i, model in enumerate(base_models):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # Initialize variables
                val_pred_clean = np.array([])
                val_actual_clean = np.array([])
                test_pred_clean = np.array([])
                test_actual_clean = np.array([])
                
                # Validation data
                if f'{model}_Validation' in predictions_df.columns:
                    val_pred = predictions_df[f'{model}_Validation'].values
                    val_actual = predictions_df['Actual_Validation'].values
                    
                    # Remove NaN values
                    mask = ~(np.isnan(val_pred) | np.isnan(val_actual))
                    val_pred_clean = val_pred[mask]
                    val_actual_clean = val_actual[mask]
                    
                    if len(val_actual_clean) > 0:
                        ax.scatter(val_actual_clean, val_pred_clean, alpha=0.6, s=30, 
                                  color='blue', label='Validation', edgecolors='black', linewidth=0.5)
                
                # Test data
                if f'{model}_Test' in predictions_df.columns:
                    test_pred = predictions_df[f'{model}_Test'].values
                    test_actual = predictions_df['Actual_Test'].values
                    
                    # Remove NaN values
                    mask = ~(np.isnan(test_pred) | np.isnan(test_actual))
                    test_pred_clean = test_pred[mask]
                    test_actual_clean = test_actual[mask]
                    
                    if len(test_actual_clean) > 0:
                        ax.scatter(test_actual_clean, test_pred_clean, alpha=0.6, s=30,
                                  color='red', label='Test', edgecolors='black', linewidth=0.5)
                
                # Perfect prediction line (only if we have data)
                if len(val_actual_clean) > 0 or len(test_actual_clean) > 0:
                    all_values = np.concatenate([val_actual_clean, test_actual_clean]) if len(val_actual_clean) > 0 and len(test_actual_clean) > 0 else (val_actual_clean if len(val_actual_clean) > 0 else test_actual_clean)
                    if len(all_values) > 0:
                        min_val, max_val = all_values.min(), all_values.max()
                        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75, linewidth=2)
                
                # Calculate R² for display
                val_r2 = r2_score(val_actual_clean, val_pred_clean) if len(val_actual_clean) > 0 else 0
                test_r2 = r2_score(test_actual_clean, test_pred_clean) if len(test_actual_clean) > 0 else 0
                
                ax.set_title(f'{model}\nVal R²: {val_r2:.3f}, Test R²: {test_r2:.3f}', fontweight='bold')
                ax.set_xlabel('Actual Revenue ($)')
                ax.set_ylabel('Predicted Revenue ($)')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Plot best ensemble in the last subplot
            ax = axes[1, 2]
            model = 'Top3_Average'
            
            # Initialize variables
            val_pred_clean = np.array([])
            val_actual_clean = np.array([])
            test_pred_clean = np.array([])
            test_actual_clean = np.array([])
            
            if f'{model}_Validation' in predictions_df.columns:
                val_pred = predictions_df[f'{model}_Validation'].values
                val_actual = predictions_df['Actual_Validation'].values
                mask = ~(np.isnan(val_pred) | np.isnan(val_actual))
                val_pred_clean = val_pred[mask]
                val_actual_clean = val_actual[mask]
                if len(val_actual_clean) > 0:
                    ax.scatter(val_actual_clean, val_pred_clean, alpha=0.6, s=30,
                              color='blue', label='Validation', edgecolors='black', linewidth=0.5)
            
            if f'{model}_Test' in predictions_df.columns:
                test_pred = predictions_df[f'{model}_Test'].values
                test_actual = predictions_df['Actual_Test'].values
                mask = ~(np.isnan(test_pred) | np.isnan(test_actual))
                test_pred_clean = test_pred[mask]
                test_actual_clean = test_actual[mask]
                if len(test_actual_clean) > 0:
                    ax.scatter(test_actual_clean, test_pred_clean, alpha=0.6, s=30,
                              color='red', label='Test', edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line (only if we have data)
            if len(val_actual_clean) > 0 or len(test_actual_clean) > 0:
                all_values = np.concatenate([val_actual_clean, test_actual_clean]) if len(val_actual_clean) > 0 and len(test_actual_clean) > 0 else (val_actual_clean if len(val_actual_clean) > 0 else test_actual_clean)
                if len(all_values) > 0:
                    min_val, max_val = all_values.min(), all_values.max()
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75, linewidth=2)
            
            val_r2 = r2_score(val_actual_clean, val_pred_clean) if len(val_actual_clean) > 0 else 0
            test_r2 = r2_score(test_actual_clean, test_pred_clean) if len(test_actual_clean) > 0 else 0
            
            ax.set_title(f'{model} (Best Ensemble)\nVal R²: {val_r2:.3f}, Test R²: {test_r2:.3f}', 
                        fontweight='bold', color='green')
            ax.set_xlabel('Actual Revenue ($)')
            ax.set_ylabel('Predicted Revenue ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = 'predictions_vs_actual_all_models.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Consolidated visualization saved: {plot_path}")

            # -----------------------------
            # Time-series plot (Actual vs Predicted)
            # -----------------------------
            # Determine the best ensemble model on the test set
            ensemble_names = list(predictions['ensemble'].keys())
            best_ensemble = max(
                ensemble_names,
                key=lambda x: r2_score(y_test, predictions['ensemble'][x]['test'])
            )

            best_test_pred = predictions['ensemble'][best_ensemble]['test']

            # Create time index for plotting (use real dates if provided)
            time_index = test_dates if hasattr(test_dates, '__iter__') else range(len(y_test))

            fig_ts, ax_ts = plt.subplots(figsize=(18, 6))
            ax_ts.plot(time_index, y_test.values, label='Actual', color='blue', linewidth=2)
            ax_ts.plot(time_index, best_test_pred, label=f'Predicted – {best_ensemble}', color='orange', linewidth=2)
            ax_ts.set_title('Actual vs Predicted Revenue – Time Series (Test Set)', fontsize=14, fontweight='bold')
            ax_ts.set_xlabel('Time')
            ax_ts.set_ylabel('Revenue ($)')
            ax_ts.legend()
            ax_ts.grid(True, alpha=0.3)

            ts_plot_path = 'time_series_actual_vs_pred.png'
            plt.tight_layout()
            plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✅ Time-series visualization saved: {ts_plot_path}")
        
        print(f"✅ Summary complete - {len(predictions_df)} predictions exported")
        return predictions_df
    
    def save_model_package(self, predictions: Dict, target_data: Dict, output_dir: str = "model_output"):
        """
        Save complete model package for production use
        """
        import os
        
        print(f"\n💾 SAVING MODEL PACKAGE TO '{output_dir}'")
        print("-" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trained models
        with open(f"{output_dir}/trained_models.pkl", 'wb') as f:
            pickle.dump(self.models, f)
        
        # Save scalers
        with open(f"{output_dir}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save feature names
        with open(f"{output_dir}/feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save ensemble weights
        with open(f"{output_dir}/ensemble_weights.pkl", 'wb') as f:
            pickle.dump(self.ensemble_weights, f)
        
        # Save predictions
        with open(f"{output_dir}/all_predictions.pkl", 'wb') as f:
            pickle.dump(predictions, f)
        
        # Save evaluation results
        if 'evaluation' in self.results:
            self.results['evaluation'].to_csv(f"{output_dir}/evaluation_results.csv", index=False)
        
        # Save model configuration summary
        model_summary = {
            'feature_count': len(self.feature_names),
            'models_trained': list(self.models.keys()),
            'ensemble_methods': list(predictions['ensemble'].keys()),
            'best_validation_r2': self.results['evaluation'][self.results['evaluation']['Model'].str.contains('Val')]['R²'].max(),
            'best_test_r2': self.results['evaluation'][self.results['evaluation']['Model'].str.contains('Test')]['R²'].max(),
            'ensemble_weights': self.ensemble_weights,
            'training_date': datetime.now().isoformat()
        }
        
        with open(f"{output_dir}/model_summary.json", 'w') as f:
            json.dump(model_summary, f, indent=2)
        
        print(f"✅ Model package saved successfully:")
        print(f"   📁 Directory: {output_dir}/")
        print(f"   🤖 Models: trained_models.pkl")
        print(f"   📊 Results: evaluation_results.csv")
        print(f"   ⚙️  Config: model_summary.json")
        print(f"   🔧 Utils: scalers.pkl, feature_names.pkl")
    
    def run_complete_pipeline(self, data_path: str, output_dir: str = "model_output"):
        """
        Run the complete ensemble modeling pipeline
        """
        print("🚀 STARTING COMPLETE ENSEMBLE MODELING PIPELINE")
        print("=" * 100)
        
        # Step 1: Load and explore data
        df = self.load_and_explore_data(data_path)
        
        # Step 2: Create temporal splits
        splits = self.create_temporal_splits(df)
        
        # Step 3: Feature engineering
        dataset = self.engineer_features(splits)
        
        # Step 4: Remove leakage sources
        clean_dataset = self.remove_leakage_sources(dataset)
        
        # Step 5: Prepare scaled datasets
        scaled_datasets = self.prepare_scaled_datasets(clean_dataset)
        
        # Step 6: Train base models
        trained_models, validation_scores = self.train_base_models(
            scaled_datasets, 
            {k: v for k, v in clean_dataset.items() if k.startswith('y_')}
        )
        
        # Step 7: Create ensemble predictions
        predictions = self.create_ensemble_predictions(
            scaled_datasets,
            {k: v for k, v in clean_dataset.items() if k.startswith('y_')}
        )
        
        # Step 8: Comprehensive evaluation
        results_df = self.evaluate_all_models(
            predictions,
            {k: v for k, v in clean_dataset.items() if k.startswith('y_')}
        )
        
        # Step 9: Create visualizations
        self.create_visualizations(
            predictions,
            {k: v for k, v in clean_dataset.items() if k.startswith('y_')}
        )
        
        # Step 10: Save model package
        self.save_model_package(
            predictions,
            {k: v for k, v in clean_dataset.items() if k.startswith('y_')},
            output_dir
        )
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        print(f"✅ All models trained and evaluated")
        print(f"✅ Ensemble methods created")
        print(f"✅ Visualizations generated")
        print(f"✅ Model package saved to '{output_dir}'")
        
        return {
            'models': trained_models,
            'predictions': predictions,
            'evaluation': results_df,
            'clean_dataset': clean_dataset
        }

def main():
    """
    Main execution function
    """
    # Initialize the ensemble model
    ensemble = HotelRevenueEnsemble(random_state=42)
    
    # Run the complete pipeline
    results = ensemble.run_complete_pipeline(
        data_path='../revenue_center_data/RevenueCenter_1_data.csv',
        output_dir='ensemble_model_output'
    )
    
    print("\n📋 FINAL SUMMARY:")
    print("-" * 50)
    print(f"Pipeline execution completed successfully!")
    print(f"Check 'ensemble_model_output/' directory for all results.")

if __name__ == "__main__":
    main() 