# OPTIMAL MODEL STRATEGY FOR HOTEL REVENUE FORECASTING
# Designed for small dataset (1,458 samples) with many features (119)

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def get_optimal_model_config(n_samples, n_features):
    """
    Get optimal model configuration based on dataset size
    
    Rules of thumb:
    - Samples per feature ratio < 10: High regularization, simple models
    - Samples per feature ratio 10-50: Moderate complexity
    - Samples per feature ratio > 50: Complex models possible
    """
    
    ratio = n_samples / n_features
    print(f"📊 Dataset Analysis:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Sample-to-feature ratio: {ratio:.1f}")
    
    if ratio < 10:
        complexity = "LOW"
        recommendation = "High regularization, feature selection critical"
    elif ratio < 50:
        complexity = "MEDIUM"
        recommendation = "Moderate regularization, some feature selection"
    else:
        complexity = "HIGH"
        recommendation = "Lower regularization, full feature set"
    
    print(f"   Complexity level: {complexity}")
    print(f"   Recommendation: {recommendation}")
    
    return complexity, ratio


def create_optimal_xgboost_config(complexity_level, n_features):
    """Create XGBoost configuration optimized for dataset size"""
    
    if complexity_level == "LOW":
        # High regularization for overfitting prevention
        config = {
            'objective': 'reg:squarederror',
            'n_estimators': 150,        # Moderate number
            'max_depth': 3,             # Shallow trees
            'learning_rate': 0.05,      # Slow learning
            'subsample': 0.6,           # Strong subsampling
            'colsample_bytree': 0.6,    # Strong feature subsampling
            'reg_alpha': 2.0,           # Strong L1 regularization
            'reg_lambda': 2.0,          # Strong L2 regularization
            'min_child_weight': 10,     # High minimum samples per leaf
            'random_state': 42,
            'verbosity': 0
        }
        feature_selection = min(20, n_features // 3)  # Use 1/3 of features max
        
    elif complexity_level == "MEDIUM":
        # Moderate regularization
        config = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.08,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_child_weight': 5,
            'random_state': 42,
            'verbosity': 0
        }
        feature_selection = min(40, n_features // 2)  # Use 1/2 of features max
        
    else:  # HIGH complexity
        # Standard configuration
        config = {
            'objective': 'reg:squarederror',
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 1,
            'random_state': 42,
            'verbosity': 0
        }
        feature_selection = n_features  # Use all features
    
    return config, feature_selection


def create_model_ensemble(X_train, y_train, X_test, y_test, complexity_level):
    """
    Create optimal model ensemble based on data size
    
    For small datasets: Simple ensemble of well-regularized models
    For large datasets: Complex ensemble with diverse models
    """
    
    print(f"\n🚀 CREATING OPTIMAL MODEL ENSEMBLE")
    print("=" * 50)
    
    models = {}
    predictions = {}
    
    # 1. OPTIMIZED XGBOOST (Primary model)
    print("🌟 Training optimized XGBoost...")
    xgb_config, _ = create_optimal_xgboost_config(complexity_level, X_train.shape[1])
    
    xgb_model = xgb.XGBRegressor(**xgb_config)
    xgb_model.fit(X_train, y_train)
    
    models['XGBoost_Optimized'] = xgb_model
    predictions['XGBoost_Optimized'] = xgb_model.predict(X_test)
    
    # 2. LIGHTGBM (Alternative gradient boosting)
    print("💡 Training LightGBM...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15 if complexity_level == "LOW" else 31,
        'learning_rate': 0.05 if complexity_level == "LOW" else 0.1,
        'feature_fraction': 0.6 if complexity_level == "LOW" else 0.8,
        'bagging_fraction': 0.6 if complexity_level == "LOW" else 0.8,
        'lambda_l1': 2.0 if complexity_level == "LOW" else 0.1,
        'lambda_l2': 2.0 if complexity_level == "LOW" else 0.1,
        'min_child_samples': 20 if complexity_level == "LOW" else 10,
        'random_state': 42,
        'verbosity': -1
    }
    
    lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=150)
    lgb_model.fit(X_train, y_train)
    
    models['LightGBM'] = lgb_model
    predictions['LightGBM'] = lgb_model.predict(X_test)
    
    # 3. RANDOM FOREST (For diversity, only if not too small dataset)
    if complexity_level != "LOW":
        print("🌲 Training Random Forest...")
        rf_params = {
            'n_estimators': 100,
            'max_depth': 8 if complexity_level == "HIGH" else 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        
        models['RandomForest'] = rf_model
        predictions['RandomForest'] = rf_model.predict(X_test)
    
    # 4. ENSEMBLE PREDICTION
    print("🎯 Creating ensemble prediction...")
    
    if complexity_level == "LOW":
        # Simple average of XGBoost and LightGBM
        ensemble_pred = (predictions['XGBoost_Optimized'] + predictions['LightGBM']) / 2
        weights = {'XGBoost_Optimized': 0.5, 'LightGBM': 0.5}
    else:
        # Weighted average based on validation performance
        # For simplicity, using equal weights (could be optimized)
        if len(predictions) == 3:
            ensemble_pred = (predictions['XGBoost_Optimized'] * 0.4 + 
                           predictions['LightGBM'] * 0.4 + 
                           predictions['RandomForest'] * 0.2)
            weights = {'XGBoost_Optimized': 0.4, 'LightGBM': 0.4, 'RandomForest': 0.2}
        else:
            ensemble_pred = (predictions['XGBoost_Optimized'] + predictions['LightGBM']) / 2
            weights = {'XGBoost_Optimized': 0.5, 'LightGBM': 0.5}
    
    predictions['Ensemble'] = ensemble_pred
    
    print(f"✅ Ensemble created with weights: {weights}")
    
    return models, predictions


def evaluate_models(y_true, predictions, model_names=None):
    """Evaluate all models with comprehensive metrics"""
    
    if model_names is None:
        model_names = list(predictions.keys())
    
    results = []
    
    print(f"\n📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'MAPE':<8}")
    print("-" * 70)
    
    for name in model_names:
        y_pred = predictions[name]
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Robust MAPE (exclude very low values)
        robust_mask = y_true >= 100
        if np.sum(robust_mask) > 0:
            mape = np.mean(np.abs((y_true[robust_mask] - y_pred[robust_mask]) / y_true[robust_mask])) * 100
        else:
            mape = np.nan
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        })
        
        print(f"{name:<20} ${mae:<9.0f} ${rmse:<9.0f} {r2:<7.3f} {mape:<7.1f}%")
    
    return pd.DataFrame(results)


def get_feature_importance_ensemble(models, feature_names):
    """Get feature importance from ensemble models"""
    
    print(f"\n🔍 ENSEMBLE FEATURE IMPORTANCE")
    print("=" * 50)
    
    importance_data = {}
    
    # Get importance from each model
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance_data[f'{name}_importance'] = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame(importance_data, index=feature_names)
    
    # Calculate average importance
    importance_cols = [col for col in importance_df.columns if 'importance' in col]
    importance_df['avg_importance'] = importance_df[importance_cols].mean(axis=1)
    importance_df['importance_std'] = importance_df[importance_cols].std(axis=1)
    
    # Sort by average importance
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    print("-" * 50)
    for i, (feature, row) in enumerate(importance_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {feature:<35} {row['avg_importance']:.4f} ±{row['importance_std']:.4f}")
    
    return importance_df


def main_optimal_strategy(X_train, y_train, X_test, y_test, feature_names):
    """
    Main function to execute optimal modeling strategy
    """
    
    print("🎯 OPTIMAL HOTEL REVENUE FORECASTING STRATEGY")
    print("=" * 60)
    
    # 1. Analyze dataset characteristics
    complexity_level, ratio = get_optimal_model_config(X_train.shape[0], X_train.shape[1])
    
    # 2. Create optimal model ensemble
    models, predictions = create_model_ensemble(X_train, y_train, X_test, y_test, complexity_level)
    
    # 3. Evaluate all models
    results_df = evaluate_models(y_test, predictions)
    
    # 4. Get feature importance
    importance_df = get_feature_importance_ensemble(models, feature_names)
    
    # 5. Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR YOUR DATASET")
    print("=" * 50)
    
    if ratio < 10:
        print("🚨 CRITICAL: Very small dataset for number of features!")
        print("   • Prioritize feature selection (use top 15-20 features)")
        print("   • Use high regularization always")
        print("   • Consider collecting more data")
        print("   • Ensemble of 2-3 simple models maximum")
        
    elif ratio < 20:
        print("⚠️  WARNING: Small dataset, high overfitting risk")
        print("   • Use moderate feature selection (top 30-40 features)")
        print("   • Apply regularization")
        print("   • Validate with time series CV")
        print("   • Simple ensemble works best")
        
    else:
        print("✅ GOOD: Reasonable dataset size")
        print("   • Can use most features")
        print("   • Moderate regularization sufficient")
        print("   • Complex ensemble possible")
    
    print(f"\n🏆 BEST MODEL: {results_df.loc[results_df['MAE'].idxmin(), 'Model']}")
    print(f"   MAE: ${results_df['MAE'].min():.0f}")
    print(f"   R²: {results_df.loc[results_df['MAE'].idxmin(), 'R2']:.3f}")
    
    return models, predictions, results_df, importance_df


# EXAMPLE USAGE
if __name__ == "__main__":
    print("🧠 OPTIMAL MODEL STRATEGY LOADED")
    print("=" * 50)
    print("📋 Available functions:")
    print("   • get_optimal_model_config(n_samples, n_features)")
    print("   • create_optimal_xgboost_config(complexity_level, n_features)")
    print("   • create_model_ensemble(X_train, y_train, X_test, y_test, complexity_level)")
    print("   • main_optimal_strategy(X_train, y_train, X_test, y_test, feature_names)")
    print("\n💡 For your dataset (1,458 samples, 119 features):")
    print("   • Ratio: 12.2 samples per feature (VERY LOW)")
    print("   • Recommended: XGBoost + LightGBM ensemble with high regularization")
    print("   • Feature selection: Use only top 15-25 most important features")
    print("   • Expect modest performance due to data limitations") 