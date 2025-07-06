# ============================================================================
# 🚀 XGBOOST HOTEL REVENUE FORECASTING - COMPREHENSIVE SOLUTION
# Automated data preparation, training, and evaluation
# ============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
import json
from datetime import datetime, timedelta
import sys

warnings.filterwarnings('ignore')

print("🚀 XGBOOST HOTEL REVENUE FORECASTING")
print("🎯 Switching from CNN-LSTM to XGBoost for better tabular data performance")
print("=" * 80)

# ============================================================================
# 📂 DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load and prepare data for XGBoost training"""
    
    print("📂 LOADING AND PREPARING DATA FOR XGBOOST")
    print("=" * 60)
    
    try:
        # Load the cleaned Revenue Center 1 data
        data_path = "filtered_revenue_data/RevenueCenter_1_cleaned.csv"
        if not os.path.exists(data_path):
            # Try alternative path
            data_path = "../filtered_revenue_data/RevenueCenter_1_cleaned.csv"
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"✅ Data loaded successfully:")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Revenue range: ${df['CheckTotal'].min():.2f} - ${df['CheckTotal'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("🔄 Trying to load from current directory...")
        
        # Try to find the data file
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'RevenueCenter_1' in file and file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        df['Date'] = pd.to_datetime(df['Date'])
                        print(f"✅ Found and loaded: {os.path.join(root, file)}")
                        return df
                    except:
                        continue
        
        print("❌ Could not find data file")
        return None

def create_advanced_features(df):
    """Create comprehensive features for XGBoost"""
    
    print("\n🔧 CREATING ADVANCED FEATURES FOR XGBOOST")
    print("=" * 60)
    
    df_features = df.copy()
    
    # ==================== TEMPORAL FEATURES ====================
    print("📅 Creating temporal features...")
    
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['Day'] = df_features['Date'].dt.day
    df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
    df_features['DayOfYear'] = df_features['Date'].dt.dayofyear
    df_features['WeekOfYear'] = df_features['Date'].dt.isocalendar().week
    df_features['Quarter'] = df_features['Date'].dt.quarter
    df_features['IsWeekend'] = df_features['DayOfWeek'].isin([5, 6]).astype(int)
    df_features['IsMonthEnd'] = df_features['Date'].dt.is_month_end.astype(int)
    df_features['IsMonthStart'] = df_features['Date'].dt.is_month_start.astype(int)
    df_features['IsQuarterEnd'] = df_features['Date'].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for periodic features
    df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
    df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
    df_features['DayOfWeek_sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
    df_features['DayOfWeek_cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
    df_features['DayOfYear_sin'] = np.sin(2 * np.pi * df_features['DayOfYear'] / 365)
    df_features['DayOfYear_cos'] = np.cos(2 * np.pi * df_features['DayOfYear'] / 365)
    
    # ==================== LAG FEATURES ====================
    print("📈 Creating lag features...")
    
    # Sort by date and meal period for proper lag calculation
    df_features = df_features.sort_values(['Date', 'MealPeriod_num']).reset_index(drop=True)
    
    # Revenue lag features (1, 2, 3, 7, 14, 30 days)
    for lag in [1, 2, 3, 7, 14, 30]:
        df_features[f'Revenue_lag_{lag}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].shift(lag)
    
    # Rolling statistics (7, 14, 30 days)
    for window in [7, 14, 30]:
        df_features[f'Revenue_rolling_mean_{window}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].rolling(window, min_periods=1).mean().values
        df_features[f'Revenue_rolling_std_{window}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].rolling(window, min_periods=1).std().values
        df_features[f'Revenue_rolling_max_{window}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].rolling(window, min_periods=1).max().values
        df_features[f'Revenue_rolling_min_{window}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].rolling(window, min_periods=1).min().values
    
    # ==================== MEAL PERIOD FEATURES ====================
    print("🍽️ Creating meal-specific features...")
    
    # Meal period statistics
    meal_stats = df_features.groupby('MealPeriod_num')['CheckTotal'].agg(['mean', 'std', 'median']).reset_index()
    meal_stats.columns = ['MealPeriod_num', 'MealPeriod_mean', 'MealPeriod_std', 'MealPeriod_median']
    df_features = df_features.merge(meal_stats, on='MealPeriod_num', how='left')
    
    # Revenue relative to meal period average
    df_features['Revenue_vs_MealAvg'] = df_features['CheckTotal'] / df_features['MealPeriod_mean']
    
    # ==================== SEASONALITY FEATURES ====================
    print("🌍 Creating seasonality features...")
    
    # Monthly statistics
    monthly_stats = df_features.groupby(['Month', 'MealPeriod_num'])['CheckTotal'].agg(['mean', 'std']).reset_index()
    monthly_stats.columns = ['Month', 'MealPeriod_num', 'Month_MealPeriod_mean', 'Month_MealPeriod_std']
    df_features = df_features.merge(monthly_stats, on=['Month', 'MealPeriod_num'], how='left')
    
    # Day of week statistics
    dow_stats = df_features.groupby(['DayOfWeek', 'MealPeriod_num'])['CheckTotal'].agg(['mean', 'std']).reset_index()
    dow_stats.columns = ['DayOfWeek', 'MealPeriod_num', 'DOW_MealPeriod_mean', 'DOW_MealPeriod_std']
    df_features = df_features.merge(dow_stats, on=['DayOfWeek', 'MealPeriod_num'], how='left')
    
    # ==================== DUBAI-SPECIFIC FEATURES ====================
    print("🏙️ Creating Dubai-specific features...")
    
    # Keep existing Dubai features if they exist
    dubai_features = [
        'IsNewYear', 'IsPreRamadan', 'IsPreEvent', 'IsLast10Ramadan', 
        'IsSummerEvent', 'IsMarathon', 'IsFoodFestival', 'IsDSF', 'IsRamadan'
    ]
    
    for feature in dubai_features:
        if feature not in df_features.columns:
            df_features[feature] = 0  # Default to 0 if not present
    
    # ==================== INTERACTION FEATURES ====================
    print("🔗 Creating interaction features...")
    
    # Weekend x Meal Period
    df_features['Weekend_Breakfast'] = df_features['IsWeekend'] * (df_features['MealPeriod_num'] == 0)
    df_features['Weekend_Lunch'] = df_features['IsWeekend'] * (df_features['MealPeriod_num'] == 1)
    df_features['Weekend_Dinner'] = df_features['IsWeekend'] * (df_features['MealPeriod_num'] == 2)
    
    # Event x Meal Period interactions
    for event in ['IsRamadan', 'IsDSF', 'IsMarathon']:
        if event in df_features.columns:
            df_features[f'{event}_Breakfast'] = df_features[event] * (df_features['MealPeriod_num'] == 0)
            df_features[f'{event}_Lunch'] = df_features[event] * (df_features['MealPeriod_num'] == 1)
            df_features[f'{event}_Dinner'] = df_features[event] * (df_features['MealPeriod_num'] == 2)
    
    # ==================== FEATURE SUMMARY ====================
    feature_cols = [col for col in df_features.columns if col not in ['Date', 'CheckTotal']]
    
    print(f"\n✅ FEATURE ENGINEERING COMPLETE!")
    print(f"   Total features created: {len(feature_cols)}")
    print(f"   Temporal features: {len([c for c in feature_cols if any(x in c for x in ['Year', 'Month', 'Day', 'Quarter', 'sin', 'cos'])])}")
    print(f"   Lag features: {len([c for c in feature_cols if 'lag' in c])}")
    print(f"   Rolling features: {len([c for c in feature_cols if 'rolling' in c])}")
    print(f"   Interaction features: {len([c for c in feature_cols if any(x in c for x in ['Weekend_', 'Ramadan_', 'DSF_', 'Marathon_'])])}")
    
    return df_features, feature_cols

def prepare_train_test_split(df, feature_cols, test_size=0.2):
    """Prepare time-based train/test split for forecasting"""
    
    print(f"\n📊 PREPARING TIME-BASED TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Sort by date
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    # Remove rows with NaN values (from lag features)
    df_clean = df_sorted.dropna().reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df_clean) * (1 - test_size))
    
    # Create splits
    train_data = df_clean.iloc[:split_idx].copy()
    test_data = df_clean.iloc[split_idx:].copy()
    
    # Prepare features and targets
    X_train = train_data[feature_cols]
    y_train = train_data['CheckTotal']
    X_test = test_data[feature_cols]
    y_test = test_data['CheckTotal']
    
    print(f"✅ Data split complete:")
    print(f"   Training samples: {len(X_train)} ({len(X_train)/len(df_clean)*100:.1f}%)")
    print(f"   Test samples: {len(X_test)} ({len(X_test)/len(df_clean)*100:.1f}%)")
    print(f"   Training period: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"   Test period: {test_data['Date'].min()} to {test_data['Date'].max()}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, train_data, test_data

# ============================================================================
# 🚀 XGBOOST MODEL TRAINING AND HYPERPARAMETER TUNING
# ============================================================================

def train_xgboost_models(X_train, X_test, y_train, y_test):
    """Train XGBoost models with comprehensive hyperparameter tuning"""
    
    print(f"\n🚀 XGBOOST MODEL TRAINING AND HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # ==================== BASELINE MODEL ====================
    print("📊 Training baseline XGBoost model...")
    
    baseline_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100
    )
    
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_mape = np.mean(np.abs((y_test - baseline_pred) / (y_test + 1e-8))) * 100
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    print(f"✅ Baseline results:")
    print(f"   MAE: ${baseline_mae:.2f}")
    print(f"   MAPE: {baseline_mape:.1f}%")
    print(f"   R²: {baseline_r2:.3f}")
    
    # ==================== HYPERPARAMETER TUNING ====================
    print(f"\n🎯 HYPERPARAMETER TUNING...")
    
    # Define parameter grid
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0]
    }
    
    # Use TimeSeriesSplit for proper validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Randomized search for efficiency
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions,
        n_iter=50,  # Try 50 random combinations
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print(f"🔄 Running randomized search with 50 configurations...")
    random_search.fit(X_train, y_train)
    
    # ==================== BEST MODEL EVALUATION ====================
    best_model = random_search.best_estimator_
    best_pred = best_model.predict(X_test)
    
    best_mae = mean_absolute_error(y_test, best_pred)
    best_mape = np.mean(np.abs((y_test - best_pred) / (y_test + 1e-8))) * 100
    best_r2 = r2_score(y_test, best_pred)
    
    print(f"\n🏆 BEST MODEL RESULTS:")
    print(f"   MAE: ${best_mae:.2f}")
    print(f"   MAPE: {best_mape:.1f}%")
    print(f"   R²: {best_r2:.3f}")
    print(f"   Improvement over baseline:")
    print(f"     MAE: {((baseline_mae - best_mae) / baseline_mae * 100):+.1f}%")
    print(f"     MAPE: {((baseline_mape - best_mape) / baseline_mape * 100):+.1f}%")
    print(f"     R²: {((best_r2 - baseline_r2) / abs(baseline_r2) * 100):+.1f}%")
    
    print(f"\n🎛️ Best hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    
    return {
        'baseline_model': baseline_model,
        'best_model': best_model,
        'best_params': random_search.best_params_,
        'baseline_metrics': {
            'mae': baseline_mae,
            'mape': baseline_mape,
            'r2': baseline_r2
        },
        'best_metrics': {
            'mae': best_mae,
            'mape': best_mape,
            'r2': best_r2
        },
        'predictions': {
            'baseline': baseline_pred,
            'best': best_pred
        }
    }

def analyze_feature_importance(model, feature_cols):
    """Analyze and visualize feature importance"""
    
    print(f"\n📊 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Display top features
    print(f"🏆 TOP 20 MOST IMPORTANT FEATURES:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importance - XGBoost Hotel Revenue Forecasting')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

def create_comprehensive_evaluation(models, X_test, y_test, test_data):
    """Create comprehensive model evaluation and visualizations"""
    
    print(f"\n📈 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 50)
    
    # Predictions
    baseline_pred = models['predictions']['baseline']
    best_pred = models['predictions']['best']
    
    # ==================== OVERALL METRICS ====================
    print(f"📊 OVERALL PERFORMANCE COMPARISON:")
    print(f"{'Metric':<15} {'Baseline':<15} {'Best Model':<15} {'Improvement':<15}")
    print("-" * 65)
    
    baseline_metrics = models['baseline_metrics']
    best_metrics = models['best_metrics']
    
    mae_improvement = (baseline_metrics['mae'] - best_metrics['mae']) / baseline_metrics['mae'] * 100
    mape_improvement = (baseline_metrics['mape'] - best_metrics['mape']) / baseline_metrics['mape'] * 100
    r2_improvement = (best_metrics['r2'] - baseline_metrics['r2']) / abs(baseline_metrics['r2']) * 100
    
    print(f"{'MAE ($)':<15} {baseline_metrics['mae']:<15.2f} {best_metrics['mae']:<15.2f} {mae_improvement:<15.1f}%")
    print(f"{'MAPE (%)':<15} {baseline_metrics['mape']:<15.1f} {best_metrics['mape']:<15.1f} {mape_improvement:<15.1f}%")
    print(f"{'R²':<15} {baseline_metrics['r2']:<15.3f} {best_metrics['r2']:<15.3f} {r2_improvement:<15.1f}%")
    
    # ==================== MEAL PERIOD ANALYSIS ====================
    print(f"\n🍽️ PERFORMANCE BY MEAL PERIOD:")
    print("-" * 50)
    
    test_results = test_data.copy()
    test_results['baseline_pred'] = baseline_pred
    test_results['best_pred'] = best_pred
    
    meal_names = {0: 'Breakfast', 1: 'Lunch', 2: 'Dinner'}
    
    for meal_num, meal_name in meal_names.items():
        meal_data = test_results[test_results['MealPeriod_num'] == meal_num]
        
        if len(meal_data) > 0:
            meal_actual = meal_data['CheckTotal']
            meal_baseline = meal_data['baseline_pred']
            meal_best = meal_data['best_pred']
            
            baseline_mae = mean_absolute_error(meal_actual, meal_baseline)
            baseline_mape = np.mean(np.abs((meal_actual - meal_baseline) / (meal_actual + 1e-8))) * 100
            
            best_mae = mean_absolute_error(meal_actual, meal_best)
            best_mape = np.mean(np.abs((meal_actual - meal_best) / (meal_actual + 1e-8))) * 100
            
            print(f"{meal_name}:")
            print(f"  Baseline - MAE: ${baseline_mae:.2f}, MAPE: {baseline_mape:.1f}%")
            print(f"  Best     - MAE: ${best_mae:.2f}, MAPE: {best_mape:.1f}%")
            print(f"  Improvement: {((baseline_mape - best_mape) / baseline_mape * 100):+.1f}%")
    
    # ==================== VISUALIZATIONS ====================
    print(f"\n📊 Creating visualizations...")
    
    # 1. Predictions vs Actual
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Baseline predictions
    axes[0,0].scatter(y_test, baseline_pred, alpha=0.6, s=20)
    axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual Revenue ($)')
    axes[0,0].set_ylabel('Predicted Revenue ($)')
    axes[0,0].set_title(f'Baseline Model\nMAE: ${baseline_metrics["mae"]:.2f}, MAPE: {baseline_metrics["mape"]:.1f}%')
    axes[0,0].grid(True, alpha=0.3)
    
    # Best model predictions
    axes[0,1].scatter(y_test, best_pred, alpha=0.6, s=20)
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Revenue ($)')
    axes[0,1].set_ylabel('Predicted Revenue ($)')
    axes[0,1].set_title(f'Best Model\nMAE: ${best_metrics["mae"]:.2f}, MAPE: {best_metrics["mape"]:.1f}%')
    axes[0,1].grid(True, alpha=0.3)
    
    # Residuals analysis
    baseline_residuals = y_test - baseline_pred
    best_residuals = y_test - best_pred
    
    axes[1,0].scatter(baseline_pred, baseline_residuals, alpha=0.6, s=20)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted Revenue ($)')
    axes[1,0].set_ylabel('Residuals ($)')
    axes[1,0].set_title('Baseline Model - Residuals')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter(best_pred, best_residuals, alpha=0.6, s=20)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Predicted Revenue ($)')
    axes[1,1].set_ylabel('Residuals ($)')
    axes[1,1].set_title('Best Model - Residuals')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Time series predictions
    plt.figure(figsize=(15, 8))
    
    # Plot last 100 predictions for clarity
    n_plot = min(100, len(y_test))
    plot_indices = range(len(y_test) - n_plot, len(y_test))
    
    plt.plot(plot_indices, y_test.iloc[-n_plot:], label='Actual', linewidth=2, alpha=0.8)
    plt.plot(plot_indices, baseline_pred[-n_plot:], label='Baseline Prediction', linewidth=1.5, alpha=0.7)
    plt.plot(plot_indices, best_pred[-n_plot:], label='Best Model Prediction', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Time Index')
    plt.ylabel('Revenue ($)')
    plt.title('Revenue Predictions - Last 100 Time Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgboost_time_series_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_results

def save_models_and_results(models, feature_importance, results_summary):
    """Save models and results for future use"""
    
    print(f"\n💾 SAVING MODELS AND RESULTS")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('xgboost_results', exist_ok=True)
    
    # Save models
    joblib.dump(models['baseline_model'], 'xgboost_results/baseline_model.pkl')
    joblib.dump(models['best_model'], 'xgboost_results/best_model.pkl')
    
    # Save results
    with open('xgboost_results/model_results.json', 'w') as f:
        json.dump({
            'best_params': models['best_params'],
            'baseline_metrics': models['baseline_metrics'],
            'best_metrics': models['best_metrics'],
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Save feature importance
    feature_importance.to_csv('xgboost_results/feature_importance.csv', index=False)
    
    print(f"✅ Saved:")
    print(f"   - Models: xgboost_results/")
    print(f"   - Results: xgboost_results/model_results.json")
    print(f"   - Feature importance: xgboost_results/feature_importance.csv")
    print(f"   - Visualizations: *.png files")

# ============================================================================
# 🚀 MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print(f"\n🚀 STARTING XGBOOST HOTEL REVENUE FORECASTING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        print("❌ Could not load data. Exiting.")
        return
    
    # Step 2: Create features
    df_features, feature_cols = create_advanced_features(df)
    
    # Step 3: Prepare train/test split
    X_train, X_test, y_train, y_test, train_data, test_data = prepare_train_test_split(
        df_features, feature_cols
    )
    
    # Step 4: Train models
    models = train_xgboost_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Analyze feature importance
    feature_importance = analyze_feature_importance(models['best_model'], feature_cols)
    
    # Step 6: Comprehensive evaluation
    test_results = create_comprehensive_evaluation(models, X_test, y_test, test_data)
    
    # Step 7: Save results
    save_models_and_results(models, feature_importance, {
        'test_results': test_results,
        'feature_cols': feature_cols
    })
    
    # ==================== FINAL SUMMARY ====================
    print(f"\n🎉 XGBOOST PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"🏆 BEST MODEL PERFORMANCE:")
    print(f"   MAE: ${models['best_metrics']['mae']:.2f}")
    print(f"   MAPE: {models['best_metrics']['mape']:.1f}%")
    print(f"   R²: {models['best_metrics']['r2']:.3f}")
    
    improvement_vs_baseline = (
        (models['baseline_metrics']['mape'] - models['best_metrics']['mape']) / 
        models['baseline_metrics']['mape'] * 100
    )
    print(f"   Improvement over baseline: {improvement_vs_baseline:.1f}%")
    
    print(f"\n📂 Results saved in 'xgboost_results/' directory")
    print(f"🎯 XGBoost provides much more stable and interpretable results than CNN-LSTM!")
    
    return models, feature_importance, test_results

if __name__ == "__main__":
    # Run the complete pipeline
    try:
        models, feature_importance, test_results = main()
        print(f"\n✅ SUCCESS! XGBoost model training completed successfully.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc() 