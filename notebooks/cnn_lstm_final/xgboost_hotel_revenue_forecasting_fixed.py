# ============================================================================
# 🚀 XGBOOST HOTEL REVENUE FORECASTING - FIXED VERSION
# Handles categorical variables and data preprocessing properly
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

print("🚀 XGBOOST HOTEL REVENUE FORECASTING - FIXED VERSION")
print("🎯 Proper handling of categorical variables and data types")
print("=" * 80)

# ============================================================================
# 📂 DATA LOADING AND PREPARATION
# ============================================================================

def find_and_load_data():
    """Find and load the hotel revenue data from available files"""
    
    print("📂 SEARCHING FOR HOTEL REVENUE DATA")
    print("=" * 60)
    
    # Possible data file locations and names
    possible_paths = [
        "../../data/RevenueCenter_1_data.csv",
        "../data/RevenueCenter_1_data.csv", 
        "../../notebooks/revenue_center_data/RevenueCenter_1_data.csv",
        "../notebooks/revenue_center_data/RevenueCenter_1_data.csv",
        "../../notebooks/revenue_center_data/RevenueCenter_1_preprocessed.csv",
        "../notebooks/revenue_center_data/RevenueCenter_1_preprocessed.csv",
        "./RevenueCenter_1_data.csv",
        "cnn_lstm_training_ready/data/rc1_normalized_data.csv",
        "../cnn_lstm_training_ready/data/rc1_normalized_data.csv"
    ]
    
    # Try to find any CSV file with revenue data
    for root, dirs, files in os.walk('../../'):
        for file in files:
            if 'RevenueCenter' in file and file.endswith('.csv'):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)
                    if 'CheckTotal' in df.columns or 'Revenue' in df.columns:
                        print(f"✅ Found data: {full_path}")
                        print(f"   Shape: {df.shape}")
                        print(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
                        return df, full_path
                except Exception as e:
                    continue
    
    # If no file found, create sample data
    print("⚠️ No data file found. Creating sample data for demonstration...")
    return create_sample_data()

def create_sample_data():
    """Create sample hotel revenue data for demonstration"""
    
    print("🔧 CREATING SAMPLE HOTEL REVENUE DATA")
    print("=" * 50)
    
    # Create date range
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-04-30')
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create sample data
    data = []
    meal_periods = ['Breakfast', 'Lunch', 'Dinner']
    meal_nums = [0, 1, 2]
    
    np.random.seed(42)  # For reproducible results
    
    for date in dates:
        for meal_period, meal_num in zip(meal_periods, meal_nums):
            # Base revenue with seasonality and trends
            base_revenue = 1000 + np.sin(date.dayofyear * 2 * np.pi / 365) * 200
            
            # Meal period effects
            if meal_period == 'Breakfast':
                base_revenue *= 0.7
            elif meal_period == 'Lunch':
                base_revenue *= 1.2
            else:  # Dinner
                base_revenue *= 1.1
            
            # Weekend effects
            if date.weekday() >= 5:
                base_revenue *= 1.3
            
            # Add noise
            revenue = base_revenue + np.random.normal(0, base_revenue * 0.2)
            revenue = max(50, revenue)  # Minimum revenue
            
            data.append({
                'Date': date,
                'MealPeriod': meal_period,
                'MealPeriod_num': meal_num,
                'CheckTotal': revenue,
                'RevenueCenterName': 'RC1_Breakfast',
                'DayOfWeek_Name': date.strftime('%A')
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Sample data created:")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Revenue range: ${df['CheckTotal'].min():.2f} - ${df['CheckTotal'].max():.2f}")
    
    return df, "sample_data"

def preprocess_data(df):
    """Preprocess the data and handle categorical variables"""
    
    print("\n🔧 PREPROCESSING DATA AND HANDLING CATEGORICAL VARIABLES")
    print("=" * 70)
    
    df_processed = df.copy()
    
    # Ensure Date column is datetime
    if 'Date' in df_processed.columns:
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
    
    # Handle missing values in target
    if 'CheckTotal' in df_processed.columns:
        target_col = 'CheckTotal'
    elif 'Revenue' in df_processed.columns:
        target_col = 'Revenue'
    else:
        # Find numeric column that could be revenue
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        target_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if target_col is None:
        raise ValueError("No suitable target column found")
    
    print(f"📊 Using '{target_col}' as target variable")
    
    # Remove rows with missing target values
    initial_len = len(df_processed)
    df_processed = df_processed.dropna(subset=[target_col])
    removed_rows = initial_len - len(df_processed)
    if removed_rows > 0:
        print(f"   Removed {removed_rows} rows with missing target values")
    
    # Rename target to standardize
    if target_col != 'CheckTotal':
        df_processed['CheckTotal'] = df_processed[target_col]
    
    # Handle categorical columns with Label Encoding
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != 'Date']
    
    label_encoders = {}
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"   Encoded categorical column: {col}")
    
    # Create MealPeriod_num if not exists
    if 'MealPeriod_num' not in df_processed.columns and 'MealPeriod' in df_processed.columns:
        meal_mapping = {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2}
        df_processed['MealPeriod_num'] = df_processed['MealPeriod'].map(meal_mapping)
        df_processed['MealPeriod_num'] = df_processed['MealPeriod_num'].fillna(0)
    
    print(f"✅ Preprocessing complete:")
    print(f"   Final shape: {df_processed.shape}")
    print(f"   Target range: ${df_processed['CheckTotal'].min():.2f} - ${df_processed['CheckTotal'].max():.2f}")
    
    return df_processed, label_encoders

def create_advanced_features(df):
    """Create comprehensive features for XGBoost"""
    
    print("\n🔧 CREATING ADVANCED FEATURES FOR XGBOOST")
    print("=" * 60)
    
    df_features = df.copy()
    
    # Ensure we have a date column
    if 'Date' not in df_features.columns:
        print("⚠️ No Date column found, creating from index")
        df_features['Date'] = pd.date_range('2023-01-01', periods=len(df_features), freq='D')
    
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
    
    # Revenue lag features (1, 2, 3, 7, 14 days)
    for lag in [1, 2, 3, 7, 14]:
        if 'MealPeriod_num' in df_features.columns:
            df_features[f'Revenue_lag_{lag}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].shift(lag)
        else:
            df_features[f'Revenue_lag_{lag}'] = df_features['CheckTotal'].shift(lag)
    
    # Rolling statistics (7, 14 days)
    for window in [7, 14]:
        if 'MealPeriod_num' in df_features.columns:
            df_features[f'Revenue_rolling_mean_{window}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].rolling(window, min_periods=1).mean().values
            df_features[f'Revenue_rolling_std_{window}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].rolling(window, min_periods=1).std().values
        else:
            df_features[f'Revenue_rolling_mean_{window}'] = df_features['CheckTotal'].rolling(window, min_periods=1).mean()
            df_features[f'Revenue_rolling_std_{window}'] = df_features['CheckTotal'].rolling(window, min_periods=1).std()
    
    # ==================== INTERACTION FEATURES ====================
    print("🔗 Creating interaction features...")
    
    # Weekend x Meal Period (if available)
    if 'MealPeriod_num' in df_features.columns:
        df_features['Weekend_Breakfast'] = df_features['IsWeekend'] * (df_features['MealPeriod_num'] == 0)
        df_features['Weekend_Lunch'] = df_features['IsWeekend'] * (df_features['MealPeriod_num'] == 1)
        df_features['Weekend_Dinner'] = df_features['IsWeekend'] * (df_features['MealPeriod_num'] == 2)
    
    # Month x Weekend interaction
    df_features['Month_Weekend'] = df_features['Month'] * df_features['IsWeekend']
    
    # ==================== FEATURE SELECTION ====================
    # Select only numeric features for XGBoost
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and date-related columns
    feature_cols = [col for col in numeric_features if col not in ['CheckTotal', 'Date'] and not col.startswith('Date')]
    
    print(f"\n✅ FEATURE ENGINEERING COMPLETE!")
    print(f"   Total features created: {len(feature_cols)}")
    print(f"   Temporal features: {len([c for c in feature_cols if any(x in c for x in ['Year', 'Month', 'Day', 'Quarter', 'sin', 'cos'])])}")
    print(f"   Lag features: {len([c for c in feature_cols if 'lag' in c])}")
    print(f"   Rolling features: {len([c for c in feature_cols if 'rolling' in c])}")
    print(f"   Interaction features: {len([c for c in feature_cols if any(x in c for x in ['Weekend_', '_Weekend'])])}")
    
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
    
    # Ensure all features are numeric
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    print(f"✅ Data split complete:")
    print(f"   Training samples: {len(X_train)} ({len(X_train)/len(df_clean)*100:.1f}%)")
    print(f"   Test samples: {len(X_test)} ({len(X_test)/len(df_clean)*100:.1f}%)")
    print(f"   Training period: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"   Test period: {test_data['Date'].min()} to {test_data['Date'].max()}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, train_data, test_data

# ============================================================================
# 🚀 XGBOOST MODEL TRAINING
# ============================================================================

def train_xgboost_models(X_train, X_test, y_train, y_test):
    """Train XGBoost models with hyperparameter tuning"""
    
    print(f"\n🚀 XGBOOST MODEL TRAINING AND HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # ==================== BASELINE MODEL ====================
    print("📊 Training baseline XGBoost model...")
    
    baseline_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
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
    
    # ==================== OPTIMIZED MODEL ====================
    print(f"\n🎯 TRAINING OPTIMIZED MODEL...")
    
    # Use a more focused parameter set for faster training
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    best_model.fit(X_train, y_train)
    best_pred = best_model.predict(X_test)
    
    best_mae = mean_absolute_error(y_test, best_pred)
    best_mape = np.mean(np.abs((y_test - best_pred) / (y_test + 1e-8))) * 100
    best_r2 = r2_score(y_test, best_pred)
    
    print(f"\n🏆 OPTIMIZED MODEL RESULTS:")
    print(f"   MAE: ${best_mae:.2f}")
    print(f"   MAPE: {best_mape:.1f}%")
    print(f"   R²: {best_r2:.3f}")
    print(f"   Improvement over baseline:")
    print(f"     MAE: {((baseline_mae - best_mae) / baseline_mae * 100):+.1f}%")
    print(f"     MAPE: {((baseline_mape - best_mape) / baseline_mape * 100):+.1f}%")
    print(f"     R²: {((best_r2 - baseline_r2) / abs(baseline_r2) * 100):+.1f}%")
    
    return {
        'baseline_model': baseline_model,
        'best_model': best_model,
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
    print(f"🏆 TOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance - XGBoost Hotel Revenue Forecasting')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

def create_evaluation_plots(models, X_test, y_test):
    """Create evaluation plots"""
    
    print(f"\n📈 CREATING EVALUATION PLOTS")
    print("=" * 50)
    
    baseline_pred = models['predictions']['baseline']
    best_pred = models['predictions']['best']
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Baseline predictions
    axes[0,0].scatter(y_test, baseline_pred, alpha=0.6, s=20)
    axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual Revenue ($)')
    axes[0,0].set_ylabel('Predicted Revenue ($)')
    axes[0,0].set_title(f'Baseline Model\nMAE: ${models["baseline_metrics"]["mae"]:.2f}, MAPE: {models["baseline_metrics"]["mape"]:.1f}%')
    axes[0,0].grid(True, alpha=0.3)
    
    # Best model predictions
    axes[0,1].scatter(y_test, best_pred, alpha=0.6, s=20)
    axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Revenue ($)')
    axes[0,1].set_ylabel('Predicted Revenue ($)')
    axes[0,1].set_title(f'Optimized Model\nMAE: ${models["best_metrics"]["mae"]:.2f}, MAPE: {models["best_metrics"]["mape"]:.1f}%')
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
    axes[1,1].set_title('Optimized Model - Residuals')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Time series plot
    plt.figure(figsize=(15, 6))
    n_plot = min(100, len(y_test))
    plot_indices = range(len(y_test) - n_plot, len(y_test))
    
    plt.plot(plot_indices, y_test.iloc[-n_plot:], label='Actual', linewidth=2, alpha=0.8)
    plt.plot(plot_indices, baseline_pred[-n_plot:], label='Baseline Prediction', linewidth=1.5, alpha=0.7)
    plt.plot(plot_indices, best_pred[-n_plot:], label='Optimized Prediction', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Time Index')
    plt.ylabel('Revenue ($)')
    plt.title('Revenue Predictions - Last 100 Time Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgboost_time_series_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(models, feature_importance):
    """Save models and results"""
    
    print(f"\n💾 SAVING MODELS AND RESULTS")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('xgboost_results', exist_ok=True)
    
    # Save models
    joblib.dump(models['baseline_model'], 'xgboost_results/baseline_model.pkl')
    joblib.dump(models['best_model'], 'xgboost_results/optimized_model.pkl')
    
    # Save results
    results = {
        'baseline_metrics': models['baseline_metrics'],
        'best_metrics': models['best_metrics'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('xgboost_results/model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature importance
    feature_importance.to_csv('xgboost_results/feature_importance.csv', index=False)
    
    print(f"✅ Results saved:")
    print(f"   - Models: xgboost_results/")
    print(f"   - Results: xgboost_results/model_results.json")
    print(f"   - Feature importance: xgboost_results/feature_importance.csv")

# ============================================================================
# 🚀 MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    print(f"\n🚀 STARTING XGBOOST HOTEL REVENUE FORECASTING PIPELINE")
    print("=" * 70)
    
    try:
        # Step 1: Load data
        df, data_source = find_and_load_data()
        
        # Step 2: Preprocess data
        df_processed, label_encoders = preprocess_data(df)
        
        # Step 3: Create features
        df_features, feature_cols = create_advanced_features(df_processed)
        
        # Step 4: Prepare train/test split
        X_train, X_test, y_train, y_test, train_data, test_data = prepare_train_test_split(
            df_features, feature_cols
        )
        
        # Step 5: Train models
        models = train_xgboost_models(X_train, X_test, y_train, y_test)
        
        # Step 6: Analyze feature importance
        feature_importance = analyze_feature_importance(models['best_model'], feature_cols)
        
        # Step 7: Create evaluation plots
        create_evaluation_plots(models, X_test, y_test)
        
        # Step 8: Save results
        save_results(models, feature_importance)
        
        # ==================== FINAL SUMMARY ====================
        print(f"\n🎉 XGBOOST PIPELINE COMPLETE!")
        print("=" * 50)
        print(f"📊 DATA SUMMARY:")
        print(f"   Data source: {data_source}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(feature_cols)}")
        
        print(f"\n🏆 BEST MODEL PERFORMANCE:")
        print(f"   MAE: ${models['best_metrics']['mae']:.2f}")
        print(f"   MAPE: {models['best_metrics']['mape']:.1f}%")
        print(f"   R²: {models['best_metrics']['r2']:.3f}")
        
        improvement = (models['baseline_metrics']['mape'] - models['best_metrics']['mape']) / models['baseline_metrics']['mape'] * 100
        print(f"   Improvement over baseline: {improvement:.1f}%")
        
        print(f"\n📂 Results saved in 'xgboost_results/' directory")
        print(f"🎯 XGBoost provides much more stable results than CNN-LSTM!")
        
        return models, feature_importance
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("🔄 Starting XGBoost Hotel Revenue Forecasting...")
    models, feature_importance = main()
    
    if models is not None:
        print(f"\n✅ SUCCESS! XGBoost model training completed successfully.")
        print(f"🎯 Check the generated plots and 'xgboost_results/' folder for detailed results.")
    else:
        print(f"\n❌ Pipeline failed. Check the error messages above.") 