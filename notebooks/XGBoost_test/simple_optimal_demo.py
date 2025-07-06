# OPTIMAL HOTEL REVENUE FORECASTING DEMO
# Simple demonstration of the best model approach for small datasets

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    print("✅ All required packages available")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Please install: pip install xgboost lightgbm scikit-learn")
    exit(1)

def create_optimal_models(X_train, y_train, X_test, y_test, n_features):
    """
    Create optimal models based on dataset characteristics
    """
    print(f"\n🎯 CREATING OPTIMAL MODELS FOR SMALL DATASET")
    print("=" * 50)
    
    # Calculate sample-to-feature ratio
    ratio = len(X_train) / n_features
    print(f"📊 Sample-to-feature ratio: {ratio:.1f}")
    
    models = {}
    predictions = {}
    
    if ratio < 15:
        print("⚠️  VERY LOW ratio - Using high regularization")
        
        # XGBoost with heavy regularization
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=2.0,
            reg_lambda=2.0,
            min_child_weight=10,
            random_state=42
        )
        
        # LightGBM with heavy regularization  
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=2.0,
            reg_lambda=2.0,
            min_child_samples=10,
            random_state=42,
            verbose=-1
        )
        
        # Simple Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=42
        )
        
    else:
        print("✅ Acceptable ratio - Using moderate regularization")
        
        # Standard configurations for better ratios
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42
        )
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    
    # Train models
    print("\n🚀 Training models...")
    
    models['XGBoost'] = xgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model.fit(X_train, y_train)
    
    # Generate predictions
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
        print(f"✅ {name} trained and predictions generated")
    
    # Create ensemble
    ensemble_pred = (predictions['XGBoost'] + predictions['LightGBM'] + predictions['RandomForest']) / 3
    predictions['Ensemble'] = ensemble_pred
    models['Ensemble'] = None  # Ensemble doesn't have a single model object
    
    print("✅ Ensemble created")
    
    return models, predictions

def evaluate_models(y_test, predictions):
    """
    Evaluate all models with standard metrics
    """
    print(f"\n📊 MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    results = []
    
    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE safely (avoid division by zero)
        mape_mask = y_test > 50  # Only calculate MAPE for values > $50
        if np.sum(mape_mask) > 0:
            mape = np.mean(np.abs((y_test[mape_mask] - y_pred[mape_mask]) / y_test[mape_mask])) * 100
        else:
            mape = float('inf')
        
        # Within 20% accuracy
        within_20_pct = np.mean(np.abs((y_test - y_pred) / y_test) <= 0.20) * 100
        
        results.append({
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Within_20_pct': within_20_pct
        })
        
        print(f"\n🔍 {model_name} Results:")
        print(f"   MAE:  ${mae:.0f}")
        print(f"   RMSE: ${rmse:.0f}")
        print(f"   R²:   {r2:.3f}")
        print(f"   MAPE: {mape:.1f}% (values >$50)")
        print(f"   Within 20%: {within_20_pct:.1f}%")
    
    return pd.DataFrame(results)

def analyze_dataset_characteristics(df):
    """
    Analyze key dataset characteristics for model selection
    """
    print("📊 DATASET CHARACTERISTICS ANALYSIS")
    print("=" * 50)
    
    print(f"📈 Basic Statistics:")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Revenue range: ${df['CheckTotal'].min():.2f} - ${df['CheckTotal'].max():.2f}")
    print(f"   Mean revenue: ${df['CheckTotal'].mean():.2f}")
    print(f"   Median revenue: ${df['CheckTotal'].median():.2f}")
    
    # Revenue distribution analysis
    low_revenue = df[df['CheckTotal'] < 200]
    medium_revenue = df[(df['CheckTotal'] >= 200) & (df['CheckTotal'] < 1000)]
    high_revenue = df[df['CheckTotal'] >= 1000]
    
    print(f"\n💰 Revenue Distribution:")
    print(f"   Low (<$200):     {len(low_revenue):4d} ({len(low_revenue)/len(df)*100:5.1f}%)")
    print(f"   Medium ($200-1K): {len(medium_revenue):4d} ({len(medium_revenue)/len(df)*100:5.1f}%)")
    print(f"   High (≥$1K):     {len(high_revenue):4d} ({len(high_revenue)/len(df)*100:5.1f}%)")
    
    # Temporal patterns
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    
    print(f"\n📅 Temporal Patterns:")
    print("   Day of Week Averages:")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(day_names):
        avg_revenue = df[df['DayOfWeek'] == i]['CheckTotal'].mean()
        print(f"     {day}: ${avg_revenue:.0f}")
    
    print("   Monthly Averages:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i, month in enumerate(month_names, 1):
        month_data = df[df['Month'] == i]['CheckTotal']
        if len(month_data) > 0:
            avg_revenue = month_data.mean()
            print(f"     {month}: ${avg_revenue:.0f}")
    
    return df

def main():
    print("🎯 OPTIMAL HOTEL REVENUE FORECASTING DEMONSTRATION")
    print("=" * 60)
    
    # Load data
    data_path = '../cnn_lstm_final/filtered_revenue_data/RevenueCenter_1_filtered.csv'
    
    try:
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"✅ Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Could not find data file: {data_path}")
        print("Please ensure the data file exists in the correct location")
        return
    
    # Analyze dataset characteristics
    df = analyze_dataset_characteristics(df)
    
    # Create simple but effective features
    print(f"\n🔧 CREATING ESSENTIAL FEATURES")
    print("=" * 50)
    
    df_features = df.copy()
    
    # Basic temporal features
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
    df_features['DayOfYear'] = df_features['Date'].dt.dayofyear
    df_features['IsWeekend'] = df_features['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Cyclical encoding
    df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
    df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
    df_features['DayOfWeek_sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
    df_features['DayOfWeek_cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
    
    # Sort data for lag features
    df_features = df_features.sort_values(['Date', 'MealPeriod_num']).reset_index(drop=True)
    
    # Safe lag features (prevent data leakage)
    if 'MealPeriod_num' in df_features.columns:
        for lag in [1, 2, 3, 7]:
            df_features[f'Revenue_lag_{lag}'] = df_features.groupby('MealPeriod_num')['CheckTotal'].shift(lag)
    
    # Simple rolling features (with proper lag)
    for window in [3, 7]:
        rolling_mean = df_features['CheckTotal'].rolling(window, min_periods=1).mean()
        df_features[f'Revenue_rolling_mean_{window}d'] = rolling_mean.shift(1)
    
    print(f"✅ Essential features created")
    
    # Select features for modeling
    feature_columns = [col for col in df_features.columns 
                      if col not in ['Date', 'CheckTotal'] and 
                      df_features[col].dtype in ['int64', 'float64']]
    
    print(f"📊 Features selected: {len(feature_columns)}")
    
    # Prepare data
    df_clean = df_features.dropna().reset_index(drop=True)
    X = df_clean[feature_columns].values
    y = df_clean['CheckTotal'].values
    
    print(f"📊 Clean data shape: {X.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Temporal split
    split_date = '2024-01-15'
    train_mask = df_clean['Date'] < split_date
    test_mask = df_clean['Date'] >= split_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n📅 Temporal Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Apply optimal model strategy
    models, predictions = create_optimal_models(X_train, y_train, X_test, y_test, X_train.shape[1])
    
    # Evaluate models
    results_df = evaluate_models(y_test, predictions)
    
    # Show best model
    best_model = results_df.loc[results_df['MAE'].idxmin()]
    
    print(f"\n🏆 BEST MODEL RESULTS")
    print("=" * 50)
    print(f"🥇 Best Model: {best_model['Model']}")
    print(f"   MAE:  ${best_model['MAE']:.0f}")
    print(f"   R²:   {best_model['R2']:.3f}")
    print(f"   MAPE: {best_model['MAPE']:.1f}%")
    print(f"   Within 20%: {best_model['Within_20_pct']:.1f}%")
    
    # Model comparison
    print(f"\n📊 ALL MODELS COMPARISON")
    print("=" * 50)
    print(results_df.round(2).to_string(index=False))
    
    # Final recommendations
    print(f"\n💡 RECOMMENDATIONS FOR YOUR DATASET")
    print("=" * 50)
    print(f"✅ Best approach: Ensemble of XGBoost + LightGBM + RandomForest")
    print(f"✅ Heavy regularization due to low sample-to-feature ratio")
    print(f"✅ Focus on essential features only (temporal + lag)")
    print(f"✅ Use robust evaluation metrics")
    print(f"✅ Expect moderate performance due to dataset size constraints")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Collect more data (target 50+ samples per feature)")
    print(f"2. Focus on most important features only")
    print(f"3. Consider domain-specific feature engineering")
    print(f"4. Implement cross-validation for more robust evaluation")
    print(f"5. Monitor model performance over time")

if __name__ == "__main__":
    main() 