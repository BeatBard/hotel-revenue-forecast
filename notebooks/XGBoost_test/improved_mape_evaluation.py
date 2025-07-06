# IMPROVED EVALUATION METRICS FOR HOTEL REVENUE FORECASTING
# Handles extreme MAPE values caused by very low revenue periods

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def calculate_robust_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate robust metrics that handle extreme MAPE values properly
    """
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 1. ROBUST MAPE CALCULATION
    # Filter out very low values that cause extreme MAPE
    min_threshold = 100  # $100 minimum for reliable percentage calculation
    robust_mask = y_true >= min_threshold
    
    if np.sum(robust_mask) > 0:
        robust_mape = np.mean(np.abs((y_true[robust_mask] - y_pred[robust_mask]) / y_true[robust_mask])) * 100
        robust_sample_size = np.sum(robust_mask)
    else:
        robust_mape = np.nan
        robust_sample_size = 0
    
    # 2. TRADITIONAL MAPE (for comparison)
    mape_mask = y_true > 0
    if np.sum(mape_mask) > 0:
        traditional_mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
    else:
        traditional_mape = np.nan
    
    # 3. SYMMETRIC MAPE (SMAPE) - handles low values better
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # 4. MEDIAN ABSOLUTE PERCENTAGE ERROR (less sensitive to outliers)
    medape = np.median(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100 if np.sum(mape_mask) > 0 else np.nan
    
    # 5. ACCURACY WITHIN THRESHOLDS
    within_10_pct = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 10)) <= 0.1) * 100
    within_20_pct = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 10)) <= 0.2) * 100
    within_50_pct = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 10)) <= 0.5) * 100
    
    # 6. VALUE RANGE ANALYSIS
    low_values = y_true < 200
    medium_values = (y_true >= 200) & (y_true < 1000)
    high_values = y_true >= 1000
    
    low_mae = mae if np.sum(low_values) == 0 else mean_absolute_error(y_true[low_values], y_pred[low_values])
    medium_mae = mae if np.sum(medium_values) == 0 else mean_absolute_error(y_true[medium_values], y_pred[medium_values])
    high_mae = mae if np.sum(high_values) == 0 else mean_absolute_error(y_true[high_values], y_pred[high_values])
    
    # Results dictionary
    metrics = {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Traditional_MAPE': traditional_mape,
        'Robust_MAPE': robust_mape,
        'SMAPE': smape,
        'MedAPE': medape,
        'Within_10_pct': within_10_pct,
        'Within_20_pct': within_20_pct,
        'Within_50_pct': within_50_pct,
        'Low_MAE': low_mae,
        'Medium_MAE': medium_mae,
        'High_MAE': high_mae,
        'Low_Count': np.sum(low_values),
        'Medium_Count': np.sum(medium_values),
        'High_Count': np.sum(high_values),
        'Robust_Sample_Size': robust_sample_size
    }
    
    return metrics

def print_detailed_evaluation(metrics):
    """Print comprehensive evaluation results"""
    print(f"\n📊 DETAILED EVALUATION: {metrics['Model']}")
    print("=" * 60)
    
    print(f"\n🎯 CORE METRICS:")
    print(f"   MAE:  ${metrics['MAE']:.2f}")
    print(f"   RMSE: ${metrics['RMSE']:.2f}")
    print(f"   R²:   {metrics['R2']:.4f}")
    
    print(f"\n📈 PERCENTAGE ERROR METRICS:")
    print(f"   Traditional MAPE: {metrics['Traditional_MAPE']:.1f}% (includes all values)")
    print(f"   Robust MAPE:      {metrics['Robust_MAPE']:.1f}% (≥$100 values only, n={metrics['Robust_Sample_Size']})")
    print(f"   SMAPE:            {metrics['SMAPE']:.1f}% (symmetric, handles low values)")
    print(f"   Median APE:       {metrics['MedAPE']:.1f}% (less sensitive to outliers)")
    
    print(f"\n✅ ACCURACY WITHIN THRESHOLDS:")
    print(f"   Within 10%: {metrics['Within_10_pct']:.1f}%")
    print(f"   Within 20%: {metrics['Within_20_pct']:.1f}%")
    print(f"   Within 50%: {metrics['Within_50_pct']:.1f}%")
    
    print(f"\n💰 PERFORMANCE BY VALUE RANGE:")
    print(f"   Low (<$200):     MAE=${metrics['Low_MAE']:.2f}     (n={metrics['Low_Count']})")
    print(f"   Medium ($200-1K): MAE=${metrics['Medium_MAE']:.2f}  (n={metrics['Medium_Count']})")
    print(f"   High (≥$1000):   MAE=${metrics['High_MAE']:.2f}    (n={metrics['High_Count']})")

def analyze_prediction_errors(y_true, y_pred, title="Prediction Analysis"):
    """Analyze where predictions go wrong"""
    print(f"\n🔍 {title}")
    print("=" * 50)
    
    errors = np.abs(y_true - y_pred)
    pct_errors = np.abs((y_true - y_pred) / np.maximum(y_true, 1)) * 100
    
    # Find problematic predictions
    extreme_errors = pct_errors > 100  # >100% error
    
    print(f"\n⚠️  EXTREME ERRORS (>100%):")
    print(f"   Count: {np.sum(extreme_errors)}/{len(y_true)} ({np.mean(extreme_errors)*100:.1f}%)")
    
    if np.sum(extreme_errors) > 0:
        print(f"   Avg actual value: ${np.mean(y_true[extreme_errors]):.2f}")
        print(f"   Avg predicted value: ${np.mean(y_pred[extreme_errors]):.2f}")
        print(f"   Avg percentage error: {np.mean(pct_errors[extreme_errors]):.1f}%")
        
        # Show worst cases
        worst_indices = np.argsort(pct_errors)[-5:]
        print(f"\n   WORST 5 PREDICTIONS:")
        for i, idx in enumerate(worst_indices):
            print(f"   {i+1}. Actual: ${y_true[idx]:.2f} | Predicted: ${y_pred[idx]:.2f} | Error: {pct_errors[idx]:.1f}%")

def compare_models_robust(results_list):
    """Compare multiple models using robust metrics"""
    print(f"\n🏆 MODEL COMPARISON (ROBUST METRICS)")
    print("=" * 80)
    
    df_results = pd.DataFrame(results_list)
    
    # Sort by robust MAPE (most important for business)
    df_results = df_results.sort_values('Robust_MAPE')
    
    print(f"{'Model':<20} {'MAE':<8} {'R²':<8} {'Robust_MAPE':<12} {'SMAPE':<8} {'Within_20%':<10}")
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        print(f"{row['Model']:<20} ${row['MAE']:<7.0f} {row['R2']:<7.3f} "
              f"{row['Robust_MAPE']:<11.1f}% {row['SMAPE']:<7.1f}% {row['Within_20_pct']:<9.1f}%")
    
    # Best model recommendation
    best_model = df_results.iloc[0]
    print(f"\n🥇 BEST MODEL: {best_model['Model']}")
    print(f"   Primary metric (Robust MAPE): {best_model['Robust_MAPE']:.1f}%")
    print(f"   Business-friendly interpretation: {best_model['Within_20_pct']:.1f}% of predictions within 20%")

def create_evaluation_visualizations(y_true, y_pred, model_name="Model"):
    """Create comprehensive evaluation visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} - Comprehensive Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Predictions vs Actual
    axes[0,0].scatter(y_true, y_pred, alpha=0.6, s=30)
    axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual Revenue ($)')
    axes[0,0].set_ylabel('Predicted Revenue ($)')
    axes[0,0].set_title('Predictions vs Actual')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_pred - y_true
    axes[0,1].scatter(y_true, residuals, alpha=0.6, s=30)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Actual Revenue ($)')
    axes[0,1].set_ylabel('Residuals ($)')
    axes[0,1].set_title('Residual Plot')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Percentage Errors Distribution
    pct_errors = np.abs((y_true - y_pred) / np.maximum(y_true, 1)) * 100
    axes[0,2].hist(pct_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0,2].set_xlabel('Absolute Percentage Error (%)')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('Distribution of Percentage Errors')
    axes[0,2].axvline(x=20, color='r', linestyle='--', label='20% threshold')
    axes[0,2].legend()
    
    # 4. Error by Value Range
    value_ranges = ['<$200', '$200-1K', '≥$1K']
    low_mask = y_true < 200
    medium_mask = (y_true >= 200) & (y_true < 1000)
    high_mask = y_true >= 1000
    
    range_errors = [
        np.mean(np.abs(y_true[low_mask] - y_pred[low_mask])) if np.sum(low_mask) > 0 else 0,
        np.mean(np.abs(y_true[medium_mask] - y_pred[medium_mask])) if np.sum(medium_mask) > 0 else 0,
        np.mean(np.abs(y_true[high_mask] - y_pred[high_mask])) if np.sum(high_mask) > 0 else 0
    ]
    
    axes[1,0].bar(value_ranges, range_errors, alpha=0.7, color=['lightcoral', 'lightblue', 'lightgreen'])
    axes[1,0].set_ylabel('Mean Absolute Error ($)')
    axes[1,0].set_title('MAE by Value Range')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Actual vs Predicted Time Series (if we have enough points)
    if len(y_true) >= 50:
        sample_indices = np.linspace(0, len(y_true)-1, min(100, len(y_true)), dtype=int)
        axes[1,1].plot(sample_indices, y_true[sample_indices], 'o-', label='Actual', alpha=0.7)
        axes[1,1].plot(sample_indices, y_pred[sample_indices], 's-', label='Predicted', alpha=0.7)
        axes[1,1].set_xlabel('Sample Index')
        axes[1,1].set_ylabel('Revenue ($)')
        axes[1,1].set_title('Sample Time Series Comparison')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Not enough data\nfor time series plot', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    # 6. Accuracy Summary
    within_thresholds = [10, 20, 30, 50]
    accuracy_rates = []
    for threshold in within_thresholds:
        accuracy = np.mean(pct_errors <= threshold) * 100
        accuracy_rates.append(accuracy)
    
    axes[1,2].bar([f'{t}%' for t in within_thresholds], accuracy_rates, alpha=0.7, color='skyblue')
    axes[1,2].set_ylabel('Accuracy Rate (%)')
    axes[1,2].set_title('Prediction Accuracy by Threshold')
    axes[1,2].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

# EXAMPLE USAGE
if __name__ == "__main__":
    print("🧪 IMPROVED EVALUATION METRICS LOADED")
    print("=" * 50)
    print("📋 Available functions:")
    print("   • calculate_robust_metrics(y_true, y_pred, model_name)")
    print("   • print_detailed_evaluation(metrics)")
    print("   • analyze_prediction_errors(y_true, y_pred, title)")
    print("   • compare_models_robust(results_list)")
    print("   • create_evaluation_visualizations(y_true, y_pred, model_name)")
    print("\n💡 Key improvements:")
    print("   • Robust MAPE excludes very low values (<$100)")
    print("   • SMAPE handles low values better")
    print("   • Value range analysis shows where model struggles")
    print("   • Business-friendly accuracy thresholds") 