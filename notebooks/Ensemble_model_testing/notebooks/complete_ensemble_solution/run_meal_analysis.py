#!/usr/bin/env python3
"""
Hotel Revenue Forecasting with Meal Period Analysis
===================================================

This script runs the ensemble model and provides detailed analysis
by meal period (Breakfast, Lunch, Dinner) to show how the model
performs for each meal type separately.
"""

from ensemble_revenue_forecasting import HotelRevenueEnsemble

def main():
    """
    Run ensemble model with detailed meal period analysis
    """
    print("🍽️ HOTEL REVENUE FORECASTING - MEAL PERIOD ANALYSIS")
    print("=" * 80)
    
    # Initialize the ensemble model
    ensemble = HotelRevenueEnsemble(random_state=42)
    
    # Run the complete pipeline with meal analysis
    results = ensemble.run_complete_pipeline(
        data_path='../revenue_center_data/RevenueCenter_1_data.csv',
        output_dir='ensemble_model_output'
    )
    
    # Display meal period results summary
    print("\n🎯 MEAL PERIOD ANALYSIS SUMMARY:")
    print("=" * 80)
    
    if 'meal_metrics' in results:
        meal_metrics = results['meal_metrics']
        
        print(f"\n📊 PERFORMANCE COMPARISON BY MEAL PERIOD:")
        print("-" * 60)
        print(f"{'Meal Period':<12} {'R²':<8} {'MAE':<10} {'Avg Revenue':<15}")
        print("-" * 60)
        
        for meal, metrics in meal_metrics.items():
            print(f"{meal:<12} {metrics['R²']:<8.3f} ${metrics['MAE']:<9.2f} ${metrics['Avg_Actual']:<14.2f}")
        
        # Find best and worst performing meal periods
        best_meal = max(meal_metrics.keys(), key=lambda x: meal_metrics[x]['R²'])
        worst_meal = min(meal_metrics.keys(), key=lambda x: meal_metrics[x]['R²'])
        
        print(f"\n🏆 BEST PERFORMING MEAL: {best_meal}")
        print(f"   R² = {meal_metrics[best_meal]['R²']:.3f}")
        print(f"   MAE = ${meal_metrics[best_meal]['MAE']:.2f}")
        
        print(f"\n⚠️  IMPROVEMENT OPPORTUNITY: {worst_meal}")
        print(f"   R² = {meal_metrics[worst_meal]['R²']:.3f}")
        print(f"   MAE = ${meal_metrics[worst_meal]['MAE']:.2f}")
        
        # Revenue insights
        print(f"\n💰 REVENUE INSIGHTS:")
        print("-" * 40)
        total_revenue = sum(metrics['Avg_Actual'] * metrics['Count'] for metrics in meal_metrics.values())
        total_samples = sum(metrics['Count'] for metrics in meal_metrics.values())
        
        for meal, metrics in meal_metrics.items():
            revenue_share = (metrics['Avg_Actual'] * metrics['Count']) / total_revenue * 100
            sample_share = metrics['Count'] / total_samples * 100
            
            print(f"{meal}: {revenue_share:.1f}% of revenue, {sample_share:.1f}% of transactions")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Check the following files for detailed results:")
    print(f"   📊 meal_period_performance_metrics.csv")
    print(f"   📈 predictions_by_meal_period.csv")
    print(f"   🖼️  meal_period_analysis.png")
    print(f"   🖼️  meal_period_time_series.png")
    print(f"   🖼️  meal_period_weekly_patterns.png")

if __name__ == "__main__":
    main() 