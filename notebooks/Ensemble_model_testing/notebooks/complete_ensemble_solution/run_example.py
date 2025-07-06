#!/usr/bin/env python3
"""
Simple Example: Hotel Revenue Forecasting Ensemble Model
========================================================

This script demonstrates how to use the ensemble model for hotel revenue forecasting.
"""

from ensemble_revenue_forecasting import HotelRevenueEnsemble
import pandas as pd
import numpy as np

def main():
    """
    Run a simple example of the ensemble model
    """
    print("🚀 Hotel Revenue Forecasting - Simple Example")
    print("=" * 60)
    
    # Initialize the ensemble model
    ensemble = HotelRevenueEnsemble(random_state=42)
    
    # Run the complete pipeline
    print("📊 Running complete ensemble pipeline...")
    results = ensemble.run_complete_pipeline(
        data_path='data/RevenueCenter_1_data.csv',
        output_dir='example_output'
    )
    
    print("\n🎉 Example completed successfully!")
    print("📁 Check 'example_output/' directory for results")
    print("📈 Check generated PNG files for visualizations")
    print("📋 Check 'predictions_vs_actual_complete.csv' for predictions")
    
    # Display key results
    evaluation_df = results['evaluation']
    
    print("\n📊 Key Performance Metrics:")
    print("-" * 40)
    
    # Best individual model
    test_results = evaluation_df[evaluation_df['Model'].str.contains('Test')]
    best_model = test_results.loc[test_results['R²'].idxmax()]
    
    print(f"🏆 Best Individual Model: {best_model['Model'].replace('_Test', '')}")
    print(f"   Test R²: {best_model['R²']:.4f}")
    print(f"   Test MAE: ${best_model['MAE']:.2f}")
    
    # Best ensemble
    ensemble_results = test_results[test_results['Model'].str.contains('Average|Ensemble')]
    if not ensemble_results.empty:
        best_ensemble = ensemble_results.loc[ensemble_results['R²'].idxmax()]
        print(f"🎯 Best Ensemble: {best_ensemble['Model'].replace('_Test', '')}")
        print(f"   Test R²: {best_ensemble['R²']:.4f}")
        print(f"   Test MAE: ${best_ensemble['MAE']:.2f}")
    
    print("\n✅ Example completed! Check the output files for detailed results.")

if __name__ == "__main__":
    main() 