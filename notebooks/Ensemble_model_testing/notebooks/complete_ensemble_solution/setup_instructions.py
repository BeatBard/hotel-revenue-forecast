#!/usr/bin/env python3
"""
Setup Instructions for Hotel Revenue Forecasting Ensemble
=========================================================

This script helps you set up and run the ensemble model.
"""

import os
import sys


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'lightgbm', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n🎉 All dependencies are installed!")
    return True


def check_data_file():
    """Check if the data file exists"""
    data_path = 'data/RevenueCenter_1_data.csv'
    if os.path.exists(data_path):
        print(f"✅ Data file found: {data_path}")
        return True
    else:
        print(f"❌ Data file missing: {data_path}")
        print("Please ensure the data file is in the correct location.")
        return False


def main():
    """Main setup function"""
    print("🚀 Hotel Revenue Forecasting - Setup Check")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}")
    else:
        print(f"⚠️ Python {python_version.major}.{python_version.minor}")
        print("Recommended: Python 3.8+")
    
    print("\n📦 Checking Dependencies:")
    deps_ok = check_dependencies()
    
    print("\n📁 Checking Data:")
    data_ok = check_data_file()
    
    if deps_ok and data_ok:
        print("\n🎉 Setup Complete! You can now run:")
        print("   python ensemble_revenue_forecasting.py")
        print("   or")
        print("   python run_example.py")
    else:
        print("\n⚠️ Setup incomplete. Please fix the issues above.")


if __name__ == "__main__":
    main() 