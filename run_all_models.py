"""
Run All Models and Generate Report Figures

This script runs all 8 implemented models on the Bitcoin price prediction task
and generates comprehensive visualizations for the capstone report.

Models:
  Baseline:  ARIMA, XGBoost, LightGBM
  Proposed:  SE-GRN, iTransformer, Times-FM, SOFTS, CNN-LSTM

Training setup (from the paper):
  - Training data: 62,809 time points (prior to 2024-01-01)
  - Prediction horizon: next 100 time steps
  - Sequence length: 100 hours for deep learning models

Usage:
    python run_all_models.py

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - NYU Capstone Project
Date: 2024
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# Create output directories
FIGURES_DIR = 'figures'
RESULTS_DIR = 'results'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_baseline_models():
    """Run all baseline models and return results."""
    results = {}
    
    # XGBoost
    print("\n" + "="*60)
    print("Running XGBoost...")
    print("="*60)
    try:
        from xgboost_model import run_xgboost_experiment
        start_time = time.time()
        results['XGBoost'] = run_xgboost_experiment()
        results['XGBoost']['time'] = time.time() - start_time
        print(f"XGBoost completed in {results['XGBoost']['time']:.2f}s")
    except Exception as e:
        print(f"XGBoost failed: {e}")
    
    # LightGBM
    print("\n" + "="*60)
    print("Running LightGBM...")
    print("="*60)
    try:
        from lightgbm_model import run_lightgbm_experiment
        start_time = time.time()
        results['LightGBM'] = run_lightgbm_experiment()
        results['LightGBM']['time'] = time.time() - start_time
        print(f"LightGBM completed in {results['LightGBM']['time']:.2f}s")
    except Exception as e:
        print(f"LightGBM failed: {e}")
    
    # ARIMA (smaller sample for speed)
    print("\n" + "="*60)
    print("Running ARIMA (may take longer)...")
    print("="*60)
    try:
        from arima_model import run_arima_experiment
        start_time = time.time()
        results['ARIMA'] = run_arima_experiment()
        results['ARIMA']['time'] = time.time() - start_time
        print(f"ARIMA completed in {results['ARIMA']['time']:.2f}s")
    except Exception as e:
        print(f"ARIMA failed: {e}")
    
    return results


def run_deep_learning_models():
    """Run all deep learning models and return results."""
    results = {}
    
    # iTransformer
    print("\n" + "="*60)
    print("Running iTransformer...")
    print("="*60)
    try:
        from itransformer_model import run_itransformer_experiment
        start_time = time.time()
        results['iTransformer'] = run_itransformer_experiment(epochs=30)
        results['iTransformer']['time'] = time.time() - start_time
        print(f"iTransformer completed in {results['iTransformer']['time']:.2f}s")
    except Exception as e:
        print(f"iTransformer failed: {e}")
    
    # SE-GRN
    print("\n" + "="*60)
    print("Running SE-GRN...")
    print("="*60)
    try:
        from se_grn_model import run_segrn_experiment
        start_time = time.time()
        results['SE-GRN'] = run_segrn_experiment(epochs=30)
        results['SE-GRN']['time'] = time.time() - start_time
        print(f"SE-GRN completed in {results['SE-GRN']['time']:.2f}s")
    except Exception as e:
        print(f"SE-GRN failed: {e}")
    
    # CNN-LSTM
    print("\n" + "="*60)
    print("Running CNN-LSTM...")
    print("="*60)
    try:
        from cnn_lstm_model import run_cnn_lstm_experiment
        start_time = time.time()
        results['CNN-LSTM'] = run_cnn_lstm_experiment(epochs=30)
        results['CNN-LSTM']['time'] = time.time() - start_time
        print(f"CNN-LSTM completed in {results['CNN-LSTM']['time']:.2f}s")
    except Exception as e:
        print(f"CNN-LSTM failed: {e}")
    
    # SOFTS
    print("\n" + "="*60)
    print("Running SOFTS...")
    print("="*60)
    try:
        from softs_model import run_softs_experiment
        start_time = time.time()
        results['SOFTS'] = run_softs_experiment(epochs=30)
        results['SOFTS']['time'] = time.time() - start_time
        print(f"SOFTS completed in {results['SOFTS']['time']:.2f}s")
    except Exception as e:
        print(f"SOFTS failed: {e}")
    
    # Times-FM Inspired
    print("\n" + "="*60)
    print("Running Times-FM Inspired...")
    print("="*60)
    try:
        from times_fm_model import run_timesfm_experiment
        start_time = time.time()
        results['Times-FM'] = run_timesfm_experiment(epochs=30)
        results['Times-FM']['time'] = time.time() - start_time
        print(f"Times-FM completed in {results['Times-FM']['time']:.2f}s")
    except Exception as e:
        print(f"Times-FM failed: {e}")
    
    return results


def generate_all_figures(all_results):
    """Generate all figures for the report."""
    from visualization import (
        generate_all_figures as gen_model_figs,
        plot_model_comparison,
        create_metrics_table
    )
    
    # Generate individual model figures
    for model_name, results in all_results.items():
        print(f"\nGenerating figures for {model_name}...")
        try:
            gen_model_figs(results, model_name, output_dir=FIGURES_DIR)
        except Exception as e:
            print(f"  Failed to generate figures for {model_name}: {e}")
    
    # Generate comparison figures
    print("\nGenerating model comparison figures...")
    
    for metric in ['MSE', 'MAE', 'RMSE']:
        try:
            plot_model_comparison(
                all_results, metric,
                save_path=os.path.join(FIGURES_DIR, f'comparison_{metric.lower()}.png')
            )
        except Exception as e:
            print(f"  Failed to generate {metric} comparison: {e}")
    
    # Create metrics table
    try:
        metrics_df = create_metrics_table(all_results)
        metrics_df.to_csv(os.path.join(RESULTS_DIR, 'metrics_comparison.csv'), index=False)
        print(f"\nMetrics table saved to {RESULTS_DIR}/metrics_comparison.csv")
        print("\nModel Comparison:")
        print(metrics_df.to_string(index=False))
    except Exception as e:
        print(f"Failed to create metrics table: {e}")


def main():
    """Main execution function."""
    print("="*60)
    print("BITCOIN PRICE PREDICTION - FULL MODEL EVALUATION")
    print("Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu")
    print("Group 36 - Capstone Project")
    print("="*60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # Run baseline models
    print("\n" + "#"*60)
    print("# BASELINE MODELS")
    print("#"*60)
    baseline_results = run_baseline_models()
    all_results.update(baseline_results)
    
    # Run deep learning models
    print("\n" + "#"*60)
    print("# DEEP LEARNING MODELS")
    print("#"*60)
    dl_results = run_deep_learning_models()
    all_results.update(dl_results)
    
    # Generate all figures
    print("\n" + "#"*60)
    print("# GENERATING FIGURES")
    print("#"*60)
    generate_all_figures(all_results)
    
    # Final summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFigures saved to: {FIGURES_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/")
    
    # Print best model
    if all_results:
        best_model = min(all_results.keys(), 
                        key=lambda x: all_results[x]['metrics']['MAE'])
        print(f"\nBest Model (by MAE): {best_model}")
        print(f"  MAE: {all_results[best_model]['metrics']['MAE']:.4f}")
        print(f"  MSE: {all_results[best_model]['metrics']['MSE']:.4f}")
        print(f"  RMSE: {all_results[best_model]['metrics']['RMSE']:.4f}")


if __name__ == '__main__':
    main()

