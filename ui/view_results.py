#!/usr/bin/env python3
import json
import os
from pathlib import Path

def view_results():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ No results directory found")
        return
    
    # Find JSON result files
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No JSON result files found")
        return
    
    # Show latest results
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📊 Latest Results: {latest_file.name}")
    print("="*50)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Display key metrics
    if 'gender_distribution' in results:
        print("\n📈 Gender Distribution:")
        for gender, count in results['gender_distribution'].items():
            print(f"  {gender}: {count}")
    
    if 'bias_analysis' in results:
        print(f"\n⚠️ Bias Score: {results['bias_analysis'].get('bias_score', 'N/A')}")
    
    print(f"\n📁 Full results in: {latest_file}")

if __name__ == "__main__":
    view_results()
