# src/clean_data.py
import pandas as pd
import numpy as np
import os

def clean_and_build_pipeline():
    print("⏳ Day 1: Starting Data Engineering Pipeline...")
    
    # 1. Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("📁 Created 'data' directory. Please place your raw CSV files inside it.")
        return

    # 2. Read Datasets safely
    try:
        battery_panel = pd.read_csv("data/ev_battery_panel.csv")
        chargers_summary = pd.read_csv("data/chargers_summary.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Ensure raw CSVs are placed inside the 'data/' folder.")
        return

    # 3. Standardize column names for merging
    # 'chargers_summary' uses country_iso3, 'ev_battery_panel' uses country_iso
    chargers_summary = chargers_summary.rename(columns={'country_iso3': 'country_iso'})
    
    # 4. Handle Missing Values (Imputation & Cleaning)
    # Group by country and use Forward Fill to propagate historical data forward, replace remaining with 0
    battery_panel['ev_stock'] = battery_panel.groupby('country_iso')['ev_stock'].ffill().fillna(0)
    battery_panel['ev_sales'] = battery_panel.groupby('country_iso')['ev_sales'].ffill().fillna(0)
    battery_panel['battery_demand_mwh_est'] = battery_panel.groupby('country_iso')['battery_demand_mwh_est'].ffill().fillna(0)
    
    # Fill specific critical missing metrics in chargers summary
    chargers_summary['total_stations'] = chargers_summary['total_stations'].fillna(0)
    chargers_summary['avg_power_kw'] = chargers_summary.groupby('country_iso')['avg_power_kw'].transform(lambda x: x.fillna(x.mean())).fillna(22.0)

    # 5. Merge Datasets on relational composite keys (Country & Year)
    print("🤝 Merging charging infrastructure with battery demand arrays...")
    master_df = pd.merge(chargers_summary, battery_panel, on=['country_iso', 'year'], how='inner')

    # 6. Advanced Feature Engineering: Lag Features
    # What happened in the previous year is the strongest predictor of infrastructure growth this year
    master_df = master_df.sort_values(by=['country_iso', 'year'])
    master_df['prev_year_stations'] = master_df.groupby('country_iso')['total_stations'].shift(1)
    master_df['prev_year_ev_stock'] = master_df.groupby('country_iso')['ev_stock'].shift(1)
    
    # Drop rows where lag features don't exist (e.g., the first year of data for a country)
    master_df.dropna(subset=['prev_year_stations', 'prev_year_ev_stock'], inplace=True)

    # 7. Save Clean Export
    master_df.to_csv("data/cleaned_master_data.csv", index=False)
    print("✅ Day 1 Complete! Clean master matrix saved to 'data/cleaned_master_data.csv'")

if __name__ == "__main__":
    clean_and_build_pipeline()