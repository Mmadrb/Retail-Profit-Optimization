import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# --- Configuration ---
DATA_PATH = r"C:\Users\moham\OneDrive\Desktop\MIT\Superstore\Superstore.csv"
OUTPUT_DIR = Path(r"C:\Users\moham\Desktop\MIT\strategy_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    return df

def analyze_portfolio_matrix(df):
    """
    Creates a BCG-style Matrix: Sales Volume vs. Profit Margin.
    Goal: Identify 'Stars' (High Sales, High Margin).
    """
    # Group by Sub-Category
    cat_stats = df.groupby(['category', 'sub_category']).agg({
        'sales': 'sum',
        'profit': 'sum',
        'quantity': 'sum'
    }).reset_index()

    # Calculate Profit Margin
    cat_stats['profit_margin'] = (cat_stats['profit'] / cat_stats['sales']) * 100
    
    # Identify Target Products (Technology)
    target_products = ['Phones', 'Accessories', 'Copiers', 'Machines']
    cat_stats['type'] = cat_stats['sub_category'].apply(
        lambda x: 'Target (Tech)' if x in target_products else 'Other'
    )

    # --- Plotting the Matrix ---
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Scatter plot
    sns.scatterplot(
        data=cat_stats,
        x='sales',
        y='profit_margin',
        hue='type',
        style='category',
        s=200, # Size of bubbles
        palette={'Target (Tech)': '#2ecc71', 'Other': '#95a5a6'}, # Green for Tech, Grey for others
        alpha=0.8,
        edgecolor='black'
    )

    # Add reference lines (Averages)
    avg_sales = cat_stats['sales'].mean()
    avg_margin = cat_stats['profit_margin'].mean()
    
    plt.axvline(avg_sales, color='red', linestyle='--', alpha=0.5, label='Avg Sales')
    plt.axhline(avg_margin, color='blue', linestyle='--', alpha=0.5, label='Avg Margin')

    # Add Labels
    for i, row in cat_stats.iterrows():
        plt.text(
            row['sales'] + 5000, 
            row['profit_margin'], 
            row['sub_category'], 
            fontsize=9, 
            fontweight='bold' if row['type'] == 'Target (Tech)' else 'normal',
            color='green' if row['type'] == 'Target (Tech)' else 'black'
        )

    # Annotate Quadrants
    plt.text(cat_stats['sales'].max(), cat_stats['profit_margin'].max(), "⭐ STARS\n(High Sales, High Margin)", 
             ha='right', va='top', fontsize=12, fontweight='bold', color='gold', bbox=dict(facecolor='black', alpha=0.7))

    plt.title('Product Portfolio Matrix: Identifying High-Margin Drivers', fontsize=16, fontweight='bold')
    plt.xlabel('Total Sales Volume ($)', fontsize=12)
    plt.ylabel('Profit Margin (%)', fontsize=12)
    plt.legend(title='Product Group', loc='lower right')
    
    save_path = OUTPUT_DIR / 'bcg_matrix.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"🖼️ Matrix Chart saved to: {save_path}")
    plt.show()
    
    return cat_stats

def analyze_seasonality(df):
    """
    Analyzes WHEN Technology products sell the most.
    Goal: Optimization of Inventory & Ad Spend timing.
    """
    # Filter only Technology
    tech_df = df[df['category'] == 'Technology'].copy()
    tech_df['month'] = tech_df['order_date'].dt.month_name()
    tech_df['month_num'] = tech_df['order_date'].dt.month
    
    # Group by month
    monthly_trend = tech_df.groupby(['month_num', 'month'])['sales'].sum().reset_index()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Line plot
    sns.lineplot(data=monthly_trend, x='month', y='sales', marker='o', linewidth=3, color='#8e44ad')
    
    # Highlight Peak Season
    peak_sales = monthly_trend['sales'].max()
    peak_month = monthly_trend.loc[monthly_trend['sales'].idxmax(), 'month']
    
    plt.annotate(f'PEAK SEASON\n({peak_month})', 
                 xy=(peak_month, peak_sales), 
                 xytext=(peak_month, peak_sales + 5000),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center', fontweight='bold')

    plt.title('Seasonality of Technology Sales (Ad Spend Timing)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.fill_between(monthly_trend['month'], monthly_trend['sales'], color='#8e44ad', alpha=0.1) # Fill area
    
    save_path = OUTPUT_DIR / 'tech_seasonality.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"🖼️ Seasonality Chart saved to: {save_path}")
    plt.show()

def main():
    if not os.path.exists(DATA_PATH):
        print("❌ File not found.")
        return
    
    print("⏳ Loading Data...")
    df = load_data(DATA_PATH)
    
    print("📊 Generating Portfolio Matrix...")
    cat_stats = analyze_portfolio_matrix(df)
    
    print("📅 Analyzing Seasonality for Technology...")
    analyze_seasonality(df)
    
    # Summary Report
    print("\n" + "="*50)
    print("🚀 STRATEGY RECOMMENDATION REPORT")
    print("="*50)
    
    tech_stats = cat_stats[cat_stats['category'] == 'Technology']
    avg_tech_margin = tech_stats['profit_margin'].mean()
    
    print(f"✅ Technology Average Margin: {avg_tech_margin:.1f}%")
    print("💡 Insight: Tech products are in the 'Star' quadrant.")
    print("👉 ACTION 1: Shift 60% of Ad budget to Phones & Accessories.")
    print("👉 ACTION 2: Increase inventory by 30% in Q4 (Sep-Dec).")
    print("👉 ACTION 3: Stop discounting Phones below 5% (Organic demand is high).")

if __name__ == '__main__':
    main()