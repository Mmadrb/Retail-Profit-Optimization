import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
DATA_PATH = r"C:\Users\moham\OneDrive\Desktop\project\Data\Superstore.csv"
OUTPUT_DIR = Path(r"C:\Users\moham\OneDrive\Desktop\MIT\Superstore\executive_dashboard_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# تنظیمات استایل 
sns.set_style("white")
plt.rcParams['figure.dpi'] = 150 

def load_data(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

# ==========================================
# 1. نمودار Kill List 
# ==========================================
def plot_low_margin_products(df):
    prod_stats = df.groupby('product_name').agg({
        'sales': 'sum', 'profit': 'sum', 'quantity': 'sum'
    }).reset_index()
    
    prod_stats = prod_stats[prod_stats['quantity'] > 10]
    prod_stats['margin'] = (prod_stats['profit'] / prod_stats['sales']) * 100
    worst_products = prod_stats.sort_values('margin', ascending=True).head(10)
    worst_products['short_name'] = worst_products['product_name'].apply(lambda x: x[:35] + '...' if len(x)>35 else x)

    plt.figure(figsize=(10, 6))
    
    bars = sns.barplot(
        data=worst_products, 
        x='margin', 
        y='short_name', 
        hue='short_name', # اضافه شده
        palette='Reds_r', 
        legend=False,     # اضافه شده
        edgecolor='black',
        linewidth=0.8
    )
    
    plt.title('The Kill List: Bottom 10 Products\n(Lowest Profit Margin)', fontsize=14, fontweight='bold', color='#c0392b', y=1.05)
    plt.xlabel('Profit Margin (%)', fontsize=8, labelpad=15)
    plt.ylabel('Product Name', fontsize=8, labelpad=15)
    plt.axvline(0, color='black', linewidth=1)
    
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    sns.despine(left=True, bottom=True)

    for i in bars.containers:
        bars.bar_label(i, fmt='%.1f%%', padding=5, fontweight='bold', color="#e26153", fontsize=8)

    plt.subplots_adjust(left=0.30, right=0.90, top=0.80, bottom=0.15)
    
    save_path = OUTPUT_DIR / '1_low_margin_products.png'
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ Chart 1 saved.")
    # plt.show() را حذف کردم تا برنامه متوقف نشود، اگر خواستید آن‌کامنت کنید
    plt.close()

# ==========================================
# 2. ماتریس BCG 
# ==========================================
def plot_bcg_matrix(df):
    cat_stats = df.groupby('sub_category').agg({'sales': 'sum', 'profit': 'sum'}).reset_index()
    cat_stats['margin'] = (cat_stats['profit'] / cat_stats['sales']) * 100
    
    avg_sales = cat_stats['sales'].mean()
    avg_margin = cat_stats['margin'].mean()
    
    def get_quadrant(row):
        if row['sales'] > avg_sales and row['margin'] > avg_margin: return "Star (High Growth/Profit)"
        if row['sales'] < avg_sales and row['margin'] < avg_margin: return "Dog (Low Growth/Profit)"
        if row['sales'] > avg_sales and row['margin'] < avg_margin: return "Volume Driver (Low Margin)"
        return "Opportunity (High Margin)"

    cat_stats['Quadrant'] = cat_stats.apply(get_quadrant, axis=1)
    
    quadrant_colors = {
        "Star (High Growth/Profit)": "#12da66",
        "Dog (Low Growth/Profit)": "#858082",
        "Volume Driver (Low Margin)": "#e67e22",
        "Opportunity (High Margin)": "#3498db"
    }

    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=cat_stats, 
        x='sales', 
        y='margin',
        size='sales',     # استفاده استاندارد از سایز
        sizes=(100, 1000), # محدوده سایز حباب‌ها
        hue='Quadrant',
        palette=quadrant_colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    plt.axvline(avg_sales, color='#8e44ad', linestyle='--', linewidth=1.5)
    plt.text(avg_sales + 1000, cat_stats['margin'].max(), "Avg Sales Threshold", 
             color='#8e44ad', rotation=0, fontweight='bold', va='top')

    plt.axhline(avg_margin, color='#2980b9', linestyle='--', linewidth=1.5)
    plt.text(cat_stats['sales'].max(), avg_margin + 0.5, "Avg Margin Threshold", 
             color='#2980b9', ha='right', fontweight='bold', va='bottom')

    for i, row in cat_stats.iterrows():
        plt.text(row['sales']+2000, row['margin'], row['sub_category'], fontsize=8, color='#34495e')

    plt.title('Strategic Product Portfolio (BCG Matrix)', fontsize=18, fontweight='bold', y=1.05)
    plt.xlabel('Total Sales Volume ($)', fontsize=8)
    plt.ylabel('Profit Margin (%)', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Strategy Zone")

    plt.subplots_adjust(left=0.10, right=0.75, top=0.85, bottom=0.15)
    
    save_path = OUTPUT_DIR / '2_bcg_matrix_enhanced.png'
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ Chart 2 saved.")
    plt.close()

# ==========================================
# 3. هیت‌ مپ 
# ==========================================
def plot_market_basket(df):
    basket = df[['order_id', 'sub_category']].drop_duplicates()
    basket_matrix = basket.merge(basket, on='order_id')
    cross_sell = basket_matrix[basket_matrix['sub_category_x'] != basket_matrix['sub_category_y']]
    pivot_table = pd.crosstab(cross_sell['sub_category_x'], cross_sell['sub_category_y'])
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        pivot_table, 
        annot=False, 
        cmap='PuBuGn', 
        linewidths=1,
        linecolor='white',
        square=True,
        cbar_kws={'label': 'Frequency', 'shrink': 0.8}
    )
    
    plt.title('Cross-Selling Patterns\n(Product Affinity Heatmap)', fontsize=14, fontweight='bold', y=1.05)
    plt.xlabel('Product B', fontsize=10, labelpad=10)
    plt.ylabel('Product A', fontsize=10, labelpad=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.subplots_adjust(left=0.20, right=0.90, top=0.85, bottom=0.20)
    
    save_path = OUTPUT_DIR / '3_market_basket_heatmap.png'
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ Chart 3 saved.")
    plt.close()

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ File not found at: {DATA_PATH}")
        return
    
    print("⏳ Processing Data...")
    df = load_data(DATA_PATH)
    
    print("📊 Generating High-Margin Charts...")
    plot_low_margin_products(df)
    plot_bcg_matrix(df)
    plot_market_basket(df)
    
    print("\n✅ All charts generated successfully.")

if __name__ == '__main__':
    main()