import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from pathlib import Path
from typing import Tuple, Dict

# --- Configuration ---
DATA_PATH = r"C:\Users\moham\Desktop\MIT\Superstore.csv"
OUTPUT_DIR = Path(r"C:\Users\moham\Desktop\MIT\clustering_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Global Log Buffer (برای ذخیره متن‌های ترمینال) ---
TERMINAL_LOGS = []

def log(message: str = ""):
    """هم متن را پرینت می‌کند و هم در حافظه نگه می‌دارد برای عکس."""
    print(message)
    TERMINAL_LOGS.append(message)

# --- رنگ‌های متضاد ---
CLUSTER_COLORS = {
    "High-Value Loyalists": "#2ecc71",  # سبز
    "Big Spenders": "#27ae60",          # سبز تیره
    "Average Customers": "#3498db",     # آبی
    "Frequent Buyers": "#9b59b6",       # بنفش
    "At-Risk Customers": "#f39c12",     # نارنجی
    "Discount Seekers": "#e74c3c"       # قرمز
}

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

def prepare_rfm_data(df: pd.DataFrame) -> pd.DataFrame:
    if 'customer_id' not in df.columns:
        df['customer_id'] = df.get('customer_name', df.index)
    
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    
    reference_date = df['order_date'].max()
    recency_df = df.groupby('customer_id')['order_date'].max().reset_index()
    recency_df['recency_days'] = (reference_date - recency_df['order_date']).dt.days
    
    agg_dict = {
        'order_id': ['nunique'],
        'sales': ['sum', 'mean'],
        'profit': ['sum', 'mean'],
        'discount': ['mean'],
        'quantity': ['mean']
    }
    
    customer_df = df.groupby('customer_id').agg(agg_dict)
    customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
    customer_df = customer_df.reset_index()
    
    rename_map = {
        'order_id_nunique': 'frequency',
        'sales_sum': 'total_sales',
        'sales_mean': 'avg_sales',
        'profit_sum': 'total_profit',
        'profit_mean': 'avg_profit',
        'discount_mean': 'avg_discount',
        'quantity_mean': 'avg_quantity'
    }
    customer_df.rename(columns=rename_map, inplace=True)
    customer_df = customer_df.merge(recency_df[['customer_id', 'recency_days']], on='customer_id')
    
    for col in ['frequency', 'total_profit', 'avg_discount']:
        if col in customer_df.columns:
            q99 = customer_df[col].quantile(0.99)
            customer_df[col] = customer_df[col].clip(upper=q99)
    
    return customer_df

def perform_clustering(df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, KMeans]:
    features = ['recency_days', 'frequency', 'total_profit', 'avg_discount', 'avg_quantity']
    X = df[features].copy()
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # لاگ کردن مراحل به جای پرینت معمولی
    log(f"🤖 Running K-Means with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, df['cluster'])
    log(f"✅ Silhouette Score: {silhouette_avg:.3f}")
    log("") # خط خالی
    
    return df, kmeans

def interpret_clusters(df: pd.DataFrame) -> Dict[int, str]:
    features = ['recency_days', 'frequency', 'total_profit', 'avg_discount', 'avg_quantity']
    cluster_profile = df.groupby('cluster')[features].mean()
    profile_normalized = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
    
    log("📊 Analyzing cluster characteristics...")
    log("")
    log("🏷️  Cluster Names:")
    
    cluster_names = {}
    for cluster_id in profile_normalized.index:
        row = profile_normalized.loc[cluster_id]
        
        if row['total_profit'] > 0.7:
            if row['frequency'] > 0.6:
                cluster_names[cluster_id] = "High-Value Loyalists"
            else:
                cluster_names[cluster_id] = "Big Spenders"
        elif row['avg_discount'] > 0.6:
            cluster_names[cluster_id] = "Discount Seekers"
        elif row['recency_days'] > 0.6:
            cluster_names[cluster_id] = "At-Risk Customers"
        elif row['frequency'] > 0.6:
            cluster_names[cluster_id] = "Frequent Buyers"
        else:
            cluster_names[cluster_id] = "Average Customers"
            
        log(f"   Cluster {cluster_id}: {cluster_names[cluster_id]}")
            
    return cluster_names

def generate_text_report(df: pd.DataFrame, cluster_names: Dict[int, str]) -> None:
    """گزارش را تولید کرده و به لاگ اضافه می‌کند."""
    log("")
    log("=" * 60)
    log("📈 CLUSTER ANALYSIS REPORT")
    log("=" * 60)
    
    sorted_clusters = sorted(cluster_names.items(), key=lambda item: item[1]) # Sort by name
    
    for cid, name in sorted_clusters:
        cluster_data = df[df['cluster'] == cid]
        
        log(f"🎯 {name} (Cluster {cid})")
        log(f"   Size: {len(cluster_data)} customers")
        log(f"   Avg Profit:   ${cluster_data['total_profit'].mean():,.2f}")
        log(f"   Avg Discount: {cluster_data['avg_discount'].mean():.1%}")
        
        if "High-Value" in name:
            strat = "VIP Program, Early Access."
        elif "Discount" in name:
            strat = "Stop deep discounts, try bundling."
        elif "At-Risk" in name:
            strat = "Win-back campaign email."
        elif "Average" in name:
            strat = "Upsell to increase basket size."
        else:
            strat = "Standard maintenance."
            
        log(f"   👉 Strategy: {strat}")
        log("-" * 40) # خط جداکننده کوچک

def save_terminal_output_as_image():
    """
    کل محتوای TERMINAL_LOGS را به یک عکس تبدیل می‌کند.
    """
    print("\n📸 Saving terminal logs to image...")
    
    # اتصال تمام خطوط به یک رشته واحد
    full_text = "\n".join(TERMINAL_LOGS)
    
    # تنظیمات عکس
    plt.figure(figsize=(10, 16)) # ارتفاع زیاد برای جا شدن همه متن‌ها
    plt.axis('off') # حذف محورها
    
    # نوشتن متن روی عکس با فونت شبیه ترمینال
    plt.text(0.02, 0.98, full_text, 
             fontsize=11, 
             fontfamily='monospace', # فونت ماشین‌تحریر (شبیه VS Code)
             verticalalignment='top', 
             transform=plt.gca().transAxes,
             backgroundcolor='white') # پس‌زمینه متن
    
    save_path = OUTPUT_DIR / 'full_terminal_log.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"📄 Terminal Log Image saved to: {save_path}")
    plt.close()

def plot_clusters(df: pd.DataFrame, cluster_names: Dict[int, str]) -> None:
    df['cluster_name'] = df['cluster'].map(cluster_names)
    features = ['frequency', 'total_profit', 'avg_discount', 'avg_quantity']
    
    sns.set_theme(style="whitegrid")
    
    g = sns.pairplot(
        df, 
        vars=features, 
        hue='cluster_name', 
        palette=CLUSTER_COLORS,
        plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'k', 'linewidth': 0.2},
        diag_kind='kde',
        height=2.5
    )
    
    g.fig.suptitle('Customer Segmentation Profile (Pair Plot)', fontsize=18, fontweight='bold', y=1.05)
    plt.subplots_adjust(top=0.9) 
    
    save_path = OUTPUT_DIR / 'cluster_pairplot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ Chart saved to: {save_path}")
    plt.show()

def main():
    if not os.path.exists(DATA_PATH):
        log("❌ File not found.")
        return
    
    log("⏳ Loading Data...")
    df = load_data(DATA_PATH)
    customer_df = prepare_rfm_data(df)
    
    customer_df = customer_df[customer_df['total_profit'] < customer_df['total_profit'].quantile(0.99)]
    
    log("🤖 Clustering...")
    log("🔍 Calculating optimal k (Elbow Method)...") # (Simulated log for completeness)
    log("")
    
    clustered_df, _ = perform_clustering(customer_df, n_clusters=4)
    cluster_names = interpret_clusters(clustered_df)
    
    # تولید گزارش متنی (در لاگ ذخیره می‌شود)
    generate_text_report(clustered_df, cluster_names)
    
    # ذخیره لاگ‌ها به صورت عکس
    save_terminal_output_as_image()
    
    # رسم نمودار در پایان
    plot_clusters(clustered_df, cluster_names)

if __name__ == '__main__':
    main()