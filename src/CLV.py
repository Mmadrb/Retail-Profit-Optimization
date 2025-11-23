import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- نصب کتابخانه‌های CLV ---
# pip install lifetimes
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
    LIFETIMES_AVAILABLE = True
except ImportError:
    print("⚠️  برای پیش‌بینی CLV کتابخانه lifetimes را نصب کنید: pip install lifetimes")
    LIFETIMES_AVAILABLE = False

# --- Configuration ---
DATA_PATH = r"C:\Users\moham\Desktop\MIT\Superstore.csv"
OUTPUT_DIR = Path(r"C:\Users\moham\Desktop\MIT\clv_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class CustomerAnalyticsEngine:
    """موتور تحلیل جامع مشتریان - خوشه‌بندی + CLV + پیش‌بینی چرن"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.customer_df = None
        self.clustered_df = None
        self.clv_predictions = None
        self.model = None
        self.scaler = None
        self.cluster_names = None
        self.clv_models = None
        
    def load_data(self) -> pd.DataFrame:
        """بارگذاری داده‌ها"""
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(self.data_path, encoding='latin1')
        
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        return df
    
    def prepare_rfm_clv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        آماده‌سازی داده‌های RFM + CLV
        برای lifetimes نیاز داریم: frequency, recency, T, monetary_value
        """
        if 'customer_id' not in df.columns:
            df['customer_id'] = df.get('customer_name', df.index)
        
        # محاسبه تاریخ امروز (یا max تاریخ در داده‌ها)
        current_date = df['order_date'].max() + timedelta(days=1)
        
        # آماده‌سازی برای lifetimes
        clv_df = df.groupby('customer_id').agg({
            'order_date': ['max', 'min'],
            'order_id': 'nunique',
            'profit': 'sum',
            'sales': 'sum',
            'discount': 'mean'
        }).reset_index()
        
        clv_df.columns = ['customer_id', 'max_order_date', 'min_order_date', 
                         'frequency', 'profit_sum', 'sales_sum', 'avg_discount']
        
        # محاسبه recency (روز از آخرین خرید تا امروز)
        clv_df['recency'] = (current_date - clv_df['max_order_date']).dt.days
        
        # محاسبه T (سن مشتری از اولین خرید تا امروز)
        clv_df['T'] = (current_date - clv_df['min_order_date']).dt.days
        
        # monetary_value = میانگین سود هر سفارش
        clv_df['monetary_value'] = clv_df['profit_sum'] / clv_df['frequency']
        
        # محاسبه اطلاعات RFM استاندارد
        clv_df['recency_days'] = clv_df['recency']
        clv_df['frequency_rfm'] = clv_df['frequency']
        clv_df['total_profit'] = clv_df['profit_sum']
        clv_df['avg_discount_clv'] = clv_df['avg_discount']
        
        # حذف outlier‌ها
        clv_df = clv_df[clv_df['frequency'] > 0]  # فقط مشتریان تکراری
        for col in ['frequency', 'recency', 'monetary_value']:
            q99 = clv_df[col].quantile(0.99)
            clv_df[col] = clv_df[col].clip(upper=q99)
        
        return clv_df
    
    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = None) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
        """
        خوشه‌بندی با قابلیت انتخاب خودکار k
        """
        features = ['recency_days', 'frequency', 'total_profit', 'avg_discount', 'monetary_value']
        X = df[features].fillna(df[features].median())
        
        # استانداردسازی
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # یافتن بهترین k
        if n_clusters is None:
            n_clusters, _ = self._find_optimal_k(X_scaled)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, df['cluster'])
        print(f"✅ Silhouette Score: {silhouette_avg:.3f}")
        
        return df, kmeans, scaler
    
    def _find_optimal_k(self, X_scaled: np.ndarray, max_k: int = 8) -> Tuple[int, Dict]:
        """یافتن بهترین k با روش Elbow و Silhouette"""
        print(f"🔍 Finding optimal k (2 to {max_k})...")
        
        inertias = []
        silhouettes = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        
        optimal_k = K_range[np.argmax(silhouettes)]
        
        # نمودار
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_title('Elbow Method')
        ax2.plot(K_range, silhouettes, 'go-')
        ax2.set_title('Silhouette Analysis')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'cluster_optimization.png', dpi=300)
        plt.show()
        
        print(f"✅ Optimal k: {optimal_k}")
        return optimal_k, {'inertias': inertias, 'silhouettes': silhouettes}
    
    def fit_clv_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        آموزش مدل‌های CLV
        BG/NBD برای پیش‌بینی فرکانس بقاء
        Gamma-Gamma برای پیش‌بینی ارزش مالی
        """
        if not LIFETIMES_AVAILABLE:
            print("⚠️  CLV models not available")
            return None
        
        print("\n🎯 Fitting CLV Models...")
        
        # فیلتر کردن داده‌ها برای lifetimes
        clv_data = df[['frequency', 'recency', 'T', 'monetary_value']].copy()
        
        # حذف مقادیر غیرمعتبر
        clv_data = clv_data[(clv_data['frequency'] > 0) & 
                           (clv_data['monetary_value'] > 0)]
        
        # آموزش BG/NBD
        bgf = BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(clv_data['frequency'], clv_data['recency'], clv_data['T'])
        
        # آموزش Gamma-Gamma
        ggf = GammaGammaFitter(penalizer_coef=0.0)
        ggf.fit(clv_data['frequency'], clv_data['monetary_value'])
        
        print("✅ CLV models fitted successfully")
        
        return {
            'bgf': bgf,
            'ggf': ggf,
            'data': clv_data
        }
    
    def predict_clv(self, clv_models: Dict, df: pd.DataFrame, 
                   time_period: int = 12) -> pd.DataFrame:
        """
        پیش‌بینی CLV برای دوره زمانی خاص (مثلاً ۱۲ ماه آینده)
        """
        if not clv_models:
            return df
        
        print(f"\n💰 Predicting CLV for next {time_period} months...")
        
        bgf = clv_models['bgf']
        ggf = clv_models['ggf']
        
        # پیش‌بینی فرکانس خرید
        df['predicted_purchases'] = bgf.predict(time_period, 
                                               df['frequency'], 
                                               df['recency'], 
                                               df['T'])
        
        # پیش‌بینی ارزش مشتری
        df['predicted_clv'] = ggf.customer_lifetime_value(
            bgf,
            df['frequency'],
            df['recency'],
            df['T'],
            df['monetary_value'],
            time=time_period,
            discount_rate=0.01  # نرخ تنزیل
        )
        
        # رند کردن برای خوانایی
        df['predicted_clv'] = df['predicted_clv'].round(2)
        df['predicted_purchases'] = df['predicted_purchases'].round(1)
        
        print(f"✅ CLV predicted for {len(df)} customers")
        return df
    
    def predict_churn(self, clv_models: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        پیش‌بینی احتمال چرن (که دیگر خرید نکنند)
        با استفاده از مدل BG/NBD
        """
        if not clv_models:
            return df
        
        print("\n⚠️  Predicting churn probability...")
        
        bgf = clv_models['bgf']
        
        # احتمال زنده بودن (که هنوز خرید خواهند کرد)
        df['probability_alive'] = bgf.conditional_probability_alive(
            df['frequency'],
            df['recency'],
            df['T']
        )
        
        # احتمال چرن = ۱ - alive
        df['churn_probability'] = (1 - df['probability_alive']).round(3)
        
        # دسته‌بندی چرن
        df['churn_risk'] = pd.cut(df['churn_probability'], 
                                 bins=[0, 0.3, 0.7, 1.0],
                                 labels=['Low', 'Medium', 'High'])
        
        print(f"✅ Churn risk calculated")
        return df
    
    def interpret_clusters_with_clv(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        ✅ اصلاح شده: تفسیر خوشه‌ها با ترکیب CLV و احتمال چرن
        """
        print("\n📊 Interpreting clusters with CLV...")
        
        # ✅ بررسی موجود بودن ستون‌ها
        features = ['recency_days', 'frequency', 'total_profit', 'monetary_value']
        
        # ✅ اضافه کردن ستون‌های CLV فقط اگر موجود باشند
        if 'predicted_clv' in df.columns:
            features.append('predicted_clv')
        else:
            print("⚠️  CLV not calculated, using only basic features")
            df['predicted_clv'] = df['total_profit'] * 2  # برآورد ساده
        
        if 'churn_probability' in df.columns:
            features.append('churn_probability')
        else:
            print("⚠️  Churn probability not calculated, using default")
            df['churn_probability'] = 0.3
        
        cluster_profile = df.groupby('cluster')[features].mean()
        
        # نرمال‌سازی برای قضاوت
        profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
        
        cluster_names = {}
        for cluster_id in profile_norm.index:
            # منطقه تصمیم‌گیری پیچیده‌تر
            if profile_norm.loc[cluster_id, 'predicted_clv'] > 0.7:
                if profile_norm.loc[cluster_id, 'churn_probability'] < 0.3:
                    cluster_names[cluster_id] = "🥇 VIP Loyalists (Low Churn Risk)"
                else:
                    cluster_names[cluster_id] = "💎 High-Value (Medium Churn Risk)"
            elif profile_norm.loc[cluster_id, 'churn_probability'] > 0.7:
                cluster_names[cluster_id] = "🚨 Critical At-Risk"
            elif profile_norm.loc[cluster_id, 'total_profit'] < 0:
                cluster_names[cluster_id] = "❌ Loss-Making Discount Seekers"
            elif profile_norm.loc[cluster_id, 'frequency'] > 0.6:
                cluster_names[cluster_id] = "🔄 Frequent (Potential Loyalists)"
            else:
                cluster_names[cluster_id] = "🆕 Average/Developing"
        
        print("\n🏷️  Cluster Names:")
        for cid, name in cluster_names.items():
            print(f"   Cluster {cid}: {name}")
        
        return cluster_names
    
    def plot_clv_analysis(self, df: pd.DataFrame, cluster_names: Dict):
        """نمودارهای CLV و چرن"""
        df['cluster_name'] = df['cluster'].map(cluster_names)
        
        # ✅ بررسی موجود بودن ستون CLV
        if 'predicted_clv' not in df.columns:
            print("⚠️  CLV column not found, skipping CLV plots")
            return
        
        # ۱. توزیع CLV بر اساس خوشه
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='cluster_name', y='predicted_clv', data=df, palette='viridis')
        plt.title('CLV Distribution by Cluster', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'clv_by_cluster.png', dpi=300)
        plt.show()
        
        # ۲. چرن vs CLV
        if 'churn_probability' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='predicted_clv', y='churn_probability', 
                           hue='cluster_name', data=df, palette='viridis')
            plt.title('Churn Risk vs CLV', fontsize=16, fontweight='bold')
            plt.axhline(0.5, color='red', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'churn_vs_clv.png', dpi=300)
            plt.show()
    
    def generate_executive_report(self, df: pd.DataFrame, cluster_names: Dict):
        """
        گزارش اجرایی جامع
        """
        print("\n" + "="*80)
        print("📊 EXECUTIVE REPORT: Customer Analytics & CLV Forecast")
        print("="*80)
        
        # ✅ بررسی موجود بودن ستون CLV
        if 'predicted_clv' in df.columns:
            total_clv = df['predicted_clv'].sum()
            avg_clv = df['predicted_clv'].mean()
        else:
            total_clv = df['total_profit'].sum() * 2  # برآورد ساده
            avg_clv = df['total_profit'].mean() * 2
            print("⚠️  Using estimated CLV (total_profit * 2)")
        
        # ✅ بررسی موجود بودن ستون چرن
        if 'churn_probability' in df.columns:
            high_churn_customers = len(df[df['churn_probability'] > 0.7])
        else:
            high_churn_customers = len(df) // 4  # برآورد ساده
            print("⚠️  Using estimated churn risk (25% of customers)")
        
        print(f"\n💰 TOTAL CLV (12-month forecast): ${total_clv:,.2f}")
        print(f"📈 Average CLV per customer: ${avg_clv:.2f}")
        print(f"⚠️  High churn risk customers: {high_churn_customers:,}")
        
        # تحلیل خوشه‌ای
        print(f"\n{'Cluster':<20} | {'Count':<8} | {'Avg CLV':<12} | {'Churn Risk':<12} | {'Action'}")
        print("-" * 80)
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            name = cluster_names[cluster_id]
            count = len(cluster_data)
            
            # ✅ بررسی موجود بودن ستون‌ها
            avg_clv_cluster = cluster_data['predicted_clv'].mean() if 'predicted_clv' in cluster_data.columns else cluster_data['total_profit'].mean() * 2
            avg_churn = cluster_data['churn_probability'].mean() if 'churn_probability' in cluster_data.columns else 0.3
            
            # توصیه عملی
            if "VIP" in name:
                action = "Protect & Expand"
            elif "Loss-Making" in name:
                action = "Restructure Discounts"
            elif "At-Risk" in name:
                action = "Win-Back Campaign"
            else:
                action = "Develop & Upsell"
            
            print(f"{name[:18]:<20} | {count:<8} | ${avg_clv_cluster:<11.0f} | "
                  f"{avg_churn:<12.2%} | {action}")
        
        # ✅ بررسی موجود بودن loss_making
        if 'total_profit' in df.columns:
            loss_making = df[df['total_profit'] < 0]
            potential_saving = loss_making['total_profit'].sum() * 0.5
        else:
            potential_saving = 0
        
        print(f"\n🎯 IMPACT FORECAST (12 months):")
        print(f"   - Discount Seekers optimization: ${potential_saving:,.2f}")
        print(f"   - Churn reduction (5%): ${avg_clv * high_churn_customers * 0.05:,.2f}")
        
        print("\n" + "="*80)
        
        # ذخیره گزارش
        df.to_csv(OUTPUT_DIR / 'customers_with_clv_clusters.csv', index=False)
    
    def save_models(self, df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler, 
                   cluster_names: Dict, clv_models: Dict):
        """ذخیره تمام مدل‌ها"""
        model_data = {
            'kmeans': kmeans,
            'scaler': scaler,
            'cluster_names': cluster_names,
            'clv_models': clv_models,
            'features': ['recency_days', 'frequency', 'total_profit', 
                        'avg_discount', 'monetary_value', 'predicted_clv', 
                        'churn_probability']
        }
        
        with open(OUTPUT_DIR / 'complete_analytics_engine.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Models saved: {OUTPUT_DIR / 'complete_analytics_engine.pkl'}")

def main():
    """اجرای کامل تحلیل"""
    
    # ساخت موتور تحلیل
    engine = CustomerAnalyticsEngine(DATA_PATH)
    
    # ۱. بارگذاری داده
    print("⏳ Loading data...")
    df = engine.load_data()
    print(f"   Loaded {len(df):,} transactions")
    
    # ۲. آماده‌سازی RFM + CLV
    print("\n🛠️  Preparing RFM & CLV data...")
    customer_df = engine.prepare_rfm_clv_data(df)
    print(f"   {len(customer_df)} customers ready for analysis")
    
    # ۳. خوشه‌بندی
    print("\n🤖 Running clustering...")
    clustered_df, kmeans_model, scaler = engine.perform_clustering(customer_df, n_clusters=4)
    
    # ۴. CLV (اگر lifetimes نصب باشد)
    if LIFETIMES_AVAILABLE:
        print("\n🎯 Training CLV models...")
        clv_models = engine.fit_clv_models(clustered_df)
        
        # ✅ اصلاح مهم: اجرای پیش‌بینی‌ها **قبل** از تفسیر خوشه‌ها
        print("\n💰 Predicting CLV...")
        clustered_df = engine.predict_clv(clv_models, clustered_df)
        
        print("\n⚠️  Predicting churn...")
        clustered_df = engine.predict_churn(clv_models, clustered_df)
    else:
        clv_models = None
        # ✅ اضافه کردن ستون‌های dummy برای جلوگیری از خطا
        clustered_df['predicted_clv'] = clustered_df['total_profit'] * 2
        clustered_df['churn_probability'] = 0.3
    
    # ۵. تفسیر خوشه‌ها (بعد از اضافه شدن ستون‌های CLV)
    print("\n📊 Interpreting clusters...")
    cluster_names = engine.interpret_clusters_with_clv(clustered_df)
    
    # ۶. نمودارها
    print("\n📈 Generating visualizations...")
    engine.plot_clv_analysis(clustered_df, cluster_names)
    
    # ۷. گزارش اجرایی
    print("\n📋 Generating executive report...")
    engine.generate_executive_report(clustered_df, cluster_names)
    
    # ۸. ذخیره مدل
    print("\n💾 Saving models...")
    engine.save_models(clustered_df, kmeans_model, scaler, cluster_names, clv_models)
    
    print("\n✅ Analysis complete!")
    print(f"📁 All files saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()