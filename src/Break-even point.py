import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import arabic_reshaper
from bidi.algorithm import get_display

# تابع تنظیم حروف فارسی
def make_farsi(text):
    reshaped_text = arabic_reshaper.reshape(text) # حروف را به هم می‌چسباند
    bidi_text = get_display(reshaped_text)        # جهت متن را درست می‌کند
    return bidi_text

#  تنظیم فونت فارسی (حیاتی)
plt.rcParams['font.family'] = 'Segoe UI'

# 2. اتصال و دیتا
engine = create_engine("postgresql://postgres:123456@localhost:5432/postgres")
query = """
SELECT discount, AVG(profit) AS avg_profit
FROM superstore
GROUP BY discount
ORDER BY discount ASC;
"""
df_disc = pd.read_sql(query, engine)

# 3. تنظیمات نمودار
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

sns.lineplot(
    data=df_disc,
    x="discount",
    y="avg_profit",
    marker="o",
    color="royalblue",
    linewidth=2.5,
    markersize=9,
    label=make_farsi("روند سودآوری") # ✅ استفاده از تابع
)

# --- پیدا کردن نقطه ضرر ---
loss_df = df_disc[df_disc['avg_profit'] < 0]
if not loss_df.empty:
    critical_discount = loss_df.iloc[0]['discount']
    
    # خط عمودی
    label_line = f"شروع ضرردهی: {critical_discount:.0%}"
    plt.axvline(x=critical_discount, color='orange', linestyle='-.', linewidth=2, label=make_farsi(label_line))
    
    # متن بالای خط
    label_text = f"آستانه بحرانی\n (Discount >= {critical_discount})"
    plt.text(
        critical_discount, 
        df_disc['avg_profit'].max(), 
        make_farsi(label_text),
        color='darkorange',
        fontweight='bold',
        ha='center'
    )
    
    plt.axvspan(critical_discount, df_disc['discount'].max(), color='red', alpha=0.1, label=make_farsi("ناحیه خطر"))

plt.axhline(0, color='black', linewidth=1)

# مقادیر روی نمودار
for x, y in zip(df_disc['discount'], df_disc['avg_profit']):
    label = f"{y:.0f}$"
    plt.text(x, y + (15 if y > 0 else -35), label, ha='center', fontsize=9, fontweight='bold', color='dimgray')

# 4. عناوین فارسی
plt.title(make_farsi("تحلیل نقطه سربه‌سر تخفیف‌ها"), fontsize=16, fontweight='bold')
plt.xlabel(make_farsi("درصد تخفیف"), fontsize=12)
plt.ylabel(make_farsi("میانگین سود (دلار)"), fontsize=12)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()