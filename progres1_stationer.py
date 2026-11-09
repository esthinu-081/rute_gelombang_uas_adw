import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI PATH LOKAL
# ============================================================
INPUT_NC = "data/raw/master_clean_3h.nc"
OUTPUT_CSV = "data/processed/timeseries_selat_makassar.csv"
OUTPUT_PLOT = "results/progres1_visual_stasioner.png"

# Titik Tengah Selat Makassar
TARGET_LAT = -4.5
TARGET_LON = 117.5

print(">>> Memulai Progres 1: Ekstraksi & Uji Stasioneritas...")

# ============================================================
# 1. EKSTRAKSI DATA
# ============================================================
if not os.path.exists(INPUT_NC):
    print(f"❌ File tidak ditemukan di: {INPUT_NC}")
    print("Pastikan file .nc sudah ditaruh di folder data/raw/")
    exit()

if not os.path.exists(OUTPUT_CSV):
    print("    Mengekstrak data dari NetCDF ke CSV...")
    ds = xr.open_dataset(INPUT_NC)
    point_data = ds.sel(lat=TARGET_LAT, lon=TARGET_LON, method='nearest')
    
    df = pd.DataFrame({
        'time': point_data.time.values,
        'Hs': point_data['hs'].values,
        'Wind_U': point_data['wind_u'].values,
        'Wind_V': point_data['wind_v'].values
    })
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_CSV)
    ds.close()
    print("    ✅ Ekstraksi selesai.")

# Load data CSV
df = pd.read_csv(OUTPUT_CSV, index_col='time', parse_dates=True)
print(f"    Total data: {len(df)} baris")

# ============================================================
# 2. VISUALISASI PEMBUKTIAN NON-STASIONER
# ============================================================
print(">>> Membuat Visualisasi Kestasioneran...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Subplot 1: Data vs Rolling Mean & Standard Deviation
window_size = 24  # 24 x 3 jam = 72 jam (3 hari)
rolling_mean = df['Hs'].rolling(window=window_size).mean()
rolling_std = df['Hs'].rolling(window=window_size).std()

axes[0].plot(df.index, df['Hs'], label='Data Hs Aktual', color='#1f77b4', alpha=0.4, linewidth=1)
axes[0].plot(df.index, rolling_mean, label=f'Rolling Mean ({window_size} step)', color='#d62728', linewidth=2.5)
axes[0].plot(df.index, rolling_std, label=f'Rolling Std Dev ({window_size} step)', color='#2ca02c', linewidth=2, linestyle='--')
axes[0].set_title('Uji Visual 1: Rolling Statistics (Fluktuasi Mean & Varians)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Tinggi Gelombang (m)', fontsize=11)
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Autocorrelation Function (ACF)
plot_acf(df['Hs'].dropna(), ax=axes[1], lags=40, color='#1f77b4')
axes[1].set_title('Uji Visual 2: Autocorrelation Function (ACF) - Pola Musiman & Autokorelasi', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Lag (Step 3-Jam)', fontsize=11)
axes[1].set_ylabel('Korelasi', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"    ✅ Grafik disimpan di: {OUTPUT_PLOT}")
plt.show()

# ============================================================
# 3. UJI STATISTIK FORMAL (ADF TEST)
# ============================================================
print("\n" + "=" * 55)
print("HASIL UJI STATISTIK AUGMENTED DICKEY-FULLER (ADF TEST)")
print("=" * 55)

adf_result = adfuller(df['Hs'].dropna())

print(f"ADF Statistic     : {adf_result[0]:.4f}")
print(f"p-value           : {adf_result[1]:.4f}")
print(f"Lags Used         : {adf_result[2]}")
print(f"Number of Obs     : {adf_result[3]}")
print(f"Critical Value 1% : {adf_result[4]['1%']:.4f}")
print(f"Critical Value 5% : {adf_result[4]['5%']:.4f}")
print(f"Critical Value 10%: {adf_result[4]['10%']:.4f}")
print("-" * 55)

if adf_result[1] > 0.05:
    print("Kesimpulan: p-value > 0.05 (Gagal Tolak H0) -> DATA NON-STASIONER.")
    print("Tindakan  : Perlu proses differencing (d) untuk pemodelan.")
else:
    print("Kesimpulan: p-value <= 0.05 (Tolak H0) -> Data STASIONER.")
    print("Tindakan  : Data siap digunakan untuk pemodelan.")