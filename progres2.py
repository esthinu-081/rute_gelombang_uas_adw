import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.filterwarnings('ignore')

CSV_PATH = "data/processed/timeseries_selat_makassar.csv"

print(">>> Memulai Progres 2: Eksplorasi Data & Persiapan Pemodelan...")

if not os.path.exists(CSV_PATH):
    print(f"❌ File {CSV_PATH} tidak ditemukan!")
    print("Pastikan kamu sudah menjalankan kode progres 1 (ekstraksi data) terlebih dahulu.")
    exit()

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH, index_col='time', parse_dates=True)
print(f"    ✅ Data berhasil dimuat. Total: {len(df)} baris.")

# ============================================================
# 2. VISUALISASI DATA AKTUAL (Untuk Poster Bagian 02)
# ============================================================
print(">>> Membuat Grafik Data Aktual...")
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['Hs'], color='#005b96', linewidth=1)
plt.title('Data Aktual Tinggi Gelombang (Hs) di Selat Makassar', fontsize=14, fontweight='bold')
plt.ylabel('Tinggi Gelombang (m)', fontsize=12)
plt.xlabel('Waktu', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
# Format sumbu X agar rapi (Bulan Tahun)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) 
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/poster_data_aktual.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 3. UJI STASIONERITAS & ACF (Validasi Karakteristik Data)
# ============================================================
print(">>> Menjalankan Uji ADF & Visualisasi ACF...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Subplot 1: Rolling Stats (Rata-rata & Standar Deviasi bergerak)
window_size = 24 
rolling_mean = df['Hs'].rolling(window=window_size).mean()
rolling_std = df['Hs'].rolling(window=window_size).std()

axes[0].plot(df.index, df['Hs'], label='Data Hs Aktual', color='#1f77b4', alpha=0.4)
axes[0].plot(df.index, rolling_mean, label=f'Rolling Mean ({window_size} step)', color='#d62728', linewidth=2)
axes[0].plot(df.index, rolling_std, label=f'Rolling Std Dev ({window_size} step)', color='#2ca02c', linewidth=2, linestyle='--')
axes[0].set_title('Uji Visual: Rolling Statistics (Fluktuasi Mean & Varians)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Tinggi Gelombang (m)')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Subplot 2: Autocorrelation Function (ACF)
plot_acf(df['Hs'].dropna(), ax=axes[1], lags=40, color='#1f77b4')
axes[1].set_title('Uji Visual: Autocorrelation Function (ACF) - Pola Musiman', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Lag (Step 3-Jam)')
axes[1].set_ylabel('Korelasi')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("results/poster_uji_stasioner.png", dpi=300, bbox_inches='tight')
plt.show()

# Uji ADF Formal
adf_result = adfuller(df['Hs'].dropna())
print("\n" + "="*40)
print("HASIL UJI ADF")
print("="*40)
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] <= 0.05:
    print("Kesimpulan: Data STASIONER (p-value <= 0.05)")
else:
    print("Kesimpulan: Data TIDAK STASIONER (p-value > 0.05)")
print("="*40)

# ============================================================
# 4. TRAIN-TEST SPLIT (Persiapan untuk Progres 3: Pemodelan)
# ============================================================
print(">>> Membagi Data Latih dan Data Uji (80% : 20%)...")
train_size = int(len(df) * 0.8)
train_data = df['Hs'].iloc[:train_size]
test_data = df['Hs'].iloc[train_size:]

plt.figure(figsize=(15, 5))
plt.plot(train_data.index, train_data, label='Data Latih (Training 80%)', color='blue')
plt.plot(test_data.index, test_data, label='Data Uji (Testing 20%)', color='orange')
plt.title('Pembagian Data Deret Waktu (Train-Test Split)', fontsize=14, fontweight='bold')
plt.ylabel('Tinggi Gelombang (m)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/poster_train_test.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Progres 2 Selesai!")
print(f"    Total observasi: {len(df)}")
print(f"    Data Latih: {len(train_data)} baris")
print(f"    Data Uji: {len(test_data)} baris")
print("    Cek folder 'results/' untuk melihat gambar yang siap masuk poster.")