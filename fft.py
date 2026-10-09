import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os  # ファイル名操作用に追加

# === 1. ファイルの読み込み ===
filename = input("読み込むファイル名を入力してください: ")
data = np.loadtxt(filename)
x = data[:, 0]
y = data[:, 1]

# 保存用ファイル名のベースを作成（拡張子を除いた名前を取得）
base_name = os.path.splitext(filename)[0]

# === 2. 解析準備 ===
N = len(y)
dt = x[1] - x[0]
fft_raw = np.fft.fft(y)
freq = np.fft.fftfreq(N, d=dt)

# === 3. スペクトル計算 ===
half = N // 2
freq_pos = freq[:half]
amp_pos = np.abs(fft_raw[:half]) * 2.0 / N
amp_pos[0] = np.abs(fft_raw[0]) / N
phase_pos = np.angle(fft_raw[:half])

# === 4. 成分の抽出関数 ===
def get_top_components(num):
    search_range = amp_pos[1:]
    actual_num = min(num, len(search_range))
    top_amp_indices = np.argsort(search_range)[-actual_num:] + 1
    return np.sort(top_amp_indices)

indices_5 = get_top_components(5)
indices_20 = get_top_components(20)

# === 5. 表の表示 ＆ CSV保存関数 ===
def process_results(indices, label, suffix):
    print(f"\n--- {label} (周波数昇順) ---")
    print(f"{'No.':>3} | {'Freq [Hz]':>12} | {'Amp':>10} | {'Phase [rad]':>10}")
    print("-" * 50)
    
    # 保存用データのリスト
    csv_data = []
    
    for i, idx in enumerate(indices):
        f = freq_pos[idx]
        A = amp_pos[idx]
        P = phase_pos[idx]
        print(f"{i+1:>3} | {f:>12.4f} | {A:>10.4f} | {P:>10.4f}")
        csv_data.append([f, A, P])
    
    # CSVファイルとして保存
    out_name = f"{base_name}_{suffix}.csv"
    header = "Frequency[Hz],Amplitude,Phase[rad]"
    np.savetxt(out_name, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"→ 保存完了: {out_name}")
    print("-" * 50)

# 5本と20本それぞれを実行
process_results(indices_5, "上位 5 成分", "top5")
process_results(indices_20, "上位 20 成分", "top20")

# === 6. 波形の合成 ===
def reconstruct(indices):
    y_rec = np.zeros_like(x) + amp_pos[0]
    for idx in indices:
        y_rec += amp_pos[idx] * np.cos(2 * np.pi * freq_pos[idx] * x + phase_pos[idx])
    return y_rec

y_5 = reconstruct(indices_5)
y_20 = reconstruct(indices_20)

# === 7. グラフ描画 ===
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x, y, 'o', label='Original', markersize=3, alpha=0.3)
plt.plot(x, y_5, label='Reconstruction (5 comps)', color='orange', linewidth=2)
plt.title(f"Comparison: Original vs 5 Components (Saved as {base_name}_top5.csv)")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x, y, 'o', label='Original', markersize=3, alpha=0.3)
plt.plot(x, y_20, label='Reconstruction (20 comps)', color='red', linewidth=2)
plt.title(f"Comparison: Original vs 20 Components (Saved as {base_name}_top20.csv)")
plt.xlabel("X")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()