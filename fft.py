import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# アプリのタイトル
st.title("高速フーリエ変換 上位20本")

# 1. ファイルアップローダー
uploaded_file = st.file_uploader(
    "解析するテキストファイル (.txt, .csv など) をアップロードしてください",
    type=["txt", "csv", "dat"],
)

if uploaded_file is not None:
    try:
        # データの読み込み
        data = np.loadtxt(uploaded_file)
        x = data[:, 0]
        y = data[:, 1]

        # 解析準備
        N = len(y)
        dt = x[1] - x[0]
        fft_raw = np.fft.fft(y)
        freq = np.fft.fftfreq(N, d=dt)

        # スペクトル計算
        half = N // 2
        freq_pos = freq[:half]
        amp_pos = np.abs(fft_raw[:half]) * 2.0 / N
        amp_pos[0] = np.abs(fft_raw[0]) / N
        phase_pos = np.angle(fft_raw[:half])

        # 上位20成分のインデックス抽出
        num_comp = 20
        search_range = amp_pos[1:]
        actual_num = min(num_comp, len(search_range))
        indices = np.argsort(search_range)[-actual_num:] + 1
        indices = np.sort(indices)  # 周波数昇順にソート

        # 結果をデータフレーム化
        csv_data = []
        for i, idx in enumerate(indices):
            f = freq_pos[idx]
            A = amp_pos[idx]
            P = phase_pos[idx]
            csv_data.append([i + 1, f, A, P])

        df_20 = pd.DataFrame(
            csv_data,
            columns=["No.", "Frequency[Hz]", "Amplitude", "Phase[rad]"],
        )

        # ---成分解析結果（上位20成分の表 ＆ CSVダウンロード） ---
        st.subheader(f" 上位 {actual_num} 成分")

        # ソート順の選択UI
        sort_col = st.selectbox(
            "並べ替えの基準（ソート項目）を選択してください:",
            ["Frequency[Hz] (昇順)", "Frequency[Hz] (降順)", 
             "Amplitude (降順)", "Amplitude (昇順)", 
             "Phase[rad] (昇順)", "Phase[rad] (降順)", "No. (昇順)"]
        )

        # 選択に応じたデータフレームのソート処理
        if sort_col == "Frequency[Hz] (昇順)":
            sorted_df = df_20.sort_values(by="Frequency[Hz]", ascending=True)
        elif sort_col == "Frequency[Hz] (降順)":
            sorted_df = df_20.sort_values(by="Frequency[Hz]", ascending=False)
        elif sort_col == "Amplitude (降順)":
            sorted_df = df_20.sort_values(by="Amplitude", ascending=False)
        elif sort_col == "Amplitude (昇順)":
            sorted_df = df_20.sort_values(by="Amplitude", ascending=True)
        elif sort_col == "Phase[rad] (昇順)":
            sorted_df = df_20.sort_values(by="Phase[rad]", ascending=True)
        elif sort_col == "Phase[rad] (降順)":
            sorted_df = df_20.sort_values(by="Phase[rad]", ascending=False)
        else:
            sorted_df = df_20.sort_values(by="No.", ascending=True)

        # ソートされた表を表示
        st.dataframe(sorted_df, hide_index=True, use_container_width=True)

        # ソートされたデータから CSV を生成
        csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="選択したソート順でCSVをダウンロード",
            data=csv_bytes,
            file_name="top20_components.csv",
            mime="text/csv",
        )

        # --- グラフ描画（元データプロット・各成分・合成波） ---
        fig = plt.figure(figsize=(12, 10))

        # 上段：元データと合成波
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(x, y, "o", label="Original", markersize=3, alpha=0.3)

        y_sum = np.zeros_like(x) + amp_pos[0]
        colors = plt.cm.tab20(np.linspace(0, 1, actual_num))

        # 下段：各成分のプロットと合成波の計算
        ax2 = plt.subplot(2, 1, 2)
        for i, idx in enumerate(indices):
            f = freq_pos[idx]
            A = amp_pos[idx]
            P = phase_pos[idx]

            y_each = A * np.cos(2 * np.pi * f * x + P)
            y_sum += y_each

            ax2.plot(
                x,
                y_each,
                label=f"{f:.1f}Hz",
                color=colors[i],
                alpha=0.6,
                linewidth=1,
            )

        # 上段に合成波を追加
        ax1.plot(
            x,
            y_sum,
            "--",
            color="red",
            label=f"Sum of {actual_num} comps",
            linewidth=1.5,
        )
        ax1.set_title(f"Reconstruction with Top {actual_num} Components")
        ax1.legend(loc="upper right", fontsize="small")
        ax1.grid(True)

        # 下段のグラフ設定
        ax2.set_title(f"Individual Top {actual_num} Components")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        ax2.legend(
            loc="upper right", fontsize="xx-small", ncol=4, framealpha=0.5
        )

        plt.tight_layout()

        # 画面に描画
        st.pyplot(fig)

        # グラフ画像のダウンロード
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        st.download_button(
            label="グラフ画像をダウンロード (PNG)",
            data=img_buffer,
            file_name="fft_analysis_result.png",
            mime="image/png",
        )

    except Exception as e:
        st.error(
            f"ファイルの読み込みまたは処理中にエラーが発生しました: {e}"
        )
else:
    st.info(
        "上のエリアにテキストファイルをドロップ（または選択）してください。"
    )
