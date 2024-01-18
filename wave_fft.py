"""
FFT calculation of wave
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy.signal import find_peaks
import math

# def FFT_main(t, x, dt, split_t_r, overlap, window_F):
#
#     #データをオーバーラップして分割する。
#     split_data = data_split(t, x, split_t_r, overlap)
#     print(split_data)
#
#     #FFTを行う。
#     FFT_result_list = []
#     for split_data_cont in split_data:
#         FFT_result_cont = FFT(split_data_cont, dt, window_F)
#         FFT_result_list.append(FFT_result_cont)
#
#     """
#     #各フレームのグラフ化
#     IDN = 0
#     for split_data_cont, FFT_result_cont in zip(split_data, FFT_result_list):
#         IDN = IDN+1
#         plot_FFT(split_data_cont[0], split_data_cont[1], FFT_result_cont[0], FFT_result_cont[1], output_FN, IDN, 0, y_label, y_unit)
#     """
#
#     #平均化
#     fq_ave = FFT_result_list[0][0]
#     F_abs_amp_ave = np.zeros(len(fq_ave))
#     for i in range(len(FFT_result_list)):
#         F_abs_amp_ave = F_abs_amp_ave + FFT_result_list[i][1]
#     F_abs_amp_ave = F_abs_amp_ave/(i+1)
#
#     return fq_ave, F_abs_amp_ave
#
def cos_taper(acc,dt):#cos20テーパーをかける
    taper_length = 30
    taper= float(taper_length/dt)
    wave=[]
    for i in range(len(acc)):
        if i <= taper:
            cos_t=math.cos(float(math.pi / 2 - i / taper * math.pi / 2))
            wave.append(acc[i]*cos_t)
        elif i >= len(acc)-taper:
            cos_t=math.cos(math.pi / 2- (len(acc)-i)/taper * math.pi / 2)
            wave.append(acc[i]*cos_t)
        else:
            wave.append(acc[i])
    return wave

def make_graph_shape(output_FN, y_label, y_unit,x_max,y_max):
    fig = plt.figure(figsize=(12, 6))

    ax2 = fig.add_subplot(111)
    title2 = "freq_" + output_FN[:-4]
    plt.xlabel('freqency(Hz)')
    plt.ylabel(y_label+"["+y_unit+"/rtHz]")
    plt.xscale("log")
    #plt.yscale("log")
    plt.title(title2)
    plt.xlim(0.1,x_max)
    plt.ylim(0,y_max)
    #plt.ylim(0,y_max)
    return 0
#
def add_plot(fq, F_abs_amp,label):
    plt.plot(fq, F_abs_amp,label = label)
#
# def FFT(data_input, dt, window_F):
#
#     N = len(data_input[0])
#
#     #窓の用意
#     if window_F == "hanning":
#         window = np.hanning(N)          # ハニング窓
#     elif window_F == "hamming":
#         window = np.hamming(N)          # ハミング窓
#     elif window_F == "blackman":
#         window = np.blackman(N)         # ブラックマン窓
#     elif window_F == "parzen":           # parzen窓
#         window = signal.parzen(N)
#     else:
#         print("Error: input window function name is not sapported. Your input: ", window_F)
#         print("Hanning window function is used.")
#         hanning = np.hanning(N)          # ハニング窓
#
#     #窓関数後の信号
#     x_windowed = data_input[1]*window
#
#     #FFT計算
#     F = np.fft.fft(x_windowed)
#     F_abs = np.abs(F)
#     F_abs_amp = F_abs / N * 2
#     fq = np.linspace(0, 1.0/dt, N)
#
#     #窓補正
#     acf=1/(sum(window)/N)
#     F_abs_amp = acf*F_abs_amp
#
#     #ナイキスト定数まで抽出
#     fq_out = fq[:int(N/2)+1]
#     F_abs_amp_out = F_abs_amp[:int(N/2)+1]
#
#     return [fq_out, F_abs_amp_out]
#
# def data_split(t, x, split_t_r, overlap):
#
#     split_data = []
#     one_frame_N = int(len(t)*split_t_r) #1フレームのサンプル数
#     overlap_N = int(one_frame_N*overlap) #オーバーラップするサンプル数
#     start_S = 0
#     end_S = start_S + one_frame_N
#
#     while True:
#         t_cont = t[start_S:end_S]
#         x_cont = x[start_S:end_S]
#         split_data.append([t_cont, x_cont])
#
#         start_S = start_S + (one_frame_N - overlap_N)
#         end_S = start_S + one_frame_N
#
#         if end_S > len(t):
#             break
#
#
#     return np.array(split_data)
#
# def fft_calc(wave,dt):#FFT calculation
#     t = np.arange(0, len(wave)*dt, dt)
#
#     split_t_r = 1.0 #1つの枠で全体のどの割合のデータを分析するか。
#     overlap = 0. #オーバーラップ率
#     window_F = "parzen" #窓関数選択: hanning, hamming, blackman, parzen
#     fq_ave, F_abs_amp_ave = FFT_main(t, wave, dt, split_t_r, overlap, window_F)
#
#     output_FN = "fourier_spectrum.png"
#     y_label = "amplitude"
#     y_unit = "V"
#     x_max= 10
#     y_max=1
#     make_graph_shape(output_FN, y_label, y_unit,x_max,y_max)
#     add_plot(fq_ave, F_abs_amp_ave,split_t_r)
#
#     split_t_r = 0.5 #1つの枠で全体のどの割合のデータを分析するか。
#     overlap = 0. #オーバーラップ率
#     window_F = "parzen" #窓関数選択: hanning, hamming, blackman, parzen
#     fq_ave, F_abs_amp_ave = FFT_main(t, wave, dt, split_t_r, overlap, window_F)
#     add_plot(fq_ave, F_abs_amp_ave,split_t_r)
#
#     split_t_r = 0.4 #1つの枠で全体のどの割合のデータを分析するか。
#     overlap = 0. #オーバーラップ率
#     window_F = "parzen" #窓関数選択: hanning, hamming, blackman, parzen
#     fq_ave, F_abs_amp_ave = FFT_main(t, wave, dt, split_t_r, overlap, window_F)
#     add_plot(fq_ave, F_abs_amp_ave,split_t_r)
#
#     plt.legend()
#     plt.savefig(output_FN, dpi=300)

#Osaki SWIN program
def SWIN(NFOLD, F, G, ND, IND, DF, BAND):

    W = np.zeros(101)
    G1 = np.zeros(60000)
    G2 = np.zeros(60000)

    T = 1.0 / DF
    if BAND != 0:#バンド幅が0以外の時実行
        #以下の条件を満たさない場合、エラー
        UDF = 1.854305 / BAND * DF
        if UDF > 0.5:
            raise ValueError("BANDWIDTH IS TOO NARROW")
        LMAX = int(2.0 / UDF) + 1
        if LMAX > 101:
            raise ValueError("BANDWIDTH IS TOO WIDE")

        #parzenのスペクトルウィンドウを作成
        W[0] = 0.75 * UDF
        for L in range(2, LMAX + 1):
            DIF = np.pi/2 * float(L - 1) * UDF
            W[L - 1] = W[0] * (np.sin(DIF) / DIF) ** 4
        df = pd.DataFrame(W)
        df.to_csv('window.csv',index=False,mode='w')

        #フーリエスペクトルをパワスペクトルに変換
        if IND == 100:
            G[0] = F[0] ** 2 / T
            for K in range(2, NFOLD):
                G[K - 1] = 2.0 * F[K - 1] ** 2 / T
            G[NFOLD - 1] = F[NFOLD - 1] ** 2 / T

        #パワスペクトルの平滑化
        if BAND != 0:
            LL = LMAX * 2 - 1
            LN = LL - 1 + NFOLD
            LT = (LL - 1) * 2 + NFOLD
            LE = LT - LMAX + 1

            G1[:LT] = 0.0
            G1[LL - 1:LL - 1 + NFOLD] = G[:NFOLD]

            for K in range(LMAX, LE + 1):
                S = W[0] * G1[K - 1]
                for L in range(2, LMAX + 1):
                    S += W[L - 1] * (G1[K - L] + G1[K + L - 2])
                G2[K - 1] = S

            for L in range(2, LMAX + 1):
                G2[LL + L - 2] += G2[LL - L ]
                G2[LN - L ] += G2[LN - L - 2]

            G[:NFOLD] = G2[LL - 1:LL - 1 + NFOLD]

        F[0] = np.sqrt(G[0] * T)
        for K in range(2, NFOLD):
            F[K - 1] = np.sqrt(G[K - 1] * T / 2.0)
        F[NFOLD - 1] = np.sqrt(G[NFOLD - 1] * T)

        return F

#メインプログラム
def calc_fft(df1,dt,data_list,graph_title):

    output_FN = "./graph/" + str(graph_title) + ".png"
    y_label = "amplitude"
    y_unit = "V"
    x_max = 10
    y_max = 200
    make_graph_shape(graph_title, y_label, y_unit, x_max, y_max)
#    add_plot(fq, F_abs_amp, 'raw_result')

    for i in data_list:
    # 対象波形のフーリエ変換
        acc =df1[i]

    # 加速度波形にcos20テーパーをかける
        acc = cos_taper(df1[i], dt)

        F = np.fft.fft(acc)
        F_abs_amp = np.abs(F) / (1/dt)
   # F_abs_amp = F_abs / len(acc) * 2
        fq = np.linspace(0, 1.0 / dt, len(acc))

    # スペクトルの平滑化
        band = 0.05
        F_pawa = np.zeros(len(fq))

        fft_smoothed = SWIN(len(F_abs_amp), F_abs_amp, F_pawa, len(F_abs_amp), 100, 1.0 / dt / len(fq), band)
        add_plot(fq, fft_smoothed, str(i))
        plt.legend()
    plt.savefig(output_FN, dpi=300)
    plt.close()
    
