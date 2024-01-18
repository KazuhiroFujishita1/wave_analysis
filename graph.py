# -*- coding: shift_jis -*-
import pandas as pd
import plotly.express as px
import glob
import os
import numpy as np
import wave_fft
import re
import matplotlib.pyplot as plt
import math

#波形のフーリエスペクトルの差分を取って逆変換
def eliminate_noize(dt,acc1,acc2):
    freq = np.fft.fftfreq(len(acc1), dt)  # 周波数軸
    # 高速フーリエ変換
    F1 = np.fft.fft(np.array(acc1))
    F2 = np.fft.fft(np.array(acc2))
    # フーリエスペクトルの差分を取る
    spectrum_diff = F1 - F2

    # ローパスフィルタなどでノイズを除去
    cutoff_frequency = 12.5  # カットオフ周波数（適切な値を設定）
    spectrum_diff[np.abs(freq) > cutoff_frequency] = 0 + 0j
    spectrum_diff[np.abs(freq) < 0.1] = 0 + 0j

    #フーリエ逆変換
    g = np.fft.ifft(spectrum_diff)
    g = [float(i) for i in g]
    return g

#波形にローパスフィルタをかける（直線境界）
def Lowpass_filter(dt,acc,limit):
    freq = np.fft.fftfreq(len(acc), dt)  # 周波数軸
    # 高速フーリエ変換
    F = np.fft.fft(np.array(acc))
    # 元波形をコピーする
    G1 = F.copy()
    G1[np.abs(freq) > limit] = 0 + 0j
    # フーリエ逆変換
    g = np.fft.ifft(G1)
    g = [float(i) for i in g]
    return g

#cos20テーパーをかける
def cos_taper(acc,dt):
    taper_length = 10
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

# read csv
    #対象ファイルの波形分析
ttt=0
rec_file_name=[]
rec_frequency1=[];rec_frequency2=[]
N_wave=20000

file = "./地震データ/230611.183000.csv"
df = pd.read_csv(file, nrows = N_wave, header=0,skiprows=range(1, 304000))#地震時含む波形データ
df2 = pd.read_csv(file, nrows = N_wave,  header=0,skiprows=range(1, 304000-N_wave))#運転時の波形データ
print(df)
#波形のdt定義
dt=0.005
#ターゲット波形の中立軸補正
base_acc_NS = list(df['BFB_NS'])-np.mean(list(df['BFB_NS']))
base_acc_EW = list(df['BFB_EW'])-np.mean(list(df['BFB_EW']))
middle1_acc_NS = list(df['4F_NS'])-np.mean(list(df['4F_NS']))
middle1_acc_EW = list(df['4F_EW'])-np.mean(list(df['4F_EW']))
middle2_acc_NS = list(df['5F_NS'])-np.mean(list(df['5F_NS']))
middle2_acc_EW = list(df['5F_EW'])-np.mean(list(df['5F_EW']))
top_acc_NS = list(df['7F_NS'])-np.mean(list(df['7F_NS']))
top_acc_EW = list(df['7F_EW'])-np.mean(list(df['7F_EW']))

base_acc_NS_or = list(df2['BFB_NS'])-np.mean(list(df2['BFB_NS']))
base_acc_EW_or = list(df2['BFB_EW'])-np.mean(list(df2['BFB_EW']))
middle1_acc_NS_or = list(df2['4F_NS'])-np.mean(list(df2['4F_NS']))
middle1_acc_EW_or = list(df2['4F_EW'])-np.mean(list(df2['4F_EW']))
middle2_acc_NS_or = list(df2['5F_NS'])-np.mean(list(df2['5F_NS']))
middle2_acc_EW_or = list(df2['5F_EW'])-np.mean(list(df2['5F_EW']))
top_acc_NS_or = list(df2['7F_NS'])-np.mean(list(df2['7F_NS']))
top_acc_EW_or = list(df2['7F_EW'])-np.mean(list(df2['7F_EW']))

#波形にローパスフィルタをかける（直線境界）
base_acc_NS = Lowpass_filter(dt,base_acc_NS,12.5)
base_acc_EW = Lowpass_filter(dt,base_acc_EW,12.5)
middle1_acc_NS = Lowpass_filter(dt,middle1_acc_NS,12.5)
middle1_acc_EW = Lowpass_filter(dt,middle1_acc_EW,12.5)
middle2_acc_NS = Lowpass_filter(dt,middle2_acc_NS,12.5)
middle2_acc_EW = Lowpass_filter(dt,middle2_acc_EW,12.5)
top_acc_NS = Lowpass_filter(dt,top_acc_NS,12.5)
top_acc_EW = Lowpass_filter(dt,top_acc_EW,12.5)

#波形のフーリエスペクトルの差分を取って逆変換
middle1_acc_NS_el = eliminate_noize(dt,middle1_acc_NS,middle1_acc_NS_or)
middle1_acc_EW_el = eliminate_noize(dt,middle1_acc_EW,middle1_acc_EW_or)
middle2_acc_NS_el = eliminate_noize(dt,middle2_acc_NS,middle2_acc_NS_or)
middle2_acc_EW_el = eliminate_noize(dt,middle2_acc_EW,middle2_acc_EW_or)
top_acc_NS_el = eliminate_noize(dt,top_acc_NS,top_acc_NS_or)
top_acc_EW_el = eliminate_noize(dt,top_acc_EW,top_acc_EW_or)

# #補正後の波形出力
df1 = pd.DataFrame()
#     for i in range(len(wave_name)):
df1['BFB_NS']=base_acc_NS
df1['BFB_EW']=base_acc_EW
df1['4F_NS']=middle1_acc_NS
df1['4F_EW']=middle1_acc_EW
df1['5F_NS']=middle2_acc_NS
df1['5F_EW']=middle2_acc_EW
df1['7F_NS']=top_acc_NS
df1['7F_EW']=top_acc_EW
df1.to_csv('Modified_wave.csv')
# #補正後の波形出力
df1 = pd.DataFrame()
#     for i in range(len(wave_name)):
df1['BFB_NS']=base_acc_NS
df1['BFB_EW']=base_acc_EW
df1['4F_NS']=middle1_acc_NS_el
df1['4F_EW']=middle1_acc_EW_el
df1['5F_NS']=middle2_acc_NS_el
df1['5F_EW']=middle2_acc_EW_el
df1['7F_NS']=top_acc_NS_el
df1['7F_EW']=top_acc_EW_el
df1.to_csv('Modified_wave_windcut.csv')


a4_width_mm = 210
a4_height_mm = 297

# サブプロットを作成
fig, axs = plt.subplots(4, 1, figsize=(a4_width_mm / 25.4, a4_height_mm / 25.4), sharex=True)

# 各サブプロットにデータをプロット
time = [i * dt for i in range(0, len(df))]
axs[0].plot(time, base_acc_NS)
axs[1].plot(time, base_acc_EW)
axs[2].plot(time, middle1_acc_NS)
axs[3].plot(time, middle1_acc_EW)

# 各サブプロットにタイトルを追加（オプション）
axs[0].set_title('BFB_NS')
axs[1].set_title('BFB_EW')
axs[2].set_title('4F_NS')
axs[3].set_title('4F_EW')

# 各サブプロットのy軸ラベルを追加（オプション）
for ax in axs:
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration (m/s2)')
# グラフをレイアウト調整
fig.tight_layout()
# グラフを表示
fig.savefig("wave1")

# サブプロットを作成
fig, axs = plt.subplots(4, 1, figsize=(a4_width_mm / 25.4, a4_height_mm / 25.4), sharex=True)

# 各サブプロットにデータをプロット
time = [i * dt for i in range(0, len(df))]
axs[0].plot(time, middle2_acc_NS)
axs[1].plot(time, middle2_acc_EW)
axs[2].plot(time, top_acc_NS)
axs[3].plot(time, top_acc_EW)

# 各サブプロットにタイトルを追加（オプション）
axs[0].set_title('5F_NS')
axs[1].set_title('5F_EW')
axs[2].set_title('7F_NS')
axs[3].set_title('7F_EW')

# 各サブプロットのy軸ラベルを追加（オプション）
for ax in axs:
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration (m/s2)')
# グラフをレイアウト調整
fig.tight_layout()
# グラフを表示
fig.savefig("wave2")

# サブプロットを作成
fig, axs = plt.subplots(4, 1, figsize=(a4_width_mm / 25.4, a4_height_mm / 25.4), sharex=True)

# 各サブプロットにデータをプロット
time = [i * dt for i in range(0, len(df))]
axs[0].plot(time, base_acc_NS)
axs[1].plot(time, base_acc_EW)
axs[2].plot(time, middle1_acc_NS_el)
axs[3].plot(time, middle1_acc_EW_el)

# 各サブプロットにタイトルを追加（オプション）
axs[0].set_title('BFB_NS')
axs[1].set_title('BFB_EW')
axs[2].set_title('4F_NS')
axs[3].set_title('4F_EW')

# 各サブプロットのy軸ラベルを追加（オプション）
for ax in axs:
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration (m/s2)')
# グラフをレイアウト調整
fig.tight_layout()
# グラフを表示
fig.savefig("wave3")

# サブプロットを作成
fig, axs = plt.subplots(4, 1, figsize=(a4_width_mm / 25.4, a4_height_mm / 25.4), sharex=True)

# 各サブプロットにデータをプロット
time = [i * dt for i in range(0, len(df))]
axs[0].plot(time, middle2_acc_NS_el)
axs[1].plot(time, middle2_acc_EW_el)
axs[2].plot(time, top_acc_NS_el)
axs[3].plot(time, top_acc_EW_el)

# 各サブプロットにタイトルを追加（オプション）
axs[0].set_title('5F_NS')
axs[1].set_title('5F_EW')
axs[2].set_title('7F_NS')
axs[3].set_title('7F_EW')

# 各サブプロットのy軸ラベルを追加（オプション）
for ax in axs:
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration (m/s2)')
# グラフをレイアウト調整
fig.tight_layout()
# グラフを表示
fig.savefig("wave4")

    #フーリエ変換
data_list = ['BFB_NS','4F_NS','5F_NS','7F_NS']
graph_title="20230604_NS_seismic"
wave_fft.calc_fft(df1,dt,data_list,graph_title)
graph_title2="20230604_EW_seismic"
data_list2 = ['BFB_EW','4F_EW','5F_EW','7F_EW']
wave_fft.calc_fft(df1,dt,data_list2,graph_title2)
data_list = ['BFB_NS','4F_NS','5F_NS','7F_NS']
graph_title="20230604_NS_ordinal"
wave_fft.calc_fft(df2,dt,data_list,graph_title)
graph_title2="20230604_EW_ordinal"
data_list2 = ['BFB_EW','4F_EW','5F_EW','7F_EW']
wave_fft.calc_fft(df2,dt,data_list2,graph_title2)


# plot figures
#plt.scatter(data[selected_column1], data[selected_column2])
#plt.xlabel(selected_column1)
#plt.ylabel(selected_column2)
#plt.title('Scatter Plot of {} vs {}'.format(selected_column1, selected_column2))
#plt.show()