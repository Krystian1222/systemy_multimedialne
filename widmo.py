import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import soundfile as sf
from scipy.interpolate import interp1d
import os
###############################################################################################################
str_dane_pierwotne = "dane_pierwotne/"
str_dane_widmo = "widmo/"
str_decymacja = "decymacja/"
str_kwantyzacja = "kwantyzacja/"
str_interpolacja = "interpolacja/"
str_dzwiek_decymacja = "dzwiek_decymacja/"
str_dzwiek_kwantyzacja = "dzwiek_kwantyzacja/"
str_dzwiek_interpolacja = "dzwiek_interpolacja/"
# os.mkdir(str_dane_widmo)
# os.mkdir(str_decymacja)
# os.mkdir(str_kwantyzacja)
# os.mkdir(str_interpolacja)
# os.mkdir(str_dzwiek_decymacja)
# os.mkdir(str_dzwiek_kwantyzacja)
# os.mkdir(str_dzwiek_interpolacja)
###############################################################################################################
pliki = ["sin_60Hz.wav", "sin_440Hz.wav", "sin_8000Hz.wav", "sin_combined.wav", "sing_high1.wav", "sing_high2.wav", "sing_low1.wav", "sing_low2.wav", "sing_medium1.wav", "sing_medium2.wav"]
liczba_bitow = [4, 8, 16, 24]
czestotliwosci = [2000, 4000, 8000, 16000, 24000, 41000, 16950]
###############################################################################################################
def usun_kanal(data):
    return data[:, 0] if len(data.shape) == 2 else data
##############################################################################################################
dane_fs = [sf.read(str_dane_pierwotne + pliki[i], dtype=np.int32) for i in range(len(pliki))]
dane = [dane_fs[i][0] for i in range(len(pliki))]
dane[4] = usun_kanal(dane[4])
fs = [dane_fs[i][1] for i in range(len(dane))]
###############################################################################################################
def rozdzielczosc_bitowa(dane, liczba_bitow):
    Min, Maks = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    nowe_min, nowe_maks = (-2 ** liczba_bitow) / 2, ((2 ** liczba_bitow) / 2) - 1
    print(nowe_min, nowe_maks)
    # przeliczenie na zakres osi y wzgledem bitow - dla 3 bitow powinno byc od -8 do 7
    Z_A = (np.int64(dane) - Min) / (np.int64(Maks - Min))
    Z_C = np.round(Z_A * (nowe_maks - nowe_min)) + nowe_min
    # przeliczenie na zakres oryginalnych danych
    Z_A = (np.int64(Z_C) - nowe_min) / (nowe_maks - nowe_min)
    Z_C = np.round(Z_A * (Maks - Min)) + Min
    return Z_C
###############################################################################################################
def decymacja(dane, stara_fs, nowa_fs):
    return dane[::int(stara_fs / nowa_fs)]
###############################################################################################################
def interpolacja(dane, stara_fs, nowa_fs, metoda):
    x = np.linspace(0, dane.size, dane.size)
    x_new = np.linspace(0, dane.size, int(dane.size / stara_fs * nowa_fs))
    return (interp1d(x, dane, metoda)(x_new)).astype(dane.dtype)
###############################################################################################################
def pliki_sin(dane, dane_mod_fs, fs, fs_mod, nazwa_pliku, do_nazwy, tytul, etykieta):
    fig1 = plt.figure(figsize=(16, 10))
    fig1.suptitle(nazwa_pliku + ": fs = " + str(fs) + "Hz, " + tytul, fontsize = 15)
    fsize = 2 ** 9
    yf = scipy.fftpack.fft(dane, fsize)
    plt.title("Połowa widma")
    plt.plot(np.arange(0, fs/2, fs/fsize), np.abs(yf[:fsize//2]), 'b', label = 'Oryginał')
    plt.legend(loc='upper right')
    plt.xlim(0, fs_mod/2)
    plt.xlabel("fs [Hz]")
    plt.ylabel("Amplituda")
    plt.savefig(str_dane_widmo + nazwa_pliku + "_" + do_nazwy + "polowa_oryginal.png")
    plt.close(fig1)
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle(nazwa_pliku + ": fs = " + str(fs) + "Hz, " + tytul, fontsize = 15)
    yf2 = scipy.fftpack.fft(dane_mod_fs, fsize)
    plt.title("Połowa widma")
    plt.plot(np.arange(0, fs_mod/2, fs_mod/fsize), np.abs(yf2[:fsize//2]), 'r', label = etykieta)
    plt.legend(loc='upper right')
    plt.xlim(0, fs_mod/2)
    plt.xlabel("fs [Hz]")
    plt.ylabel("Amplituda")
    plt.savefig(str_dane_widmo + nazwa_pliku + "_" + do_nazwy + "polowa" + etykieta + ".png")
    plt.close(fig2)
    fig3 = plt.figure(figsize = (16, 10))
    fig3.suptitle(nazwa_pliku + ": fs = " + str(fs) + "Hz, " + tytul, fontsize = 15)
    plt.title("Widmo w skali decybelowej")
    plt.plot(np.arange(0, fs/2, fs/fsize), 20*np.log10(np.abs(yf[:fsize//2])), 'b', label = 'Oryginał')
    plt.legend(loc = 'upper right')
    plt.xlim(0, fs_mod/2)
    plt.xlabel("fs [Hz]")
    plt.ylabel("Amplituda")
    plt.savefig(str_dane_widmo + nazwa_pliku + "_" + do_nazwy + "decybelowa_oryginal.png")
    plt.close(fig3)
    fig4 = plt.figure(figsize=(16, 10))
    fig4.suptitle(nazwa_pliku + ": fs = " + str(fs) + "Hz, " + tytul, fontsize = 15)
    plt.title("Widmo w skali decybelowej")
    if nazwa_pliku =="sin_8000Hz.wav":
        plt.plot(np.arange(0, fs_mod / 2, fs_mod / fsize), 20 * np.log10(np.abs(yf2[:fsize // 2]) + 1E-6), 'r', label = etykieta)
    else:
        plt.plot(np.arange(0, fs_mod / 2, fs_mod / fsize), 20 * np.log10(np.abs(yf2[:fsize // 2])), 'r', label=etykieta)
    plt.legend(loc = 'upper right')
    plt.xlim(0, fs_mod/2)
    plt.xlabel("fs [Hz]")
    plt.ylabel("Amplituda")
    plt.savefig(str_dane_widmo + nazwa_pliku + "_" + do_nazwy + "_decybelowa_" + etykieta + ".png")
    plt.close(fig4)
###############################################################################################################
def pliki_kwantyzacja():
    xlimy = [0.007, 0.0015, 0.0003, 0.0004]
    for i in range(len(pliki)):
        fig = plt.figure(figsize = (20, 15))
        for j in range(len(liczba_bitow)):
            y = rozdzielczosc_bitowa(dane = dane[i], liczba_bitow = liczba_bitow[j])
            x = np.linspace(0, y.shape[0] / len(y), y.shape[0])
            if i < 4:
                fig.suptitle(pliki[i], fontsize=15)
                ax = fig.add_subplot(2, 2, j + 1)
                plt.plot(x, y, 'r', label = 'Kwantyzacja')
                plt.plot(x, dane[i], 'b', label = 'Oryginał')
                plt.title("Kwantyzacja - " + str(liczba_bitow[j]) + " bitów")
                plt.xlim([0, xlimy[i]])
                plt.xlabel("Czas [s]")
                plt.ylabel("Amplituda")
                ax.legend(loc = 'upper left')
            sf.write(str_dzwiek_kwantyzacja  + pliki[i] + '_kwantyzacja_' + str(liczba_bitow[j]) + '.wav', y.astype(np.int32), fs[i])
        if i < 4:
            plt.savefig(str_kwantyzacja + pliki[i] + "_kwantyzacja_" + str(liczba_bitow[i]) + ".png")
            plt.close()
    for i in range(4):
        for j in range(len(liczba_bitow)):
            y = rozdzielczosc_bitowa(dane=dane[i], liczba_bitow=liczba_bitow[j])
            tytul = 'kwantyzacja ' + str(liczba_bitow[j]) + ' bitów'
            do_nazwy = 'kwantyzacja_' + str(liczba_bitow[j])
            pliki_sin(dane[i], y, fs[j], fs[j], pliki[i], do_nazwy, tytul, 'Kwantyzacja')
###############################################################################################################
def pliki_interpolacja():
    xlimy = [0.01, 0.005, 0.001, 0.02]
    for i in range(len(pliki)):
        for j in range(1):
            y_linear = interpolacja(dane[i], fs[i], czestotliwosci[0], 'linear')
            x_linear = np.linspace(0, y_linear.shape[0] / len(y_linear), y_linear.shape[0])
            y_cubic = interpolacja(dane[i], fs[i], czestotliwosci[0], 'cubic')
            x_cubic = np.linspace(0, y_cubic.shape[0] / len(y_cubic), y_cubic.shape[0])
            x = np.linspace(0, dane[i].shape[0] / len(dane[i]), dane[i].shape[0])
            #y = dane[i]
            # if i < 4:
            #     fig = plt.figure(figsize=(20, 10))
            #     fig.suptitle("Plik: " + pliki[i] + " " + "interpolacja - fs = " + str(czestotliwosci[6]) + ' Hz', fontsize = 15)
            #     ax1 = fig.add_subplot(1, 2, 1)
            #     ax2 = fig.add_subplot(1, 2, 2)
            #     ax1.set_title("Linear")
            #     ax2.set_title("Cubic")
            #     ax1.plot(x, y, 'b', label = 'oryginał')
            #     ax1.plot(x_linear, y_linear, 'r', label = 'linear')
            #     ax2.plot(x, y, 'b', label = 'oryginał')
            #     ax2.plot(x_cubic, y_cubic, 'r', label = 'cubic')
            #     ax1.set_xlabel("Czas [s]")
            #     ax1.set_ylabel("Amplituda")
            #     ax2.set_xlabel("Czas [s]")
            #     ax2.set_ylabel("Amplituda")
            #     ax1.legend(loc = 'upper left')
            #     ax2.legend(loc = 'upper left')
            #     ax1.set_xlim([0, xlimy[i]])
            #     ax2.set_xlim([0, xlimy[i]])
            #     plt.savefig(str_interpolacja + pliki[i] + "_interpolacja_" + str(czestotliwosci[6]) + ".png")
            #     plt.close()
            sf.write(str_dzwiek_interpolacja + pliki[i] + '_interpolacja_linear_' + str(czestotliwosci[0]) + '.wav', y_linear.astype(np.int32), czestotliwosci[0])
            sf.write(str_dzwiek_interpolacja + pliki[i] + '_interpolacja_cubic_' + str(czestotliwosci[0]) + '.wav', y_cubic.astype(np.int32), czestotliwosci[0])
    # for i in range(4):
    #     for j in range(1):
    #         y_linear = interpolacja(dane[i], fs[i], czestotliwosci[0], 'linear')
    #         tytul = 'interpolacja linear ' + str(czestotliwosci[0]) + ' Hz'
    #         do_nazwy = 'interpolacja_' + str(czestotliwosci[0]) + 'linear'
    #         pliki_sin(dane[i], y_linear, fs[6], czestotliwosci[0], pliki[i], do_nazwy, tytul, 'Interpolacja')
    # for i in range(4):
    #     for j in range(1):
    #         y_cubic = interpolacja(dane[i], fs[i], czestotliwosci[6], 'cubic')
    #         tytul = 'interpolacja cubic ' + str(czestotliwosci[6]) + ' Hz'
    #         do_nazwy = 'interpolacja_' + str(czestotliwosci[6]) + 'cubic'
    #         pliki_sin(dane[i], y_cubic, fs[5], czestotliwosci[6], pliki[i], do_nazwy, tytul, 'Interpolacja')
###############################################################################################################
def pliki_decymacja():
    xlimy = [0.004, 0.01, 0.0003, 0.01]
    for i in range(len(pliki)):
        for j in range(len(czestotliwosci) - 2):
            y = decymacja(dane = dane[i], stara_fs = fs[i], nowa_fs = czestotliwosci[j])
            x_d = np.linspace(0, y.shape[0] / len(y), y.shape[0])
            x = np.linspace(0, dane[i].shape[0] / len(dane[i]), dane[i].shape[0])
            if i < 4:
                fig = plt.figure(figsize = (20, 10))
                fig.suptitle("Plik: " + pliki[i] + " " + "decymacja - fs = " + str(czestotliwosci[j]) + " Hz", fontsize = 15)
                plt.plot(x, dane[i], 'b', label = 'oryginał')
                plt.plot(x_d, y, 'r', label = 'decymacja')
                plt.xlim(0, xlimy[i])
                plt.xlabel("Czas [s]")
                plt.ylabel("Amplituda")
                plt.legend(loc = 'upper left')
                plt.savefig(str_decymacja + pliki[i] + "_decymacja_" + str(czestotliwosci[j]) + ".png")
                plt.close()
            sf.write(str_dzwiek_decymacja + pliki[i] + '_decymacja_' + str(czestotliwosci[j]) + '.wav', y.astype(np.int32), czestotliwosci[j])
    for i in range(4):
        for j in range(len(czestotliwosci) - 2):
            y = decymacja(dane[i], fs[i], czestotliwosci[j])
            tytul = 'decymacja ' + str(czestotliwosci[j]) + ' Hz'
            do_nazwy = 'decymacja_' + str(czestotliwosci[j])
            pliki_sin(dane[i], y, fs[j], czestotliwosci[j], pliki[i], do_nazwy, tytul, 'Decymacja')
###############################################################################################################
def glowny():
    # y = np.arange(np.iinfo(np.int32).min,np.iinfo(np.int32).max,1000,dtype=np.int32)
    # y_kw = rozdzielczosc_bitowa(y, 1)
    # #plt.plot(y)
    # plt.plot(y_kw, 'k')
    # plt.show()
    # pliki_kwantyzacja()
    # pliki_decymacja()
    pliki_interpolacja()
###############################################################################################################
if __name__ == "__main__":
    glowny()