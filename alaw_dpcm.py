import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
###############################################################################################################
def usun_kanal(data):
    return data[:, 0] if len(data.shape) == 2 else data
###############################################################################################################
str_dane_pierwotne = "dane_pierwotne/"
str_dane_wyjsciowe = "dane_wyjsciowe/"
pliki = ["sin_60Hz.wav", "sin_440Hz.wav", "sing_high1.wav", "sing_high2.wav", "sing_low1.wav", "sing_low2.wav", "sing_medium1.wav", "sing_medium2.wav", "sin_combined.wav"]
liczba_bitow = [i for i in range(2, 9)]
dane_fs = [sf.read(str_dane_pierwotne + pliki[i], dtype = np.int32) for i in range(len(pliki))]
dane = [dane_fs[i][0] for i in range(len(pliki))]
dane[2] = usun_kanal(dane[2])
fs = [dane_fs[i][1] for i in range(len(dane))]
###############################################################################################################
def kwantyzacja(dane, liczba_bitow):
    dane = dane.copy()
    Min, Max = np.min(dane), np.max(dane)
    Z_A = (dane - Min) / (Max - Min)
    nMin, nMax = -(2 ** liczba_bitow) / 2, ((2 ** liczba_bitow) / 2) - 1
    Z_C = np.round(Z_A * (nMax - nMin)) + nMin
    return Z_C / ((2 ** liczba_bitow) / 2)
###############################################################################################################
def Alaw_kompresja(dane, bezuzyteczny, A = 87.6):
    dane = dane / np.iinfo(np.int32).max
    dane_alaw = []
    for i in range(len(dane)):
        if np.abs(dane[i]) < (1 / A):
            dane_alaw.append(np.sign(dane[i]) * ((A * np.abs(dane[i])) / (1 + np.log(A))))
        if (1 / A) <= np.abs(dane[i]) <= 1:
            dane_alaw.append(np.sign(dane[i]) * ((1 + np.log(A * np.abs(dane[i]))) / (1 + np.log(A))))
    return np.array(dane_alaw)
###############################################################################################################
def Alaw_dekompresja(dane_alaw, bezuzyteczny, A = 87.6):
    dane_alaw_kopia = dane_alaw.copy()
    dane = []
    for i in range(len(dane_alaw_kopia)):
        if np.abs(dane_alaw_kopia[i]) < 1 / (1 + np.log(A)):
            dane.append(np.sign(dane_alaw_kopia[i]) * ((np.abs(dane_alaw_kopia[i]) * (1 + np.log(A))) / A))
        if (1 / (1 + np.log(A))) <= np.abs(dane_alaw_kopia[i]) <= 1:
            dane.append(np.sign(dane_alaw_kopia[i]) * ((np.exp(np.abs(dane_alaw_kopia[i]) * (1 + np.log(A)) - 1)) / (A)))
    return np.array(dane) * np.iinfo(np.int32).max
###############################################################################################################
def DPCM_kompresja(dane, liczba_bitow):
    dane_2 = dane.copy()
    granica = 2 ** (liczba_bitow - 1) - 1
    maks_int = np.iinfo(np.int32).max
    wyj = np.zeros(dane.shape[0])
    E = dane_2[0]
    wyj[0] = dane_2[0] / (maks_int - 1) * granica
    for i in range(1, len(dane_2)):
        y1 = round((dane_2[i] - E) / (maks_int - 1) * granica)
        wyj[i] = y1
        y2 = y1 / granica * (maks_int - 1)
        E += y2
    return wyj
###############################################################################################################
def DPCM_dekompresja(dane, liczba_bitow):
    dane_2 = dane.copy()
    granica = 2 ** (liczba_bitow - 1) - 1
    x = np.zeros(dane_2.shape)
    x[0] = dane_2[0]
    for i in range(1, dane_2.shape[0]):
        x[i] = x[i - 1] + dane_2[i]
    x = x / granica * (np.iinfo(np.int32).max - 1)
    x = x.astype(np.int32)
    return x
###############################################################################################################
kompresje = [Alaw_kompresja, DPCM_kompresja]
nazwy = ["ALAW", "DPCM"]
###############################################################################################################
def glowny():
    for i in range(len(kompresje)):
        for j in range(len(liczba_bitow)):
            for k in range(2, len(dane)):
                dzwiek = kompresje[i](dane[k], liczba_bitow[j])
                if i == 0:
                    dzwiek_k = kwantyzacja(dzwiek, liczba_bitow[j])
                    sf.write(str_dane_wyjsciowe + nazwy[i] + '_' + str(liczba_bitow[j]) + '_' + pliki[k], dzwiek_k, fs[0])
                else:
                    sf.write(str_dane_wyjsciowe + nazwy[i] + '_' + str(liczba_bitow[j]) + '_' + pliki[k], dzwiek, fs[0])
###############################################################################################################
def wykresy():
    pliki_wykresy = [pliki[-1]]
    dane_n =[dane[-1]]
    liczba_bitow = [3, 4, 5, 6, 7, 8]
    osx = [[0, 0.008]]
    rozmiar_czcionki = 15
    for i in range(len(pliki_wykresy)):
        for j in range(len(liczba_bitow)):
            fig1 = plt.figure(figsize = (8, 12))
            plt.tight_layout()
            fig1.suptitle(pliki_wykresy[i] + ', liczba bitów: ' + str(liczba_bitow[j]), fontsize = 25)
            x = np.linspace(0, dane_n[i].shape[0] / len(dane_n[i]), dane_n[i].shape[0])
            ax1 = fig1.add_subplot(3, 1, 1)
            plt.plot(x, dane_n[i])
            ax1.set_title("Oryginalny sygnał", fontsize = rozmiar_czcionki)
            plt.xlim(osx[i])
            ax2 = fig1.add_subplot(3, 1, 2)
            plt.plot(x, kwantyzacja(Alaw_kompresja(dane_n[i], 0), liczba_bitow[j]))
            ax2.set_title("Kompresja A-law", fontsize = rozmiar_czcionki)
            plt.xlim(osx[i])
            ax3 = fig1.add_subplot(3, 1, 3)
            plt.plot(x, Alaw_dekompresja(kwantyzacja(Alaw_kompresja(dane_n[i], 0), liczba_bitow[j]), 0))
            ax3.set_title("Dekompresja A-law", fontsize = rozmiar_czcionki)
            plt.xlim(osx[i])
            #plt.show()
            plt.savefig(str_dane_wyjsciowe + 'ALAW_' + pliki_wykresy[i] + '_' + str(liczba_bitow[j]) + '.png')
            plt.close(fig1)
            fig2 = plt.figure(figsize=(8, 12))
            plt.tight_layout()
            fig2.suptitle(pliki_wykresy[i] + ', liczba bitów: ' + str(liczba_bitow[j]), fontsize = 25)
            ax4 = fig2.add_subplot(3, 1, 1)
            plt.plot(x, dane_n[i])
            ax4.set_title("Oryginalny sygnał", fontsize = rozmiar_czcionki)
            plt.xlim(osx[i])
            ax5 = fig2.add_subplot(3, 1, 2)
            plt.plot(x, DPCM_kompresja(dane_n[i], liczba_bitow[j]))
            #print("liczba bitów: " + str(liczba_bitow[j]) + ", liczba unikalnych wartości: " + str(np.unique(DPCM_kompresja(dane_n[i], liczba_bitow[j])).size))
            ax5.set_title("Kompresja DPCM", fontsize = rozmiar_czcionki)
            plt.xlim(osx[i])
            ax6 = fig2.add_subplot(3, 1, 3)
            plt.plot(x, DPCM_dekompresja(DPCM_kompresja(dane_n[i], liczba_bitow[j]), liczba_bitow[j]))
            ax6.set_title("Dekompresja DPCM", fontsize = rozmiar_czcionki)
            plt.xlim(osx[i])
            #plt.show()
            plt.savefig(str_dane_wyjsciowe + 'DPCM_' + pliki_wykresy[i] + '_' + str(liczba_bitow[j]) + '.png')
            plt.close(fig2)
            print('wykres')

###############################################################################################################
if __name__ == "__main__":
    #glowny()
    wykresy()