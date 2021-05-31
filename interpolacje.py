import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
import random
###################################################################################################
def najblizszy_sasiad(obraz_wej, skala):
    obraz_wyj = np.zeros((int(obraz_wej.shape[0] * skala), int(obraz_wej.shape[1] * skala),
                          obraz_wej.shape[2])).astype(np.uint8)
    for x in range(obraz_wyj.shape[0]):
        for y in range(obraz_wyj.shape[1]):
            x_wej, y_wej = floor(x / skala), floor(y / skala)
            for z in range(obraz_wej.shape[2]):
                obraz_wyj[x][y][z] = obraz_wej[x_wej][y_wej][z]
    return obraz_wyj
###################################################################################################
def interpolacja_dwuliniowa(obraz_wej, skala):
    obraz_wyj = np.zeros((int(obraz_wej.shape[0] * skala), int(obraz_wej.shape[1] * skala), obraz_wej.shape[2])).astype(np.uint8)
    wys = np.linspace(0, obraz_wej.shape[0] - 1, int(obraz_wej.shape[0] * skala))
    szer = np.linspace(0, obraz_wej.shape[1] - 1, int(obraz_wej.shape[1] * skala))
    for x in range(obraz_wyj.shape[0]):
        for y in range(obraz_wyj.shape[1]):
            x_d, y_d, x_g, y_g = floor(wys[x]), floor(szer[y]), ceil(wys[x]), ceil(szer[y])
            xn, yn = wys[x] - x_d, szer[y] - y_d
            for z in range(obraz_wej.shape[2]):
                obraz_wyj[x][y][z] = obraz_wej[x_d][y_d][z] * (1 - xn) * (1 - yn) + obraz_wej[x_g][y_d][z] * xn * (1 - yn) + obraz_wej[x_d][y_g][z] * (1 - xn) * yn + obraz_wej[x_g][y_g][z] * xn * yn
    return obraz_wyj
###################################################################################################
def srednia(obraz_wej, skala):
    obraz_wyj = np.zeros((int(obraz_wej.shape[0] * skala), int(obraz_wej.shape[1] * skala), obraz_wej.shape[2])).astype(np.uint8)
    for x in range(obraz_wyj.shape[0]):
        for y in range(obraz_wyj.shape[1]):
            xn = [a for a in list(np.arange(int(x * 1 / skala - 1 / skala), int(x * 1 / skala + 1 / skala))) if a >= 0]
            yn = [a for a in list(np.arange(int(y * 1 / skala - 1 / skala), int(y * 1 / skala + 1 / skala))) if a >= 0]
            tab = [obraz_wej[i][j] for j in yn for i in xn]
            wynik = list()
            for k in range(obraz_wej.shape[2]):
                suma = 0
                for l in range(len(tab)):
                    suma += tab[l][k]
                wynik.append(suma / len(tab))
            obraz_wyj[x][y] = wynik
    return obraz_wyj
###################################################################################################
def srednia_wazona(obraz_wej, skala):
    obraz_wyj = np.zeros((int(obraz_wej.shape[0] * skala), int(obraz_wej.shape[1] * skala), obraz_wej.shape[2])).astype(np.uint8)
    for x in range(obraz_wyj.shape[0]):
        for y in range(obraz_wyj.shape[1]):
            xn = [a for a in list(np.arange(int(x * 1 / skala - 1 / skala), int(x * 1 / skala + 1 / skala))) if a >= 0]
            yn = [a for a in list(np.arange(int(y * 1 / skala - 1 / skala), int(y * 1 / skala + 1 / skala))) if a >= 0]
            tab = [obraz_wej[i][j] for j in yn for i in xn]
            wagi, wynik, suma_wag = list(), list(), 0
            for i in range(len(xn)):
                for j in range(len(yn)):
                    war = random.randint(1, 10) / 10
                    wagi.append(war)
                    suma_wag += war
            for k in range(obraz_wej.shape[2]):
                suma = 0
                for l in range(len(tab)):
                    suma += (tab[l][k] * wagi[l])
                wynik.append(suma / suma_wag)
            obraz_wyj[x][y] = wynik
    return obraz_wyj
###################################################################################################
def mediana(obraz_wej, skala):
    obraz_wyj = np.zeros((int(obraz_wej.shape[0] * skala), int(obraz_wej.shape[1] * skala), obraz_wej.shape[2])).astype(np.uint8)
    for x in range(obraz_wyj.shape[0]):
        for y in range(obraz_wyj.shape[1]):
            xn = [a for a in list(np.arange(int(x * 1 / skala - 1 / skala), int(x * 1 / skala + 1 / skala))) if a >= 0]
            yn = [a for a in list(np.arange(int(y * 1 / skala - 1 / skala), int(y * 1 / skala + 1 / skala))) if a >= 0]
            tab = [obraz_wej[i][j] for j in yn for i in xn]
            wynik, do_mediany = list(), np.zeros((len(tab)))
            for k in range(obraz_wej.shape[2]):
                for l in range(len(tab)):
                    do_mediany = tab[l][k]
                wynik.append(np.median(do_mediany))
            obraz_wyj[x][y] = wynik
    return obraz_wyj
###################################################################################################
def obrazy_powiekszone(obraz, skala, nazwa):
    rozmiar_tytulu = 7
    fig = plt.figure(figsize = (16, 7))
    obrazy = [obraz, najblizszy_sasiad(obraz, skala), cv2.resize(obraz, (int(obraz.shape[1] * skala), int(obraz.shape[0] * skala)), interpolation=cv2.INTER_NEAREST), interpolacja_dwuliniowa(obraz, skala), cv2.resize(obraz, (int(obraz.shape[1] * skala), int(obraz.shape[0] * skala)), interpolation=cv2.INTER_LINEAR)]
    tytuly = [
        "Obraz oryginalny", "Najbliższy sąsiad - skala = {}".format(skala), "Najbliższy sąsiad - wbudowany - skala = {}".format(skala),
        "Interpolacja dwuliniowa - skala = {}".format(skala), "Interpolacja dwuliniowa - wbudowana - skala = {}".format(skala)]
    for i in range(len(tytuly)):
        tytuly.append(tytuly[i])
    obrazy += [cv2.Canny(obrazy[i], 100, 200) for i in range(len(obrazy))]
    for i in range(len(tytuly)):
        ax = fig.add_subplot(2, 5, i + 1)
        plt.imshow(obrazy[i]) if i < 5 else plt.imshow(obrazy[i], cmap = plt.cm.gray)
        ax.set_title(tytuly[i], fontsize = rozmiar_tytulu)
    plt.savefig("powiekszanie_2_skala_{}_{}.png".format(skala, nazwa))
###################################################################################################
def obrazy_pomniejszone(obraz, skala, nazwa):
    rozmiar_tytulu = 7
    fig = plt.figure(figsize = (20, 7))
    obrazy = [obraz, najblizszy_sasiad(obraz, skala), interpolacja_dwuliniowa(obraz, skala), srednia(obraz, skala), srednia_wazona(obraz, skala), mediana(obraz, skala)]
    tytuly = ["Obraz oryginalny".format(skala), "Najbliższy sąsiad - skala = {}".format(skala), "Interpolacja dwuliniowa - skala = {}".format(skala),
        "Średnia - skala = {}".format(skala), "Średnia ważona - skala = {}".format(skala), "Mediana - skala = {}".format(skala)]
    for i in range(len(tytuly)):
        tytuly.append(tytuly[i])
    obrazy += [cv2.Canny(obrazy[i], 100, 200) for i in range(len(obrazy))]
    for i in range(len(tytuly)):
        ax = fig.add_subplot(2, 6, i + 1)
        plt.tight_layout()
        plt.imshow(obrazy[i]) if i < 6 else plt.imshow(obrazy[i], cmap = plt.cm.gray)
        ax.set_title(tytuly[i], fontsize = rozmiar_tytulu)
    plt.savefig("pomniejszanie_2_skala_{}_{}.png".format(skala, nazwa))
###################################################################################################
def main_test():
    obrazy = [plt.imread("0001.jpg"), plt.imread("0002.jpg"), plt.imread("0003.jpg"), plt.imread("0004.jpg"), plt.imread("0005.jpg"), plt.imread("0006.jpg"), plt.imread("0007.jpg"), plt.imread("0008.tif")]
    nazwy = ["000{}".format(i) for i in range(1, 9)]
    skale = [1.5, 2, 4, 8, 16, 0.05, 0.15, 0.25, 0.5, 0.75]
    for i in range(len(obrazy)):
        for j in range(len(skale)):
            if i < 4 and j < 5:
                obrazy_powiekszone(obrazy[i], skale[j], nazwy[i])
            elif i >= 4 and j >= 5:
                pass
                obrazy_pomniejszone(obrazy[i], skale[j], nazwy[i])
###################################################################################################
main_test()