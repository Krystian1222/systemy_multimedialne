import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
Palety_szare = [np.linspace(0, 1, 2 ** i) for i in [1, 2, 4]]
Palety_kolorowe = [np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.],
                             [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]),
                   np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0.5, 0],
                             [0.5, 0.5, 0.5], [0, 1, 0], [0.5, 0, 0], [0, 0, 0.5], [0.5, 0.5, 0],
                             [0.5, 0, 0.5], [1, 0, 0], [0.75, 0.75, 0.75], [0, 0.5, 0.5], [1, 1, 1], [1, 1, 0]])]
Paleta_papryka = np.array([[133.0, 134.0, 83.0],
[184.0, 167.0, 58.0],
[192.0, 35.0, 18.0],
[109.0, 159.0, 63.0],
[57.0, 0.0, 0.0],
[166.0, 55.0, 39.0],
[20.0, 0.0, 0.0],
[192.0, 220.0, 204.0],
[187.0, 196.0, 85.0],
[49.0, 0.0, 4.0],
[107.0, 123.0, 57.0],
[148.0, 193.0, 111.0],
[177.0, 164.0, 131.0],
[89.0, 20.0, 5.0],
[102.0, 64.0, 15.0],
[109.0, 97.0, 29.0],
[170.0, 95.0, 28.0],
[86.0, 85.0, 16.0],
[189.0, 225.0, 153.0],
[222.0, 222.0, 214.0],
[107.0, 125.0, 70.0],
[191.0, 191.0, 63.0],
[86.0, 85.0, 16.0],
[168.0, 214.0, 188.0]])
for i in range(Paleta_papryka.shape[0]):
    for j in range(Paleta_papryka.shape[1]):
        Paleta_papryka[i][j] /= 255.0
###########################################################################
def macierz_Bayera(bity):
    M = np.array([[0.5]])
    if bity != 1:
        M = np.array([[0, 2], [3, 1]])
        i = 2
        while (i != bity):
            M = np.vstack((np.hstack((M * 4, M * 4 + 2)),
                           np.hstack((M * 4 + 3, M * 4 + 1))))
            i *= 2
        M = M / bity ** 2 - 0.5
    return M
###########################################################################
def colorFit_szary(wartosc, paleta):
    return paleta[np.argmin(np.abs(paleta - wartosc))]
###########################################################################
def colorFit(wartosc, paleta):
    return paleta[np.argmin(np.linalg.norm(paleta - wartosc, axis=1))]
###########################################################################
def dopasowanie_do_palety(obraz_wej, paleta, kolor=False):
    obraz_wyj = np.zeros((obraz_wej.shape))
    for x in range(obraz_wej.shape[0]):
        for y in range(obraz_wej.shape[1]):
            obraz_wyj[x][y] = colorFit(obraz_wej[x][y], paleta) if kolor else colorFit_szary(obraz_wej[x][y], paleta)
            print(obraz_wej[x][y])
    return obraz_wyj
###########################################################################
def dithering_losowy(obraz_wej):
    obraz_wyj = np.random.uniform(0, 1, obraz_wej.shape)
    for x in range(obraz_wej.shape[0]):
        for y in range(obraz_wej.shape[1]):
            obraz_wyj[x][y] = 0 if obraz_wyj[x][y] >= obraz_wej[x][y] else 1
    return obraz_wyj
###########################################################################
def dithering_zorganizowany(obraz_wej, paleta, bity, kolor=False):
    M = macierz_Bayera(bity)
    obraz_wyj = np.zeros(obraz_wej.shape)
    for x in range(obraz_wej.shape[0]):
        for y in range(obraz_wej.shape[1]):
            obraz_wyj[x][y] = colorFit(obraz_wej[x][y] + M[x % M.shape[0]][y % M.shape[0]], paleta) if kolor else colorFit_szary(obraz_wej[x][y] + M[x % M.shape[0]][y % M.shape[0]], paleta)
    return obraz_wyj
###########################################################################
def dithering_Floyda_Steinberga(obraz_wej, paleta, kolor=False):
    obraz_wyj = np.copy(obraz_wej)
    for x in range(obraz_wej.shape[0]):
        for y in range(obraz_wej.shape[1]):
            oldpixel = np.copy(obraz_wyj[x][y])
            newpixel = colorFit(oldpixel, paleta) if kolor else colorFit_szary(oldpixel, paleta)
            obraz_wyj[x][y] = newpixel
            quant_error = oldpixel - newpixel
            if x + 1 < obraz_wej.shape[0]:
                obraz_wyj[x + 1][y] += quant_error * 7 / 16
                if y + 1 < obraz_wej.shape[1]:
                    obraz_wyj[x + 1][y + 1] += quant_error * 1 / 16
            if y + 1 < obraz_wej.shape[1]:
                obraz_wyj[x][y + 1] += quant_error * 5 / 16
                obraz_wyj[x - 1][y + 1] += quant_error * 3 / 16
    return np.clip(obraz_wyj, 0, 1)
###########################################################################
def normalizacja(obraz):
    obraz_tmp = Image.open(obraz)
    Z_C = np.array(obraz_tmp)
    m, n = np.min(Z_C), np.max(Z_C)
    Z_A = (Z_C - m) / (n - m)
    return Z_A
###########################################################################
def obrazy_1_bit(obraz, nazwa):
    liczba_bitow = 8
    rozmiar_tytulu = 10
    fig = plt.figure(figsize=(20, 6))
    obraz = normalizacja(obraz)
    obrazy = [obraz, dopasowanie_do_palety(obraz, Palety_szare[0]), dithering_losowy(obraz),
              dithering_zorganizowany(obraz, Palety_szare[0], liczba_bitow), dithering_Floyda_Steinberga(obraz, Palety_szare[0])]
    tytuly = ["Obraz oryginalny", "Progowanie 1 bit", "Dithering losowy", "1 bit Dithering zorganizowany", "1 bit Dithering Floyda-Steinberga"]
    for i in range(len(tytuly)):
        ax = fig.add_subplot(1, 5, i + 1)
        plt.tight_layout()
        plt.imshow(obrazy[i], cmap='gray')
        ax.set_title(tytuly[i], fontsize=rozmiar_tytulu)
    plt.savefig("obraz_binarny_{}.png".format(nazwa))
###########################################################################
def obrazy_szare(obraz, nazwa):
    liczba_bitow = 8
    bity = [2, 4]
    rozmiar_tytulu = 10
    fig = plt.figure(figsize=(20, 12))
    obraz = normalizacja(obraz)
    obrazy = [[obraz, dopasowanie_do_palety(obraz, Palety_szare[i]), dithering_zorganizowany(obraz, Palety_szare[i], liczba_bitow), dithering_Floyda_Steinberga(obraz, Palety_szare[i])] for i in [1, 2]]
    tytuly = [["Obraz oryginalny", "Progowanie {} bity".format(i), "{} bity Dithering zorganizowany".format(i), "{} bity Dithering Floyda-Steinberga".format(i)] for i in bity]
    for i in range(len(bity)):
        for j in range(len(tytuly[0])):
            print(j + 1 if i < 1 else j + 5)
            ax = fig.add_subplot(2, 4, j + 1 if i < 1 else j + 5)
            plt.tight_layout()
            plt.imshow(obrazy[i][j], cmap='gray')
            ax.set_title(tytuly[i][j], fontsize=rozmiar_tytulu)
    plt.savefig("obraz_szary_{}.png".format(nazwa))
###########################################################################
def obrazy_kolorowe(obraz, nazwa):
    liczba_bitow = 8
    rozmiar_tytulu = 10
    obraz = normalizacja(obraz)
    indeksy_palet = [0, 1]

    obrazy = [[obraz, dopasowanie_do_palety(obraz, Palety_kolorowe[i], kolor=True),
               dithering_zorganizowany(obraz, Palety_kolorowe[i], liczba_bitow, kolor=True),
               dithering_Floyda_Steinberga(obraz, Palety_kolorowe[i], kolor=True)] for i in indeksy_palet]
    tytuly = [["Obraz oryginalny", "Obraz z pikselami dopasowanymi do palety {} kolorów".format(8 if i == 0 else 16), "Dithering zorganizowany", "Dithering Floyda-Steinberga"] for i in indeksy_palet]
    fig = plt.figure(figsize=(20, 12))
    for j in indeksy_palet:
        for k in range(len(tytuly[0])):
            print(k + 1 if j < 1 else k + 5)
            ax = fig.add_subplot(2, 4, k + 1 if j < 1 else k + 5)
            plt.tight_layout()
            plt.imshow(obrazy[j][k])
            ax.set_title(tytuly[j][k], fontsize=rozmiar_tytulu)
    plt.savefig("obraz_kolorowy_{}.png".format(nazwa))
###########################################################################
def papryka_obraz(obraz, nazwa):
    obraz = normalizacja(obraz)
    rozmiar_tytulu = 15
    fig = plt.figure(figsize=(20, 12))
    obrazy = [obraz, dithering_Floyda_Steinberga(obraz, Paleta_papryka, kolor=True)]
    tytuly = ["Obraz oryginalny", "Dithering Floyda-Steinberga z wybraną paletą"]
    for i in range(len(tytuly)):
        ax = fig.add_subplot(1, 2, i + 1)
        plt.tight_layout()
        plt.imshow(obrazy[i])
        ax.set_title(tytuly[i], fontsize = rozmiar_tytulu)
    plt.savefig("papryka_dopasowanie_wlasne_3.png")
###########################################################################
def glowny():
    obrazy_sz = ["0007.tif", "0008.png", "0009.png", "0010.jpg"]
    nazwy_sz = ["0007", "0008", "0009", "0010"]
    obrazy = ["0001.jpg", "0002.jpg", "0003.jpg","0004.png", "0005.tif", "0006.tif", "0011.jpg", "0012.jpg", "0013.jpg", "0014.jpg", "0015.jpg", "0016.jpg"]
    nazwy_obrazy = ["0001", "0002", "0003", "0004", "0005", "0006", "0011", "0012", "0013", "0014", "0015", "0016"]
    papryka_obraz("0006.tif", "0006")
    for j in range(len(obrazy_sz)):
        print(nazwy_sz[j])
        obrazy_1_bit(obrazy_sz[j], nazwy_sz[j])
        obrazy_szare(obrazy_sz[j], nazwy_sz[j])
    for i in range(len(obrazy)):
        print(nazwy_obrazy[i])
        obrazy_kolorowe(obrazy[i], nazwy_obrazy[i])
###########################################################################
if __name__ == '__main__':
    glowny()