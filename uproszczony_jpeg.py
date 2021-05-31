import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2
###############################################################################################################
class JPEG:
    pass
###############################################################################################################
str_dane_pierwotne = "dane_pierwotne/"
str_dane_wyjsciowe = "dane_wyjsciowe/"
###############################################################################################################
QY = [np.array([
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 36, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
        ]).reshape(8, 8, 1),
      np.ones((8, 8, 1))]
###############################################################################################################
QC = [np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        ]).reshape(8, 8, 1),
      np.ones((8, 8, 1))]
###############################################################################################################
def zigzag(A, bezuzyteczny):
    template = n = np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8,1))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c,0]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B
###############################################################################################################
def kodowanie_RLE(dane, bezuzyteczny):
    wyj = np.zeros([dane.size * 2])
    licznik = 1
    poprzednia_wartosc = dane[0]
    indeks = 0
    for i in range(1, dane.size):
        if dane[i] == poprzednia_wartosc:
            licznik += 1
        else:
            wyj[indeks] = licznik
            wyj[indeks + 1] = poprzednia_wartosc
            indeks += 2
            poprzednia_wartosc = dane[i]
            licznik = 1
    wyj[indeks] = licznik
    wyj[indeks + 1] = poprzednia_wartosc
    return (np.resize(wyj, [indeks + 2])[0: indeks + 2]).astype(int)
###############################################################################################################
def dekodowanie_RLE(dane, bezuzyteczny):
    tab = [0] * int(dane.size / 2)
    for i in range(0, dane.size, 2):
        tab[int(i / 2)] = np.repeat(dane[i + 1], dane[i])
    return np.concatenate(tab).astype(int)
###############################################################################################################
def dct2(a, bezuzyteczny):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.astype(float) - 128, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
###############################################################################################################
def idct2(a, bezuzyteczny):
    return scipy.fftpack.idct(scipy.fftpack.idct(a.astype(float), axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho') + 128
###############################################################################################################
def subsampling(warstwa, typ):
    return warstwa if typ == "4:4:4" else warstwa[:, ::2, :]
###############################################################################################################
def resampling(warstwa, typ):
    return warstwa if typ == "4:4:4" else np.repeat(warstwa, 2, axis = 1)
###############################################################################################################
def kwantyzacja(wartosci, macierz_kwantyzacji):
    return np.round(wartosci / macierz_kwantyzacji).astype(int)
###############################################################################################################
def dekwantyzacja(wartosci, macierz_dekwantyzacji):
    return wartosci * macierz_dekwantyzacji
###############################################################################################################
def operacja(lista_blokow, funkcja, lista_Q):
    lista, nowe_bloki = [[], [], []], [[], [], []]
    if lista_Q != []:
        for i in range(len(lista_Q)):
            lista[i] = lista_Q[i]
    for i in range(len(nowe_bloki)):
        for j in lista_blokow[i]:
            nowe_bloki[i].append(funkcja(np.array(j), lista[i]))
    return nowe_bloki
###############################################################################################################
def podziel_na_bloki(lista_warstw):
    bloki = [[], [], []]
    for i in range(len(lista_warstw)):
        podzial_pionowy = np.vsplit(lista_warstw[i], lista_warstw[i].shape[0] // 8)
        for blok_pionowy in podzial_pionowy:
            podzial_poziomy = np.hsplit(blok_pionowy, lista_warstw[i].shape[1] // 8)
            for blok_poziomy in podzial_poziomy:
                bloki[i].append(blok_poziomy)
    return bloki
###############################################################################################################
def polacz_bloki(lista_blokow, lista_wymiarow):
    warstwy = [np.zeros(lista_wymiarow[i]) for i in range(len(lista_blokow))]
    for i in range(len(lista_wymiarow)):
        for y in range(0, lista_wymiarow[i][0], 8):
            for x in range(0, lista_wymiarow[i][1], 8):
                warstwy[i][y:y + 8, x:x + 8] = lista_blokow[i][y // 8 * (lista_wymiarow[i][1] // 8) + x // 8]
    return [warstwy[i].clip(0, 255).astype(np.uint8) for i in range(len(lista_wymiarow))]
###############################################################################################################
def jotpeg_kompresja(obraz, indeks_jakosci, chrominancja):
    jotpeg = JPEG()
    jotpeg.QY, jotpeg.QC = QY[indeks_jakosci], QC[indeks_jakosci]
    jotpeg.wymiary, jotpeg.chrominancja = obraz.shape, chrominancja
    obraz = np.pad(obraz, ((0, 16 - obraz.shape[0] % 16), (0, 16 - obraz.shape[1] % 16), (0, 0)), mode = 'edge')
    y, Cr, Cb = np.dsplit(cv2.cvtColor(obraz, cv2.COLOR_RGB2YCrCb), (1, 2))
    y, Cr, Cb = y.astype(int), Cr.astype(int), Cb.astype(int)
    Cr, Cb = subsampling(Cr, chrominancja), subsampling(Cb, chrominancja)
    jotpeg.lista, lista_pusta = podziel_na_bloki([y, Cr, Cb]), []
    jotpeg.wymiar_y, jotpeg.wymiar_Cr, jotpeg.wymiar_Cb = y.shape, Cr.shape, Cb.shape
    lista_Q = [jotpeg.QY, jotpeg.QC, jotpeg.QC]
    jotpeg.lista = operacja(jotpeg.lista, dct2, lista_pusta)
    jotpeg.lista = operacja(jotpeg.lista, kwantyzacja, lista_Q)
    jotpeg.lista = operacja(jotpeg.lista, zigzag, lista_pusta)
    jotpeg.lista = operacja(jotpeg.lista, kodowanie_RLE, lista_pusta)
    return jotpeg
###############################################################################################################
def jotpeg_dekompresja(jotpeg):
    lista2, lista_pusta, lista_Q = jotpeg.lista, [], [jotpeg.QY, jotpeg.QC, jotpeg.QC]
    lista2 = operacja(jotpeg.lista, dekodowanie_RLE, lista_pusta)
    lista2 = operacja(lista2, zigzag, lista_pusta)
    lista2 = operacja(lista2, dekwantyzacja, lista_Q)
    lista2 = operacja(lista2, idct2, lista_pusta)
    bloki = polacz_bloki(lista2, [jotpeg.wymiar_y, jotpeg.wymiar_Cr, jotpeg.wymiar_Cb])
    bloki[1], bloki[2] = resampling(bloki[1], jotpeg.chrominancja), resampling(bloki[2], jotpeg.chrominancja)
    return cv2.cvtColor(np.dstack((bloki[0], bloki[1], bloki[2])), cv2.COLOR_YCrCb2RGB)[0:jotpeg.wymiary[0], 0:jotpeg.wymiary[1]]
###############################################################################################################
def rysuj_obraz(nazwa, wycinek):
    nazwa = str_dane_pierwotne + nazwa
    print(nazwa, wycinek)
    obraz = cv2.cvtColor(cv2.imread(nazwa), cv2.COLOR_BGR2RGB)
    nazwa = nazwa.split('/')[1]
    if wycinek:
        obraz = obraz[wycinek[0]: (wycinek[0] + wycinek[2]), wycinek[1]:(wycinek[1] + wycinek[3])]
    lista_jakosci = [0, 1]
    lista_chrominancji = ["4:4:4", "4:2:2"]
    jakosc_tytuly = [50, 100]
    plt.figure(figsize = (28, 7))
    plt.suptitle(nazwa, fontsize = 25)
    plt.tight_layout()
    plt.subplot(1, 5, 1)
    plt.imshow(obraz)
    plt.title("Obraz oryginalny")
    k = 1
    for i in range(len(lista_jakosci)):
        for j in range(len(lista_chrominancji)):
            k += 1
            jotpeg = jotpeg_kompresja(obraz, lista_jakosci[i], lista_chrominancji[j])
            obraz_odtworzony = jotpeg_dekompresja(jotpeg)
            plt.subplot(1, 5, k)
            plt.imshow(obraz_odtworzony)
            plt.title("Obraz odtworzony\njakość: {}%, chrominancja: {}".format(jakosc_tytuly[i], lista_chrominancji[j]))
    if wycinek:
        plt.savefig(str_dane_wyjsciowe + nazwa + str(wycinek[0]) + '_' + str(wycinek[1]) + '_' + str(wycinek[2]) + '_' + str(wycinek[3]) + '.jpeg')
    else:
        plt.savefig(str_dane_wyjsciowe + nazwa + '.jpeg')
    plt.close()
###############################################################################################################
def przygotuj_obrazy():
    lista_obrazow = ['rysunek_techniczny.png', 'rysunek_techniczny.png', 'rysunek_techniczny.png', 'krajobraz1.png', 'krajobraz1.png', 'krajobraz1.png', 'krajobraz2.png', 'krajobraz2.png', 'krajobraz2.png', 'krajobraz3.png', 'krajobraz3.png', 'krajobraz3.png', 'krajobraz4.png', 'krajobraz4.png', 'krajobraz4.png', 'podpis1.png', 'podpis1.png', 'podpis1.png', 'rysunek_techniczny.png', 'krajobraz1.png', 'krajobraz2.png', 'krajobraz3.png', 'krajobraz4.png', 'podpis1.png']
    lista_wycinkow = [[78,87,512,512], [582,750,256,256], [1467,1710,128,128], [120,530,512,512], [474,635,256,256], [465,990,128,128], [402,12,512,512], [687,546,256,256], [411,1329,128,128], [6,6,256,256], [348,8,128,128], [101,302,512,512], [255,8,128,128], [389,489,256,256], [44,456,512,512], [117,275,128,128], [27,576,256,256], [104,798,512,512], None, None, None, None, None, None]
    for i in range(len(lista_obrazow)):
        rysuj_obraz(lista_obrazow[i], lista_wycinkow[i])
###############################################################################################################
if __name__ == '__main__':
    przygotuj_obrazy()