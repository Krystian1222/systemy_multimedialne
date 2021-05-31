import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pympler.asizeof as asizeof
import sys
from tqdm import tqdm
import cv2
###############################################################################################################
def kodowanie_RLE(dane):
    wyj = np.zeros([dane.size * 2])
    licznik = 1
    poprzednia_wartosc = dane[0]
    indeks = 0
    for i in tqdm(range(1, dane.size)):
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
def dekodowanie_RLE(dane):
    tab = [0] * int(dane.size / 2)
    for i in tqdm(range(0, dane.size, 2)):
        tab[int(i / 2)] = np.repeat(dane[i + 1], dane[i])
    return np.concatenate(tab).astype(int)
###############################################################################################################
class Drzewo_czworkowe:
    def podziel_obraz(self, obraz):
        podzial = np.array_split(obraz, 2, axis = 0)
        return (*np.array_split(podzial[0], 2, axis = 1), *np.array_split(podzial[1], 2, axis = 1))

    def polacz_obraz(self, kierunki):
        return np.concatenate((np.concatenate((kierunki[0], kierunki[1]), axis = 1), np.concatenate((kierunki[2], kierunki[3]), axis = 1)), axis = 0)

    def __init__(self, obraz):
        self.rozmiar = (obraz.shape[0], obraz.shape[1])
        self.koniec = True
        if obraz.size == 0:
            self.srednia_kafelka = 0 if len(obraz.shape) != 3 else np.zeros((3,)).astype(np.uint8)
            return
        self.srednia_kafelka = np.mean(obraz, axis = (0, 1)).astype(np.uint8)
        if not (obraz == (self.srednia_kafelka)).all():
            podzielony_img = self.podziel_obraz(obraz)
            self.koniec = False
            self.kier = [Drzewo_czworkowe(podzielony_img[i]) for i in range(4)]

    def wez_obraz(self):
        if self.koniec:
            return np.tile(self.srednia_kafelka, (*self.rozmiar, 1))
        return self.polacz_obraz([self.kier[i].wez_obraz() for i in range(4)]).astype(int)
###############################################################################################################
def glowny():
    nazwy_obrazow = ['rysunek-techniczny.jpg', 'skan-dokumentu.jpg', 'kolorowe-zdjecie-2.jpeg']
    obrazy_do_modyfikacji = [np.asarray(Image.open(nazwy_obrazow[i])).astype(int) for i in range(len(nazwy_obrazow))]
    wymiary = [[obrazy_do_modyfikacji[i].shape] for i in range(len(nazwy_obrazow))]
    obrazy_splaszczone = [obrazy_do_modyfikacji[i].flatten() for i in range(len(nazwy_obrazow))]
    zakodowany_RLE = [kodowanie_RLE(obrazy_splaszczone[i]) for i in range(len(nazwy_obrazow))]
    zdekodowany_RLE = [(dekodowanie_RLE(zakodowany_RLE[i])).reshape(wymiary[i][0]) for i in range(len(nazwy_obrazow))]
    zakodowany_DC = [Drzewo_czworkowe(obrazy_do_modyfikacji[i]) for i in range(len(nazwy_obrazow))]
    zdekodowany_DC = [zakodowany_DC[i].wez_obraz() for i in range(len(nazwy_obrazow))]
    nazwy = ['rysunek_techniczny', 'skan_dokumentu', 'kolorowe_zdjecie']
    for i in range(len(obrazy_do_modyfikacji)):
        print("(RLE(" + nazwy[i] + " - oryginał).nbytes) = " + str(obrazy_do_modyfikacji[i].nbytes))
        print("(RLE(" + nazwy[i] + " - zakodowany).nbytes) = " + str(zakodowany_RLE[i].nbytes))
        print("(RLE(" + nazwy[i] + " - zdekodowany).nbytes) = " + str(zdekodowany_RLE[i].nbytes))
        print("asizeof.asizeof(DC(" + nazwy[i] + " - oryginał)) = " + str(asizeof.asizeof(obrazy_do_modyfikacji[i])))
        print("asizeof.asizeof(DC(" + nazwy[i] + " - zakodowany)) = " + str(asizeof.asizeof(zakodowany_DC[i])))
        print("asizeof.asizeof(DC(" + nazwy[i] + " - zdekodowany)) = " + str(asizeof.asizeof(zdekodowany_DC[i])))
        fig1 = plt.figure(figsize = (15, 10))
        plt.subplot(1, 3, 1)
        plt.title("Obraz oryginalny")
        plt.imshow(obrazy_do_modyfikacji[i])
        plt.subplot(1, 3, 2)
        plt.title("Obraz zdekodowany\nRLE\nStopień kompresji: " + str(round(obrazy_do_modyfikacji[i].nbytes / zakodowany_RLE[i].nbytes, 2)) + ",\n czyli " + str(round(100 * zakodowany_RLE[i].nbytes / obrazy_do_modyfikacji[i].nbytes, 2)) + "%")
        plt.imshow(zdekodowany_RLE[i])
        plt.subplot(1, 3, 3)
        plt.title("Obraz zdekodowany\nDrzewo czwórkowe\nStopień kompresji: " + str(round(asizeof.asizeof(obrazy_do_modyfikacji[i]) / asizeof.asizeof(zakodowany_DC[i]), 2)) + ",\n czyli " + str(round(100 * asizeof.asizeof(zakodowany_DC[i]) / asizeof.asizeof(obrazy_do_modyfikacji[i]), 2)) + "%")
        plt.imshow(zdekodowany_DC[i].astype(int))
        plt.savefig(nazwy[i] + '_RLE_DC.png')
        #plt.show()
        plt.close(fig1)
        roznica = cv2.subtract(obrazy_do_modyfikacji[i], zdekodowany_RLE[i])
        b, g, r = cv2.split(roznica)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("Obraz oryginalny i zdekodowany RLE są sobie równe")
        else:
            print("Obraz oryginalny i zdekodowany RLE nie są sobie równe")
        roznica_DC = cv2.subtract(obrazy_do_modyfikacji[i], zdekodowany_DC[i])
        bdc, gdc, rdc = cv2.split(roznica_DC)
        if cv2.countNonZero(bdc) == 0 and cv2.countNonZero(gdc) == 0 and cv2.countNonZero(rdc) == 0:
            print("Obraz oryginalny i zdekodowany drzewem czwórkowym są sobie równe")
        else:
            print("Obraz oryginalny i zdekodowany drzewem czwórkowym nie są sobie równe")

if __name__ == "__main__":
    glowny()
