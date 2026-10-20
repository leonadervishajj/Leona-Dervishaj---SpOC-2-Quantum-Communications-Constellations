
def crowding_distance(fronti):
    """
    Llogarit Crowding Distance per nje front Pareto.

    fronti: lista e konfigurimeve me J1 dhe J2.
    Kthen: distancat ne rendin origjinal.
    """

    n = len(fronti)

    if n == 0:
        return []

    if n <= 2:
        return [float("inf")] * n

    distancat = [0.0] * n

    # Dy objektivat qe minimizohen
    for objektivi in ("J1", "J2"):

        renditja = sorted(
            range(n),
            key=lambda i: fronti[i][objektivi]
        )

        minimumi = fronti[renditja[0]][objektivi]
        maksimumi = fronti[renditja[-1]][objektivi]

        # Ruajme konfigurimet skajore
        distancat[renditja[0]] = float("inf")
        distancat[renditja[-1]] = float("inf")

        if maksimumi == minimumi:
            continue

        # Llogarisim distancat per konfigurimet e tjera
        for j in range(1, n - 1):
            indeksi = renditja[j]

            para = fronti[renditja[j - 1]][objektivi]
            pas = fronti[renditja[j + 1]][objektivi]

            if distancat[indeksi] != float("inf"):
                distancat[indeksi] += (
                    (pas - para) / (maksimumi - minimumi)
                )

    return distancat


import random


def binary_tournament_selection(popullata, ranks, distances, rng=None):
    """
    Zgjedh nje konfigurim per riprodhim duke perdorur
    Pareto rank dhe Crowding Distance.

    Rank me i vogel = konfigurim me i mire.
    Distance me e madhe = diversitet me i larte.
    """
    if rng is None:
        rng = random

    if not popullata:
        raise ValueError("Popullata nuk mund te jete bosh.")

    if len(popullata) != len(ranks) or len(popullata) != len(distances):
        raise ValueError("Rangs dhe distances duhet te perputhen me popullaten.")

    if len(popullata) == 1:
        return popullata[0]

    # Zgjedhim dy konkurrente te ndryshem
    a, b = rng.sample(range(len(popullata)), 2)

    # Preferojme rank me te vogel
    if ranks[a] < ranks[b]:
        return popullata[a]

    if ranks[b] < ranks[a]:
        return popullata[b]

    # Nese rank eshte i njejte, preferojme crowding distance
    if distances[a] > distances[b]:
        return popullata[a]

    if distances[b] > distances[a]:
        return popullata[b]

    # Barazim i plote: zgjedhje e rastit
    return popullata[rng.choice([a, b])]
