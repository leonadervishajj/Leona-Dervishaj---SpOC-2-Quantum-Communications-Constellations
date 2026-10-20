
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
