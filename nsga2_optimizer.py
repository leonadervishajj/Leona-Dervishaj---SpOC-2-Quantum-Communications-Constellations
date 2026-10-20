
import random
from constellation import udp

# Gjenerator per rezultate te perseritshme
rng = random.Random(42)

# Kufijte e percaktuar nga ESA
lower_bounds, upper_bounds = udp.get_bounds()

# 10 parametrat e fundit jane numra te plote
INTEGER_START = 10


def gjenero_konfigurim():
    """Gjeneron nje konfigurim brenda kufijve te ESA-s."""
    x = []

    for i in range(len(lower_bounds)):
        minimumi = lower_bounds[i]
        maksimumi = upper_bounds[i]

        if i >= INTEGER_START:
            vlera = rng.randint(int(minimumi), int(maksimumi))
        else:
            vlera = rng.uniform(minimumi, maksimumi)

        x.append(vlera)

    return x


def vlereso_konfigurim(x):
    """Vlereson konfigurimin me modelin zyrtar te ESA-s."""
    J1, J2, C1, C2 = udp.fitness(x)

    return {
        "x": x,
        "J1": J1,
        "J2": J2,
        "C1": C1,
        "C2": C2,
        "valid": C1 <= 0 and C2 <= 0
    }


if __name__ == "__main__":
    x = gjenero_konfigurim()
    rezultati = vlereso_konfigurim(x)

    print("Konfigurimi:", x)
    print("J1:", rezultati["J1"])
    print("J2:", rezultati["J2"])
    print("C1:", rezultati["C1"])
    print("C2:", rezultati["C2"])
    print("I vlefshem:", rezultati["valid"])
