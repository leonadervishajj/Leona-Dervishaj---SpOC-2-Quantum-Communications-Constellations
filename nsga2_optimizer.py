
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
    MADHESIA_POPULLATES = 20
    popullata = []

    print("--- GJENERIMI I POPULLATES ---")

    for i in range(MADHESIA_POPULLATES):
        x = gjenero_konfigurim()
        rezultati = vlereso_konfigurim(x)
        popullata.append(rezultati)

        print(
            f"Konfigurimi {i + 1}: "
            f"J1={rezultati['J1']:.4f}, "
            f"J2={rezultati['J2']:.4f}, "
            f"C1={rezultati['C1']:.2f}, "
            f"C2={rezultati['C2']:.2f}, "
            f"Valid={rezultati['valid']}"
        )

    te_vlefshme = [
        z for z in popullata if z["valid"]
    ]

    print("\n--- PERMBLEDHJA ---")
    print("Gjithsej konfigurime:", len(popullata))
    print("Konfigurime te vlefshme:", len(te_vlefshme))
    print(
        "Konfigurime jo te vlefshme:",
        len(popullata) - len(te_vlefshme)
    )
