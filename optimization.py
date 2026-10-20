
from constellation import udp

# Konfigurimi fillestar i ESA-s
x = udp.example()

# Kufijte e parametrave
lower_bounds, upper_bounds = udp.get_bounds()

print("Numri i parametrave:", len(x))
print("Kufijte e poshtem:", lower_bounds)
print("Kufijte e siperm:", upper_bounds)

# Vleresimi fillestar
J1_fillestar, J2_fillestar, C1_fillestar, C2_fillestar = udp.fitness(x)

print("\nRezultatet fillestare:")
print("J1 =", J1_fillestar)
print("J2 =", J2_fillestar)
print("C1 =", C1_fillestar)
print("C2 =", C2_fillestar)


import random

# Gjenerojme 5 konfigurime te reja
print("\n--- TESTIMI I KONFIGURIMEVE TE REJA ---")

random.seed(42)
zgjidhjet = []
for i in range(5):
    konfigurimi = []

    for j in range(len(lower_bounds)):
        minimumi = lower_bounds[j]
        maksimumi = upper_bounds[j]

        # Parametrat 10-19 jane numra te plote
        if j >= 10:
            vlera = random.randint(int(minimumi), int(maksimumi))
        else:
            vlera = random.uniform(minimumi, maksimumi)

        konfigurimi.append(vlera)

    J1, J2, C1, C2 = udp.fitness(konfigurimi)
    if C1 <= 0 and C2 <= 0:
            zgjidhjet.append({
            "emri": f"Konfigurimi {i + 1}",
            "J1": J1,
            "J2": J2
        })

    print(f"\nKonfigurimi {i + 1}")
    print(f"J1 = {J1:.4f}")
    print(f"J2 = {J2:.4f}")
    print(f"C1 = {C1:.4f}")
    print(f"C2 = {C2:.4f}")

    if C1 <= 0 and C2 <= 0:
        print("Konfigurimi eshte i vlefshem")
    else:
        print("Konfigurimi nuk eshte i vlefshem")
        
# Shtojme konfigurimin fillestar
if C1_fillestar <= 0 and C2_fillestar <= 0:
    zgjidhjet.append({
        "emri": "Fillestar",
        "J1": J1_fillestar,
        "J2": J2_fillestar
    })


# Kontrollojme nese A dominon B
def dominon(A, B):
    return (
        A["J1"] <= B["J1"]
        and A["J2"] <= B["J2"]
        and (
            A["J1"] < B["J1"]
            or A["J2"] < B["J2"]
        )
    )


# Gjejme zgjidhjet jo te dominuara
pareto = []

for A in zgjidhjet:
    dominohet = False

    for B in zgjidhjet:
        if dominon(B, A):
            dominohet = True
            break

    if not dominohet:
        pareto.append(A)


print("\n--- ZGJIDHJET PARETO ---")

for z in pareto:
    print(
        f"{z['emri']}: "
        f"J1 = {z['J1']:.4f}, "
        f"J2 = {z['J2']:.4f}"
    )

# Vizualizimi i zgjidhjeve Pareto
import matplotlib.pyplot as plt

if pareto:
    j1_values = [z["J1"] for z in pareto]
    j2_values = [z["J2"] for z in pareto]

    plt.figure(figsize=(8, 5))

    plt.scatter(j1_values, j2_values, color="blue", s=80)

    for z in pareto:
        plt.annotate(
            z["emri"],
            (z["J1"], z["J2"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("J1 - Kostoja e komunikimit")
    plt.ylabel("J2 - Kostoja e infrastruktures")
    plt.title("Pareto Front - ESA SpOC 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Nuk ka zgjidhje te vlefshme per grafik.")


