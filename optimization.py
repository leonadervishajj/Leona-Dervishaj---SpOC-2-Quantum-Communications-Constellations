
from constellation import udp

# Konfigurimi fillestar i ESA-s
x = udp.example()

# Kufijte e parametrave
lower_bounds, upper_bounds = udp.get_bounds()

print("Numri i parametrave:", len(x))
print("Kufijte e poshtem:", lower_bounds)
print("Kufijte e siperm:", upper_bounds)

# Vleresimi fillestar
J1, J2, C1, C2 = udp.fitness(x)

print("\nRezultatet fillestare:")
print("J1 =", J1)
print("J2 =", J2)
print("C1 =", C1)
print("C2 =", C2)


import random

# Gjenerojme 5 konfigurime te reja
print("\n--- TESTIMI I KONFIGURIMEVE TE REJA ---")

random.seed(42)

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

    print(f"\nKonfigurimi {i + 1}")
    print(f"J1 = {J1:.4f}")
    print(f"J2 = {J2:.4f}")
    print(f"C1 = {C1:.4f}")
    print(f"C2 = {C2:.4f}")

    if C1 <= 0 and C2 <= 0:
        print("Konfigurimi eshte i vlefshem")
    else:
        print("Konfigurimi nuk eshte i vlefshem")

