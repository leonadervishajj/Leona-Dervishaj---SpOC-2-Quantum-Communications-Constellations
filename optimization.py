
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
