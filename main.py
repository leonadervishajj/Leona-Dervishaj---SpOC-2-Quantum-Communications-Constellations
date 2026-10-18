print("TESTI 1: Programi ka filluar")

from constellation import udp

print("TESTI 2: Kodi i ESA-s u ngarkua")

x = udp.example()

print("TESTI 3: Konfigurimi u krijua")
print(x)

J1, J2, C1, C2 = udp.fitness(x)

print("TESTI 4: Llogaritja perfundoi")
print("J1 =", J1)
print("J2 =", J2)
print("C1 =", C1)
print("C2 =", C2)
from matplotlib import pyplot as plt

# Paraqitja 3D e komunikimit
ax, path = udp.plot(x, src=1, dst=1)

plt.show()