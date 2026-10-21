
import random


INTEGER_START = 10


def crossover(parent1, parent2, lower_bounds, upper_bounds, rng=None):
    """
    Krijon dy konfigurime te reja nga dy prinder.
    Perdor kombinim aritmetik per parametrat real
    dhe uniform crossover per parametrat integer.
    """
    if rng is None:
        rng = random

    n = len(parent1)

    if not (
        len(parent2) == len(lower_bounds) == len(upper_bounds) == n
    ):
        raise ValueError("Dimensionet e kromozomeve nuk perputhen.")

    child1 = []
    child2 = []

    for i in range(n):
        a = parent1[i]
        b = parent2[i]

        if i < INTEGER_START:
            alpha = rng.random()

            v1 = alpha * a + (1 - alpha) * b
            v2 = alpha * b + (1 - alpha) * a

            v1 = max(lower_bounds[i], min(upper_bounds[i], v1))
            v2 = max(lower_bounds[i], min(upper_bounds[i], v2))
        else:
            if rng.random() < 0.5:
                v1, v2 = int(a), int(b)
            else:
                v1, v2 = int(b), int(a)

        child1.append(v1)
        child2.append(v2)

    return child1, child2


def mutation(x, lower_bounds, upper_bounds, probability=0.1, rng=None):
    """
    Ndryshon disa parametra te nje konfigurimi,
    duke respektuar kufijte e ESA-s.
    """
    if rng is None:
        rng = random

    if not (
        len(x) == len(lower_bounds) == len(upper_bounds)
    ):
        raise ValueError("Dimensionet e kromozomit nuk perputhen.")

    mutated = x.copy()

    for i in range(len(mutated)):
        if rng.random() >= probability:
            continue

        low = lower_bounds[i]
        high = upper_bounds[i]

        if i < INTEGER_START:
            sigma = 0.1 * (high - low)
            value = mutated[i] + rng.gauss(0, sigma)
            mutated[i] = max(low, min(high, value))
        else:
            mutated[i] = rng.randint(int(low), int(high))

    return mutated
