
import random

INTEGER_START = 10

def adaptive_mutation(x, lower_bounds, upper_bounds,
                      generation, total_generations,
                      probability=0.1, rng=None):
    if rng is None:
        rng = random

    mutated = list(x)

    progress = min(1.0, generation / max(1, total_generations))
    sigma_factor = 0.10 * (1.0 - progress) + 0.01

    for i in range(len(mutated)):
        if rng.random() >= probability:
            continue

        low = lower_bounds[i]
        high = upper_bounds[i]

        if i < INTEGER_START:
            sigma = sigma_factor * (high - low)
            value = mutated[i] + rng.gauss(0, sigma)
            mutated[i] = max(low, min(high, value))
        else:
            if rng.random() < 0.7:
                step = rng.choice([-1, 1])
                value = int(mutated[i]) + step
                mutated[i] = max(int(low), min(int(high), value))
            else:
                mutated[i] = rng.randint(int(low), int(high))

    return mutated
