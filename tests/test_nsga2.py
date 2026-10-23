
import unittest
import random

from src.nsga2 import dominates, sort_fronts
from src.selection import (
    crowding_distance,
    binary_tournament_selection
)
from src.genetic_operators import crossover, mutation


class TestNSGA2(unittest.TestCase):

    # TESTI 1 - Dominimi Pareto
    def test_pareto_dominance(self):
        a = {
            "J1": 1.0,
            "J2": 2.0,
            "C1": -1.0,
            "C2": -1.0
        }

        b = {
            "J1": 2.0,
            "J2": 3.0,
            "C1": -1.0,
            "C2": -1.0
        }

        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))

    # TESTI 2 - Respektimi i kufizimeve
    def test_constraints(self):
        valid = {
            "J1": 2.0,
            "J2": 2.0,
            "C1": -1.0,
            "C2": -1.0
        }

        invalid = {
            "J1": 1.0,
            "J2": 1.0,
            "C1": 10.0,
            "C2": -1.0
        }

        self.assertTrue(dominates(valid, invalid))
        self.assertFalse(dominates(invalid, valid))

    # TESTI 3 - Ndarja ne fronte Pareto
    def test_pareto_fronts(self):
        population = [
            {"J1": 1, "J2": 3, "C1": -1, "C2": -1},
            {"J1": 2, "J2": 2, "C1": -1, "C2": -1},
            {"J1": 3, "J2": 1, "C1": -1, "C2": -1},
            {"J1": 4, "J2": 4, "C1": -1, "C2": -1}
        ]

        fronts = sort_fronts(population)

        self.assertEqual(set(fronts[0]), {0, 1, 2})
        self.assertEqual(fronts[1], [3])

    # TESTI 4 - Crowding Distance
    def test_crowding_distance(self):
        front = [
            {"J1": 1, "J2": 3},
            {"J1": 2, "J2": 2},
            {"J1": 3, "J2": 1}
        ]

        distances = crowding_distance(front)

        self.assertEqual(
            distances,
            [float("inf"), 2.0, float("inf")]
        )

    # TESTI 5 - Tournament Selection
    def test_tournament_selection(self):
        population = ["A", "B"]

        winner = binary_tournament_selection(
            population,
            ranks=[1, 2],
            distances=[0.5, 10.0],
            rng=random.Random(42)
        )

        self.assertEqual(winner, "A")

    # TESTI 6 - Crossover dhe Mutation
    def test_genetic_operators(self):
        lower_bounds = [0.0] * 10 + [0] * 10
        upper_bounds = [1.0] * 10 + [9] * 10

        parent1 = [0.2] * 10 + [2] * 10
        parent2 = [0.8] * 10 + [8] * 10

        child1, child2 = crossover(
            parent1,
            parent2,
            lower_bounds,
            upper_bounds,
            rng=random.Random(42)
        )

        mutated = mutation(
            child1,
            lower_bounds,
            upper_bounds,
            probability=1.0,
            rng=random.Random(123)
        )

        for chromosome in (child1, child2, mutated):
            self.assertEqual(len(chromosome), 20)

            for i, value in enumerate(chromosome):
                self.assertGreaterEqual(
                    value, lower_bounds[i]
                )
                self.assertLessEqual(
                    value, upper_bounds[i]
                )

                if i >= 10:
                    self.assertIsInstance(value, int)


if __name__ == "__main__":
    unittest.main()
