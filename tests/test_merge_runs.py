
import contextlib
import csv
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import experiments.merge_runs as merge_runs
import experiments.export_submission as export_submission
import src.nsga2 as nsga2
from experiments.merge_runs import (
    REF_POINT,
    select_solutions,
    _hv_contributions,
    main
)

FIELDS = (
    ["J1", "J2", "C1", "C2", "valid"]
    + [f"x{i}" for i in range(20)]
)


def hv_2d(points):
    """Hypervolume 2D i pavarur (reference e njejte me ESA)."""
    ref1, ref2 = REF_POINT
    inside = sorted(p for p in points if p[0] < ref1 and p[1] < ref2)
    volume, best = 0.0, ref2

    for j1, j2 in inside:
        if j2 < best:
            volume += (ref1 - j1) * (best - j2)
            best = j2

    return volume


def sol(j1, j2, tag=0):
    return {
        "x": [0.0] * 9 + [float(tag)] + [1] * 10,
        "J1": j1, "J2": j2, "C1": -1.0, "C2": -1.0, "valid": True
    }


def make_front(count, seed=0):
    """Front Pareto i lemuar: J1 rritet, J2 ulet."""
    rng = random.Random(seed)
    j1s = sorted(rng.uniform(0.05, 1.1) for _ in range(count))
    front = []
    previous = 1.35

    for i, j1 in enumerate(j1s):
        previous = previous - rng.uniform(0.001, 0.9 / count)
        front.append(sol(j1, previous, tag=i))

    return front


class TestSelectSolutions(unittest.TestCase):

    def test_small_front_returned_unchanged(self):
        front = make_front(30)
        result = select_solutions(front, 100)

        self.assertEqual(result, front)

    def test_exactly_limit_is_unchanged(self):
        front = make_front(100)

        self.assertEqual(select_solutions(front, 100), front)

    def test_reduces_to_limit_without_duplicates(self):
        front = make_front(150)
        result = select_solutions(front, 100)

        self.assertEqual(len(result), 100)
        keys = [tuple(s["x"]) for s in result]
        self.assertEqual(len(set(keys)), 100)

        original = {tuple(s["x"]) for s in front}
        self.assertTrue(set(keys) <= original)

    def test_is_deterministic(self):
        front = make_front(150, seed=3)
        shuffled = front[:]
        random.Random(9).shuffle(shuffled)

        self.assertEqual(
            select_solutions(front, 100),
            select_solutions(shuffled, 100)
        )

    def test_removes_weakest_contribution_first(self):
        # Pika e mesme (0.50, 0.72) kontribuon shume pak
        front = [
            sol(0.2, 1.0, 0),
            sol(0.5, 0.7, 1),
            sol(0.501, 0.699, 2),
            sol(0.9, 0.3, 3)
        ]

        result = select_solutions(front, 3)
        tags = sorted(int(s["x"][9]) for s in result)

        self.assertEqual(len(tags), 3)
        self.assertIn(0, tags)
        self.assertIn(3, tags)

    def test_points_outside_reference_are_dropped_first(self):
        front = [
            sol(0.3, 1.0, 0),
            sol(0.8, 0.5, 1),
            sol(1.3, 0.2, 2)   # J1 mbi 1.2 -> kontribut 0
        ]

        result = select_solutions(front, 2)

        self.assertEqual(
            sorted(int(s["x"][9]) for s in result), [0, 1]
        )

    def test_hypervolume_matches_reference_and_beats_subsampling(self):
        front = make_front(150, seed=5)
        selected = select_solutions(front, 100)

        def pts(solutions):
            return [(s["J1"], s["J2"]) for s in solutions]

        # Nenzgjedhje uniforme me 100 pika nga 150
        uniform = [front[int(i * 1.5)] for i in range(100)]

        self.assertGreaterEqual(
            hv_2d(pts(selected)), hv_2d(pts(uniform))
        )
        # Nuk humb me shume se 1% nga fronti i plote
        self.assertGreater(
            hv_2d(pts(selected)), 0.99 * hv_2d(pts(front))
        )

    def test_contribution_equals_hypervolume_lost_on_removal(self):
        pts = [(0.2, 1.0), (0.5, 0.7), (0.9, 0.3)]

        # Heqja e nje pike humbet saktesisht kontributin e saj
        contributions = _hv_contributions(pts)
        for i in range(3):
            reduced = pts[:i] + pts[i + 1:]
            self.assertAlmostEqual(
                hv_2d(pts) - hv_2d(reduced),
                contributions[i],
                places=12
            )

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            select_solutions(make_front(5), 0)


class FakeUdp:
    """Front i pastert: J1 = 1.1*x0, J2 = 1.3 - 1.2*x0."""

    def get_bounds(self):
        return [0.0] * 10 + [0] * 10, [1.0] * 10 + [9] * 10

    def fitness(self, x):
        c1 = 1.0 if x[1] > 0.95 else -1.0
        return 1.1 * x[0], 1.3 - 1.2 * x[0], c1, -1.0


def write_run(folder, x0_values, infeasible_x0=()):
    folder.mkdir(parents=True)
    path = folder / "nsga2_final_population.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        rows = [(x0, 0.5) for x0 in x0_values]
        # x1 > 0.95 -> FakeUdp kthen C1 > 0 (e pavlefshme)
        rows += [(x0, 0.99) for x0 in infeasible_x0]

        for x0, x1 in rows:
            row = {"J1": 0, "J2": 0, "C1": -1, "C2": -1, "valid": True}
            for i in range(20):
                if i == 0:
                    row["x0"] = x0
                elif i == 1:
                    row["x1"] = x1
                else:
                    row[f"x{i}"] = 0.5 if i < 10 else 1
            writer.writerow(row)

    return path


class TestMergeMain(unittest.TestCase):

    def setUp(self):
        self.score = mock.Mock(return_value=-1000.0)

        for target, name, value in (
            (merge_runs, "udp", FakeUdp()),
            (merge_runs, "combine_scores", self.score),
            (export_submission, "udp", FakeUdp()),
            (export_submission, "combine_scores", self.score),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_main(self, *args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(list(args))
        return buffer.getvalue()

    def read_rows(self, folder):
        path = Path(folder) / "nsga2_final_population.csv"
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_small_merge_keeps_everything(self):
        a = write_run(self.root / "a", [i / 100 for i in range(1, 31)])
        b = write_run(self.root / "b", [i / 100 for i in range(31, 61)])

        output = self.run_main(
            "--run-dirs", str(a.parent), str(b.parent),
            "--output-dir", str(self.root / "m")
        )

        self.assertEqual(len(self.read_rows(self.root / "m")), 60)
        self.assertIn("Selected solutions: 60", output)

    def test_over_limit_selects_exactly_100(self):
        a = write_run(self.root / "a", [i / 200 for i in range(1, 91)])
        b = write_run(self.root / "b", [i / 200 + 0.0003 for i in range(91, 161)])

        output = self.run_main(
            "--run-dirs", str(a.parent), str(b.parent),
            "--output-dir", str(self.root / "m")
        )

        rows = self.read_rows(self.root / "m")
        self.assertEqual(len(rows), 100)
        self.assertIn("Pareto solutions: 160", output)
        self.assertIn("Selected solutions: 100", output)

    def test_max_solutions_option(self):
        a = write_run(self.root / "a", [i / 100 for i in range(1, 41)])

        self.run_main(
            "--run-dirs", str(a.parent),
            "--output-dir", str(self.root / "m"),
            "--max-solutions", "10"
        )

        self.assertEqual(len(self.read_rows(self.root / "m")), 10)

    def test_invalid_max_solutions(self):
        a = write_run(self.root / "a", [0.1, 0.2])

        with self.assertRaises(SystemExit):
            main(["--run-dirs", str(a.parent),
                  "--output-dir", str(self.root / "m"),
                  "--max-solutions", "0"])

    def test_duplicates_and_infeasible_are_removed(self):
        a = write_run(self.root / "a", [0.1, 0.2, 0.3])
        b = write_run(
            self.root / "b", [0.2, 0.4], infeasible_x0=[0.5]
        )

        self.run_main(
            "--run-dirs", str(a.parent), str(b.parent),
            "--output-dir", str(self.root / "m")
        )

        x0 = sorted(
            float(r["x0"]) for r in self.read_rows(self.root / "m")
        )
        self.assertEqual(x0, [0.1, 0.2, 0.3, 0.4])

    def test_nonempty_output_dir_is_refused_and_untouched(self):
        a = write_run(self.root / "a", [0.1, 0.2])
        out = self.root / "m"
        out.mkdir()
        keep = out / "keep.txt"
        keep.write_text("vlefshme", encoding="utf-8")

        with self.assertRaises(SystemExit):
            main(["--run-dirs", str(a.parent),
                  "--output-dir", str(out)])

        self.assertEqual(keep.read_text(encoding="utf-8"), "vlefshme")

    def test_input_runs_are_not_modified(self):
        a = write_run(self.root / "a", [i / 200 for i in range(1, 161)])
        before = a.read_bytes()
        files_before = sorted(p.name for p in a.parent.iterdir())

        self.run_main(
            "--run-dirs", str(a.parent),
            "--output-dir", str(self.root / "m")
        )

        self.assertEqual(a.read_bytes(), before)
        self.assertEqual(
            sorted(p.name for p in a.parent.iterdir()), files_before
        )

    def test_final_json_can_be_exported_outside_merged_folder(self):
        a = write_run(self.root / "a", [i / 100 for i in range(1, 21)])
        merged = self.root / "results" / "merged_final"
        final = self.root / "results" / "esa_submission_final.json"

        self.run_main(
            "--run-dirs", str(a.parent),
            "--output-dir", str(merged)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            export_submission.main([
                "--run-dir", str(merged),
                "--output", str(final)
            ])

        data = json.loads(final.read_text(encoding="utf-8"))
        self.assertEqual(len(data[0]["decisionVector"]), 20)
        self.assertFalse((merged / "esa_submission.json").exists())


class TestZeroHypervolume(unittest.TestCase):

    def test_zero_score_is_positive_zero(self):
        import math

        with mock.patch.object(
            nsga2, "combine_scores", mock.Mock(return_value=0.0)
        ):
            hv, score = nsga2.population_hypervolume([{
                "x": [], "J1": 1.5, "J2": 1.0,
                "C1": -1.0, "C2": -1.0, "valid": True
            }])

        self.assertEqual(hv, 0.0)
        self.assertEqual(math.copysign(1.0, hv), 1.0)


@unittest.skipUnless(
    os.environ.get("RUN_ESA_TESTS") == "1",
    "Vendos RUN_ESA_TESTS=1 per testet me modelin real ESA"
)
class TestWithRealCombineScores(unittest.TestCase):
    """Krahason HV-ne e pavarur me combine_scores real (pygmo)."""

    def test_reference_point_and_hypervolume_match_esa(self):
        from constellation import combine_scores

        self.assertAlmostEqual(
            combine_scores([[1.19, 1.39, -1, -1]]), -1.0, places=6
        )

        front = make_front(120, seed=11)
        selected = select_solutions(front, 100)
        points = [[s["J1"], s["J2"], -1, -1] for s in selected]

        self.assertAlmostEqual(
            -combine_scores(points) / 10000.0,
            hv_2d([(s["J1"], s["J2"]) for s in selected]),
            places=9
        )


if __name__ == "__main__":
    unittest.main()