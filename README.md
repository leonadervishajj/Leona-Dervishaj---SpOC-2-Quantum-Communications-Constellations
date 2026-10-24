# Quantum Communications Constellations

## ESA SpOC 2 – University Project

This project is based on the European Space Agency (ESA) SpOC 2 challenge, *Quantum Communications Constellations*.

### Project Overview

The goal is to optimize satellite constellations that support communication between seven motherships and four rovers on the fictional planet New Mars.

The optimization problem considers two objectives:

- **J1:** Minimize the communication cost between motherships and rovers.
- **J2:** Minimize the infrastructure cost of the satellite constellations.

Solutions must also satisfy the distance constraints for rover placement and satellite separation.

Each candidate solution is represented by 20 decision variables, including orbital parameters, constellation characteristics, and rover selections.

### Technologies

- Python
- NumPy and SciPy
- PyKEP and PyGMO
- Matplotlib
- NetworkX and SGP4

### Optimization Approach

The project implements **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** to search for trade-offs between communication cost and infrastructure cost.

The optimization workflow includes:

- Random initialization of satellite configurations
- Evaluation using the ESA-provided fitness function
- Constraint-aware Pareto dominance
- Non-dominated sorting
- Crowding distance
- Binary tournament selection
- Crossover and mutation
- Elitist survivor selection

The algorithm maintains a population of candidate solutions and iteratively generates new configurations over multiple generations.

### Current Progress

- [x] Set up the Python development environment
- [x] Integrated the ESA-provided evaluation model
- [x] Loaded the rover dataset
- [x] Evaluated the example satellite configuration
- [x] Generated a 3D visualization
- [x] Implemented a multi-objective optimization algorithm (NSGA-II)
- [x] Generated and analyzed non-dominated Pareto solutions
- [x] Compared optimized solutions with the ESA baseline
- [x] Implemented multi-seed experiments
- [x] Calculated the ESA Hypervolume score
- [x] Implemented six automated unit tests
- [x] Validated 15 exported configurations using the ESA fitness function
- [x] Generated a submission JSON using the official ESA submission helper
- [ ] Perform larger experiments and additional statistical validation
- [ ] Finalize reproducibility and project documentation

### Initial Simulation Results

| Metric | Baseline value |
|---|---:|
| J1 – Communication cost | 8.3579 |
| J2 – Infrastructure cost | 0.0140 |
| C1 – Rover distance constraint | -4526.99 |
| C2 – Satellite separation constraint | -1027.46 |

Both inequality constraints are satisfied by the ESA example configuration.

### NSGA-II Experimental Results

A preliminary experiment was performed using a population of 20 candidate configurations over five generations.

| Metric | Result |
|---|---:|
| Final population size | 20 |
| Feasible configurations | 20/20 |
| Non-dominated configurations | 15 |
| Best J1 | 0.763822 |
| Best J2 | 0.133554 |
| Hypervolume | 0.524638 |
| ESA scaled score | -5246.376155 |

The best values of J1 and J2 may correspond to different satellite configurations.

The results demonstrate the ability of NSGA-II to produce feasible trade-off solutions. They do not establish global optimality.

### Multi-Seed Analysis

The algorithm was evaluated with two different random seeds, each using a population of 20 and three generations.

| Metric | Seed 42 | Seed 123 |
|---|---:|---:|
| Best J1 | 0.763822 | 0.795793 |
| Best J2 | 0.133554 | 0.105800 |
| Non-dominated solutions | 15 | 7 |
| Feasible configurations | 20/20 | 20/20 |

Further experiments with additional random seeds are required for stronger statistical conclusions.

### Project Structure

The project is organized into separate components:

- `constellation.py` – ESA satellite simulation and fitness evaluation
- `src/nsga2.py` – NSGA-II optimization algorithm
- `src/selection.py` – Pareto sorting support, crowding distance, and selection
- `src/genetic_operators.py` – Crossover and mutation
- `experiments/` – Experiment execution, baseline comparison, multi-seed analysis, and submission export
- `tests/test_nsga2.py` – Automated unit tests
- `results/` – Experiment CSV files, visualizations, and exported submission
- `submission_helper.py` – ESA submission helper

### Running the Project

The project requires compatible scientific Python dependencies and an activated environment.

**Run the initial ESA simulation:**

`python -u main.py`

**Run NSGA-II:**

`python -m experiments.run_nsga2 --population 20 --generations 5`

**Compare with the ESA baseline:**

`python -m experiments.compare_baseline`

**Run the multi-seed analysis:**

`python -u -m experiments.multi_seed_analysis --population 20 --generations 3 --seeds 42 123`

**Run automated tests:**

`python -m unittest discover -s tests -v`

**Export and validate the ESA submission:**

`python -u -m experiments.export_submission`

### Validation and Submission

Six automated unit tests passed, covering Pareto dominance, constraints, non-dominated sorting, crowding distance, tournament selection, and genetic operators.

The 15 exported non-dominated configurations were independently re-evaluated using the ESA fitness function and satisfied both inequality constraints.

The submission file is generated at:

`results/esa_submission.json`

It is created using the official ESA submission helper. The submission has been validated locally but has not been uploaded to the ESA platform.

### Next Steps

- Perform additional experiments with larger populations and more generations
- Evaluate Hypervolume across multiple independent runs
- Expand automated tests and verify reproducibility
- Review dependency management and clean up experimental files
- Finalize project documentation and GitHub organization

### References

- [ESA SpOC 2 – Official GitHub Repository](https://github.com/esa/SpOC2)
- [ESA Quantum Communications Constellations Challenge](https://optimize.esa.int/challenge/spoc-2-quantum-communications-constellations/About)

### Acknowledgements

The original challenge definition, satellite constellation evaluation code, and rover dataset are provided by the European Space Agency (ESA). The NSGA-II optimization implementation, experimental analysis, and testing workflow were developed as part of this university project.