# Quantum Communications Constellations

## ESA SpOC 2 – University Project

This project is based on the European Space Agency (ESA) SpOC 2 challenge, *Quantum Communications Constellations*.

### Project Overview

The goal is to optimize satellite constellations that support communication between seven motherships and four rovers on the fictional planet New Mars.

The optimization problem considers two objectives:

- **J1:** Minimize the communication cost between motherships and rovers.
- **J2:** Minimize the infrastructure cost of the satellite constellations.

Solutions must also satisfy the distance constraints for rover placement and satellite separation.

### Technologies

- Python
- NumPy and SciPy
- PyKEP and PyGMO
- Matplotlib
- NetworkX and SGP4

### Current Progress

- [x] Set up the Python development environment
- [x] Integrated the ESA-provided evaluation model
- [x] Loaded the rover dataset
- [x] Evaluated the example satellite configuration
- [x] Generated a 3D visualization
- [ ] Implement a multi-objective optimization algorithm
- [ ] Generate and analyze Pareto-optimal solutions
- [ ] Compare optimized solutions with the baseline

### Initial Simulation Results

| Metric | Baseline value |
|---|---:|
| J1 – Communication cost | 8.3579 |
| J2 – Infrastructure cost | 0.0140 |
| C1 – Rover distance constraint | -4526.99 |
| C2 – Satellite separation constraint | -1027.46 |

Both inequality constraints are satisfied by the example configuration.

### Running the Project

The project uses the ESA evaluation code and requires compatible scientific Python dependencies.

From the activated Conda environment, run:

`python -u main.py`

The script evaluates the ESA example configuration and displays a 3D visualization.

### Next Steps

The next development phase will focus on implementing and testing a multi-objective optimization method such as NSGA-II.

### Acknowledgements

The original challenge definition, evaluation code, and rover dataset are provided by the European Space Agency (ESA). The optimization experiments and further implementation are being developed as part of a university project.