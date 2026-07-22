# OPUS

**Open Process Uncertainty Simulator**

A Python framework for dynamic process simulation, engineering analysis, and uncertainty quantification.

> **Status:** Early development. The deterministic simulation core is complete and validated. Monte Carlo and uncertainty-analysis capabilities are planned next.

## Current capabilities

- Validated physical parameter models
- Reproducible simulation configuration
- Dynamic process models
- Generic SciPy-backed ODE solver
- Engineering performance calculations
- Unit and integration test coverage

## Installation

```bash
git clone https://github.com/parisisv/OPUS
cd opus
python -m pip install -e .
```

## Example

The end-to-end CSTR example demonstrates the complete deterministic workflow:

```text
examples/cstr_end_to_end_simulation.ipynb
```

Run it with:

```bash
jupyter lab
```

## Project structure

```text
src/       OPUS source code
tests/     Unit and integration tests
examples/  Executable examples
```

## Roadmap

**Completed**

- Deterministic simulation core
- CSTR model
- ODE solver
- Reproducible configuration
- Engineering metrics

**Next**

- Sampling methods
- Monte Carlo simulation
- Statistical analysis
- Sensitivity analysis

**Future**

- Optimization
- Surrogate models
- Interactive dashboards

## License

MIT Licence
