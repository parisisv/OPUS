# ODE Solver Quick Reference

## Installation

The ODE Solver is part of OPUS. Make sure required dependencies are installed:

```bash
pip install numpy scipy
```

## Quick Start

### 1. Create a Model

```python
from opus.core.parameters import SimulationParameters, ReactorParameters, FeedParameters, CoolingParameters, KineticParameters
from opus.models.cstr import CSTR

params = SimulationParameters(
    reactor=ReactorParameters(volume=1.0, density=1000.0, heat_capacity=4186.0, UA=5000.0),
    feed=FeedParameters(flow_rate=0.001, concentration=1000.0, temperature=298.15),
    cooling=CoolingParameters(coolant_temperature=288.15),
    kinetics=KineticParameters(k0=7.2e10, activation_energy=75000.0, heat_of_reaction=-50000.0, gas_constant=8.314)
)

model = CSTR(params)
```

### 2. Create a Solver

```python
from opus.solvers import ODESolver

solver = ODESolver(model, method="RK45")
```

### 3. Set Initial Conditions

```python
from opus.models.state import ReactorState

initial_state = ReactorState(concentration=100.0, temperature=300.0)
```

### 4. Solve

```python
solution = solver.solve(
    initial_state=initial_state,
    t_span=(0.0, 1000.0),  # Start and end time in seconds
    max_step=10.0          # Optional: maximum step size
)
```

### 5. Access Results

```python
# Get final state
final_state = solver.get_final_state(solution)
print(f"Final concentration: {final_state.concentration} mol/m³")

# Get state at specific time (requires dense_output=True)
state_at_500 = solver.get_state_at_time(solution, 500.0)

# Get all time points and states
times = solution.t
concentrations = solution.y[0, :]
temperatures = solution.y[1, :]
```

## Common Tasks

### Evaluate at Specific Times

```python
import numpy as np

t_eval = np.linspace(0, 1000, 101)  # 101 points from 0 to 1000
solution = solver.solve(
    initial_state=initial_state,
    t_span=(0.0, 1000.0),
    t_eval=t_eval
)
```

### Change Integration Method

```python
# For stiff problems:
solver = ODESolver(model, method="BDF")

# For high accuracy:
solver = ODESolver(model, method="DOP853")

# Automatic stiff detection:
solver = ODESolver(model, method="LSODA")
```

### Adjust Tolerances

```python
# Loose tolerances (faster)
solution = solver.solve(
    initial_state=initial_state,
    t_span=(0.0, 1000.0),
    rtol=1e-2,   # Relative tolerance
    atol=1e-5    # Absolute tolerance
)

# Tight tolerances (slower, more accurate)
solution = solver.solve(
    initial_state=initial_state,
    t_span=(0.0, 1000.0),
    rtol=1e-6,
    atol=1e-9
)
```

### Detect Events

```python
# Define event (e.g., when temperature exceeds 350 K)
def temp_threshold(t, y):
    return y[1] - 350.0

temp_threshold.terminal = True
temp_threshold.direction = -1

solution = solver.solve_to_events(
    initial_state=initial_state,
    t_span=(0.0, 10000.0),
    events=temp_threshold
)
print(f"Event at time: {solution.t[-1]}")
```

### Disable Dense Output

```python
solver = ODESolver(model, dense_output=False)
solution = solver.solve(...)
# Cannot use get_state_at_time() or evaluate_solution()
# But uses less memory
```

## Integration Methods

```
RK45   = Runge-Kutta 4/5 (default, good for most cases)
RK23   = Runge-Kutta 2/3 (faster but less accurate)
BDF    = Backward Differentiation Formula (stiff systems)
LSODA  = Automatic switching (best general choice)
Radau  = Implicit Radau 3 (very stiff systems)
DOP853 = Dormand-Prince 8 (high accuracy needed)
```

## Default Values

```python
ODESolver(
    model,
    method="RK45",              # Integration method
    dense_output=True,          # Continuous solution
    events=None                 # Event detection
)

solve(
    initial_state,
    t_span,
    t_eval=None,                # Auto time selection
    max_step=None,              # No step limit
    rtol=1e-3,                  # Relative tolerance
    atol=1e-6                   # Absolute tolerance
)
```

## Error Handling

```python
from opus.solvers import ODESolver

try:
    solution = solver.solve(initial_state, (0, 1000))
except RuntimeError as e:
    print(f"Integration failed: {e}")

try:
    state = solver.get_state_at_time(solution, 5000)
except ValueError as e:
    print(f"Time out of range: {e}")

try:
    y = solver.evaluate_solution(solution, [10, 20, 30])
except ValueError as e:
    print(f"Dense output not available: {e}")
```

## Typical Configurations

### Conservative (Accurate)
```python
solver = ODESolver(model, method="RK45")
solution = solver.solve(
    initial_state, (0, 1000),
    rtol=1e-5, atol=1e-8, max_step=5
)
```

### Standard (Default)
```python
solver = ODESolver(model)
solution = solver.solve(initial_state, (0, 1000))
```

### Fast (Less Accurate)
```python
solver = ODESolver(model, method="RK23")
solution = solver.solve(
    initial_state, (0, 1000),
    rtol=1e-2, atol=1e-5, max_step=50
)
```

### Stiff Problems
```python
solver = ODESolver(model, method="LSODA")
solution = solver.solve(initial_state, (0, 1000))
```

## Accessing Solution Data

```python
# Time points
t = solution.t                 # 1D array of times

# State values
y = solution.y                 # 2D array: (2, len(t))
concentrations = solution.y[0, :]
temperatures = solution.y[1, :]

# Solution object
sol_func = solution.sol         # Callable for any time (if dense_output=True)

# Metadata
print(solution.status)         # 0 = success
print(solution.message)        # Description
```

## Performance Tips

1. **Use appropriate method**
   - Non-stiff → RK45
   - Stiff → LSODA or BDF
   - High accuracy → DOP853

2. **Adjust tolerances**
   - Looser tolerances → faster but less accurate
   - Tighter tolerances → slower but more accurate

3. **Use t_eval for specific times**
   - More efficient than evaluating solution later

4. **Set max_step appropriately**
   - Too small → slow integration
   - Too large → may miss dynamics

5. **Disable dense_output if not needed**
   - Saves memory if not interpolating

## See Also

- Full documentation: `docs/ODE_SOLVER.md`
- Examples: `examples/ode_solver_example.py`
- Tests: `tests/unit/test_ode_solver.py`
