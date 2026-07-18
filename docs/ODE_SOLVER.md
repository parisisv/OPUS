# ODE Solver Documentation

## Overview

The ODE Solver module provides a high-level interface for solving the ordinary differential equations (ODEs) that describe reactor dynamics in OPUS. It wraps SciPy's `solve_ivp` function with reactor-specific functionality.

## Key Features

- **Multiple Integration Methods**: Support for RK45, RK23, BDF, DOP853, LSODA, and Radau methods
- **Dense Output**: Continuous solution functions for interpolation
- **Event Detection**: Support for detecting special events during integration
- **Physical State Management**: Direct integration with `ReactorState` objects
- **Error Handling**: Clear error messages and validation

## Core Classes

### ODESolver

The main solver class for integrating ODE systems.

#### Initialization

```python
from opus.solvers import ODESolver
from opus.models.cstr import CSTR

solver = ODESolver(
    model=cstr_model,
    method="RK45",              # Integration method
    dense_output=True,          # Enable continuous output
    events=None                 # Event detection functions
)
```

**Parameters:**
- `model` (CSTR): Reactor model with `rhs()` method
- `method` (str): Integration method ('RK45', 'RK23', 'BDF', 'DOP853', 'LSODA', 'Radau')
- `dense_output` (bool): Enable continuous solution functions (default: True)
- `events` (callable, optional): Event detection functions

#### Main Methods

##### solve()

Solve the ODE system over a time interval.

```python
solution = solver.solve(
    initial_state=initial_state,
    t_span=(t_start, t_end),
    t_eval=None,                # Specific time points (optional)
    max_step=None,              # Maximum step size (optional)
    rtol=1e-3,                  # Relative tolerance
    atol=1e-6                   # Absolute tolerance
)
```

**Parameters:**
- `initial_state` (ReactorState): Initial reactor conditions
- `t_span` (tuple): Time interval (t_start, t_end) in seconds
- `t_eval` (ndarray, optional): Specific time points to evaluate
- `max_step` (float, optional): Maximum allowed step size (seconds)
- `rtol` (float): Relative error tolerance (default: 1e-3)
- `atol` (float): Absolute error tolerance (default: 1e-6)

**Returns:** ODESolution object

##### get_final_state()

Extract the final reactor state from a solution.

```python
final_state = solver.get_final_state(solution)
# final_state is a ReactorState object
```

##### get_state_at_time()

Get the reactor state at a specific time.

```python
state_at_500s = solver.get_state_at_time(solution, 500.0)
```

**Requires:** `dense_output=True`

##### evaluate_solution()

Evaluate the solution at arbitrary times.

```python
t_new = np.array([10.0, 25.0, 50.0, 75.0])
y_new = solver.evaluate_solution(solution, t_new)
# Returns shape (n_states, len(t_new))
```

**Requires:** `dense_output=True`

##### solve_to_events()

Solve until specific events are detected.

```python
def temperature_threshold(t, y):
    return y[1] - 350.0  # Temperature threshold

temperature_threshold.terminal = True
temperature_threshold.direction = -1

solution = solver.solve_to_events(
    initial_state=initial_state,
    t_span=(0.0, 1000.0),
    events=temperature_threshold
)
```

### ODESolution

Data class containing the solution from ODE integration.

**Attributes:**
- `t` (ndarray): Time points [s]
- `y` (ndarray): State values at each time point, shape (n_states, n_timepoints)
- `sol` (OdeSolution or None): Continuous solution object (if dense_output=True)
- `status` (int): Integration status (0 = success)
- `message` (str): Integration message

## Integration Methods

Different methods are suitable for different problem types:

| Method | Type | Best For |
|--------|------|----------|
| RK45 | Non-stiff | Default choice, most problems |
| RK23 | Non-stiff | Low accuracy required, fast |
| BDF | Stiff | Stiff systems with sharp transitions |
| LSODA | Adaptive | Automatically handles stiff/non-stiff |
| Radau | Stiff | Very stiff systems |
| DOP853 | Non-stiff | Very high accuracy required |

## Usage Examples

### Basic Integration

```python
from opus.core.parameters import SimulationParameters
from opus.models.cstr import CSTR
from opus.models.state import ReactorState
from opus.solvers import ODESolver

# Create model with parameters
params = SimulationParameters(...)
model = CSTR(params)

# Create solver
solver = ODESolver(model)

# Define initial state
initial_state = ReactorState(
    concentration=100.0,  # mol/m³
    temperature=300.0     # K
)

# Solve for 1000 seconds
solution = solver.solve(
    initial_state=initial_state,
    t_span=(0.0, 1000.0),
    max_step=10.0
)

# Get final state
final_state = solver.get_final_state(solution)
print(f"Final concentration: {final_state.concentration} mol/m³")
print(f"Final temperature: {final_state.temperature} K")
```

### Integration with Time Points

```python
import numpy as np

# Define specific time points
t_eval = np.linspace(0.0, 1000.0, 101)

solution = solver.solve(
    initial_state=initial_state,
    t_span=(0.0, 1000.0),
    t_eval=t_eval  # Solution at these specific times
)

# Access time data
times = solution.t
concentrations = solution.y[0, :]
temperatures = solution.y[1, :]
```

### Event Detection

```python
# Define an event function (concentration threshold)
def concentration_minimum(t, y):
    return y[0] - 10.0  # Trigger when C < 10 mol/m³

concentration_minimum.terminal = True
concentration_minimum.direction = -1

solution = solver.solve_to_events(
    initial_state=initial_state,
    t_span=(0.0, 10000.0),
    events=concentration_minimum
)

print(f"Event detected at t = {solution.t[-1]} s")
```

### Comparing Integration Methods

```python
for method in ["RK45", "BDF", "LSODA"]:
    solver = ODESolver(model, method=method)
    solution = solver.solve(
        initial_state=initial_state,
        t_span=(0.0, 1000.0)
    )
    final_state = solver.get_final_state(solution)
    print(f"{method}: {len(solution.t)} steps, "
          f"final T = {final_state.temperature:.2f} K")
```

## Performance Considerations

### Choosing Tolerances

- **Default** (rtol=1e-3, atol=1e-6): Good for most applications
- **Loose** (rtol=1e-2, atol=1e-5): Faster integration, for rough estimates
- **Tight** (rtol=1e-6, atol=1e-9): High accuracy, slower integration

### Memory Usage

- **Dense Output On**: Stores continuous solution function (~20-50 KB extra)
- **Dense Output Off**: Only stores computed points

### Step Size Control

- **Automatic (default)**: Solver adjusts steps based on local error
- **max_step parameter**: Limit maximum step size for safety constraints

## Error Handling

The solver provides clear error messages for common issues:

```python
try:
    solution = solver.solve(...)
except RuntimeError as e:
    print(f"Integration failed: {e}")

try:
    state = solver.get_state_at_time(solution, 5000.0)
except ValueError as e:
    print(f"Invalid time request: {e}")
```

## Validation

The ODE solver validates:

1. **Initial conditions**: Must be physical (C ≥ 0, T > 0)
2. **Time span**: t_start < t_end
3. **Method selection**: Only valid scipy methods
4. **Dense output operations**: Requires dense_output=True
5. **Time bounds**: Requested times within solution range

## Physical Constraints

The solver maintains:

- **Non-negative concentrations**: C(t) ≥ 0
- **Positive temperatures**: T(t) > 0 K
- **Consistent state vectors**: 2 elements per state

## See Also

- [CSTR Model Documentation](../models/README.md)
- [Kinetics Module](../kinetics/README.md)
- [SciPy ODE Integration](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
