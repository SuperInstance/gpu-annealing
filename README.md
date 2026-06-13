# gpu-annealing: GPU-Accelerated Simulated Annealing Orchestration

A pure-Rust orchestration layer for GPU-accelerated simulated annealing. Provides cooling schedules, parallel trial management, topology-aware annealing over DAGs, and conservation-constrained optimization. The crate contains **no GPU code** — it manages the state, scheduling, and decision logic that drives a GPU kernel executing thousands of annealing trials in parallel.

## Why It Matters

Simulated annealing (SA) is one of the most versatile **global optimization** algorithms — it can escape local minima that trap gradient-based methods. But SA is slow: it requires millions of function evaluations. GPU parallelism solves this by running thousands of independent annealing chains simultaneously, then selecting the best. This crate provides the **CPU-side orchestration** that decides when to cool, reheat, checkpoint, and terminate — the kernels themselves run on the GPU.

Applications: VLSI layout, traveling salesman, protein folding, neural architecture search, Ising model ground states, and combinatorial optimization over DAGs.

## How It Works

### The Metropolis Criterion

SA accepts worse solutions with probability:

$$P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta E < 0 \\ e^{-\Delta E / T} & \text{if } \Delta E \geq 0 \end{cases}$$

where ΔE is the energy increase and T is the current temperature. This crate uses a deterministic proxy (`P > 0.5`) for library use; GPU kernels replace this with random sampling.

### Cooling Schedules

Four schedules control how temperature decreases:

| Schedule | Update Rule | Convergence Guarantee |
|----------|-------------|----------------------|
| **Linear** | T_{n+1} = T_n − rate | None |
| **Exponential** | T_{n+1} = T_n × factor | Geometric, fast |
| **Logarithmic** | T_n = T₀ / ln(n+1) | **Proven**: Hajek (1988) |
| **Adaptive** | Adjust based on acceptance rate | Empirical |

The **logarithmic schedule** is theoretically significant: Hajek proved that SA converges to the global optimum if and only if the cooling is no faster than c/log(n) where c is the maximum depth of any local minimum.

### Parallel Annealing

Manages N independent trials, each with its own `AnnealingState`:

```
ParallelAnnealing {
    trials: Vec<AnnealingState<S>>,
    schedule: CoolingSchedule,    // shared
    best_trial_idx: usize,
}
```

Operations: `cool_all()` (one step for all trials), `reheat_stuck(threshold)` (reheat low-acceptance trials), `into_best_result()` (select winner).

**Complexity**: O(N) per cooling step for N parallel trials.

### Topology-Aware Annealing

Annealing over a dependency DAG where node assignments must respect topological order. Moves are only valid if all dependencies remain satisfied. Uses `petgraph` for DAG operations:

```
TopologyAwareAnnealing {
    graph: DiGraph<TopoNode, ()>,
    cost: f64,
    temperature: f64,
    schedule: CoolingSchedule,
}
```

Cycle detection at construction: if `toposort` fails, the graph is rejected.

### Conservation-Constrained Annealing

Extends SA with a **conservation constraint** (γ + η = C): the total "mass" of the solution must remain constant. Any proposed move that violates conservation is rejected. This is analogous to:

- **Energy conservation** in physics simulations
- **Budget constraints** in resource allocation
- **Probability axioms** (Σp = 1) in distribution optimization

### Reheating

When acceptance rate drops below a threshold, temperature is multiplied by a reheat factor. This prevents premature convergence — the analog of **restarting** in random-restart hill climbing, but gentler.

### Complexity Summary

| Operation | Time | Notes |
|-----------|------|-------|
| `try_accept(ΔE)` | O(1) | Metropolis criterion |
| `cool_all(iteration)` | O(N) | N parallel trials |
| `reheat_stuck(threshold)` | O(N) | Check all trials |
| `into_best_result()` | O(N) | Find min cost |
| Topology annealing step | O(V + E) | Toposort validation |

## Quick Start

```rust
use gpu_annealing::{AnnealingState, CoolingSchedule, ParallelAnnealing};

// Create 4 parallel trials at T=1000
let trials: Vec<AnnealingState<Vec<f64>>> = (0..4).map(|_| {
    AnnealingState::new(vec![0.0; 100], 42.0, 1000.0)
}).collect();

let mut pa = ParallelAnnealing::new(trials, CoolingSchedule::Exponential { factor: 0.95 });

pa.cool_all(1);
assert!(pa.best_cost() <= 42.0);
let stuck = pa.reheat_stuck(0.01, 1.5); // reheat trials with <1% acceptance
```

## API

| Type | Purpose |
|------|---------|
| `CoolingSchedule` | Linear / Exponential / Logarithmic / Adaptive |
| `AnnealingState<S>` | Single trial: solution, cost, temperature, stats |
| `AnnealingResult<S>` | Final result with convergence metrics |
| `ParallelAnnealing<S>` | N trials with shared schedule |
| `TopologyAwareAnnealing` | DAG-constrained annealing |
| `ConservationAnnealing` | Mass-conserving variant |

## Architecture Notes

This crate is the **η (eta)** layer — orchestration and decision logic — in the γ + η = C framework. The **γ** is the GPU kernel that evaluates the cost function and generates random perturbations. This separation is essential: the GPU handles the compute-intensive inner loop (millions of Metropolis evaluations per second), while this crate handles the outer loop (when to cool, reheat, checkpoint, and terminate).

## References

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing*. Science 220, 671–680.
- Hajek, B. (1988). *Cooling Schedules for Optimal Annealing*. Mathematics of Operations Research 13(2), 311–329.
- Ingber, L. (1993). *Simulated Annealing: Practice versus Theory*. Math. Comput. Modelling 18(11).

## License

MIT
