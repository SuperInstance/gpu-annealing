# gpu-annealing

GPU-accelerated simulated annealing orchestration layer — pure data structures for scheduling, parallel trial management, topology-aware annealing over DAGs, and conservation-constrained optimization.

## Overview

This crate provides the **orchestration layer** for GPU-accelerated simulated annealing. It implements pure data structures and scheduling logic — no actual GPU code — designed to be the host-side controller that dispatches work to GPU kernels.

## Features

- **`AnnealingState<S>`** — Generic state wrapper with f64 cost, temperature tracking, Metropolis acceptance criterion, and reheat support
- **`CoolingSchedule`** — Enum with Linear, Exponential, Logarithmic, and Adaptive cooling strategies
- **`ParallelAnnealing<S>`** — Manages N concurrent annealing trials with best-selection and stuck-trial reheat
- **`AnnealingResult<S>`** — Final result with cost history, convergence metrics, acceptance rate estimate
- **`TopologyAwareAnnealing`** — Annealing over a dependency DAG (uses `petgraph`), only proposes valid topological moves
- **`ConservationAnnealing`** — Annealing enforcing `∑components = constant` via transfer moves

## Usage

```rust
use gpu_annealing::*;

// Create a simple annealing state
let mut state = AnnealingState::new(vec![1.0, 2.0, 3.0], 100.0, 50.0);

// Try accepting an improving move
let accepted = state.try_accept(80.0, vec![1.5, 2.0, 2.5]);
assert!(accepted);

// Run parallel trials
let trials = vec![
    AnnealingState::new(0, 50.0, 100.0),
    AnnealingState::new(1, 30.0, 100.0),
    AnnealingState::new(2, 70.0, 100.0),
];
let mut parallel = ParallelAnnealing::new(trials, CoolingSchedule::Exponential { factor: 0.95 });
parallel.cool_all(1);
let result = parallel.into_best_result();
```

## Conservation Annealing

```rust
use gpu_annealing::*;

let mut ann = ConservationAnnealing::new(
    vec![50.0, 50.0], // components
    100.0,            // conserved total
    100.0,            // initial cost
    50.0,             // initial temperature
    CoolingSchedule::Linear { rate: 1.0 },
);

// Transfer 10 units from component 0 to 1 (cost improves to 80)
ann.try_transfer(0, 1, 10.0, 80.0);
assert!(ann.verify_conservation());
```

## Topology-Aware Annealing

```rust
use gpu_annealing::*;
use petgraph::graph::DiGraph;

let mut graph = DiGraph::new();
let a = graph.add_node(TopoNode { id: "a".into(), value: 1.0, cost_contribution: 10.0 });
let b = graph.add_node(TopoNode { id: "b".into(), value: 2.0, cost_contribution: 20.0 });
graph.add_edge(a, b, ());

let mut ann = TopologyAwareAnnealing::new(graph, 100.0, CoolingSchedule::Linear { rate: 1.0 }).unwrap();
ann.try_modify_node(a, 5.0, 5.0); // improving → accepted
```

## License

MIT
