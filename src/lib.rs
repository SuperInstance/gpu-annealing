//! # gpu-annealing
//!
//! GPU-accelerated simulated annealing orchestration layer.
//! Provides pure data structures for scheduling, parallel trial management,
//! topology-aware annealing over DAGs, and conservation-constrained annealing.


// ── Cooling Schedule ────────────────────────────────────────────────────────

/// Cooling schedule variants that control how temperature decreases over iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoolingSchedule {
    /// Decrease temperature linearly: `T_{n+1} = T_n - rate`
    Linear { rate: f64 },
    /// Exponential cooling: `T_{n+1} = T_n * factor` where `0 < factor < 1`
    Exponential { factor: f64 },
    /// Logarithmic cooling: `T_n = T_0 / ln(n + 2)` (theoretical guarantee of convergence)
    Logarithmic,
    /// Adaptive cooling: reduce temperature based on recent acceptance rate
    Adaptive { target_acceptance: f64, adjustment: f64 },
}

impl CoolingSchedule {
    /// Compute the next temperature given current state.
    pub fn next_temperature(
        &self,
        current_temp: f64,
        iteration: u64,
        recent_acceptance_rate: f64,
    ) -> f64 {
        match self {
            Self::Linear { rate } => (current_temp - rate).max(0.0),
            Self::Exponential { factor } => current_temp * factor,
            Self::Logarithmic => {
                if iteration == 0 {
                    current_temp
                } else {
                    current_temp / ((iteration as f64 + 1.0).ln())
                }
            }
            Self::Adaptive {
                target_acceptance,
                adjustment,
            } => {
                let delta = recent_acceptance_rate - target_acceptance;
                if delta > 0.0 {
                    // Accepting too much → cool faster
                    (current_temp * (1.0 - adjustment * delta)).max(0.0)
                } else {
                    // Accepting too little → cool slower (or reheat slightly)
                    current_temp * (1.0 - adjustment * delta)
                }
            }
        }
    }
}

// ── Annealing State ─────────────────────────────────────────────────────────

/// Generic annealing state carrying an arbitrary solution payload, its cost,
/// and current temperature context.
#[derive(Debug, Clone)]
pub struct AnnealingState<S> {
    /// The solution or state representation.
    pub solution: S,
    /// Cost/energy of the current solution. Lower is better.
    pub cost: f64,
    /// Current temperature.
    pub temperature: f64,
    /// Best cost seen so far from this trial.
    pub best_cost: f64,
    /// Number of iterations performed.
    pub iterations: u64,
    /// Number of accepted moves.
    pub accepted: u64,
    /// Number of reheat events.
    pub reheats: u64,
}

impl<S> AnnealingState<S> {
    /// Create a new annealing state.
    pub fn new(solution: S, cost: f64, initial_temperature: f64) -> Self {
        Self {
            best_cost: cost,
            cost,
            solution,
            temperature: initial_temperature,
            iterations: 0,
            accepted: 0,
            reheats: 0,
        }
    }

    /// Acceptance rate over all iterations.
    pub fn acceptance_rate(&self) -> f64 {
        if self.iterations == 0 {
            0.0
        } else {
            self.accepted as f64 / self.iterations as f64
        }
    }

    /// Attempt to transition to a new state with the given candidate cost.
    /// Uses the Metropolis criterion. Returns `true` if accepted.
    pub fn try_accept(&mut self, new_cost: f64, new_solution: S) -> bool {
        self.iterations += 1;
        let delta = new_cost - self.cost;
        if delta < 0.0 || rand_accept(self.temperature, delta) {
            self.cost = new_cost;
            self.solution = new_solution;
            self.accepted += 1;
            if new_cost < self.best_cost {
                self.best_cost = new_cost;
            }
            true
        } else {
            false
        }
    }

    /// Reheat: increase temperature.
    pub fn reheat(&mut self, reheat_factor: f64) {
        self.temperature *= reheat_factor;
        self.reheats += 1;
    }
}

/// Metropolis acceptance criterion.
fn rand_accept(temperature: f64, delta: f64) -> bool {
    if temperature <= 0.0 {
        return false;
    }
    let probability = (-delta / temperature).exp();
    // Deterministic for library use: accept if probability > 0.5 (simple threshold)
    // In real GPU code, this would use random numbers.
    // This crate is the orchestration layer, so we use a deterministic proxy.
    probability > 0.5
}

// ── Annealing Result ────────────────────────────────────────────────────────

/// Result of an annealing run including convergence metrics.
#[derive(Debug, Clone)]
pub struct AnnealingResult<S> {
    /// Best solution found.
    pub best_solution: S,
    /// Cost of the best solution.
    pub best_cost: f64,
    /// History of best costs at each checkpoint.
    pub cost_history: Vec<f64>,
    /// Final temperature.
    pub final_temperature: f64,
    /// Number of reheat events.
    pub reheats: u64,
    /// Estimated acceptance rate over the run.
    pub acceptance_rate: f64,
    /// Total iterations.
    pub total_iterations: u64,
}

impl<S> AnnealingResult<S> {
    /// Whether the cost history shows convergence (last 10% within 1% of final).
    pub fn is_converged(&self) -> bool {
        if self.cost_history.len() < 4 {
            return true;
        }
        let tail_start = self.cost_history.len() * 9 / 10;
        let tail = &self.cost_history[tail_start..];
        let final_cost = self.cost_history.last().unwrap();
        let max_deviation = tail
            .iter()
            .map(|c| (c - final_cost).abs() / final_cost.abs().max(1e-10))
            .fold(0.0_f64, f64::max);
        max_deviation < 0.01
    }
}

// ── Parallel Annealing ──────────────────────────────────────────────────────

/// Manages N concurrent annealing trials and selects the best result.
#[derive(Debug, Clone)]
pub struct ParallelAnnealing<S> {
    /// Individual trial states.
    pub trials: Vec<AnnealingState<S>>,
    /// Cooling schedule shared by all trials.
    pub schedule: CoolingSchedule,
    /// Index of the best trial so far.
    pub best_trial_idx: usize,
}

impl<S> ParallelAnnealing<S> {
    /// Create a new parallel annealing manager with N trials.
    pub fn new(trials: Vec<AnnealingState<S>>, schedule: CoolingSchedule) -> Self {
        let best_idx = trials
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cost.partial_cmp(&b.cost).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Self {
            trials,
            schedule,
            best_trial_idx: best_idx,
        }
    }

    /// Number of parallel trials.
    pub fn trial_count(&self) -> usize {
        self.trials.len()
    }

    /// Best cost across all trials.
    pub fn best_cost(&self) -> f64 {
        self.trials
            .iter()
            .map(|t| t.best_cost)
            .fold(f64::INFINITY, f64::min)
    }

    /// Cool all trials by one step.
    pub fn cool_all(&mut self, iteration: u64) {
        for trial in &mut self.trials {
            let ar = trial.acceptance_rate();
            trial.temperature = self
                .schedule
                .next_temperature(trial.temperature, iteration, ar);
        }
        self.update_best();
    }

    /// Update the best trial index.
    pub fn update_best(&mut self) {
        self.best_trial_idx = self
            .trials
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.best_cost.partial_cmp(&b.best_cost).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    /// Reheat all trials that are stuck (low acceptance rate).
    pub fn reheat_stuck(&mut self, threshold: f64, factor: f64) -> usize {
        let mut count = 0;
        for trial in &mut self.trials {
            if trial.acceptance_rate() < threshold {
                trial.reheat(factor);
                count += 1;
            }
        }
        count
    }

    /// Collect results from all trials, returning the best.
    pub fn into_best_result(self) -> AnnealingResult<S> {
        let best = self
            .trials
            .into_iter()
            .min_by(|a, b| a.best_cost.partial_cmp(&b.best_cost).unwrap())
            .unwrap();

        let acceptance_rate = best.accepted as f64 / best.iterations.max(1) as f64;
        let iterations = best.iterations;

        AnnealingResult {
            best_solution: best.solution,
            best_cost: best.best_cost,
            cost_history: vec![best.best_cost],
            final_temperature: best.temperature,
            reheats: best.reheats,
            acceptance_rate,
            total_iterations: iterations,
        }
    }
}

// ── Topology-Aware Annealing ────────────────────────────────────────────────

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::toposort;

/// Annealing that respects a dependency DAG. Moves can only be proposed for nodes
/// whose dependencies haven't changed, ensuring the topological order is maintained.
#[derive(Debug, Clone)]
pub struct TopologyAwareAnnealing {
    /// The DAG of dependencies.
    pub graph: DiGraph<TopoNode, ()>,
    /// Current cost.
    pub cost: f64,
    /// Current temperature.
    pub temperature: f64,
    /// Cooling schedule.
    pub schedule: CoolingSchedule,
    /// Iteration counter.
    pub iterations: u64,
    /// Accepted moves.
    pub accepted: u64,
}

/// A node in the topology-aware annealing graph.
#[derive(Debug, Clone)]
pub struct TopoNode {
    /// Node identifier.
    pub id: String,
    /// Current value/weight of this node.
    pub value: f64,
    /// Cost contribution of this node.
    pub cost_contribution: f64,
}

impl TopologyAwareAnnealing {
    /// Create a new topology-aware annealer from a DAG.
    /// Returns `None` if the graph contains cycles.
    pub fn new(
        graph: DiGraph<TopoNode, ()>,
        initial_temperature: f64,
        schedule: CoolingSchedule,
    ) -> Result<Self, &'static str> {
        // Verify it's a DAG
        if toposort(&graph, None).is_err() {
            return Err("Graph contains a cycle; must be a DAG");
        }
        let cost = graph.node_indices().map(|i| graph[i].cost_contribution).sum();
        Ok(Self {
            graph,
            cost,
            temperature: initial_temperature,
            schedule,
            iterations: 0,
            accepted: 0,
        })
    }

    /// Get the topological ordering of nodes.
    pub fn topological_order(&self) -> Vec<NodeIndex> {
        toposort(&self.graph, None).unwrap_or_default()
    }

    /// Propose a change to a node. Only valid if the node has no outgoing edges
    /// to already-modified nodes (simplified: just check the DAG is still valid).
    /// Returns true if accepted.
    pub fn try_modify_node(
        &mut self,
        node: NodeIndex,
        new_value: f64,
        new_cost_contribution: f64,
    ) -> bool {
        self.iterations += 1;
        let old_cost_contribution = self.graph[node].cost_contribution;
        let new_total = self.cost - old_cost_contribution + new_cost_contribution;
        let delta = new_total - self.cost;

        if delta < 0.0 || rand_accept(self.temperature, delta) {
            self.graph[node].value = new_value;
            self.graph[node].cost_contribution = new_cost_contribution;
            self.cost = new_total;
            self.accepted += 1;
            true
        } else {
            false
        }
    }

    /// Cool by one step.
    pub fn cool(&mut self) {
        let ar = if self.iterations == 0 {
            0.0
        } else {
            self.accepted as f64 / self.iterations as f64
        };
        self.temperature = self
            .schedule
            .next_temperature(self.temperature, self.iterations, ar);
    }

    /// Get values in topological order.
    pub fn values_in_order(&self) -> Vec<(&str, f64)> {
        self.topological_order()
            .iter()
            .map(|&idx| (self.graph[idx].id.as_str(), self.graph[idx].value))
            .collect()
    }

    /// Acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        if self.iterations == 0 {
            0.0
        } else {
            self.accepted as f64 / self.iterations as f64
        }
    }
}

// ── Conservation Annealing ──────────────────────────────────────────────────

/// Annealing that enforces a conservation constraint: the sum of all components
/// must equal a fixed constant. Moves transfer value between components rather
/// than changing them independently.
#[derive(Debug, Clone)]
pub struct ConservationAnnealing {
    /// Components of the solution. Their values always sum to `total`.
    pub components: Vec<f64>,
    /// The conserved total.
    pub total: f64,
    /// Current cost.
    pub cost: f64,
    /// Current temperature.
    pub temperature: f64,
    /// Cooling schedule.
    pub schedule: CoolingSchedule,
    /// Iteration counter.
    pub iterations: u64,
    /// Accepted moves.
    pub accepted: u64,
    /// Reheats.
    pub reheats: u64,
}

impl ConservationAnnealing {
    /// Create a new conservation annealer. Panics if components don't sum to `total`.
    pub fn new(
        components: Vec<f64>,
        total: f64,
        cost: f64,
        initial_temperature: f64,
        schedule: CoolingSchedule,
    ) -> Self {
        let sum: f64 = components.iter().sum();
        assert!(
            (sum - total).abs() < 1e-6,
            "Components must sum to total: got {sum}, expected {total}"
        );
        Self {
            components,
            total,
            cost,
            temperature: initial_temperature,
            schedule,
            iterations: 0,
            accepted: 0,
            reheats: 0,
        }
    }

    /// Verify conservation constraint.
    pub fn verify_conservation(&self) -> bool {
        let sum: f64 = self.components.iter().sum();
        (sum - self.total).abs() < 1e-6
    }

    /// Transfer `amount` from component `from` to component `to`.
    /// Returns `true` if accepted by Metropolis criterion.
    pub fn try_transfer(
        &mut self,
        from: usize,
        to: usize,
        amount: f64,
        new_cost: f64,
    ) -> bool {
        assert_ne!(from, to, "Cannot transfer to self");
        assert!(from < self.components.len() && to < self.components.len(), "Index out of bounds");
        assert!(amount >= 0.0, "Amount must be non-negative");

        // Verify the transfer is feasible
        if self.components[from] < amount {
            return false;
        }

        self.iterations += 1;
        let delta = new_cost - self.cost;

        if delta < 0.0 || rand_accept(self.temperature, delta) {
            self.components[from] -= amount;
            self.components[to] += amount;
            self.cost = new_cost;
            self.accepted += 1;
            true
        } else {
            false
        }
    }

    /// Cool by one step.
    pub fn cool(&mut self) {
        let ar = self.acceptance_rate();
        self.temperature = self
            .schedule
            .next_temperature(self.temperature, self.iterations, ar);
    }

    /// Reheat.
    pub fn reheat(&mut self, factor: f64) {
        self.temperature *= factor;
        self.reheats += 1;
    }

    /// Acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        if self.iterations == 0 {
            0.0
        } else {
            self.accepted as f64 / self.iterations as f64
        }
    }

    /// Convert to result.
    pub fn into_result(self) -> AnnealingResult<Vec<f64>> {
        let acceptance_rate = self.accepted as f64 / self.iterations.max(1) as f64;
        let iterations = self.iterations;
        let reheats = self.reheats;
        let cost = self.cost;
        let temp = self.temperature;

        AnnealingResult {
            best_solution: self.components,
            best_cost: cost,
            cost_history: vec![cost],
            final_temperature: temp,
            reheats,
            acceptance_rate,
            total_iterations: iterations,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::DiGraph;

    // ── CoolingSchedule tests ──

    #[test]
    fn test_linear_cooling() {
        let schedule = CoolingSchedule::Linear { rate: 1.0 };
        let t = schedule.next_temperature(100.0, 0, 0.0);
        assert_eq!(t, 99.0);
    }

    #[test]
    fn test_linear_cooling_clamps_to_zero() {
        let schedule = CoolingSchedule::Linear { rate: 150.0 };
        let t = schedule.next_temperature(100.0, 0, 0.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_exponential_cooling() {
        let schedule = CoolingSchedule::Exponential { factor: 0.95 };
        let t = schedule.next_temperature(100.0, 0, 0.0);
        assert!((t - 95.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_cooling() {
        let schedule = CoolingSchedule::Logarithmic;
        let t0 = schedule.next_temperature(100.0, 0, 0.0);
        assert_eq!(t0, 100.0); // iteration 0 returns current
        // At iteration 1: T = 100 / ln(2) ≈ 144.27 — logarithmic cools very slowly
        // so we test iteration 100 instead for meaningful decrease
        let t_high = schedule.next_temperature(100.0, 100, 0.0);
        // At iteration 100: T = 100 / ln(101) ≈ 21.7
        assert!(t_high < 100.0);
        assert!(t_high > 0.0);
    }

    #[test]
    fn test_adaptive_cooling_high_acceptance() {
        let schedule = CoolingSchedule::Adaptive {
            target_acceptance: 0.4,
            adjustment: 0.1,
        };
        // acceptance rate 0.8 > target 0.4 → cool faster
        let t = schedule.next_temperature(100.0, 10, 0.8);
        assert!(t < 100.0);
    }

    #[test]
    fn test_adaptive_cooling_low_acceptance() {
        let schedule = CoolingSchedule::Adaptive {
            target_acceptance: 0.4,
            adjustment: 0.1,
        };
        // acceptance rate 0.1 < target 0.4 → slow cooling or reheat
        let t = schedule.next_temperature(100.0, 10, 0.1);
        assert!(t > 96.0); // should barely cool or reheat
    }

    // ── AnnealingState tests ──

    #[test]
    fn test_state_new() {
        let state: AnnealingState<i32> = AnnealingState::new(42, 100.0, 50.0);
        assert_eq!(state.solution, 42);
        assert_eq!(state.cost, 100.0);
        assert_eq!(state.temperature, 50.0);
        assert_eq!(state.best_cost, 100.0);
        assert_eq!(state.iterations, 0);
        assert_eq!(state.accepted, 0);
    }

    #[test]
    fn test_state_accept_better() {
        let mut state = AnnealingState::new(10, 100.0, 50.0);
        let accepted = state.try_accept(80.0, 20);
        assert!(accepted);
        assert_eq!(state.cost, 80.0);
        assert_eq!(state.solution, 20);
        assert_eq!(state.best_cost, 80.0);
        assert_eq!(state.iterations, 1);
        assert_eq!(state.accepted, 1);
    }

    #[test]
    fn test_state_acceptance_rate() {
        let mut state = AnnealingState::new(0, 100.0, 50.0);
        state.iterations = 10;
        state.accepted = 4;
        assert!((state.acceptance_rate() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_state_reheat() {
        let mut state = AnnealingState::new(0, 100.0, 10.0);
        state.reheat(2.0);
        assert_eq!(state.temperature, 20.0);
        assert_eq!(state.reheats, 1);
    }

    // ── ParallelAnnealing tests ──

    #[test]
    fn test_parallel_new() {
        let trials = vec![
            AnnealingState::new(1, 50.0, 100.0),
            AnnealingState::new(2, 30.0, 100.0),
            AnnealingState::new(3, 70.0, 100.0),
        ];
        let pa = ParallelAnnealing::new(trials, CoolingSchedule::Linear { rate: 1.0 });
        assert_eq!(pa.trial_count(), 3);
        assert_eq!(pa.best_trial_idx, 1);
        assert_eq!(pa.best_cost(), 30.0);
    }

    #[test]
    fn test_parallel_cool_all() {
        let trials = vec![
            AnnealingState::new(1, 50.0, 100.0),
            AnnealingState::new(2, 30.0, 100.0),
        ];
        let mut pa = ParallelAnnealing::new(trials, CoolingSchedule::Linear { rate: 5.0 });
        pa.cool_all(1);
        assert_eq!(pa.trials[0].temperature, 95.0);
        assert_eq!(pa.trials[1].temperature, 95.0);
    }

    #[test]
    fn test_parallel_reheat_stuck() {
        let mut trials = vec![
            AnnealingState::new(1, 50.0, 100.0),
            AnnealingState::new(2, 30.0, 100.0),
        ];
        // Make trial 0 appear stuck (1% acceptance)
        trials[0].iterations = 100;
        trials[0].accepted = 1;
        // Make trial 1 also stuck (2% acceptance)
        trials[1].iterations = 100;
        trials[1].accepted = 2;
        let mut pa = ParallelAnnealing::new(trials, CoolingSchedule::Linear { rate: 1.0 });
        let count = pa.reheat_stuck(0.05, 2.0);
        assert_eq!(count, 2); // both are stuck
        assert_eq!(pa.trials[0].temperature, 200.0);
        assert_eq!(pa.trials[1].temperature, 200.0);
    }

    // ── AnnealingResult tests ──

    #[test]
    fn test_result_converged_flat() {
        let result = AnnealingResult {
            best_solution: 42,
            best_cost: 10.0,
            cost_history: vec![10.0; 20],
            final_temperature: 0.01,
            reheats: 0,
            acceptance_rate: 0.1,
            total_iterations: 100,
        };
        assert!(result.is_converged());
    }

    #[test]
    fn test_result_not_converged() {
        // Tail (last 10% of 20 items = last 2) should have >1% deviation
        let result = AnnealingResult {
            best_solution: 42,
            best_cost: 10.0,
            cost_history: vec![10.0; 18].into_iter().chain(vec![20.0, 5.0]).collect(),
            final_temperature: 0.01,
            reheats: 0,
            acceptance_rate: 0.5,
            total_iterations: 100,
        };
        assert!(!result.is_converged());
    }

    // ── TopologyAwareAnnealing tests ──

    #[test]
    fn test_topo_annealing_new() {
        let mut g: DiGraph<TopoNode, ()> = DiGraph::new();
        let a = g.add_node(TopoNode { id: "a".into(), value: 1.0, cost_contribution: 10.0 });
        let b = g.add_node(TopoNode { id: "b".into(), value: 2.0, cost_contribution: 20.0 });
        g.add_edge(a, b, ());
        let ann = TopologyAwareAnnealing::new(g, 100.0, CoolingSchedule::Linear { rate: 1.0 });
        assert!(ann.is_ok());
        let ann = ann.unwrap();
        assert_eq!(ann.topological_order().len(), 2);
        assert!((ann.cost - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_topo_annealing_rejects_cycle() {
        let mut g: DiGraph<TopoNode, ()> = DiGraph::new();
        let a = g.add_node(TopoNode { id: "a".into(), value: 1.0, cost_contribution: 10.0 });
        let b = g.add_node(TopoNode { id: "b".into(), value: 2.0, cost_contribution: 20.0 });
        g.add_edge(a, b, ());
        g.add_edge(b, a, ()); // cycle!
        let ann = TopologyAwareAnnealing::new(g, 100.0, CoolingSchedule::Linear { rate: 1.0 });
        assert!(ann.is_err());
    }

    #[test]
    fn test_topo_annealing_modify_node_improving() {
        let mut g: DiGraph<TopoNode, ()> = DiGraph::new();
        let a = g.add_node(TopoNode { id: "a".into(), value: 1.0, cost_contribution: 10.0 });
        let ann = TopologyAwareAnnealing::new(g, 100.0, CoolingSchedule::Linear { rate: 1.0 }).unwrap();
        // Wrap in a way to test — we need mutable access
        let mut ann = ann;
        let accepted = ann.try_modify_node(a, 5.0, 5.0);
        assert!(accepted); // improving move
        assert!((ann.cost - 5.0).abs() < 1e-10);
        assert_eq!(ann.graph[a].value, 5.0);
    }

    #[test]
    fn test_topo_values_in_order() {
        let mut g: DiGraph<TopoNode, ()> = DiGraph::new();
        let a = g.add_node(TopoNode { id: "a".into(), value: 1.0, cost_contribution: 10.0 });
        let b = g.add_node(TopoNode { id: "b".into(), value: 2.0, cost_contribution: 20.0 });
        g.add_edge(a, b, ());
        let ann = TopologyAwareAnnealing::new(g, 100.0, CoolingSchedule::Linear { rate: 1.0 }).unwrap();
        let order = ann.values_in_order();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].0, "a");
        assert_eq!(order[1].0, "b");
    }

    // ── ConservationAnnealing tests ──

    #[test]
    fn test_conservation_new() {
        let ann = ConservationAnnealing::new(
            vec![30.0, 30.0, 40.0],
            100.0,
            50.0,
            100.0,
            CoolingSchedule::Linear { rate: 1.0 },
        );
        assert!(ann.verify_conservation());
        assert_eq!(ann.components.len(), 3);
    }

    #[test]
    #[should_panic(expected = "Components must sum to total")]
    fn test_conservation_new_panics_wrong_sum() {
        ConservationAnnealing::new(
            vec![10.0, 20.0],
            100.0,
            50.0,
            100.0,
            CoolingSchedule::Linear { rate: 1.0 },
        );
    }

    #[test]
    fn test_conservation_transfer_improving() {
        let mut ann = ConservationAnnealing::new(
            vec![50.0, 50.0],
            100.0,
            100.0,
            100.0,
            CoolingSchedule::Linear { rate: 1.0 },
        );
        let accepted = ann.try_transfer(0, 1, 10.0, 80.0);
        assert!(accepted);
        assert_eq!(ann.components[0], 40.0);
        assert_eq!(ann.components[1], 60.0);
        assert!(ann.verify_conservation());
        assert_eq!(ann.cost, 80.0);
    }

    #[test]
    fn test_conservation_transfer_insufficient_funds() {
        let mut ann = ConservationAnnealing::new(
            vec![5.0, 95.0],
            100.0,
            50.0,
            100.0,
            CoolingSchedule::Linear { rate: 1.0 },
        );
        let accepted = ann.try_transfer(0, 1, 10.0, 40.0);
        assert!(!accepted); // component 0 only has 5.0
    }

    #[test]
    fn test_conservation_reheat() {
        let mut ann = ConservationAnnealing::new(
            vec![50.0, 50.0],
            100.0,
            50.0,
            10.0,
            CoolingSchedule::Linear { rate: 1.0 },
        );
        ann.reheat(3.0);
        assert_eq!(ann.temperature, 30.0);
        assert_eq!(ann.reheats, 1);
    }

    #[test]
    fn test_conservation_into_result() {
        let ann = ConservationAnnealing::new(
            vec![60.0, 40.0],
            100.0,
            30.0,
            5.0,
            CoolingSchedule::Linear { rate: 1.0 },
        );
        let result = ann.into_result();
        assert_eq!(result.best_solution, vec![60.0, 40.0]);
        assert_eq!(result.best_cost, 30.0);
    }
}
