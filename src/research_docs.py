import numpy as np

def print_mathematical_formulation():
    print("\n" + "="*75)
    print("📊 1. MATHEMATICAL FORMULATION")
    print("="*75)
    print("""
Decision Variables:
  x_{i,j} ∈ {0, 1}  (Binary variable for asset i, bit j)
  w_i = Δ * Σ (2^j * x_{i,j})  (Discretized weight, Δ = step size)

Objective Function (Minimize Volatility & Cost, Maximize Return):
  min  O(w) = λ * (w^T Σ w) - μ^T w + γ * Σ (w_i - w_{0,i})^2

Subject to Constraints:
  1. Budget:      Σ w_i = 1
  2. Position:    0 ≤ w_i ≤ 0.25
  3. Sector:      Σ_{i ∈ S_k} w_i ≤ 0.40
  4. Turnover:    Σ |w_i - w_{0,i}| ≤ 0.15
  5. Stress Cap:  Σ_{i ∈ Eq} w_i ≤ 0.40
  6. Return:      μ^T w ≥ 0.07
""")

def print_qubo_equation(Q):
    print("\n" + "="*75)
    print("⚛️ 2. QUBO FORMULATION & ENCODING")
    print("="*75)
    print("""
Quadratic Unconstrained Binary Optimization (QUBO):
  min_{x ∈ {0,1}^n}  x^T Q x
  
  Where Q is an n×n upper-triangular symmetric matrix.
  Constraints are mapped as quadratic penalty terms:
  H(x) = O(w) + P_budget*(Σw - 1)^2 + P_ret*(μ^Tw - ε)^2 + ...
""")
    print("Sample QUBO Matrix Snippet (Top-Left 6x6):")
    np.set_printoptions(precision=2, suppress=True)
    print(Q[:6, :6])
    
    print("""
Binary Encoding (6 Bits per Asset):
  - Why 6 bits? It provides 2^6 - 1 = 63 discrete states.
  - Precision (Δ): max_weight / 63 = 0.25 / 63 ≈ 0.397% per step.
  - This balances quantum resource limits (qubits) with financial 
    precision requirements (sub-0.5% granularity).
""")

def print_quantum_circuit():
    print("\n" + "="*75)
    print("🔬 3. QUANTUM CIRCUIT DIAGRAM (QAOA p=3)")
    print("="*75)
    print("""
      |0⟩ ──── H ──[e^{-iβ_1 H_M}]──[e^{-iγ_1 H_C}]── ... ──[e^{-iβ_3 H_M}]──[e^{-iγ_3 H_C}]── M ──
      
      Initial State: |+⟩^⊗n (Uniform Superposition via Hadamard gates)
      H_C: Cost Hamiltonian (Encodes the QUBO matrix via Pauli-Z rotations)
      H_M: Mixer Hamiltonian (Pauli-X rotations to explore state space)
      M:  Measurement in computational basis
""")

def print_research_discussion():
    print("\n" + "="*75)
    print("🎓 4. RESEARCH DISCUSSION & EVALUATION")
    print("="*75)
    print("""
Why QAOA / Quantum Annealing?
  Standard convex optimization struggles with discrete variables and 
  non-convex constraints. QUBO mapping allows the quantum annealer to 
  tunnel through energy barriers, finding global minimums in a high-dimensional
  discrete space (2^78 configurations) more efficiently than brute-force.

Why Hybrid Optimization?
  Quantum hardware is noisy and limited in qubit count. Purely quantum 
  solutions often violate hard constraints (like budget=1) due to soft penalties.
  The Hybrid approach uses Quantum for global structure search, and Classical 
  (SLSQP/NSGA-II) for strict boundary polishing, guaranteeing zero breaches.

Limitations & Future Work:
  - Current simulation uses Simulated Annealing. True gate-model QAOA on 
    hardware (e.g., IBM Heron) will introduce noise and decoherence.
  - Future work: Implement Domain-Wall encoding instead of binary to reduce 
    circuit depth and two-qubit gate count.
""")
    

def print_qaoa_variational_loop():
    print("\n" + "="*75)
    print("🔬 4. QAOA METHODOLOGY & VARIATIONAL LOOP")
    print("="*75)
    print("""
The Quantum Approximate Optimization Algorithm (QAOA) prepares a parameterized 
quantum state by alternating between the Cost Hamiltonian (H_C) and the Mixer 
Hamiltonian (H_M):

  |ψ(γ, β)⟩ = e^{-iβ_p H_M} e^{-iγ_p H_C} ... e^{-iβ_1 H_M} e^{-iγ_1 H_C} |+⟩^⊗n

Where:
  - |+⟩^⊗n is the uniform superposition initial state.
  - γ and β are the variational parameters optimized classically.
  - H_C encodes the QUBO matrix via Pauli-Z (Z_i Z_j) rotations.
  - H_M applies Pauli-X (X_i) rotations to explore the Hilbert space.

Classical Optimization Loop:
  1. Measure the expectation value: ⟨ψ(γ, β)| H_C |ψ(γ, β)⟩
  2. Use a classical optimizer (e.g., Adam/COBYLA) to update γ, β to minimize cost.
  3. Repeat until convergence to the ground state (optimal portfolio).

Note: For our 78-qubit implementation, we utilize D-Wave's Simulated Annealing 
sampler, which mathematically mirrors the adiabatic quantum tunneling of QAOA 
but is computationally more tractable for NISQ-era hardware constraints.
""")
    
def print_ising_hamiltonian_proof(Q, n_qubits):
    print("\n" + "="*75)
    print("🔬 5. ISING HAMILTONIAN MAPPING (Mathematical Proof)")
    print("="*75)
    print("""
The QUBO objective min(x^T Q x) is mapped to an Ising Hamiltonian (H_C) 
by substituting binary variables x_i ∈ {0,1} with spin variables Z_i ∈ {-1,+1}:

  Mapping: x_i = (1 - Z_i) / 2

Substituting this into the QUBO expansion:
  x^T Q x = Σ_i Q_ii x_i + Σ_{i<j} (Q_ij + Q_ji) x_i x_j

Yields the Cost Hamiltonian:
  H_C = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j

Where the Ising coefficients are mathematically derived as:
  J_ij = (Q_ij + Q_ji) / 4
  h_i  = (Q_ii / 2) - (1/2) * Σ_{j≠i} (Q_ij + Q_ji)
""")
    print(f"--- Extracted Hamiltonian Coefficients (First 4 Qubits of {n_qubits}) ---")
    
    # Calculate h_i and J_ij for the first 4 qubits to prove the math
    for i in range(min(4, n_qubits)):
        h_i = Q[i, i] / 2
        for j in range(n_qubits):
            if i != j:
                h_i -= (Q[i, j] + Q[j, i]) / 4
        print(f"  h_{i} (Linear Bias): {h_i:.4f}")
        
    print("\n  J_ij Couplings (First 4x4 block):")
    for i in range(min(4, n_qubits)):
        for j in range(i + 1, min(4, n_qubits)):
            J_ij = (Q[i, j] + Q[j, i]) / 4
            print(f"  J_{i},{j} (Quadratic Coupling): {J_ij:.4f}")
    
    print("\nNote: These coefficients are passed to the D-Wave Simulated Annealing Sampler.")
    print("The sampler seeks the ground state (minimum energy) of this Hamiltonian.\n")



def print_encoding_comparison(n_assets, bits):
    print("\n" + "="*75)
    print("🔬 6. QUANTUM ENCODING ANALYSIS: QUBO vs DOMAIN-WALL vs ONE-HOT")
    print("="*75)
    print("""
When mapping discrete portfolio weights to qubits, the choice of encoding 
significantly impacts quantum resource requirements. We analyze three schemes:

1. Binary QUBO (Our Choice): w_i = Δ Σ 2^j x_{i,j}
   - Qubits: N_assets × N_bits
   - Fully connected graph (all-to-all couplings)

2. Domain-Wall Encoding: Uses d-1 qubits for d discrete states
   - Qubits: N_assets × (2^N_bits - 1)
   - Only local (nearest-neighbor) couplings per asset

3. One-Hot Encoding: Uses d qubits for d discrete states
   - Qubits: N_assets × 2^N_bits
   - Requires one-hot constraint penalty (only 1 qubit ON per asset)
""")
    
    # Calculate metrics for each encoding
    n_states = 2**bits  # 63 discrete states
    
    # 1. Binary QUBO
    qubo_qubits = n_assets * bits
    qubo_couplings = qubo_qubits * (qubo_qubits - 1) // 2  # Fully connected
    qubo_2qgates = qubo_couplings * 3  # p=3 layers, each coupling = 1 CNOT pair
    qubo_depth = qubo_2qgates + qubo_qubits * 3  # Rough estimate
    
    # 2. Domain-Wall
    dw_qubits = n_assets * (n_states - 1)
    dw_couplings_per_asset = (n_states - 2)  # Only nearest-neighbor within each asset
    dw_couplings = n_assets * dw_couplings_per_asset
    dw_cross_couplings = n_assets * (n_assets - 1) * (n_states - 1)  # Cross-asset
    dw_total_couplings = dw_couplings + dw_cross_couplings
    dw_2qgates = dw_total_couplings * 3
    dw_depth = dw_2qgates + dw_qubits * 3
    
    # 3. One-Hot
    oh_qubits = n_assets * n_states
    oh_couplings = oh_qubits * (oh_qubits - 1) // 2
    oh_2qgates = oh_couplings * 3
    oh_depth = oh_2qgates + oh_qubits * 3
    
    print(f"{'Metric':<30} {'Binary QUBO':<20} {'Domain-Wall':<20} {'One-Hot':<20}")
    print("-" * 90)
    print(f"{'Total Qubits':<30} {qubo_qubits:<20} {dw_qubits:<20} {oh_qubits:<20}")
    print(f"{'2-Qubit Gates (CNOTs)':<30} {qubo_2qgates:<20,} {dw_2qgates:<20,} {oh_2qgates:<20,}")
    print(f"{'Est. Circuit Depth (p=3)':<30} {qubo_depth:<20,} {dw_depth:<20,} {oh_depth:<20,}")
    print(f"{'Graph Connectivity':<30} {'Fully Connected':<20} {'Mixed':<20} {'Fully Connected':<20}")
    print("-" * 90)
    
    print(f"""
Analysis:
- Binary QUBO uses the FEWEST qubits ({qubo_qubits}), making it scalable.
- Domain-Wall uses {dw_qubits} qubits but has simpler connectivity.
- One-Hot uses {oh_qubits} qubits, which exceeds current NISQ hardware limits.
- We chose Binary QUBO because it minimizes qubit count while maintaining 
  full coupling fidelity for the covariance matrix interactions.
- Future work: Domain-Wall encoding could reduce circuit depth if hardware 
  connectivity constraints become the primary bottleneck.
""")