import numpy as np

def analyze_quantum_resources(Q, n_qubits):
    """
    Analyzes the QUBO matrix to extract Quantum Resource metrics 
    and theoretical Ising/QAOA mapping.
    """
    print("\n🔬 Quantum Resource & Hamiltonian Analysis:")
    print("-" * 50)
    
    # 1. QUBO to Ising Mapping (Theoretical)
    # Linear terms (h_i) and Quadratic terms (J_ij)
    h = np.zeros(n_qubits)
    J = np.zeros((n_qubits, n_qubits))
    
    for i in range(n_qubits):
        h[i] = Q[i, i] / 2
        for j in range(i+1, n_qubits):
            J[i, j] = Q[i, j] / 4
            h[i] += Q[i, j] / 4
            h[j] += Q[i, j] / 4
            
    # 2. Graph Density (Coupling connectivity)
    num_couplings = np.count_nonzero(J)
    max_couplings = n_qubits * (n_qubits - 1) / 2
    density = num_couplings / max_couplings if max_couplings > 0 else 0
    
    # 3. QAOA Circuit Depth Estimation
    # p=3 layers. Each layer requires 2-qubit gates for every coupling.
    p_layers = 3
    two_qubit_gates_per_layer = num_couplings
    total_two_qubit_gates = p_layers * two_qubit_gates_per_layer
    
    print(f"Total Qubits (Variables): {n_qubits}")
    print(f"Ising Linear Biases (h_i): {np.count_nonzero(h)} non-zero")
    print(f"Ising Quadratic Couplings (J_ij): {num_couplings} non-zero")
    print(f"Graph Density: {density*100:.2f}%")
    print(f"Theoretical QAOA Layers (p): {p_layers}")
    print(f"Theoretical 2-Qubit Gates (CNOTs): {total_two_qubit_gates}")
    print(f"Estimated Circuit Depth: ~{total_two_qubit_gates + n_qubits*p_layers}")
    print("-" * 50)
    print("Cost Hamiltonian: H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j")
    print("Mixer Hamiltonian: H_M = Σ X_i")
    print("-" * 50)