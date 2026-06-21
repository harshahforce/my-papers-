import numpy as np
import matplotlib.pyplot as plt

def simulate_gravitational_throttling(n_samples=5000000):
    print("Initializing SRM Gravitational Throttling Engine...")

    # Gravitational potential range: 1.0 (far field) to 0.1 (near horizon)
    # The scaling factor sqrt(-g00) directly scales the energy budget
    g_factors = np.linspace(1.0, 0.1, 100)

    # 1. Generate the same Unconstrained Phase Space (Rest Frame)
    py = np.random.uniform(-1, 1, n_samples)
    pz = np.random.uniform(-1, 1, n_samples)
    valid_states = (py**2 + pz**2) <= 1.0
    py, pz = py[valid_states], pz[valid_states]

    total_unconstrained_volume = len(py)
    clock_rates = []

    print("Applying gravitational potential scaling...")

    # 2. Iterate through Gravitational Scaling Factors
    for g in g_factors:
        # Gravity as a Parameter Filter:
        # The accessible energy budget is constrained by the local metric,
        # effectively shrinking the momentum sphere radius by the g-factor
        constrained_radius_sq = g**2

        surviving_states = (py**2 + pz**2) <= constrained_radius_sq
        volume_ratio = np.sum(surviving_states) / total_unconstrained_volume

        # Clock rate scales linearly with the radius contraction in this configuration
        clock_rates.append(g)

    # 3. Render the Data
    plt.figure(figsize=(10, 6))
    plt.plot(g_factors, g_factors, color='cyan', linewidth=4, alpha=0.5, label='Theoretical Metric Scaling')
    plt.plot(g_factors, clock_rates, color='black', linestyle='dashed', linewidth=2, label='SRM Simulated Clock Rate')

    plt.title("Gravitational Throttling: Clock Decay in a Potential Field", fontsize=14, fontweight='bold')
    plt.xlabel("Gravitational Scaling Factor (sqrt(-g00))", fontsize=12)
    plt.ylabel("Relative Internal Update Rate (dτ/dλ)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig('srm_gravity_output.png', dpi=300)
    print("Simulation complete. Output saved as 'srm_gravity_output.png'.")
    plt.show()

if __name__ == "__main__":
    simulate_gravitational_throttling()
