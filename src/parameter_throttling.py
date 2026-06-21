import numpy as np
import matplotlib.pyplot as plt

def simulate_parameter_throttling(n_samples=5000000):
    print(f"Initializing Static Reality Model Simulation...")
    print(f"Generating unconstrained continuous phase space with {n_samples} microstates...")

    # Universal Constants
    c = 1.0
    v_steps = np.linspace(0, 0.999 * c, 100)

    # 1. Generate the Unconstrained Phase Space
    # Simulating a 2D internal momentum cross-section (py, pz) representing the maximum energy budget.
    py = np.random.uniform(-1, 1, n_samples)
    pz = np.random.uniform(-1, 1, n_samples)

    # The Sieve: Filter to create a perfect hypersphere of radius 1 (E_total = 1)
    valid_states = (py**2 + pz**2) <= 1.0
    py = py[valid_states]
    pz = pz[valid_states]

    total_unconstrained_volume = len(py)
    simulation_curve = []

    print("Applying external kinematic constraints (v -> c)...")

    # 2. Iterate through Macroscopic Velocities
    for v in v_steps:
        # Parameter Throttling: Velocity consumes total energy capacity
        remaining_radius_sq = 1.0 - (v/c)**2

        # Count how many microstates survive the constrained boundary
        surviving_states = (py**2 + pz**2) <= remaining_radius_sq
        surviving_volume = np.sum(surviving_states)

        # The ratio of surviving states is the relative phase-space volume
        volume_ratio = surviving_volume / total_unconstrained_volume

        # Emergent clock rate scales with the square root of the 2D volume ratio
        emergent_clock_rate = np.sqrt(volume_ratio)
        simulation_curve.append(emergent_clock_rate)

    # 3. The Theoretical Benchmark (Lorentz Factor Inverse)
    theoretical_curve = np.sqrt(1 - (v_steps/c)**2)

    # 4. Render the Data
    print("Rendering output...")
    plt.figure(figsize=(10, 6))
    plt.plot(v_steps, theoretical_curve, color='cyan', linewidth=4, alpha=0.5, label='Theoretical (Lorentz)')
    plt.plot(v_steps, simulation_curve, color='black', linestyle='dashed', linewidth=2, label='SRM Simulated Clock Rate')

    plt.title("Parameter Throttling: Emergent Time Dilation via Phase-Space Contraction", fontsize=14, fontweight='bold')
    plt.xlabel("Macroscopic Velocity (v/c)", fontsize=12)
    plt.ylabel("Relative Internal Update Rate (dτ/dλ)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig('srm_throttling_output.png', dpi=300)
    print("Simulation complete. Output saved as 'srm_throttling_output.png'.")
    plt.show()

if __name__ == "__main__":
    simulate_parameter_throttling()
