import numpy as np
import matplotlib.pyplot as plt

def simulate_kinetic_engine(N=2000, steps=500):
    print("Initializing SRM Kinetic Engine...")
    print(f"Simulating continuous boundary interactions for {N} localized nodes...")

    # Universal Limits and Parameters
    c = 1.0
    v_macro_steps = np.linspace(0, 0.99 * c, 50)
    box_size = 10.0
    dt = 0.1

    simulated_clock_rates = []

    # Run the simulation across increasing macroscopic velocities
    for v_macro in v_macro_steps:
        # The Throttling Mechanism:
        # Energy budget is fixed. Macroscopic velocity drains internal thermal capacity.
        v_thermal = np.sqrt(c**2 - v_macro**2)

        # Initialize random continuous positions within the observable macro-state (box)
        positions = np.random.uniform(0, box_size, (N, 2))

        # Initialize internal velocity vectors distributed uniformly in all directions
        angles = np.random.uniform(0, 2 * np.pi, N)
        velocities = np.column_stack((np.cos(angles), np.sin(angles))) * v_thermal

        total_state_transitions = 0

        # Execute the internal clock engine for a fixed external duration
        for _ in range(steps):
            # Advance the deterministic continuous positions
            positions += velocities * dt

            # The Sieve: Detect boundary collisions (macro-state updates)
            hit_x = (positions[:, 0] <= 0) | (positions[:, 0] >= box_size)
            hit_y = (positions[:, 1] <= 0) | (positions[:, 1] >= box_size)

            # Count the total number of emergent "ticks"
            total_state_transitions += np.sum(hit_x) + np.sum(hit_y)

            # Apply elastic reflections to keep the system bounded
            velocities[hit_x, 0] *= -1
            velocities[hit_y, 1] *= -1

            # Correct positions to prevent algorithmic escape
            positions[:, 0] = np.clip(positions[:, 0], 0, box_size)
            positions[:, 1] = np.clip(positions[:, 1], 0, box_size)

        simulated_clock_rates.append(total_state_transitions)

    print("Rendering mechanical output...")

    # Normalize the observed ticks against the rest-frame baseline (v=0)
    simulated_clock_rates = np.array(simulated_clock_rates)
    relative_clock_rates = simulated_clock_rates / simulated_clock_rates[0]

    # The Theoretical Benchmark
    theoretical_curve = np.sqrt(1 - (v_macro_steps / c)**2)

    # Render the Data
    plt.figure(figsize=(10, 6))
    plt.plot(v_macro_steps, theoretical_curve, color='cyan', linewidth=4, alpha=0.5, label='Theoretical (Lorentz)')
    plt.plot(v_macro_steps, relative_clock_rates, color='black', linestyle='dashed', linewidth=2, label='SRM Simulated Tick Rate')

    plt.title("The Kinetic Engine: Operational Time Dilation via Resource Depletion", fontsize=14, fontweight='bold')
    plt.xlabel("Macroscopic Velocity (v/c)", fontsize=12)
    plt.ylabel("Relative Internal Update Rate (dτ/dλ)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig('srm_kinetic_output.png', dpi=300)
    print("Simulation complete. Output saved as 'srm_kinetic_output.png'.")
    plt.show()

if __name__ == "__main__":
    simulate_kinetic_engine()
