import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

def main():
    print("Initializing SRM System Stability Engine...")
    print("Tracking data shedding across 1,000,000 localized lattice nodes...")

    # System Parameters
    N = 1_000_000         # Total information nodes
    ticks = 10_000        # Execution loop duration
    B_max = 100           # Global Processing Limit (Ceiling)
    I_base = 200          # Initial node load (set above capacity to force saturation)
    noise_std = 5.0       # Gaussian processing variance
    P_drop = 0.005        # Baseline packet shedding probability
    drop_amount = 10      # Data volume cleared per shed event

    # Initialize States
    I_total = np.full(N, I_base, dtype=np.int32)
    has_dropped = np.zeros(N, dtype=bool) 
    undecayed_counts = np.zeros(ticks, dtype=np.int32)
    
    rng = np.random.default_rng()
    start_time = time.time()
    
    # The Execution Loop
    for t in range(ticks):
        # Processing Noise: Generate fluctuations in local operational bandwidth
        B_local = B_max + rng.standard_normal(N, dtype=np.float32) * noise_std
        
        # The Throttling Check: Detect localized grid saturation/buffer overflow
        overloaded = I_total > B_local
        num_overloaded = np.sum(overloaded)
        
        if num_overloaded > 0:
            # Automated Garbage Collection: Shed data blocks to prevent grid corruption
            drop_mask_overloaded = rng.random(num_overloaded, dtype=np.float32) < P_drop
            
            overloaded_indices = np.nonzero(overloaded)[0]
            actual_drop_indices = overloaded_indices[drop_mask_overloaded]
            
            # Execute data clearance routine
            I_total[actual_drop_indices] -= drop_amount
            has_dropped[actual_drop_indices] = True
            
        # Log the surviving structural integrity of the lattice
        undecayed_counts[t] = N - np.sum(has_dropped)
        
        if (t + 1) % 1000 == 0:
            print(f"Tick {t+1}/{ticks} completed. Intact Nodes: {undecayed_counts[t]} | Elapsed: {time.time() - start_time:.2f}s")

    # Analytical Verification Curve Fitting
    x_data = np.arange(ticks)
    y_data = undecayed_counts

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
        
    try:
        p0 = [N, 0.001, 0]
        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=p0)
        fit_y = exp_decay(x_data, *popt)
        fit_label = f"SRM Decay Fit: {popt[0]:.2e} * e^(-{popt[1]:.2e} * t)"
    except Exception as e:
        print(f"Curve fitting matrix uncomputable: {e}")
        coefs = np.polyfit(x_data, y_data, 2)
        poly = np.poly1d(coefs)
        fit_y = poly(x_data)
        fit_label = "Lattice Anisotropy Poly Fit"

    # Render Stability Wavefronts
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, color='black', label="Observed Lattice Ticks", alpha=0.8)
    plt.plot(x_data, fit_y, color='cyan', linestyle='dashed', label=fit_label, linewidth=2)
    
    plt.title("System Stability: Information Decay Waves Under Spatial Saturation", fontsize=14, fontweight='bold')
    plt.xlabel("Lattice Ticks", fontsize=12)
    plt.ylabel("Undecayed Information Nodes", fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('srm_stability_output.png', dpi=300)
    print("Lattice metrics exported safely as 'srm_stability_output.png'.")
    plt.show()

if __name__ == "__main__":
    main()
