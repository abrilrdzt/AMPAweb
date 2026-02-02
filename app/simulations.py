#Sarah Mory 261119392
#Abril Rodriguez 261175850
#Ekin Celtikcioglu


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from numba import njit, prange
from joblib import Parallel, delayed, parallel_backend
from scipy.linalg import expm
import time as time_module
from collections import OrderedDict


# ─────────────────────────────────────────────────────────────────────────────
#  1) CORE SIMULATION FUNCTIONS 
# ─────────────────────────────────────────────────────────────────────────────

# Indices for open states 
IDX_O2, IDX_O3, IDX_O4 = 5, 6, 7

# Default values for all 28 transition rates + global parameters
default_params = {
    # R‑chain binding/unbinding
    'k_R0_R1': 4 * 2e7,   'k_R1_R0': 9000.0,
    'k_R1_R2': 3 * 2e7,   'k_R2_R1': 2 * 9000.0,
    'k_R2_R3': 2 * 2e7,   'k_R3_R2': 3 * 9000.0,
    'k_R3_R4': 1 * 2e7,   'k_R4_R3': 4 * 9000.0,

    # D‑chain (desensitized) binding/unbinding
    'k_D1_D2': 3 * 2e7,   'k_D2_D1': 9000.0,
    'k_D2_D3': 2 * 2e7,   'k_D3_D2': 2 * 9000.0,
    'k_D3_D4': 1 * 2e7,   'k_D4_D3': 3 * 9000.0,

    # Gating transitions
    'k_R2_O2': 2 * 8000.0, 'k_O2_R2': 3100.0,
    'k_R3_O3': 3 * 8000.0, 'k_O3_R3': 3100.0,
    'k_R4_O4': 4 * 8000.0, 'k_O4_R4': 3100.0,

    # Desensitization transitions
    'k_R1_D1': 1800.0,      'k_D1_R1': 7.6,
    'k_R2_D2': 2 * 1800.0,  'k_D2_R2': 7.6,
    'k_R3_D3': 3 * 1800.0,  'k_D3_R3': 7.6,
    'k_R4_D4': 4 * 1800.0,  'k_D4_R4': 7.6,

    # Global simulation parameters
    '∆t':               2e-6,
    'Agonist Concentration':   0.01,    # M
    'Pulse Duration':   0.001,   # s
    'Conductance O2':              10e-12,  # S
    'Conductance O3':              15e-12,
    'Conductance O4':              20e-12,

    # DR‐specific
    'Maximum time':             0.25,    # total DR simulation time (s)
    'Agonist starting time':        0.005,   # (s)
    'Agonist ending time':          0.245,   # (s)
    'Agonist concentrations':            "1,10,25,50,100,200,250,500,750,1000,2000,5000,10000",

    # LPPR‐specific
    'Starting time (baseline pulse) Long Paired Pulse':             -0.2,    # (s)
    'Buffer time':      0.01,
    'Pulse intervals (long)': "5,10,15,20,25,30,35,40,45,50,55,60,65,70,90,110,130,150,170,190,210,230,250,270,300,350,400,450,500,550,600,650,700,750,800,850,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,2000",

    # SPPR‐specific
    'Starting time (baseline pulse) Short Paired Pulse':        -0.0011, # (s)
    'Pulse intervals (short)':  ",".join(str(int(x)) for x in np.unique(np.round(np.logspace(np.log10(2), np.log10(1000), 70)).astype(int))[:50]),

    # SSI‐specific
    'Baseline duration': 2.0,    # (s)
    'Baseline agonist concentrations': "1,10,50,100,200,600,1000,1600,2000,2500,5000,10000,12500",
    
    #Train-specific
    'Train frequencies (Hz)': "10,25,50,100,200",
    'Number of pulses': 20,
    'Pre time': 0.01,
    'Post time': 0.05,
    'Pulse width': 0.001
}

def make_rates(params):
    return (
      params['k_R0_R1'], params['k_R1_R0'],
      params['k_R1_R2'], params['k_R2_R1'],
      params['k_R2_R3'], params['k_R3_R2'],
      params['k_R3_R4'], params['k_R4_R3'],

      params['k_D1_D2'], params['k_D2_D1'],
      params['k_D2_D3'], params['k_D3_D2'],
      params['k_D3_D4'], params['k_D4_D3'],

      params['k_R2_O2'], params['k_O2_R2'],
      params['k_R3_O3'], params['k_O3_R3'],
      params['k_R4_O4'], params['k_O4_R4'],

      params['k_R1_D1'], params['k_D1_R1'],
      params['k_R2_D2'], params['k_D2_R2'],
      params['k_R3_D3'], params['k_D3_R3'],
      params['k_R4_D4'], params['k_D4_R4'],
    )

@njit
def build_Q_numba(glu, rates):
    
    Q = np.zeros((12, 12), dtype=np.float64)

    # Unpack “effective” ligand‐binding rates
    R01_eff = rates[0] * glu
    R12_eff = rates[2] * glu
    R23_eff = rates[4] * glu
    R34_eff = rates[6] * glu

    D12_eff = rates[8]  * glu
    D23_eff = rates[10] * glu
    D34_eff = rates[12] * glu

    # R0 ↔ R1
    Q[0, 1], Q[1, 0] = 4 * R01_eff, rates[1]
    # R1 ↔ R2
    Q[1, 2], Q[2, 1] = 3 * R12_eff, 2 * rates[3]
    # R2 ↔ R3
    Q[2, 3], Q[3, 2] = 2 * R23_eff, 3 * rates[5]
    # R3 ↔ R4
    Q[3, 4], Q[4, 3] = R34_eff, 4 * rates[7]

    # Desensitized chain: D1 ↔ D2, D2 ↔ D3, D3 ↔ D4
    Q[8,  9],  Q[9,  8]  = 3 * D12_eff, rates[9]
    Q[9,  10], Q[10, 9]  = 2 * D23_eff, 2 * rates[11]
    Q[10, 11], Q[11, 10] = D34_eff,    3 * rates[13]

    # Gating: R2 ↔ O2, R3 ↔ O3, R4 ↔ O4
    Q[2,  5],  Q[5,  2]  = 2 * rates[14], rates[15]
    Q[3,  6],  Q[6,  3]  = 3 * rates[16], rates[17]
    Q[4,  7],  Q[7,  4]  = 4 * rates[18], rates[19]

    # Desensitization: R1 ↔ D1, R2 ↔ D2, R3 ↔ D3, R4 ↔ D4
    Q[1,  8],  Q[8,  1]  = rates[20], rates[21]
    Q[2,  9],  Q[9,  2]  = 2 * rates[22], rates[23]
    Q[3,  10], Q[10, 3]  = 3 * rates[24], rates[25]
    Q[4,  11], Q[11, 4]  = 4 * rates[26], rates[27]

    # Fill diagonals so each row sums to zero
    for i in range(12):
        row_sum = 0.0
        for j in range(12):
            if i != j:
                row_sum += Q[i, j]
        Q[i, i] = -row_sum

    return Q

# (Run run_dr, run_ppr_sppr, run_ssi functions)

def precompute_transition_matrices_DR(gluts, params):
    rates = make_rates(params)
    dt = params['∆t']

    Q_off = build_Q_numba(0.0, rates)
    M_off = expm(Q_off * dt)

    M_gluts = np.zeros((len(gluts), 12, 12), dtype=np.float64)
    for i, g in enumerate(gluts):
        Qg = build_Q_numba(g, rates)
        M_gluts[i] = expm(Qg * dt)

    return M_gluts, M_off

@njit
def simulate_glut_DR(glu_id, M_glut, M_off, is_on, nT):
    P = np.zeros((nT, 12), dtype=np.float64)
    P[0, 0] = 1.0
    for t in range(nT - 1):
        M = M_glut if is_on[t] else M_off
        P_next = np.zeros(12, dtype=np.float64)
        for i in range(12):
            for j in range(12):
                P_next[i] += P[t, j] * M[j, i]
        # Clip and normalize
        s = 0.0
        for i in range(12):
            v = max(P_next[i], 0.0)
            P[t + 1, i] = v
            s += v
        if s > 0.0:
            P[t + 1, :] /= s
    return P

def run_dr(params):
    t0 = time_module.perf_counter()
    tmax      = float(params['Maximum time'])
    glu_start = float(params['Agonist starting time'])
    glu_end   = float(params['Agonist ending time'])
    dt        = float(params['∆t'])
    glu_str   = params['Agonist concentrations']

    # Parse gluts string (in µM), convert to M
    try:
        arr = [float(x) for x in glu_str.split(',') if x.strip() != ""]
        gluts = np.array(arr, dtype=np.float64) / 1e6
        print(f"Parsed Agonist concentrations: {arr} µM")
    except Exception as e:
        # Fall back to defaults if parse fails
        print(f"Error parsing Agonist concentrations ('{glu_str}'): {e}, using default")
        gluts = np.array([1,10,25,50,100,200,250,500,750,1000,2000,5000,10000], dtype=np.float64) / 1e6

    total_time = np.arange(0, tmax + dt, dt)
    is_on = (total_time >= glu_start) & (total_time <= glu_end)
    nT = len(total_time)

    print("⏳ Precomputing transition matrices for Dose Response…")
    M_gluts, M_off = precompute_transition_matrices_DR(gluts, params)
    
    P0 = simulate_glut_DR(0, M_gluts[0], M_off, is_on, nT) #fixes segmentation fault
    rest_indices = list(range(1, len(gluts)))

    print("⏳ Running Dose Response sims in parallel…")
    with parallel_backend('threading', n_jobs=4):
        restP = Parallel()(
            delayed(simulate_glut_DR)(i, M_gluts[i], M_off, is_on, nT)
            for i in rest_indices
        )
    
    allP = [P0] + restP
    t1 = time_module.perf_counter()
    print(f"✅ Completed {len(gluts)} Dose Response runs in {(t1 - t0):.3f}s")

    Pall = np.array(allP)
    open_traces = Pall[:, :, IDX_O2] + Pall[:, :, IDX_O3] + Pall[:, :, IDX_O4]
    max_probs = open_traces.max(axis=1)

    # Save CSV
    from flask import current_app
    import os
    static_path = os.path.join(current_app.static_folder, 'plot_data_dose_response.csv')
    with open(static_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Glutamate (M)", "Max P(open)"])
        for g, m in zip(gluts, max_probs):
            w.writerow([g, m])

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(gluts)))
    for trace, g, c in zip(open_traces, gluts, colors):
        ax1.plot(total_time, trace, color=c, label=f"{g*1e6:.0f} µM")
    ax1.axvspan(glu_start, glu_end, color='lightgray', alpha=0.3)
    ax1.set(title="Dose–Response: P(open) vs Time",xlabel="Time (s)", ylabel="P(open)")
    ax1.legend(loc='upper right', fontsize='x-small')
    ax1.grid(True)

    ax2.semilogx(gluts * 1e6, max_probs, 'o-', color='C1')
    ax2.set(title="Max P(open) vs [Glu]", xlabel="[Glu] (µM)", ylabel="Max P(open)")
    ax2.grid(True)
    
    param_text_lines = ["Simulation Parameters:"]
    for key, value in params.items():
        if not key.startswith('k_') and key not in ['Agonist concentrations']:
            param_text_lines.append(f"{key}: {value}")
    param_text = "\n".join(param_text_lines)
    plt.gcf().text(0.97, 0.110, param_text, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    static_path = os.path.join(current_app.static_folder, 'dose_response.png')
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("▶️ Saved Dose Response plot.")
    return "Dose Response simulation complete."

# ─────────────────────────────────────────────────────────────────────────────

def precompute_transition_matrices_PPR(params):
    rates = make_rates(params)
    Q_off = build_Q_numba(0.0, rates)
    Q_on  = build_Q_numba(params['Agonist Concentration'], rates)
    M_off = expm(Q_off * params['∆t'])
    M_on  = expm(Q_on  * params['∆t'])
    return M_on, M_off

@njit
def simulate_segment_PPR(initial_state, time_points, glut_profile, M_on, M_off):
    nT = len(time_points)
    P = np.zeros((nT, 12), dtype=np.float64)
    P[0] = initial_state.copy()
    for t in prange(nT - 1):
        M = M_on if glut_profile[t] > 0 else M_off
        P[t + 1] = P[t] @ M
        # Clip & normalize
        s = 0.0
        for k in range(12):
            v = max(P[t + 1, k], 0.0)
            P[t + 1, k] = v
            s += v
        if s > 0.0:
            P[t + 1] /= s
        else:
            P[t + 1] = 0.0
    return P

def run_ppr_sppr(params, is_sppr=False):
    total_start = time_module.perf_counter()
    tmin = params['Starting time (baseline pulse) Short Paired Pulse'] if is_sppr else params['Starting time (baseline pulse) Long Paired Pulse']
    buffer_time = params['Buffer time']
    
    if is_sppr:
        s = params['Pulse intervals (short)']
        try:
            arr = [float(x) for x in s.split(',') if x.strip() != ""]
            pulse_intervals = np.array(arr, dtype=np.float64) / 1000.0
            print(f"Parsed Pulse intervals (short): {arr} ms")
        except Exception as e:
            print(f"Error parsing Pulse intervals (short) ('{s}'): {str(e)}, using default")
            pulse_intervals = np.unique(np.round(np.logspace(np.log10(2), np.log10(1000), 70)).astype(int))[:50] / 1000.0
    else:
        s = params['Pulse intervals (long)']
        try:
            arr = [float(x) for x in s.split(',') if x.strip() != ""]
            pulse_intervals = np.array(arr, dtype=np.float64) / 1000.0
        except:
            pulse_intervals = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,90,110,130,150,170,190,210,230,250,270,300,350,400,450,500,550,600,650,700,750,800,850,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,2000]) / 1000.0

    baseline_steps = int(round((0 - tmin) / params['∆t'])) + 1
    time_base = np.linspace(tmin, 0, baseline_steps)
    glut_base = np.where(time_base >= tmin, params['Agonist Concentration'], 0)

    initial_state = np.zeros(12)
    initial_state[0] = 1.0
    print("⏳ Precomputing transition matrices for PPR/SPPR…")
    M_on, M_off = precompute_transition_matrices_PPR(params)
    
    baseline_probs = simulate_segment_PPR(initial_state, time_base, glut_base, M_on, M_off)

    G_base = baseline_probs[:, IDX_O2] * params['Conductance O2'] + baseline_probs[:, IDX_O3] * params['Conductance O3'] + baseline_probs[:, IDX_O4] * params['Conductance O4']
    G_max = np.max(G_base)

    max_interval = pulse_intervals[-1]
    recovery_steps = int(round(max_interval / params['∆t'])) + 1
    time_recovery = np.linspace(0, max_interval, recovery_steps)
    glut_recovery = np.zeros_like(time_recovery)

    recovery_probs = simulate_segment_PPR(baseline_probs[-1], time_recovery, glut_recovery, M_on, M_off)
    print("⏳ Simulating second pulses…")
    sim_start = time_module.perf_counter()
    
    def process_interval(interval):
        recovery_idx = int(round(interval / params['∆t']))
        initial_state = recovery_probs[recovery_idx]
        
        pulse_steps = int(round((params['Pulse Duration'] + buffer_time) / params['∆t'])) + 1
        time_pulse = np.linspace(0, params['Pulse Duration'] + buffer_time, pulse_steps)
        glut_pulse = np.where(time_pulse <= params['Pulse Duration'], params['Agonist Concentration'], 0)
        
        pulse_probs = simulate_segment_PPR(initial_state, time_pulse, glut_pulse, M_on, M_off)
        
        G_pulse = pulse_probs[:, IDX_O2] * params['Conductance O2'] + pulse_probs[:, IDX_O3] * params['Conductance O3'] + pulse_probs[:, IDX_O4] * params['Conductance O4']  
        G_pulse_max = np.max(G_pulse[:int(round(params['Pulse Duration'] / params['∆t']))])
        
        full_time = np.concatenate((time_base, time_recovery[:recovery_idx] + 0, time_pulse + interval))
        full_probs = np.vstack((
            baseline_probs,
            recovery_probs[1:recovery_idx + 1],
            pulse_probs[1:]
        ))
        open_probs = full_probs[:, IDX_O2] + full_probs[:, IDX_O3] + full_probs[:, IDX_O4]
        
        min_length = min(len(full_time), len(open_probs))
        return full_time[:min_length], open_probs[:min_length], G_pulse_max / G_max
    
    with parallel_backend('threading', n_jobs=4):  # Limit to 4 jobs
        results = Parallel()(delayed(process_interval)(interval) for interval in pulse_intervals)
    
    sim_time = time_module.perf_counter() - sim_start
    print(f"✅ Second pulses simulated in {sim_time:.3f} seconds")
    
    # Unpack results
    all_time = [r[0] for r in results]
    all_open_probs = [r[1] for r in results]
    conductances = [r[2] for r in results]

    # Plotting
    plt.figure(figsize=(12, 12))
    
    # First subplot: Recovery from desensitization
    plt.subplot(2, 1, 1)
    colors = plt.cm.plasma(np.linspace(0, 1, len(pulse_intervals)))
    for t, op, interval, c in zip(all_time, all_open_probs, pulse_intervals, colors):
        plt.plot(t, op, color=c, label=f"Interval {interval*1000:.0f} ms", linewidth=1.5)
    plt.title("Recovery from Desensitization - Paired Pulse")
    plt.xlabel("Time (s)")
    plt.ylabel("Open State Probability")
    plt.grid(True)
     #this is to not show repeated interval values on the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
            
    plt.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize='small'
    )
    # Second subplot: Conductance ratio
    plt.subplot(2, 1, 2)
    plt.plot(
        [interval * 1000 for interval in pulse_intervals], 
        conductances, 
        marker='o', 
        linestyle='-', 
        label='Conductance Ratio (G_pulse / G_max)',
        color = 'purple'
    )
    plt.xscale('log')
    plt.title("Conductance Ratio vs Paired Pulse Interval")
    plt.xlabel("Interval (ms, log scale)")
    plt.ylabel("Conductance Ratio (G_pulse / G_max)")
    plt.grid(True)
    
    param_text_lines = ["Simulation Parameters:"]
    for key, value in params.items():
        if not key.startswith('k_') and key not in ['Pulse intervals (short)', 'Pulse intervals (long)']:
            param_text_lines.append(f"{key}: {value}")
    param_text = "\n".join(param_text_lines)
    plt.gcf().text(0.82, 0.08, param_text, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    from flask import current_app
    import os

    static_path = os.path.join(current_app.static_folder, 'ppr_conductance_ratio_vs_interval.csv' if not is_sppr else 'sppr_conductance_ratio_vs_interval.csv')
    with open(static_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Paired Pulse Interval (ms)", "Conductance Ratio (G_pulse / G_max)"])
        for interval, conductance in zip(pulse_intervals, conductances):
            writer.writerow([interval * 1000, conductance])
    static_path = os.path.join(current_app.static_folder, 'ppr_sppr.png' if not is_sppr else 'sppr_sppr.png')
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    total_time = time_module.perf_counter() - total_start
    print(f"Total runtime: {total_time:.3f} seconds")
    print("▶️ Saved PPR/SPPR plot.")
    return f"{'Short' if is_sppr else 'Long'} Paired-Pulse Recovery simulation complete."
   

# ─────────────────────────────────────────────────────────────────────────────

def precompute_transition_matrices_SSI(baselines, pulse_glu, params):
    rates = make_rates(params)
    Q_off   = build_Q_numba(0.0, rates)
    Q_pulse = build_Q_numba(pulse_glu, rates)
    M_off   = expm(Q_off * params['∆t'])
    M_pulse = expm(Q_pulse * params['∆t'])

    M_bases = np.zeros((len(baselines), 12, 12), dtype=np.float64)
    for i, b in enumerate(baselines):
        Qb = build_Q_numba(b, rates)
        M_bases[i] = expm(Qb * params['∆t'])
    return M_bases, M_pulse, M_off

@njit(parallel=True)
def simulate_protocol_all_SSI(baselines, ref_max, M_bases, M_pulse, M_off,
                             time_points, baseline_duration,
                             pulse_duration, dt, GO1, GO2, GO3):
    n = baselines.size
    open_traces = np.zeros((n, time_points), dtype=np.float64)
    ratios = np.zeros(n, dtype=np.float64)

    baseline_steps = int(round(baseline_duration / dt))
    pulse_steps = int(round(pulse_duration / dt))

    for i in prange(n):
        P_cur = np.zeros(12, dtype=np.float64)
        P_cur[0] = 1.0
        max_cond = 0.0
        for t in range(time_points):
            op = P_cur[IDX_O2] + P_cur[IDX_O3] + P_cur[IDX_O4]
            cond = P_cur[IDX_O2] * GO1 + P_cur[IDX_O3] * GO2 + P_cur[IDX_O4] * GO3
            open_traces[i, t] = op
            if baseline_steps <= t < baseline_steps + pulse_steps:
                if cond > max_cond:
                    max_cond = cond
            if t == time_points - 1:
                break
            if t < baseline_steps:
                M = M_bases[i]
            elif t < baseline_steps + pulse_steps:
                M = M_pulse
            else:
                M = M_off
            P_next = np.zeros(12, dtype=np.float64)
            for j in range(12):
                for k in range(12):
                    P_next[j] += P_cur[k] * M[k, j]
            s = 0.0
            for k in range(12):
                v = max(P_next[k], 0.0)
                P_cur[k] = v
                s += v
            if s > 0.0:
                P_cur /= s
        ratios[i] = max_cond / ref_max
    return open_traces, ratios

def run_ssi(params):
    total_start = time_module.perf_counter()
    dt = params['∆t']
    baseline_duration = float(params['Baseline duration'])
    pulse_duration = float(params['Pulse Duration'])

    total_time = baseline_duration + pulse_duration + 0.05
    time_points = int(round(total_time / dt)) + 1

    # Parse baseline glutamate concs (in nM), convert to M
    s = params['Baseline agonist concentrations']
    try:
        arr = [float(x) for x in s.split(',') if x.strip() != ""]
        bas_nM = np.array(arr, dtype=np.float64)
        bas_M = bas_nM / 1e9
        print(f"Parsed Baseline agonist concentrations: {arr} nM")
    except Exception as e:
        print(f"Error parsing Baseline agonist concentrations ('{s}'): {str(e)}, using default")
        bas_M = np.array([1,10,50,100,200,600,1000,1600,2000,2500,5000,10000,12500], dtype=np.float64) / 1e9

    pulse_glu = float(params['Agonist Concentration'])

    print("⏳ Precomputing transition matrices for reference SSI…")
    M_bases_ref, M_pulse_ref, M_off_ref = precompute_transition_matrices_SSI(np.array([0.0]), pulse_glu, params)
    _, raw_ref = simulate_protocol_all_SSI(np.array([0.0]), 1.0, M_bases_ref, M_pulse_ref, M_off_ref,
                                           time_points, baseline_duration, pulse_duration, dt,
                                           params['Conductance O2'], params['Conductance O3'], params['Conductance O4'])
    ref_max = raw_ref[0]

    print(f"⏳ Precomputing transition matrices for SSI ({len(bas_M)} baselines)…")
    M_bases, M_pulse, M_off = precompute_transition_matrices_SSI(bas_M, pulse_glu, params)

    print("⏳ Running SSI simulations in parallel…")
    open_traces, max_ratios = simulate_protocol_all_SSI(bas_M, ref_max,
        M_bases, M_pulse, M_off, time_points,
        baseline_duration, pulse_duration,
        dt, params['Conductance O2'], params['Conductance O3'], params['Conductance O4'])

    t1 = time_module.perf_counter()
    print(f"✅ Completed SSI runs in {(t1 - total_start):.3f}s")

    # Save CSV
    from flask import current_app
    import os
    static_path = os.path.join(current_app.static_folder, 'plot_data_ssi_optimized.csv')
    with open(static_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Baseline (M)", "Norm Max"])
        for b, r in zip(bas_M, max_ratios):
            w.writerow([b, r])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    times = np.linspace(0, total_time, time_points)
    colors = plt.cm.cividis(np.linspace(0, 1, len(bas_M)))
    for i, b in enumerate(bas_M):
        ax1.plot(times, open_traces[i], color=colors[i], label=f"{b*1e9:.0f} nM")
    ax1.set(title="Steady-State Inactivation: P(open) vs Time", xlabel="Time (s)", ylabel="P(open)")
    ax1.legend(loc='upper right', fontsize='x-small')
    ax1.grid(True)

    ax2.semilogx(bas_M * 1e6, max_ratios, 'o-', color='C3')
    ax2.set(title="Normalized Max vs Baseline [Glu]", xlabel="[Glu] baseline (µM)", ylabel="Norm Max")
    ax2.grid(True)

    param_text_lines = ["Simulation Parameters:"]
    for key, value in params.items():
        if not key.startswith('k_') and key not in ['Baseline agonist concentrations']:
            param_text_lines.append(f"{key}: {value}")
    param_text = "\n".join(param_text_lines)
    plt.gcf().text(0.97, 0.26, param_text, fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    static_path = os.path.join(current_app.static_folder, 'ssi.png')
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("▶️ Saved SSI plot.")
    return "Steady-State Inactivation simulation complete."
# ─────────────────────────────────────────────────────────────────────────────

def precompute_transition_matrices_TRAIN(params):
    rates = make_rates(params)
    dt = params['∆t']

    Q_off = build_Q_numba(0.0, rates)
    Q_on  = build_Q_numba(params['Agonist Concentration'], rates)

    M_off = expm(Q_off * dt)
    M_on  = expm(Q_on * dt)

    return M_on, M_off

def build_train_sweep(freq, params):
    dt = params['∆t']
    pre = max(0.0, params['Pre time'])               # force non-negative
    post = max(0.0, params['Post time'])
    pulse_width = max(0.0, params['Pulse width'])
    n_pulses = max(1, int(params['Number of pulses']))  # at least 1

    if freq <= 0:
        freq = 1.0  # fallback to avoid inf

    isi = 1.0 / freq
    t_between = (n_pulses - 1) * isi
    tmax = pre + t_between + post + pulse_width  # include last pulse end

    if tmax <= 0 or dt <= 0:
        raise ValueError(
            f"Simulation time invalid (tmax={tmax:.6f} s). "
            f"Check Pre={pre}, Post={post}, Pulse width={pulse_width}, pulses={n_pulses}, freq={freq} Hz."
        )

    nT = int(round(tmax / dt)) + 1
    if nT < 10:  # arbitrary small minimum to catch tiny times
        raise ValueError(f"Too few time steps (nT={nT}). Increase durations or decrease ∆t.")

    t = np.arange(nT) * dt
    is_on = np.zeros(nT, dtype=np.uint8)

    for p in range(n_pulses):
        t0 = pre + p * isi
        t1 = t0 + pulse_width
        i0 = int(t0 / dt)
        i1 = int(t1 / dt)
        is_on[i0:i1] = 1

    return t, is_on

@njit
def simulate_train(M_on, M_off, is_on, GO2, GO3, GO4):
    nT = len(is_on)
    P = np.zeros(12)
    P[0] = 1.0

    G = np.zeros(nT)
    Pop = np.zeros(nT)

    for t in range(nT - 1):
        M = M_on if is_on[t] else M_off

        P_next = np.zeros(12)
        for j in range(12):
            for k in range(12):
                P_next[j] += P[k] * M[k, j]
        P = P_next / np.sum(P_next)
    

        o2, o3, o4 = P[IDX_O2], P[IDX_O3], P[IDX_O4]
        Pop[t] = o2 + o3 + o4
        G[t]   = o2 * GO2 + o3 * GO3 + o4 * GO4

    return G, Pop

def run_train(params):
    from flask import current_app
    import os
    t0 = time_module.perf_counter()
    # Parse frequencies
    s = params['Train frequencies (Hz)']
    freqs = np.array([float(x) for x in s.split(',') if x.strip() != ""])
    print("⏳ Precomputing transition matrices for TRAIN…")
    M_on, M_off = precompute_transition_matrices_TRAIN(params)
    sweeps = [build_train_sweep(f, params) for f in freqs]
    print("⏳ Running TRAIN simulations…")
    with parallel_backend("threading", n_jobs=4):
        results = Parallel()(
            delayed(simulate_train)(
                M_on, M_off, is_on,
                params['Conductance O2'],
                params['Conductance O3'],
                params['Conductance O4']
            )
            for (_, is_on) in sweeps
        )
    t_list = [sweeps[i][0] for i in range(len(freqs))]
    G_list = [r[0] for r in results]
    P_list = [r[1] for r in results]
    # --------------------------------------------------
    # Pulse ratios (normalized to pulse 1)
    # --------------------------------------------------
    pulse_ratios_all = []
    dt = params['∆t']
    pre = params['Pre time']
    pulse_width = params['Pulse width']
    n_pulses = params['Number of pulses']
    for idx, f in enumerate(freqs):
        isi = 1.0 / f
        G = G_list[idx]
        ratios = np.zeros(n_pulses)
        for p in range(n_pulses):
            t0p = pre + p * isi
            t1p = t0p + pulse_width
            i0 = int(round(t0p / dt))  # Safer rounding
            i1 = int(round(t1p / dt))
            i0 = max(0, i0)
            i1 = min(len(G), i1)
            if i1 <= i0:
                ratios[p] = 0.0
            else:
                ratios[p] = np.max(G[i0:i1])
        if ratios[0] > 0:
            ratios /= ratios[0]
        else:
            ratios /= 1.0
        pulse_ratios_all.append(np.clip(ratios, 0.0, 1.0))
    t1 = time_module.perf_counter()
    print(f"✅ TRAIN completed in {(t1 - t0):.3f}s")
    # Save CSV
    def save_train_csv(filename, freqs, t_list, G_list, pulse_ratios_all, stride=10):
        t_ds = [t[::stride] for t in t_list]
        G_ds = [G[::stride] for G in G_list]
        lengths = [len(t) for t in t_ds]
        maxlen = max(lengths)
        header = []
        for f in freqs:
            header.append(f"time_{int(f)}Hz_s")
            header.append(f"G_{int(f)}Hz_pS")
        header.append("")
        header.append("pulse_number")
        for f in freqs:
            header.append(f"{int(f)}Hz_ratio")
        with open(filename, "w", newline="") as fh:
            fh.write("# Train simulation results\n")
            fh.write(",".join(header) + "\n")
            for row in range(maxlen):
                fields = []
                for i in range(len(freqs)):
                    if row < lengths[i]:
                        fields.append(f"{t_ds[i][row]:.9g}")
                        fields.append(f"{G_ds[i][row]:.9g}")
                    else:
                        fields.append("")
                        fields.append("")
                fields.append("")
                if row < n_pulses:
                    fields.append(str(row + 1))
                    for i in range(len(freqs)):
                        fields.append(f"{pulse_ratios_all[i][row]:.9g}")
                else:
                    fields.append("")
                    for _ in freqs:
                        fields.append("")
                fh.write(",".join(fields) + "\n")
    csv_path = os.path.join(current_app.static_folder, 'ephys_train_sim_stride10.csv')
    save_train_csv(csv_path, freqs, t_list, G_list, pulse_ratios_all)
    print("▶️ Saved Train CSV.")
    # Plot
    n_freqs = len(freqs)
    ncols = 2
    nrows = int(np.ceil(n_freqs / ncols))
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=(5.2 * ncols, 5.0 * nrows), squeeze=False)
    colors = plt.cm.winter(np.linspace(0, 1, n_freqs))
    for idx, f in enumerate(freqs):
        row = idx // ncols
        col = idx % ncols
        ax_G = axes[2 * row, col]
        ax_P = axes[2 * row + 1, col]
        t_i = t_list[idx]
        G = G_list[idx]
        ratios = pulse_ratios_all[idx]
        ax_G.plot(t_i, G, color=colors[idx], lw=1.5)
        ax_G.set_title(f"{int(f)} Hz")
        ax_G.set_ylabel("Conductance (pS)")
        ax_G.set_xlim(0.0, t_i.max())
        ax_G.grid(True)
        # Scaling y-ticks to pS (multiply displayed values by 1e12)
        ax_G.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 1e12:.2g}"))
        ax_P.plot(np.arange(1, n_pulses + 1), ratios, marker='o', lw=1.5, color=colors[idx])
        ax_P.set_xlabel("Pulse number")
        ax_P.set_ylabel("Ratio (to pulse 1,\n" "clipped 0-1)")
        ax_P.set_xlim(1, n_pulses)
        ax_P.set_ylim(0.0, 1.0)
        ax_P.grid(True)
    total_slots = nrows * ncols
    for empty in range(n_freqs, total_slots):
        row = empty // ncols
        col = empty % ncols
        axes[2 * row, col].axis("off")
        axes[2 * row + 1, col].axis("off")
    fig.suptitle(f"Train simulations ({n_pulses} pulses, width = {pulse_width*1e3:.1f} ms)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(current_app.static_folder, 'train.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("▶️ Saved Train plot.")
    return "Train simulation complete."