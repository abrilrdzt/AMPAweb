from flask import render_template, request, send_file, flash, redirect, url_for
from app import app
from .simulations import default_params, run_dr, run_ppr_sppr, run_ssi
import io
import sys
from contextlib import redirect_stdout

RATE_COORDS = {
    'k_R0_R1': (60, 120),
    'k_R1_R0': (60, 165),
    'k_R1_R2': (220, 120),
    'k_R2_R1': (220, 165),
    'k_R2_R3': (390, 120),
    'k_R3_R2': (390, 165),
    'k_R3_R4': (570, 120),
    'k_R4_R3': (570, 165),
    'k_R1_D1': (180, 220),
    'k_D1_R1': (90, 220),
    'k_R2_D2': (350, 220),
    'k_D2_R2': (255, 220),
    'k_R3_D3': (520, 220),
    'k_D3_R3': (430, 220),
    'k_R4_D4': (690, 220),
    'k_D4_R4': (600, 220),
    'k_D1_D2': (220, 265),
    'k_D2_D1': (220, 315),
    'k_D2_D3': (390, 265),
    'k_D3_D2': (390, 315),
    'k_D3_D4': (570, 265),
    'k_D4_D3': (570, 315),
    'k_R2_O2': (255, 60),
    'k_O2_R2': (345, 60),
    'k_R3_O3': (430, 60),
    'k_O3_R3': (520, 60),
    'k_R4_O4': (600, 60),
    'k_O4_R4': (690, 60),
}

def get_unit_for_rate(key):
    if key.startswith("k_R") or key.startswith("k_D"):
        parts = key.split("_")
        if len(parts) == 3:
            src, tgt = parts[1], parts[2]
            if src[0] in {"R", "D"} and tgt[0] in {"R", "D"} and int(src[1]) < int(tgt[1]):
                return "M⁻⁻¹·s⁻⁻¹"
            else:
                return "s⁻⁻¹"
    return "s⁻⁻¹"

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    output = ""
    plot_file = None
    csv_file = None
    if request.method == 'POST':
        try:
            params = {}
            # Collect transition rates
            for key in default_params:
                if key.startswith('k_'):
                    val = request.form.get(key, str(default_params[key]))
                    try:
                        params[key] = float(val) if val.strip() else default_params[key]
                    except ValueError:
                        params[key] = default_params[key]
            # Collect general parameters
            for key in ['∆t', 'Agonist Concentration', 'Pulse Duration', 'Conductance O2', 'Conductance O3', 'Conductance O4']:
                val = request.form.get(key, str(default_params[key]))
                unit = request.form.get(f'unit_{key}', 'ms' if key in ['∆t', 'Pulse Duration'] else 'mM' if key == 'Agonist Concentration' else 'pS')
                try:
                    val_float = float(val) if val.strip() else default_params[key]
                    if key in ['∆t', 'Pulse Duration']:
                        params[key] = val_float if unit == 's' else val_float / 1000.0
                    elif key == 'Agonist Concentration':
                        params[key] = val_float if unit == 'M' else val_float / 1000.0
                    elif key in ['Conductance O2', 'Conductance O3', 'Conductance O4']:
                        params[key] = val_float if unit == 'S' else val_float / 1e12
                except ValueError:
                    params[key] = default_params[key]
            # Collect simulation-specific parameters
            sim_type = request.form.get('simulation_type', 'Dose Response')
            if sim_type == 'Dose Response':
                for key in ['Maximum time', 'Agonist starting time', 'Agonist ending time']:
                    val = request.form.get(key, str(default_params[key]))
                    unit = request.form.get(f'unit_{key}', 'ms')
                    try:
                        params[key] = float(val) if unit == 's' else float(val) / 1000.0
                    except ValueError:
                        params[key] = default_params[key]
                val = request.form.get('Agonist concentrations', default_params['Agonist concentrations'])
                try:
                    params['Agonist concentrations'] = [float(x.strip()) for x in val.split(',') if x.strip()]
                except ValueError:
                    params['Agonist concentrations'] = default_params['Agonist concentrations']
                plot_file = 'dose_response.png'
                csv_file = 'plot_data_dose_response.csv'
            elif sim_type == 'Long Paired-Pulse Recovery':
                for key in ['Starting time (baseline pulse) Long Paired Pulse', 'Buffer time']:
                    val = request.form.get(key, str(default_params[key]))
                    unit = request.form.get(f'unit_{key}', 'ms')
                    try:
                        params[key] = float(val) if unit == 's' else float(val) / 1000.0
                    except ValueError:
                        params[key] = default_params[key]
                val = request.form.get('Pulse intervals (long)', default_params['Pulse intervals (long)'])
                try:
                    params['Pulse intervals (long)'] = [float(x.strip()) for x in val.split(',') if x.strip()]
                except ValueError:
                    params['Pulse intervals (long)'] = default_params['Pulse intervals (long)']
                plot_file = 'ppr_sppr.png'
                csv_file = 'ppr_conductance_ratio_vs_interval.csv'
            elif sim_type == 'Short Paired-Pulse Recovery':
                for key in ['Starting time (baseline pulse) Short Paired Pulse', 'Buffer time']:
                    val = request.form.get(key, str(default_params[key]))
                    unit = request.form.get(f'unit_{key}', 'ms')
                    try:
                        params[key] = float(val) if unit == 's' else float(val) / 1000.0
                    except ValueError:
                        params[key] = default_params[key]
                val = request.form.get('Pulse intervals (short)', default_params['Pulse intervals (short)'])
                try:
                    params['Pulse intervals (short)'] = [float(x.strip()) for x in val.split(',') if x.strip()]
                except ValueError:
                    params['Pulse intervals (short)'] = default_params['Pulse intervals (short)']
                plot_file = 'sppr_sppr.png'
                csv_file = 'sppr_conductance_ratio_vs_interval.csv'
            elif sim_type == 'Steady-State Inactivation':
                val = request.form.get('Baseline duration', str(default_params['Baseline duration']))
                unit = request.form.get('unit_Baseline duration', 'ms')
                try:
                    params['Baseline duration'] = float(val) if unit == 's' else float(val) / 1000.0
                except ValueError:
                    params['Baseline duration'] = default_params['Baseline duration']
                val = request.form.get('Baseline agonist concentrations', default_params['Baseline agonist concentrations'])
                try:
                    params['Baseline agonist concentrations'] = [float(x.strip()) for x in val.split(',') if x.strip()]
                except ValueError:
                    params['Baseline agonist concentrations'] = default_params['Baseline agonist concentrations']
                plot_file = 'ssi.png'
                csv_file = 'plot_data_ssi_optimized.csv'

            # Capture console output
            f = io.StringIO()
            with redirect_stdout(f):
                if sim_type == 'Dose Response':
                    output = run_dr(params)
                elif sim_type == 'Long Paired-Pulse Recovery':
                    output = run_ppr_sppr(params, is_sppr=False)
                elif sim_type == 'Short Paired-Pulse Recovery':
                    output = run_ppr_sppr(params, is_sppr=True)
                elif sim_type == 'Steady-State Inactivation':
                    output = run_ssi(params)
            output += f.getvalue()
        except Exception as e:
            flash(f"Error: {str(e)}")
            output = f"Error: {str(e)}"
    return render_template('index.html',
                          default_params=default_params,
                          output=output,
                          plot_file=plot_file,
                          csv_file=csv_file,
                          rate_coords=RATE_COORDS,
                          get_unit_for_rate=get_unit_for_rate)