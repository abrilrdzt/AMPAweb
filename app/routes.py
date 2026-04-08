from flask import render_template, request, send_file, flash, redirect, url_for
from app import app
from .simulations import default_params, run_dr, run_ppr_sppr, run_ssi, run_train
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


def parse_pulse_duration(request_form, params, default_params, errors):
    """Parse only Pulse Duration from form."""
    val = request_form.get('Pulse Duration', '').strip()
    unit = request_form.get('unit_Pulse Duration', 'ms')
    if not val:
        params['Pulse Duration'] = default_params['Pulse Duration']
        errors.append(f"Missing Pulse Duration, using default: {default_params['Pulse Duration']}")
    else:
        try:
            val_f = float(val)
            params['Pulse Duration'] = val_f if unit == 's' else val_f / 1000.0
        except ValueError:
            params['Pulse Duration'] = default_params['Pulse Duration']
            errors.append(f"Invalid Pulse Duration, using default: {default_params['Pulse Duration']}")


def parse_agonist_concentration(request_form, params, default_params, errors):
    """Parse only Agonist Concentration from form."""
    val = request_form.get('Agonist Concentration', '').strip()
    unit = request_form.get('unit_Agonist Concentration', 'mM')
    if not val:
        params['Agonist Concentration'] = default_params['Agonist Concentration']
        errors.append(f"Missing Agonist Concentration, using default: {default_params['Agonist Concentration']}")
    else:
        try:
            val_f = float(val)
            params['Agonist Concentration'] = val_f if unit == 'M' else val_f / 1000.0
        except ValueError:
            params['Agonist Concentration'] = default_params['Agonist Concentration']
            errors.append(f"Invalid Agonist Concentration, using default: {default_params['Agonist Concentration']}")


def parse_agonist_and_pulse(request_form, params, default_params, errors):
    """Parse both Agonist Concentration and Pulse Duration from form."""
    parse_agonist_concentration(request_form, params, default_params, errors)
    parse_pulse_duration(request_form, params, default_params, errors)


def get_unit_for_rate(key):
    if key.startswith("k_R") or key.startswith("k_D"):
        parts = key.split("_")
        if len(parts) == 3:
            src, tgt = parts[1], parts[2]
            if src[0] in {"R", "D"} and tgt[0] in {"R", "D"} and int(src[1]) < int(tgt[1]):
                return "M⁻¹·s⁻¹"
            else:
                return "s⁻¹"
    return "s⁻¹"


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    output = ""
    plot_file = None
    csv_file = None
    errors = []

    if request.method == 'POST':
        print(f"Received form data: {request.form}")

        params = {}

        # ── Transition rates ──────────────────────────────────────────────────
        for key in default_params:
            if key.startswith('k_'):
                val = request.form.get(key, '').strip()
                if not val:
                    params[key] = default_params[key]
                    errors.append(f"Missing value for {key}, using default: {default_params[key]}")
                else:
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = default_params[key]
                        errors.append(f"Invalid value for {key} ('{val}'), using default: {default_params[key]}")

        # ── General parameters shared by all protocols: ∆t + conductances ────
        for key in ['∆t', 'Conductance O2', 'Conductance O3', 'Conductance O4']:
            val = request.form.get(key, '').strip()
            unit = request.form.get(f'unit_{key}', 'ms' if key == '∆t' else 'pS')
            if not val:
                params[key] = default_params[key]
                errors.append(f"Missing value for {key}, using default: {default_params[key]}")
            else:
                try:
                    val_float = float(val)
                    if key == '∆t':
                        params[key] = val_float if unit == 's' else val_float / 1000.0
                    elif key in ['Conductance O2', 'Conductance O3', 'Conductance O4']:
                        params[key] = val_float if unit == 'S' else val_float / 1e12
                except ValueError:
                    params[key] = default_params[key]
                    errors.append(f"Invalid value for {key} ('{val}'), using default: {default_params[key]}")

        # ── Simulation-specific parameters ────────────────────────────────────
        sim_type = request.form.get('simulation_type', 'Dose Response')

        # ── DOSE RESPONSE ─────────────────────────────────────────────────────
        if sim_type == 'Dose Response':
            # DR does NOT use a single Agonist Concentration — the sweep list is what matters.
            # Only Pulse Duration is needed here.
            parse_pulse_duration(request.form, params, default_params, errors)

            for key in ['Maximum time', 'Agonist starting time', 'Agonist ending time']:
                val = request.form.get(key, '').strip()
                unit = request.form.get(f'unit_{key}', 'ms')
                if not val:
                    params[key] = default_params[key]
                    errors.append(f"Missing value for {key}, using default: {default_params[key]}")
                else:
                    try:
                        params[key] = float(val) if unit == 's' else float(val) / 1000.0
                    except ValueError:
                        params[key] = default_params[key]
                        errors.append(f"Invalid value for {key} ('{val}'), using default: {default_params[key]}")

            val = request.form.get('Agonist concentrations', default_params['Agonist concentrations']).strip()
            if not val:
                params['Agonist concentrations'] = default_params['Agonist concentrations']
                errors.append(f"Missing Agonist concentrations, using default: {default_params['Agonist concentrations']}")
            else:
                try:
                    _ = [float(x) for x in val.split(',') if x.strip()]
                    params['Agonist concentrations'] = val
                except ValueError:
                    params['Agonist concentrations'] = default_params['Agonist concentrations']
                    errors.append(f"Invalid format for Agonist concentrations ('{val}'), using default: {default_params['Agonist concentrations']}")

            plot_file = 'dose_response.png'
            csv_file = 'plot_data_dose_response.csv'

        # ── LONG PAIRED-PULSE RECOVERY ────────────────────────────────────────
        elif sim_type == 'Long Paired-Pulse Recovery':
            # Agonist Concentration (both pulses) + Pulse Duration (2nd pulse)
            parse_agonist_and_pulse(request.form, params, default_params, errors)

            # Duration of conditioning (1st) pulse — user enters positive ms, sim expects negative seconds
            key = 'Starting time (baseline pulse) Long Paired Pulse'
            val = request.form.get(key, '').strip()
            unit = request.form.get(f'unit_{key}', 'ms')
            if not val:
                params[key] = default_params[key]
                errors.append(f"Missing Duration of Conditioning Pulse, using default: {default_params[key]}")
            else:
                try:
                    val_f = float(val)
                    duration = val_f if unit == 's' else val_f / 1000.0
                    params[key] = -duration  # negate: user enters positive duration, sim expects negative start time
                except ValueError:
                    params[key] = default_params[key]
                    errors.append(f"Invalid Duration of Conditioning Pulse ('{val}'), using default: {default_params[key]}")

            # Buffer time — stays positive
            key = 'Buffer time'
            val = request.form.get(key, '').strip()
            unit = request.form.get(f'unit_{key}', 'ms')
            if not val:
                params[key] = default_params[key]
                errors.append(f"Missing Buffer time, using default: {default_params[key]}")
            else:
                try:
                    params[key] = float(val) if unit == 's' else float(val) / 1000.0
                except ValueError:
                    params[key] = default_params[key]
                    errors.append(f"Invalid Buffer time ('{val}'), using default: {default_params[key]}")

            # Inter-pulse intervals
            val = request.form.get('Pulse intervals (long)', default_params['Pulse intervals (long)']).strip()
            if not val:
                params['Pulse intervals (long)'] = default_params['Pulse intervals (long)']
                errors.append(f"Missing Pulse intervals (long), using default.")
            else:
                try:
                    _ = [float(x) for x in val.split(',') if x.strip()]
                    params['Pulse intervals (long)'] = val
                except ValueError:
                    params['Pulse intervals (long)'] = default_params['Pulse intervals (long)']
                    errors.append(f"Invalid format for Pulse intervals (long) ('{val}'), using default.")

            plot_file = 'ppr_sppr.png'
            csv_file = 'ppr_conductance_ratio_vs_interval.csv'

        # ── SHORT PAIRED-PULSE RECOVERY ───────────────────────────────────────
        elif sim_type == 'Short Paired-Pulse Recovery':
            # Agonist Concentration (both pulses) + Pulse Duration (2nd pulse)
            parse_agonist_and_pulse(request.form, params, default_params, errors)

            # Duration of conditioning (1st) pulse — same negation logic as LPPR
            key = 'Starting time (baseline pulse) Short Paired Pulse'
            val = request.form.get(key, '').strip()
            unit = request.form.get(f'unit_{key}', 'ms')
            if not val:
                params[key] = default_params[key]
                errors.append(f"Missing Duration of Conditioning Pulse, using default: {default_params[key]}")
            else:
                try:
                    val_f = float(val)
                    duration = val_f if unit == 's' else val_f / 1000.0
                    params[key] = -duration  # negate: user enters positive duration, sim expects negative start time
                except ValueError:
                    params[key] = default_params[key]
                    errors.append(f"Invalid Duration of Conditioning Pulse ('{val}'), using default: {default_params[key]}")

            # Buffer time — stays positive
            key = 'Buffer time'
            val = request.form.get(key, '').strip()
            unit = request.form.get(f'unit_{key}', 'ms')
            if not val:
                params[key] = default_params[key]
                errors.append(f"Missing Buffer time, using default: {default_params[key]}")
            else:
                try:
                    params[key] = float(val) if unit == 's' else float(val) / 1000.0
                except ValueError:
                    params[key] = default_params[key]
                    errors.append(f"Invalid Buffer time ('{val}'), using default: {default_params[key]}")

            # Inter-pulse intervals
            val = request.form.get('Pulse intervals (short)', default_params['Pulse intervals (short)']).strip()
            if not val:
                params['Pulse intervals (short)'] = default_params['Pulse intervals (short)']
                errors.append(f"Missing Pulse intervals (short), using default.")
            else:
                try:
                    _ = [float(x) for x in val.split(',') if x.strip()]
                    params['Pulse intervals (short)'] = val
                except ValueError:
                    params['Pulse intervals (short)'] = default_params['Pulse intervals (short)']
                    errors.append(f"Invalid format for Pulse intervals (short) ('{val}'), using default.")

            plot_file = 'sppr_sppr.png'
            csv_file = 'sppr_conductance_ratio_vs_interval.csv'

        # ── STEADY-STATE INACTIVATION ─────────────────────────────────────────
        elif sim_type == 'Steady-State Inactivation':
            # Agonist Concentration (test pulse, fixed) + Pulse Duration (test pulse)
            parse_agonist_and_pulse(request.form, params, default_params, errors)

            # Pre-conditioning baseline duration
            val = request.form.get('Baseline duration', '').strip()
            unit = request.form.get('unit_Baseline duration', 'ms')
            if not val:
                params['Baseline duration'] = default_params['Baseline duration']
                errors.append(f"Missing Baseline duration, using default: {default_params['Baseline duration']}")
            else:
                try:
                    params['Baseline duration'] = float(val) if unit == 's' else float(val) / 1000.0
                except ValueError:
                    params['Baseline duration'] = default_params['Baseline duration']
                    errors.append(f"Invalid Baseline duration ('{val}'), using default: {default_params['Baseline duration']}")

            # Pre-conditioning concentrations sweep
            val = request.form.get('Baseline agonist concentrations', default_params['Baseline agonist concentrations']).strip()
            if not val:
                params['Baseline agonist concentrations'] = default_params['Baseline agonist concentrations']
                errors.append(f"Missing Baseline agonist concentrations, using default.")
            else:
                try:
                    _ = [float(x) for x in val.split(',') if x.strip()]
                    params['Baseline agonist concentrations'] = val
                except ValueError:
                    params['Baseline agonist concentrations'] = default_params['Baseline agonist concentrations']
                    errors.append(f"Invalid format for Baseline agonist concentrations ('{val}'), using default.")

            plot_file = 'ssi.png'
            csv_file = 'plot_data_ssi_optimized.csv'

        # ── TRAIN ─────────────────────────────────────────────────────────────
        elif sim_type == 'Train':
            errors_train = []

            # Pulse agonist concentration
            parse_agonist_concentration(request.form, params, default_params, errors_train)

            # Stimulation frequencies
            freq_input = request.form.get('Train frequencies (Hz)', '').strip()
            if not freq_input:
                freq_input = default_params['Train frequencies (Hz)']
                errors_train.append("No frequencies provided → using default")
            try:
                freq_list = [float(x.strip()) for x in freq_input.split(',') if x.strip()]
                if not freq_list:
                    raise ValueError("empty after parsing")
                if any(f <= 0 for f in freq_list):
                    raise ValueError("frequencies must be > 0 Hz")
                params['Train frequencies (Hz)'] = freq_input
            except Exception as e:
                errors_train.append(f"Invalid frequencies '{freq_input}': {e} → using default")
                params['Train frequencies (Hz)'] = default_params['Train frequencies (Hz)']

            # Number of pulses
            npulses_input = request.form.get('Number of pulses', '').strip()
            try:
                if not npulses_input:
                    raise ValueError("missing")
                n = int(float(npulses_input))
                if n < 1:
                    raise ValueError("must be at least 1")
                params['Number of pulses'] = n
            except Exception as e:
                errors_train.append(f"Invalid 'Number of pulses' '{npulses_input}': {e} → using default")
                params['Number of pulses'] = default_params['Number of pulses']

            # Pre time, Post time, Pulse width — all must be positive
            for key in ['Pre time', 'Post time', 'Pulse width']:
                val_str = request.form.get(key, '').strip()
                unit = request.form.get(f'unit_{key}', 'ms')
                try:
                    if not val_str:
                        raise ValueError("missing")
                    val_f = float(val_str)
                    if val_f <= 0:
                        raise ValueError("must be positive")
                    params[key] = val_f if unit == 's' else val_f / 1000.0
                except Exception as e:
                    errors_train.append(f"Invalid '{key}' value '{val_str}' ({e}) → using default")
                    params[key] = default_params[key]

            for msg in errors_train:
                flash(msg)

            plot_file = 'train.png'
            csv_file = 'ephys_train_sim_stride10.csv'

        # ── Debug + user-facing errors ────────────────────────────────────────
        print(f"Parsed parameters: {params}")
        for error in errors:
            flash(error)

        # ── Run simulation ────────────────────────────────────────────────────
        try:
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
                elif sim_type == 'Train':
                    output = run_train(params)
            captured_output = f.getvalue()
            output += captured_output
            print(f"Captured simulation output: {captured_output}")
        except Exception as e:
            flash(f"Simulation error: {str(e)}")
            output = f"Simulation error: {str(e)}"
            print(f"Simulation exception: {str(e)}")

    return render_template('index.html',
                           default_params=default_params,
                           output=output,
                           plot_file=plot_file,
                           csv_file=csv_file,
                           rate_coords=RATE_COORDS,
                           get_unit_for_rate=get_unit_for_rate)