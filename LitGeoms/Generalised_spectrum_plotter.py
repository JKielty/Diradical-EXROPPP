import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required in WSL and headless environments
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import re
from itertools import cycle
from pathlib import Path


def parse_spectrum(inputfile):
    '''
    Parse a gnuplot file and return the plot command string.

    Args:
        inputfile (str): Path to the gnuplot file.
    Returns:
        plot_cmd (str): Cleaned plot command string.
    '''
    with open(inputfile, 'r') as file:
        lines = file.readlines()

    last_line = lines[-1]
    start_index = last_line.find('p ') + 2
    end_index = last_line.find(' lw 3 dt 1')

    plot_cmd = last_line[start_index:end_index]
    plot_cmd = plot_cmd.replace('0inf', '0')
    print(f"Cleaned expression: {plot_cmd[:100]}...")
    return plot_cmd


def evaluate_spectrum(plot_cmd, wavelength):
    '''
    Evaluate a gnuplot Lorentzian expression over a wavelength array.

    x is set to the full wavelength array before calling eval(), so every
    arithmetic operation acts as a numpy ufunc and the entire spectrum is
    returned in one vectorised call — no per-point loop required.

    Args:
        plot_cmd (str): Cleaned gnuplot plot command string.
        wavelength (ndarray): Array of wavelength values in nm.
    Returns:
        broad (ndarray): Array of broadened spectral intensities.
    '''
    x = wavelength          # eval() references this variable
    return np.asarray(eval(plot_cmd), dtype=float)


def load_spectrum(filepath, wavelength):
    '''
    Load a spectrum from a .gp (gnuplot Lorentzian) or .dat (two-column) file,
    returning intensities on the given wavelength grid.

    .gp  → parse the Lorentzian expression and evaluate it via evaluate_spectrum.
    .dat → load the two-column file with np.loadtxt and interpolate onto
           wavelength using np.interp so the returned array always has the same
           length regardless of the grid stored in the file.

    Args:
        filepath (str): Path to a .gp or .dat spectrum file.
        wavelength (ndarray): Target wavelength grid in nm.
    Returns:
        broad (ndarray): Spectral intensities on wavelength.
    '''
    ext = Path(filepath).suffix.lower()
    if ext == '.dat':
        data = np.loadtxt(filepath, comments='#')
        return np.interp(wavelength, data[:, 0], data[:, 1])
    else:
        plot_cmd = parse_spectrum(filepath)
        return evaluate_spectrum(plot_cmd, wavelength)


# Colour-blind-friendly (Okabe–Ito) palette, cycled if more spectra than colours
# are supplied. The first three match the original F^eff / ³OS / ¹OS colours.
COLOURS = ['#D55E00',   # vermillion
           '#009E73',   # teal green
           '#0072B2',   # deep blue
           '#CC79A7',   # reddish purple
           '#E69F00',   # orange
           '#56B4E9',   # sky blue
           '#F0E442']   # yellow

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Plots any number of ExROPPP spectra, each with its own label, "
                "alongside an optional experimental spectrum. "
                "Accepts .gp (gnuplot Lorentzian) or .dat (two-column) files for "
                "each spectrum — useful when passing Boltzmann-combined .dat files "
                "produced by Hybrid_spectrum_plotter.py."
)
parser.add_argument('molecule_name', type=str, help="The name of the molecule")
parser.add_argument('-s', '--spectrum', nargs=2, action='append',
                    metavar=('FILE', 'LABEL'), required=True,
                    help="A spectrum file (.gp or .dat) and its legend label. "
                         "Repeat for each spectrum, e.g. "
                         "-s feff.gp '$F^{\\mathrm{eff}}$' -s other.dat 'My label'. "
                         "Labels support matplotlib mathtext (wrap in $...$).")
parser.add_argument('--xmin', type=float, default=250,
                    help="Minimum wavelength to plot in nm. Default 250.")
parser.add_argument('--xmax', type=float, default=800,
                    help="Maximum wavelength to plot in nm. Default 800.")
parser.add_argument('--expfile', type=str, help="Experimental data file (CSV, TSV, or space-separated .txt)")
args = parser.parse_args()

# Evaluate over the full spectral range (200-1000 nm) regardless of plot window,
# so that normalisation is never affected by the chosen x-axis limits.
wavelength = np.linspace(200, 1000, 1601)
mask = (wavelength >= args.xmin) & (wavelength <= args.xmax)

# Plot
mpl.rcParams['font.family']      = 'serif'
mpl.rcParams['font.serif']       = ['Times New Roman', 'Times']
mpl.rcParams['mathtext.fontset']  = 'stix'

plt.style.use('default')
fig, ax = plt.subplots(figsize=(6, 5))

# Load, normalise, and plot each spectrum.
# load_spectrum() handles both .gp and .dat files transparently.
for (filepath, label), colour in zip(args.spectrum, cycle(COLOURS)):
    broad = load_spectrum(filepath, wavelength)
    broad = (broad - np.min(broad)) / (np.max(broad[mask]) - np.min(broad))
    ax.plot(wavelength, broad, color=colour, linestyle='solid', linewidth=1.5, label=label)

if args.expfile is not None:
    exp_wl, exp_abs = [], []
    with open(args.expfile, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip().strip('"')
            tokens = [t.strip().strip('"') for t in re.split(r'[,\s]+', line) if t.strip()]
            if len(tokens) < 2:
                continue
            try:
                exp_wl.append(float(tokens[0]))
                exp_abs.append(float(tokens[1]))
            except ValueError:
                pass
    print(f"Parsed {len(exp_wl)} rows from '{args.expfile}'")
    exp_wl  = np.array(exp_wl)
    exp_abs = np.array(exp_abs)
    exp_abs = (exp_abs - np.min(exp_abs)) / (np.max(exp_abs) - np.min(exp_abs))
    ax.plot(exp_wl, exp_abs, color='black', linestyle='dashdot', linewidth=1.5, label='Experiment')

ax.set_xlabel('Wavelength / nm', fontsize=12)
ax.set_ylabel('Normalised Absorbance', fontsize=12)
ax.set_xlim(args.xmin, args.xmax)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, framealpha=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{args.molecule_name}_spectrum.png', dpi=600, bbox_inches='tight', transparent=True)
print(f"Spectrum saved to {args.molecule_name}_spectrum.png")