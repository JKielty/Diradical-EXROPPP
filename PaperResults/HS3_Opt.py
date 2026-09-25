from quantel.wfn.csf import CSF
from quantel.ints.fcidump_integrals import FCIDUMP
from quantel.opt.lbfgs import LBFGS
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('FCIDUMP', type=str, help='FCIDUMP file containing PPP integrals')
args = parser.parse_args()
fcidump = args.FCIDUMP

fcidump_ints = FCIDUMP(fcidump)
fcidump_ints.print()
nmo = fcidump_ints.nmo()

n_trials = 10

if __name__ == "__main__":

    best_energy = np.inf
    best_wfn    = None
    best_label  = None

    # ------------------------------------------------------------------
    # Named initial guesses: gwh and core
    # ------------------------------------------------------------------
    print("===============================================")
    print(" Testing CSF optimisation with named guesses")
    print("===============================================")

    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")

        # Create a fresh CSF object for each guess (open-shell singlet)
        wfn = CSF(fcidump_ints, '++')
        wfn.get_orbital_guess(method=guess, localise=False)
        LBFGS().run(wfn)

        energy = wfn.energy          # <-- verify attribute name with: print(dir(wfn))
        print(f" Converged energy ({guess}): {energy:.10f}")

        if energy < best_energy:
            best_energy = energy
            best_wfn    = wfn
            best_label  = guess

    # ------------------------------------------------------------------
    # Random coefficient initial guesses
    # ------------------------------------------------------------------
    print("\n===============================================")
    print(" Testing CSF optimisation with random coefficients")
    print("===============================================")

    for trial in range(n_trials):
        print("\n************************************************")
        print(f" Random coefficient initial guess, Trial {trial}")
        print("************************************************")

        wfn = CSF(fcidump_ints, '++')
        rng = np.random.default_rng()          # fresh seed each trial
        Cguess = rng.standard_normal((nmo, nmo))
        wfn.initialise(Cguess)
        LBFGS().run(wfn)

        energy = wfn.energy
        print(f" Converged energy (trial {trial}): {energy:.10f}")

        if energy < best_energy:
            best_energy = energy
            best_wfn    = wfn
            best_label  = f"random_trial_{trial}"

    # ------------------------------------------------------------------
    # Post-processing: canonicalise, Hessian check, print, and save
    # — performed once, for the lowest-energy solution only
    # ------------------------------------------------------------------
    print("\n===============================================")
    print(f" Best solution: '{best_label}'")
    print(f" Energy        = {best_energy:.10f}")
    print("===============================================")

    best_wfn.canonicalize()
    best_wfn.get_davidson_hessian_index(approx_hess=False)
    best_wfn.print(verbose=5)

    print(best_wfn.mo_coeff)
    outfile = f"{fcidump}_HS3_mo_coeff.npy"
    np.save(outfile, best_wfn.mo_coeff)
    print(f" Saved MO coefficients to {outfile}")