from quantel.wfn.csf import CSF
from quantel.ints.fcidump_integrals import FCIDUMP
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('FCIDUMP', type = str, help = 'FCIDUMP file containing PPP integrals')
args = parser.parse_args()
fcidump = args.FCIDUMP

fcidump_ints = FCIDUMP(fcidump)
fcidump_ints.print()
nmo = fcidump_ints.nmo()

n_trials = 10
results = []

if __name__ == "__main__":

    for guess in ("gwh", "core"):
        print("\n************************************************")
        print(f" Testing '{guess}' initial guess method")
        print("************************************************")
        
        wfn = CSF(fcidump_ints, '++')
        
        from quantel.opt.lbfgs import LBFGS
        wfn.get_orbital_guess(method=guess, localise=False)
        LBFGS().run(wfn)
        
        # Test canonicalisation 
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index(approx_hess=False)
        wfn.print(verbose=3)
    
    print("===============================================")
    print(f" Testing CSF optimisation with random coefficients")
    print("===============================================")
    
    
    for trial in range(n_trials):
        print("\n************************************************")
        print(f" Testing random coefficient initial guess method, Trial {trial}")
        print("************************************************")
        from quantel.opt.lbfgs import LBFGS
        
        wfn = CSF(fcidump_ints, '++')
        
        rng = np.random.default_rng()   # fresh seed each trial
        Cguess = rng.standard_normal((nmo, nmo))
        
        wfn.initialise(Cguess)
        LBFGS().run(wfn)
        
        # Test canonicalisation 
        wfn.canonicalize()
        # Test Hessian index
        wfn.get_davidson_hessian_index(approx_hess=False)
        wfn.print(verbose=3)
        
        '''
        print(wfn.mo_coeff)
        outfile = f"{fcidump}_{guess}_mo_coeff.npy"
        np.save(outfile, wfn.mo_coeff)
        print(f" Saved MO coefficients to {outfile}")
        '''