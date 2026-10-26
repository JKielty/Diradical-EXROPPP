import numpy as np
import scipy.optimize as optimize
import scipy.sparse.linalg as sp
import scipy.linalg as linalg
from datetime import datetime
from subprocess import getoutput
import sys
from ExROPPP_settings_opt import *
from SCF import *
from DipBuilder import *
from CIBuilderNew import *
import os





# parser = argparse.ArgumentParser()
# parser.add_argument('geometry', type = str, help = 'file containing geometry')
# args = parser.parse_args()
# optimized_geometry = args.geometry

# params=[[-22.53982183,   1.70115484 ,  8.47994446 ,  1.17367777,   0.        ],
#  [ -3.25983816 ,-24.50393011 ,  1.76554162 , 13.0486315  ,  1.18938422],
#  [-17.50211252 ,-23.67958463 ,  1.43383449 , 18.08184948 ,  1.12853335],
#  [-10.18396573 ,-26.36242115 ,  1.45855408 ,  9.61199125 ,  2.23245629]] 





def transform(two_body, hf_orbs):
    '''
    Places two-body terms (V_ij) into a four-index tensor (ij|kl) and performs a four-index transformation to the molecular orbital basis.
    
    Args:
        two_body: 2D array of two-body repulsion integrals in the atomic orbital basis. Usually repulsion array from v_term function. Shape (Natoms, Natoms)
        hf_orbs: 2D array of Hartree-Fock orbital coefficients in the atomic orbital basis. Shape (Natoms, Natoms)
    
    Returns:
        two_body_mo: 4D array of two-body repulsion integrals in the molecular orbital basis. Shape (Natoms, Natoms, Natoms, Natoms)
    '''
    Natoms = hf_orbs.shape[0]
    two_body_4i = np.zeros((Natoms, Natoms, Natoms, Natoms))
    ia = np.arange(Natoms)
    two_body_4i[ia[:, None], ia[:, None], ia[None, :], ia[None, :]] = two_body
    #four index transformation
    two_body_mo = np.einsum("ia, jb, kc, ld, ijkl -> abcd",
                             hf_orbs, hf_orbs, hf_orbs, hf_orbs, two_body_4i, optimize= 'optimal' )
    return two_body_mo



def write_gnu(strng,file):
    base_dir = os.path.dirname(file)
    mol_name = os.path.basename(file)
    output_path = os.path.join(base_dir, 'Gnuplots', f'gnuplot_script_{mol_name}')
    f=open(output_path,'w')
    f.write("#simulated spectrum\n")
    f.write("set term pdf size 6,4\n")
    f.write("unset key\n")
    f.write("set output '%s.pdf'\n" %(file))
    f.write("set xrange [200:700]\n")
    f.write("set samples 10000\n")
    f.write("set xlabel 'Wavelength / nm' font ',18'\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units' font ',18'\n")
    f.write("set xtics font ',18'\n")
    f.write("set ytics font ',18'\n")
    f.write("set bmargin 4\n")
    f.write("p %s lw 3 dt 1" %strng)
    f.close()
    return

   

def diagonalise_xcis(ham_blocks, rng, nstates, out, ci_level):
    """
    Diagonalise the XCIS Hamiltonian by exploiting its block-diagonal structure
    (singlet / triplet / quintet blocks), then merge and sort the results by
    ascending energy.

    Args:
        ham_rot  : (nstates, nstates) ndarray — full XCIS Hamiltonian
        ndocc    : int — number of doubly-occupied orbitals
        nvirt    : int — number of virtual orbitals
        rng      : int — number of lowest states requested (sparse path if rng < nstates)
        nstates  : int — total number of states
        out      : file handle for log output
        ci_type  : str — type of CI calculation ('XCIS' or 'XCISD')

    Returns:
        ci_energies : (nstates,) or (rng,) ndarray — eigenvalues sorted low→high
        ci_coeffs   : (nstates, nstates) or (nstates, rng) ndarray — eigenvectors,
                      each column is a CI state in the full CSF basis
    """

    # Slice the three diagonal blocks
    H_s = ham_blocks[0]
    H_t = ham_blocks[1]
    n_singlet = H_s.shape[0]
    n_triplet = H_t.shape[0]
    
    if ci_level > 1:
        H_q = ham_blocks[2]
        n_quintet = H_q.shape[0]

    # ------------------------------------------------------------------
    # 1. Diagonalise each block
    # ------------------------------------------------------------------
    if rng < nstates:
        # Sparse path — request enough states from each block.
        # We over-request proportionally then trim after merging.
        # At minimum request 1 from each block, at most the full block size.
        k_s = max(1, min(n_singlet - 1, int(np.ceil(rng * n_singlet / nstates)) + 10))
        k_t = max(1, min(n_triplet - 1, int(np.ceil(rng * n_triplet / nstates)) + 10))
        k_q = max(1, min(n_quintet - 1, int(np.ceil(rng * n_quintet / nstates)) + 10))

        msg = (
            f"Using sparse solver (eigsh) — requesting "
            f"{k_s} singlets, {k_t} triplets, {k_q} quintets "
            f"(targeting {rng} states total)\n"
        )
        print(msg)
        out.write(msg)

        e_s, v_s = sp.eigsh(H_s, k=k_s, which="SA")
        e_t, v_t = sp.eigsh(H_t, k=k_t, which="SA")
        if ci_level > 1:
            e_q, v_q = sp.eigsh(H_q, k=k_q, which="SA")

    else:
        # Dense path — full diagonalisation of each block
        msg = "Using dense solver (eigh) on each block ...\n"
        print(msg)
        out.write(msg)

        e_s, v_s = linalg.eigh(H_s)
        e_t, v_t = linalg.eigh(H_t)
        if ci_level > 1:
            e_q, v_q = linalg.eigh(H_q)

    # ------------------------------------------------------------------
    # 2. Embed block eigenvectors into the full CSF basis
    #    Each column of ci_coeffs_block is a state vector of length nstates,
    #    with zeros outside the relevant block.
    # ------------------------------------------------------------------
    def embed(v, start, total):
        """Pad eigenvector matrix v into the full basis of size `total`."""
        n_basis, n_vecs = v.shape
        full = np.zeros((total, n_vecs))
        full[start:start + n_basis, :] = v
        return full          # shape: (nstates, n_vecs)

    V_s = embed(v_s, 0, nstates)   # (nstates, k_s or n_singlet)
    V_t = embed(v_t, n_singlet, nstates)   # (nstates, k_t or n_triplet)
    if ci_level > 1:
        V_q = embed(v_q, n_singlet+n_triplet, nstates)   # (nstates, k_q or n_quintet)

    # ------------------------------------------------------------------
    # 3. Concatenate all eigenvalues/vectors and sort by energy
    # ------------------------------------------------------------------
    
    if ci_level > 1:
        all_energies = np.concatenate([e_s, e_t, e_q])
        all_coeffs   = np.concatenate([V_s, V_t, V_q], axis=1)  # (nstates, total_vecs)
    else:
        all_energies = np.concatenate([e_s, e_t])
        all_coeffs   = np.concatenate([V_s, V_t], axis=1)  # (nstates, total_vecs)
        
    sort_idx = np.argsort(all_energies)
    all_energies = all_energies[sort_idx]
    all_coeffs   = all_coeffs[:, sort_idx]

    # ------------------------------------------------------------------
    # 4. Trim to rng states if using the sparse path
    # ------------------------------------------------------------------
    if rng < nstates:
        # Guard: if we didn't get enough states across blocks, warn and use what we have
        n_available = len(all_energies)
        if n_available < rng:
            msg = (
                f"Warning: only {n_available} states available after merging blocks "
                f"(requested {rng}). Consider increasing over-request buffer.\n"
            )
            print(msg)
            out.write(msg)
            rng = n_available

        all_energies = all_energies[:rng]
        all_coeffs   = all_coeffs[:, :rng]

    ci_energies = all_energies
    ci_coeffs   = all_coeffs

    msg = f"Diagonalisation complete. Returning {ci_energies.shape[0]} states.\n"
    print(msg)
    out.write(msg)

    return ci_energies, ci_coeffs



def ci_rot(ndocc,norbs,coords,atoms,energy0,rep_tens,fock_mat,hf_orbs, file, ci_level):
    '''
    Calculates monoradical excited states in rotated (CSF) basis using the CIS or XCIS method. Used for molecules without Nitrogen or Chlorine present.
    
    Args:
        ndocc (int): Number of doubly occupied orbitals
        norbs (int): Total number of orbitals
        coords (array): Array of atomic coordinates
        atoms (array): Array of atomic symbols
        energy0 (float): Ground state energy
        rep_tens (array): 4-index two-electron repulsion integrals in MO basis
        fock_mat (array): Fock matrix in MO basis
        hf_orbs (array): HF molecular orbitals
        file (str): Name of file to write output to (without extension)

    '''
    base_dir = os.path.dirname(file)
    mol_name = os.path.basename(file)
    output_path = os.path.join(base_dir, 'Excited_States', f'{mol_name}_excitedstates.xyz')
    with open(output_path,'w') as out:
        print("")
        print("------------------------")
        print("Starting ExROPPP calculation for diradical in rotated basis")
        print("------------------------\n")

        out.write("")
        out.write("------------------------")
        out.write("Starting ExROPPP calculation for diradical in rotated basis")
        out.write("------------------------\n")
        
        # Construct CIS Hamiltonian
        ham_rot, ham_blocks = get_full_CIMatrix(ndocc, norbs, energy0, fock_mat, rep_tens, ci_level)
            
        print('Dimensions of CI matrix:', ham_rot.shape)
        print("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... ")
        print("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))

        out.write("Checking that the Hamiltonian is symmetric (a value of zero means matrix is symmetric) ... \n")
        out.write("Frobenius norm of matrix - matrix transpose = %f.\n" %(linalg.norm(ham_rot-ham_rot.T)))
        
        # Print energies of CSFs
        #print_csf_info(out, ham_rot, ndocc, norbs, ci_level)
        
        # Set rng and cutoff_energy
        nstates = ham_rot.shape[0]
        if states_cutoff_option == 'states' and states_to_print <= nstates:
            rng = states_to_print
            print('Lowest %d states. WARNING - Some states may not be included in the spectrum.\n'%states_to_print)
            out.write('Lowest %d states. WARNING - Some states may not be included in the spectrum.\n'%states_to_print)
        else:
            rng = nstates
        if states_cutoff_option == 'energy':
            cutoff_energy = energy_cutoff
            print('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
            out.write('Used energy cutoff of %04.2f eV for states. WARNING - Some states may not be included in spectrum.\n'%cutoff_energy)
        else:
            cutoff_energy = 100
        
        ci_energies, ci_coeffs = diagonalise_xcis(ham_blocks, rng, nstates, out, ci_level)

        # Calculate transition dipole moment matrix
        dip_array = get_full_TDM(ndocc, norbs, coords, hf_orbs, ci_level)[0]
        
        
        print("Summary of reference states before CI")
        
        print("<OS1|H|OS1> = ", ham_rot[0,0])
        print("<OS1|H|ZW-> = ", ham_rot[0,1])
        print("<OS1|H|ZW+> = ", ham_rot[0,2])
        print("<ZW-|H|ZW-> = ", ham_rot[1,1])
        print("<ZW-|H|ZW+> = ", ham_rot[1,2])
        print("<ZW+|H|ZW+> = ", ham_rot[2,2])
        
        
        print("<OS1|mu|OS1> = ", dip_array[0,0,:])
        print("<OS1|mu|ZW-> = ", dip_array[0,1,:])
        print("<OS1|mu|ZW+> = ", dip_array[0,2,:])
        print("<ZW-|mu|ZW-> = ", dip_array[1,1,:])
        print("<ZW-|mu|ZW+> = ", dip_array[1,2,:])
        print("<ZW+|mu|ZW+> = ", dip_array[2,2,:])
        
        nvirt = norbs - ndocc - 2
        
        print("")
        print("Summary of low-energy single excitations before CI")
        print("<DS(1-->0)|H|DS(1-->0)> = ", ham_rot[ndocc+2,ndocc+2])
        print("<DS(1-->0)|H|OS1> = ", ham_rot[ndocc+2,0])
        print("<DS(1-->0)|H|ZW-> = ", ham_rot[ndocc+2,1])
        print("<DS(1-->0)|H|ZW+> = ", ham_rot[ndocc+2,2])
        print("<DS(2-->0)|H|DS(2-->0)> = ", ham_rot[ndocc+1,ndocc+1])
        print("<DS(2-->0)|H|OS1> = ", ham_rot[ndocc+1,0])
        print("<DS(2-->0)|H|ZW-> = ", ham_rot[ndocc+1,1])
        print("<DS(2-->0)|H|ZW+> = ", ham_rot[ndocc+1,2])
        print("<DS(3-->0)|H|DS(3-->0)> = ", ham_rot[ndocc,ndocc])
        print("<DS(3-->0)|H|OS1> = ", ham_rot[ndocc,0])
        print("<DS(3-->0)|H|ZW-> = ", ham_rot[ndocc,1])
        print("<DS(3-->0)|H|ZW+> = ", ham_rot[ndocc,2])
        print("")
        print("<DS(1-->0')|H|DS(1-->0')> = ", ham_rot[2*ndocc+2,2*ndocc+2])
        print("<DS(1-->0')|H|OS1> = ", ham_rot[2*ndocc+2,0])
        print("<DS(1-->0')|H|ZW-> = ", ham_rot[2*ndocc+2,1])
        print("<DS(1-->0')|H|ZW+> = ", ham_rot[2*ndocc+2,2])
        print("<DS(2-->0')|H|DS(2-->0')> = ", ham_rot[2*ndocc+1,2*ndocc+1])
        print("<DS(2-->0')|H|OS1> = ", ham_rot[2*ndocc+1,0])
        print("<DS(2-->0')|H|ZW-> = ", ham_rot[2*ndocc+1,1])
        print("<DS(2-->0')|H|ZW+> = ", ham_rot[2*ndocc+1,2])
        print("<DS(3-->0')|H|DS(3-->0')> = ", ham_rot[2*ndocc,2*ndocc])
        print("<DS(3-->0')|H|OS1> = ", ham_rot[2*ndocc,0])
        print("<DS(3-->0')|H|ZW-> = ", ham_rot[2*ndocc,1])
        print("<DS(3-->0')|H|ZW+> = ", ham_rot[2*ndocc,2])
        print("")
        print("<SV(0-->1')|H|SV(0-->1')> = ", ham_rot[2*ndocc+3,2*ndocc+3])
        print("<SV(0-->1')|H|OS1> = ", ham_rot[2*ndocc+3,0])
        print("<SV(0-->1')|H|ZW-> = ", ham_rot[2*ndocc+3,1])
        print("<SV(0-->1')|H|ZW+> = ", ham_rot[2*ndocc+3,2])
        print("<SV(0-->2')|H|SV(0-->2')> = ", ham_rot[2*ndocc+4,2*ndocc+4])
        print("<SV(0-->2')|H|OS1> = ", ham_rot[2*ndocc+4,0])
        print("<SV(0-->2')|H|ZW-> = ", ham_rot[2*ndocc+4,1])
        print("<SV(0-->2')|H|ZW+> = ", ham_rot[2*ndocc+4,2])
        print("<SV(0-->3')|H|SV(0-->3')> = ", ham_rot[2*ndocc+5,2*ndocc+5])
        print("<SV(0-->3')|H|OS1> = ", ham_rot[2*ndocc+5,0])
        print("<SV(0-->3')|H|ZW-> = ", ham_rot[2*ndocc+5,1])
        print("<SV(0-->3')|H|ZW+> = ", ham_rot[2*ndocc+5,2])
        print("")
        print("<SV(0'-->1')|H|SV(0'-->1')> = ", ham_rot[nvirt+2*ndocc+3,nvirt+2*ndocc+3])
        print("<SV(0'-->1')|H|OS1> = ", ham_rot[nvirt+2*ndocc+3,0])
        print("<SV(0'-->1')|H|ZW-> = ", ham_rot[nvirt+2*ndocc+3,1])
        print("<SV(0'-->1')|H|ZW+> = ", ham_rot[nvirt+2*ndocc+3,2])
        print("<SV(0'-->2')|H|SV(0-->2')> = ", ham_rot[nvirt+2*ndocc+4,nvirt+2*ndocc+4])
        print("<SV(0'-->2')|H|OS1> = ", ham_rot[nvirt+2*ndocc+4,0])
        print("<SV(0'-->2')|H|ZW-> = ", ham_rot[nvirt+2*ndocc+4,1])
        print("<SV(0'-->2')|H|ZW+> = ", ham_rot[nvirt+2*ndocc+4,2])
        print("<SV(0'-->3')|H|SV(0-->3')> = ", ham_rot[nvirt+2*ndocc+5,nvirt+2*ndocc+5])
        print("<SV(0'-->3')|H|OS1> = ", ham_rot[nvirt+2*ndocc+5,0])
        print("<SV(0'-->3')|H|ZW-> = ", ham_rot[nvirt+2*ndocc+5,1])
        print("<SV(0'-->3')|H|ZW+> = ", ham_rot[nvirt+2*ndocc+5,2])
        print("")
        print("<DS(1-->0)|mu|OS1> = ", dip_array[ndocc+2,0])
        print("<DS(1-->0)|mu|ZW-> = ", dip_array[ndocc+2,1])
        print("<DS(1-->0)|mu|ZW+> = ", dip_array[ndocc+2,2])
        print("<DS(2-->0)|mu|OS1> = ", dip_array[ndocc+1,0])
        print("<DS(2-->0)|mu|ZW-> = ", dip_array[ndocc+1,1])
        print("<DS(2-->0)|mu|ZW+> = ", dip_array[ndocc+1,2])
        print("<DS(3-->0)|mu|OS1> = ", dip_array[ndocc,0])
        print("<DS(3-->0)|mu|ZW-> = ", dip_array[ndocc,1])
        print("<DS(3-->0)|mu|ZW+> = ", dip_array[ndocc,2])
        print("")
        print("<DS(1-->0')|mu|OS1> = ", dip_array[2*ndocc+2,0])
        print("<DS(1-->0')|mu|ZW-> = ", dip_array[2*ndocc+2,1])
        print("<DS(1-->0')|mu|ZW+> = ", dip_array[2*ndocc+2,2])
        print("<DS(2-->0')|mu|OS1> = ", dip_array[2*ndocc+1,0])
        print("<DS(2-->0')|mu|ZW-> = ", dip_array[2*ndocc+1,1])
        print("<DS(2-->0')|mu|ZW+> = ", dip_array[2*ndocc+1,2])
        print("<DS(3-->0')|mu|OS1> = ", dip_array[2*ndocc,0])
        print("<DS(3-->0')|mu|ZW-> = ", dip_array[2*ndocc,1])
        print("<DS(3-->0')|mu|ZW+> = ", dip_array[2*ndocc,2])
        print("")
        print("<SV(0-->1')|mu|OS1> = ", dip_array[2*ndocc+3,0])
        print("<SV(0-->1')|mu|ZW-> = ", dip_array[2*ndocc+3,1])
        print("<SV(0-->1')|mu|ZW+> = ", dip_array[2*ndocc+3,2])
        print("<SV(0-->2')|mu|OS1> = ", dip_array[2*ndocc+4,0])
        print("<SV(0-->2')|mu|ZW-> = ", dip_array[2*ndocc+4,1])
        print("<SV(0-->2')|mu|ZW+> = ", dip_array[2*ndocc+4,2])
        print("<SV(0-->3')|mu|OS1> = ", dip_array[2*ndocc+5,0])
        print("<SV(0-->3')|mu|ZW-> = ", dip_array[2*ndocc+5,1])
        print("<SV(0-->3')|mu|ZW+> = ", dip_array[2*ndocc+5,2])
        print("")
        print("<SV(0'-->1')|mu|OS1> = ", dip_array[nvirt+2*ndocc+3,0])
        print("<SV(0'-->1')|mu|ZW-> = ", dip_array[nvirt+2*ndocc+3,1])
        print("<SV(0'-->1')|mu|ZW+> = ", dip_array[nvirt+2*ndocc+3,2])
        print("<SV(0'-->2')|mu|OS1> = ", dip_array[nvirt+2*ndocc+4,0])
        print("<SV(0'-->2')|mu|ZW-> = ", dip_array[nvirt+2*ndocc+4,1])
        print("<SV(0'-->2')|mu|ZW+> = ", dip_array[nvirt+2*ndocc+4,2])
        print("<SV(0'-->3')|mu|OS1> = ", dip_array[nvirt+2*ndocc+5,0])
        print("<SV(0'-->3')|mu|ZW-> = ", dip_array[nvirt+2*ndocc+5,1])
        print("<SV(0'-->3')|mu|ZW+> = ", dip_array[nvirt+2*ndocc+5,2])


        
        print("Checking that the Dipole matrix is symmetric (a value of zero means matrix is symmetric) ... ")
        print(f"Frobenius norm of matrix - matrix transpose = {linalg.norm(dip_array[:, :, 0]-dip_array[:,:,0].T):.5f} \
            {linalg.norm(dip_array[:, :, 1]-dip_array[:,:,1].T):.5f}, {linalg.norm(dip_array[:, :, 2]-dip_array[:,:,2].T):.5f}.\n")
        out.write("Checking that the Dipole matrix is symmetric (a value of zero means matrix is symmetric) ... \n")
        out.write(f"Frobenius norm of matrix - matrix transpose = {linalg.norm(dip_array[:, :, 0]-dip_array[:,:,0].T):.5f} \
            {linalg.norm(dip_array[:, :, 1]-dip_array[:,:,1].T):.5f}, {linalg.norm(dip_array[:, :, 2]-dip_array[:,:,2].T):.5f}.\n")
        
        dip_couplings = np.einsum("ijx,jk",dip_array,ci_coeffs)
        state0_tdms = np.einsum("j,jix",ci_coeffs[:,0].T, dip_couplings)
        state1_tdms = np.einsum("j,jix",ci_coeffs[:,1].T, dip_couplings)
        tdms = (state0_tdms, state1_tdms) 
        
        # Print information about CI states
        strngs, osc_arrays = print_ci_info(out, ci_energies, ci_coeffs, ndocc, norbs, tdms, dip_array, rng, cutoff_energy, ci_level, csf_tol=0.05)
        strngs = (strngs[0][1:], strngs[1][1:])
    return strngs, ci_energies - ci_energies[0], osc_arrays





def rad_calc(file,params,rotation_matrix=None,converged_orbs=None):
    filename = os.path.basename(file)
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy0,rep_tens,hf_orbs,fock_mat = main_scf(file,params,rotation_matrix,converged_orbs)
    com,coord = re_center(coord,atoms_array,coord_w_h)
    natoms=np.shape(coord)[0]

    hf_orbs, fock_mo, order = sort_orbitals_by_fock_energy(hf_orbs, fock_mat)
    rep_tens = reorder_mo_tensor(rep_tens, order)
    print('\nOrbitals reordered by Fock-matrix diagonal (orig idx -> new idx):')
    for new_i, old_i in enumerate(order):
        marker = '  <-- moved' if new_i != old_i else ''
        print("  %d -> %d%s" % (old_i+1, new_i+1, marker))
    
    dens_mat = density(hf_orbs, ndocc)
    dens_mo = np.dot(hf_orbs.T, np.dot(dens_mat, hf_orbs))
    print('\nOrbital occupation numbers:')
    for i in range(dens_mo.shape[0]):
        print("%d: %f"%(i+1,dens_mo[i,i]))
    strngs, ci_energies_array, osc_arrays = ci_rot(ndocc, natoms, coord, atoms_array, energy0, rep_tens, fock_mo, hf_orbs, file, ci_level=3)
    return strngs, ci_energies_array, osc_arrays


def sort_orbitals_by_fock_energy(hf_orbs, fock_mat):
    """
    Reorder MOs by the diagonal elements of the Fock matrix in the MO basis.

    Orbitals coming from an SCF that was optimized for an open-shell
    singlet diradical aren't guaranteed to come out in aufbau (energy)
    order. Downstream code that assumes orbital index <-> occupation
    (e.g. "first ndocc columns are doubly occupied") can then silently
    swap occupied/virtual character. Diagonalizing F in the current MO
    basis and sorting by the diagonal gives an approximate orbital-energy
    ordering (exact if F is already diagonal in this basis, i.e. these
    are true canonical orbitals; approximate otherwise, but still a good
    proxy for occupied-vs-virtual character since occupied orbitals sit
    much lower than virtuals).

    Parameters
    ----------
    hf_orbs : ndarray, shape (natoms, natoms)
        MO coefficients, one orbital per column, in their original order.
    fock_mat : ndarray, shape (natoms, natoms)
        Fock matrix in the AO basis.

    Returns
    -------
    hf_orbs_sorted : ndarray, shape (natoms, natoms)
        Columns of hf_orbs reordered by increasing Fock-diagonal "energy".
    fock_mo_sorted : ndarray, shape (natoms, natoms)
        Fock matrix in the MO basis, rows/cols permuted to match.
    order : ndarray, shape (natoms,)
        Permutation applied, i.e. hf_orbs_sorted[:, k] = hf_orbs[:, order[k]].
    """
    fock_mo = np.dot(hf_orbs.T, np.dot(fock_mat, hf_orbs))
    orb_energies = np.diag(fock_mo)

    # stable sort so that near-degenerate orbitals (equal to within
    # numerical noise) keep their relative original order rather than
    # being shuffled arbitrarily
    order = np.argsort(orb_energies, kind='stable')

    hf_orbs_sorted = hf_orbs[:, order]
    fock_mo_sorted = fock_mo[np.ix_(order, order)]
    return hf_orbs_sorted, fock_mo_sorted, order

def reorder_mo_tensor(tensor, order):
    """
    Permute every axis of an MO-indexed array/tensor by the same
    orbital ordering. Use this for anything whose indices all run
    over orbitals -- e.g. a 4-index two-electron integral tensor
    (pq|rs) in the MO basis -- so that after sorting the orbitals,
    integral element [i,j,k,l] still refers to the same four
    (relabelled) orbitals it did before sorting.

    Parameters
    ----------
    tensor : ndarray, shape (natoms,)*ndim
        MO-basis tensor, e.g. two-electron repulsion integrals.
    order : ndarray, shape (natoms,)
        Permutation from sort_orbitals_by_fock_energy, i.e. the same
        one applied to hf_orbs' columns.

    Returns
    -------
    ndarray, same shape as tensor, with every axis permuted by order.
    """
    idx = np.ix_(*([order] * tensor.ndim))
    return tensor[idx]