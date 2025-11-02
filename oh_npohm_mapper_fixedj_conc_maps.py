# -*- coding: utf-8 -*-
"""
oh_npohm_mapper_fixedj_conc_maps.py
-----------------------------------
Create 2D [OH-] maps for a fixed current density j = -1.00 A/cm^2
at initial KOH concentrations c0 in {1, 2, 4} M.
COMMON color scale: 0 ~ 21.21 M for all three maps.

Outputs (saved to the current folder):
  - map_fixedj_m1p00_conc{1|2|4}M.png
  - fixedj_m1p00_conc{1|2|4}M_c.csv  (concentration field, mol/m^3)
  - fixedj_m1p00_conc{1|2|4}M_phi.csv (potential field)
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

F, R, T = 96485.0, 8.314, 298.15

Lx, Ly = 2.5e-2, 2.5e-2
Nx, Ny = 151, 151
dx, dy = Lx/(Nx-1), Ly/(Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

cm = 1e-2
cx_we, cx_ce = 0.25*cm, 2.25*cm
t = 0.20*cm
y1, y2 = 0.75*cm, 1.75*cm
faces_we = [cx_we - t/2, cx_we + t/2]
faces_ce = [cx_ce - t/2, cx_ce + t/2]
j_y_idx = np.where((y >= y1) & (y <= y2))[0]

kappa0_1M = 30.0
beta_kappa = 0.25
D0 = 5.27e-9
eddy = 10.0

outer_loops, max_iters_phi, max_iters_c = 4, 800, 1200
tol_phi, tol_c = 5e-6, 5e-6
omega_phi, omega_c = 0.8, 0.8

j_A_cm2 = -1.00
j_abs_Am2 = abs(j_A_cm2) * 1e4

cmin_plot = 0.0
cmax_plot = 21.21 * 1000.0   # mol/m^3

def clamp_pos(a, eps=1e-9):
    a = np.asarray(a); a[a<eps]=eps; return a

def kappa_of_c(c):
    return kappa0_1M * (clamp_pos(c)/1000.0)**beta_kappa

def D_mol_of_c(c):
    return D0 * np.ones_like(c)

def build_species_source():
    s = np.zeros((Ny, Nx))
    if j_abs_Am2 == 0.0: return s
    rate = (j_abs_Am2/F)/dx
    for xf in faces_we + faces_ce:
        i = int(round(xf/dx)); i = max(0, min(Nx-1, i))
        if xf in faces_we: s[j_y_idx, i] += rate
        else:              s[j_y_idx, i] -= rate
    return s

def build_current_source():
    s = np.zeros((Ny, Nx))
    if j_abs_Am2 == 0.0: return s
    rate = (j_abs_Am2)/dx
    for xf in faces_we + faces_ce:
        i = int(round(xf/dx)); i = max(0, min(Nx-1, i))
        if xf in faces_we: s[j_y_idx, i] += rate
        else:              s[j_y_idx, i] -= rate
    return s

def solve_phi(kappa_field, s_phi, phi0=None):
    phi = np.zeros((Ny, Nx)) if phi0 is None else phi0.copy()
    inv_dx2, inv_dy2 = 1.0/dx**2, 1.0/dy**2
    for _ in range(max_iters_phi):
        kE = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[1:-1,2:])
        kW = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[1:-1,0:-2])
        kN = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[0:-2,1:-1])
        kS = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[2:,1:-1])

        phiE = phi[1:-1,2:]; phiW = phi[1:-1,0:-2]
        phiN = phi[0:-2,1:-1]; phiS = phi[2:,1:-1]

        num = (kE*phiE + kW*phiW)*inv_dx2 + (kN*phiN + kS*phiS)*inv_dy2 - s_phi[1:-1,1:-1]
        den = ((kE + kW)*inv_dx2 + (kN + kS)*inv_dy2)

        phi_new = phi.copy()
        phi_new[1:-1,1:-1] = num/den

        phi_new[:,0]  = phi_new[:,1]
        phi_new[:,-1] = phi_new[:,-2]
        phi_new[0,:]  = phi_new[1,:]
        phi_new[-1,:] = phi_new[-2,:]

        delta = phi_new - phi
        phi += omega_phi*delta
        if np.max(np.abs(delta)) < tol_phi: break
    phi -= phi.mean()
    return phi

def solve_c(c_init, D_eff, u_x, u_y, s_species):
    c = c_init.copy()
    for _ in range(max_iters_c):
        cim1 = c[1:-1,0:-2];  cip1 = c[1:-1,2:]
        cjm1 = c[0:-2,1:-1];  cjp1 = c[2:,1:-1]
        Dcell = D_eff[1:-1,1:-1]
        ux = u_x[1:-1,1:-1];  uy = u_y[1:-1,1:-1]
        ux_pos = np.maximum(ux,0.0);  ux_neg = np.minimum(ux,0.0)
        uy_pos = np.maximum(uy,0.0);  uy_neg = np.minimum(uy,0.0)

        num = (Dcell*(cip1+cim1)/dx**2
             + Dcell*(cjp1+cjm1)/dy**2
             + (ux_pos/dx)*cim1 + (-ux_neg/dx)*cip1
             + (uy_pos/dy)*cjm1 + (-uy_neg/dy)*cjp1
             + s_species[1:-1,1:-1])

        den = (2*Dcell/dx**2 + 2*Dcell/dy**2 + (np.abs(ux)/dx) + (np.abs(uy)/dy))

        c_new = c.copy()
        c_new[1:-1,1:-1] = num/den

        c_new[:,0]  = c_new[:,1]
        c_new[:,-1] = c_new[:,-2]
        c_new[0,:]  = c_new[1,:]
        c_new[-1,:] = c_new[-2,:]

        delta = c_new - c
        c += 0.8*delta
        c[c<0.0] = 0.0
        if np.max(np.abs(delta)) < 5e-6: break
    return c

def draw_map_fixedscale(c, path_png):
    plt.figure(figsize=(7,6))
    plt.imshow(c, origin="lower", extent=[0, Lx*100, 0, Ly*100], aspect="equal",
               vmin=cmin_plot, vmax=cmax_plot)
    cb = plt.colorbar()
    ticks_M = [0.0, 5.0, 10.0, 15.0, 20.0, 21.21]
    cb.set_ticks([t*1000.0 for t in ticks_M])
    cb.set_ticklabels([f"{t:g}" for t in ticks_M])
    ax = plt.gca(); ax.set_xlabel(""); ax.set_ylabel(""); plt.title("")
    ax.add_line(Line2D([cx_we*100, cx_we*100], [y1*100, y2*100], linewidth=2.5, color='red'))
    ax.add_line(Line2D([cx_ce*100, cx_ce*100], [y1*100, y2*100], linewidth=2.5, color='red'))
    plt.tight_layout(); plt.savefig(path_png, dpi=180); plt.close()

def run_case(c0_M):
    c0 = c0_M * 1000.0
    c   = np.full((Ny, Nx), c0)
    phi = np.zeros((Ny, Nx))
    s_species = build_species_source()
    s_phi     = build_current_source()

    for _ in range(outer_loops):
        kappa = kappa_of_c(c)
        Dmol  = D_mol_of_c(c)
        D_eff = eddy * Dmol

        phi = solve_phi(kappa, s_phi, phi0=phi)

        dphidx = np.zeros_like(phi); dphidy = np.zeros_like(phi)
        dphidx[:,1:-1] = (phi[:,2:] - phi[:,:-2])/(2*dx)
        dphidy[1:-1,:] = (phi[2:,:] - phi[:-2,:])/(2*dy)
        u_mig_x = (Dmol*F/(R*T))*dphidx
        u_mig_y = (Dmol*F/(R*T))*dphidy

        u_x = u_mig_x; u_y = u_mig_y
        c = solve_c(c, D_eff, u_x, u_y, s_species)
    return c, phi

def main():
    conc_list = [1.0, 2.0, 4.0]
    for cM in conc_list:
        c, phi = run_case(cM)
        np.savetxt(f"fixedj_m1p00_conc{int(cM)}M_c.csv",  c,  delimiter=",")
        np.savetxt(f"fixedj_m1p00_conc{int(cM)}M_phi.csv", phi, delimiter=",")
        draw_map_fixedscale(c, f"map_fixedj_m1p00_conc{int(cM)}M.png")
    print("Done: maps saved with common scale 0–21.21 M.")

if __name__ == "__main__":
    main()
