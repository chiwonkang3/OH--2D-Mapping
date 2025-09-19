# -*- coding: utf-8 -*-
"""
oh_npohm_mapper_with_zero_local.py
- Saves all outputs to the *current working directory* (same folder as this script).
- Includes j = 0.00 A/cm^2 plus {-0.25, -0.50, -0.75, -1.00}.
- Closed-cell NP–Ohm with two-sided line sources/sinks (physics), but draw only one centerline per electrode (graphics).
- Produces common-scale maps (0..max at -1.00), and y=1.25 cm profiles; omits labels/titles.
"""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Physical constants
F,R,T = 96485.0, 8.314, 298.15

# Domain/grid
Lx,Ly = 2.5e-2, 2.5e-2
Nx,Ny = 151,151
dx,dy = Lx/(Nx-1), Ly/(Ny-1)
x = np.linspace(0,Lx,Nx); y = np.linspace(0,Ly,Ny)

# Electrolyte & properties
c_bulk = 1000.0     # mol/m^3 (1 M)
D0 = 5.27e-9        # m^2/s
kappa0 = 30.0       # S/m
beta_D = 0.0
beta_kappa = 0.25
eddy = 10.0         # mixing augmentation (e.g., 300 rpm)

# Geometry (cm → m)
cm = 1e-2
cx_pt, cx_ni = 0.25*cm, 2.25*cm
t = 0.20*cm
y1, y2 = 0.75*cm, 1.75*cm
faces_pt = [cx_pt - t/2, cx_pt + t/2]
faces_ni = [cx_ni - t/2, cx_ni + t/2]
j_y = np.where((y>=y1)&(y<=y2))[0]

# Currents (A/cm^2) including 0
currents = [0.00, -0.25, -0.50, -0.75, -1.00]

# Flow (constant; wall-normal set to zero)
u_flow_x_const = 0.0
u_flow_y_const = 0.0

# Iterations
outer_loops = 4
max_iters_phi = 800
max_iters_c = 1200
tol_phi = 5e-6
tol_c = 5e-6
omega_phi = 0.8
omega_c = 0.8

def clamp_pos(a, eps=1e-9):
    a = np.asarray(a); a[a<eps] = eps; return a

def D_mol_c(c):
    return D0 * (clamp_pos(c)/c_bulk)**beta_D

def kappa_c(c):
    return kappa0 * (clamp_pos(c)/c_bulk)**beta_kappa

def build_species_source(jabs):
    s = np.zeros((Ny,Nx))
    if jabs==0.0: return s
    rate = (jabs/F)/dx
    for xf in faces_pt + faces_ni:
        i = int(round(xf/dx)); i = max(0, min(Nx-1, i))
        if xf in faces_pt: s[j_y,i] += rate  # WE generation
        else:              s[j_y,i] -= rate  # CE consumption
    return s

def build_current_source(jabs):
    s = np.zeros((Ny,Nx))
    if jabs==0.0: return s
    rate = (jabs)/dx
    for xf in faces_pt + faces_ni:
        i = int(round(xf/dx)); i = max(0, min(Nx-1, i))
        if xf in faces_pt: s[j_y,i] += rate  # inject at WE
        else:              s[j_y,i] -= rate  # extract at CE
    return s

def solve_phi(kappa_field, s_phi, phi0=None):
    phi = np.zeros((Ny,Nx)) if phi0 is None else phi0.copy()
    inv_dx2, inv_dy2 = 1.0/dx**2, 1.0/dy**2
    for _ in range(max_iters_phi):
        kE = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[1:-1,2:])
        kW = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[1:-1,0:-2])
        kN = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[0:-2,1:-1])
        kS = 0.5*(kappa_field[1:-1,1:-1] + kappa_field[2:  ,1:-1])

        phiE = phi[1:-1,2:]; phiW = phi[1:-1,0:-2]
        phiN = phi[0:-2,1:-1]; phiS = phi[2:  ,1:-1]

        num = (kE*phiE + kW*phiW)*inv_dx2 + (kN*phiN + kS*phiS)*inv_dy2 - s_phi[1:-1,1:-1]
        den = ((kE + kW)*inv_dx2 + (kN + kS)*inv_dy2)

        phi_new = phi.copy()
        phi_new[1:-1,1:-1] = num/den

        # Neumann(0) at outer walls
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
        cim1 = c[1:-1,0:-2]; cip1 = c[1:-1,2:]; cjm1 = c[0:-2,1:-1]; cjp1 = c[2:,1:-1]
        Dcell = D_eff[1:-1,1:-1]
        ux = u_x[1:-1,1:-1]; uy = u_y[1:-1,1:-1]
        ux_pos = np.maximum(ux,0.0); ux_neg = np.minimum(ux,0.0)
        uy_pos = np.maximum(uy,0.0); uy_neg = np.minimum(uy,0.0)
        num = (Dcell*(cip1+cim1)/dx**2
              + Dcell*(cjp1+cjm1)/dy**2
              + (ux_pos/dx)*cim1 + (-ux_neg/dx)*cip1
              + (uy_pos/dy)*cjm1 + (-uy_neg/dy)*cjp1
              + s_species[1:-1,1:-1])
        den = (2*Dcell/dx**2 + 2*Dcell/dy**2 + (np.abs(ux)/dx) + (np.abs(uy)/dy))
        c_new = c.copy(); c_new[1:-1,1:-1] = num/den
        # no-flux walls (and u·n=0 enforced externally)
        c_new[:,0] = c_new[:,1]; c_new[:,-1] = c_new[:,-2]
        c_new[0,:] = c_new[1,:]; c_new[-1,:] = c_new[-2,:]
        delta = c_new - c; c += 0.8*delta; c[c<0.0] = 0.0
        if np.max(np.abs(delta)) < 5e-6: break
    return c

def run_case(j):
    jabs = abs(j)*1e4
    s_c  = build_species_source(jabs)
    s_phi= build_current_source(jabs)
    c = np.full((Ny,Nx), c_bulk); phi = np.zeros((Ny,Nx))
    ufx = np.full((Ny,Nx), u_flow_x_const); ufy = np.full((Ny,Nx), u_flow_y_const)
    ufx[:,0]=ufx[:,-1]=0.0; ufy[0,:]=ufy[-1,:]=0.0
    for _ in range(outer_loops):
        kappa = kappa_c(c); Dmol = D_mol_c(c); Deff = eddy*Dmol
        phi = solve_phi(kappa, s_phi, phi0=phi)
        dphidx = np.zeros_like(phi); dphidy = np.zeros_like(phi)
        dphidx[:,1:-1] = (phi[:,2:] - phi[:,:-2])/(2*dx)
        dphidy[1:-1,:] = (phi[2:,:] - phi[:-2,:])/(2*dy)
        u_mx = (Dmol*F/(R*T))*dphidx; u_my = (Dmol*F/(R*T))*dphidy
        u_x = ufx + u_mx; u_y = ufy + u_my
        c = solve_c(c, Deff, u_x, u_y, s_c)
    return c, phi

def draw_map_common(c, cmax_target, path_png):
    plt.figure(figsize=(7,6))
    plt.imshow(c, origin="lower", extent=[0, Lx*100, 0, Ly*100], aspect="equal",
               vmin=0.0, vmax=cmax_target)
    cb = plt.colorbar()
    cmax_M = cmax_target/1000.0
    ticks_M = list(np.arange(0.0, np.floor(cmax_M/5.0)*5.0 + 0.1, 5.0))
    if abs(cmax_M - 5.0*round(cmax_M/5.0)) > 1e-8:
        if ticks_M and abs(ticks_M[-1] - cmax_M) > 1e-8: ticks_M.append(cmax_M)
    cb.set_ticks([t*1000.0 for t in ticks_M])
    cb.set_ticklabels([f"{t:.0f}" if abs(t-round(t))<1e-9 else f"{t:.2f}" for t in ticks_M])
    ax = plt.gca(); ax.set_xlabel(""); ax.set_ylabel(""); plt.title("")
    ax.add_line(Line2D([cx_pt*100,cx_pt*100],[y1*100,y2*100],linewidth=2.5,color='red'))
    ax.add_line(Line2D([cx_ni*100,cx_ni*100],[y1*100,y2*100],linewidth=2.5,color='red'))
    plt.tight_layout(); plt.savefig(path_png, dpi=180); plt.close()

def save_profile_y12(c, cmax_target, j, path_png, path_csv):
    y_target_cm = 1.25; y_target_m = y_target_cm*1e-2
    jrow = int(np.argmin(np.abs(y - y_target_m)))
    x_cm = np.linspace(0, Lx, Nx)*100.0
    c_row_M = c[jrow, :]/1000.0
    np.savetxt(path_csv, np.column_stack([x_cm, c_row_M]), delimiter=",", header="x_cm,c_M", comments="")
    plt.figure(figsize=(7,3))
    plt.plot(x_cm, c_row_M)
    plt.xlim(0.0, 2.5); plt.ylim(0.0, cmax_target/1000.0)
    cmax_M = cmax_target/1000.0
    ticks_M = list(np.arange(0.0, np.floor(cmax_M/5.0)*5.0 + 0.1, 5.0))
    if abs(cmax_M - 5.0*round(cmax_M/5.0)) > 1e-8:
        if ticks_M and abs(ticks_M[-1] - cmax_M) > 1e-8: ticks_M.append(cmax_M)
    plt.yticks(ticks_M, [f"{t:.0f}" if abs(t-round(t))<1e-9 else f"{t:.2f}" for t in ticks_M])
    ax = plt.gca(); ax.set_xlabel(""); ax.set_ylabel(""); plt.title("")
    plt.tight_layout(); plt.savefig(path_png, dpi=180); plt.close()

def main():
    outputs = []
    for j in currents:
        c, phi = run_case(j); outputs.append((j, c, phi))
        # save raw fields in current folder
        np.savetxt(f"oh_npohm_closed_centerline_{j:+.2f}_Acm2_c.csv".replace("+","p").replace("-","m"), c, delimiter=",")
        np.savetxt(f"oh_npohm_closed_centerline_{j:+.2f}_Acm2_phi.csv".replace("+","p").replace("-","m"), phi, delimiter=",")
    # common max from j=-1.00
    cmax_target = None
    for (j, c, phi) in outputs:
        if abs(j + 1.00) < 1e-12:
            cmax_target = float(np.max(c)); break
    if cmax_target is None:
        raise RuntimeError("j = -1.00 A/cm^2 case not found")
    # draw maps & profiles in current folder
    for (j, c, phi) in outputs:
        draw_map_common(c, cmax_target, f"map_commonScale_j_{j:+.2f}_Acm2.png".replace("+","p").replace("-","m"))
        save_profile_y12(c, cmax_target, j,
                         f"profile_y1p25cm_j_{j:+.2f}_Acm2.png".replace("+","p").replace("-","m"),
                         f"profile_y1p25cm_j_{j:+.2f}_Acm2.csv".replace("+","p").replace("-","m"))
    print("Done. Outputs saved to the current folder. Common vmax = {:.2f} M".format(cmax_target/1000.0))

if __name__ == "__main__":
    main()
