# -*- coding: utf-8 -*-
"""
OH- 2D steady-state concentration mapping for a 2.5 cm × 2.5 cm cell
with finite-thickness internal electrodes (Pt working and Ni foam counter).

Model (steady):
    div( -D_eff grad(c) + u_eff c ) + S(x,y) = 0
    u_eff = u_flow + u_mig x-hat,  where  u_mig = (D_eff*F/(R*T))*(|j|/kappa)
Source term S is a volumetric representation of interfacial flux across
electrode strips of thickness w:
    S = (+|j|/F)/w  in the Pt strip (OH- generation)
    S = (-|j|/F)/w  in the Ni strip (OH- consumption)
Outer boundary condition: c = c_bulk on all four edges (well-stirred reservoir).

Numerics:
    - Uniform grid, finite-difference (Jacobi-like update with upwind convection)
    - One figure per current density; electrodes outlined in red
Dependencies: numpy, matplotlib
"""
import json, os, numpy as np, matplotlib.pyplot as plt

F, R, T = 96485.0, 8.314, 298.15

def rect_mask(x, y, x0, x1, y0, y1):
    ix = np.where((x >= x0) & (x <= x1))[0]
    iy = np.where((y >= y0) & (y <= y1))[0]
    m = np.zeros((y.size, x.size), dtype=bool)
    if ix.size>0 and iy.size>0:
        m[np.ix_(iy, ix)] = True
    return m

def draw_electrodes(ax, geom_cm):
    import matplotlib.patches as patches
    # Pt strip
    ax.add_patch(patches.Rectangle(
        (geom_cm["pt_x0_cm"], geom_cm["pt_y0_cm"]),
        geom_cm["pt_thick_cm"], geom_cm["pt_y1_cm"]-geom_cm["pt_y0_cm"],
        fill=False, edgecolor='red', linewidth=2))
    # Ni strip
    ax.add_patch(patches.Rectangle(
        (geom_cm["ni_x0_cm"], geom_cm["ni_y0_cm"]),
        geom_cm["ni_thick_cm"], geom_cm["ni_y1_cm"]-geom_cm["ni_y0_cm"],
        fill=False, edgecolor='red', linewidth=2))

def solve_case(cfg, j_A_per_m2, x, y):
    # Unpack
    D_eff   = cfg["transport"]["D_eff_m2_per_s"]
    kappa   = cfg["transport"]["kappa_S_per_m"]
    c_bulk  = cfg["electrolyte"]["c_bulk_mol_per_m3"]
    u_flow  = cfg["transport"].get("u_flow_x_m_per_s", 0.0)

    # Geometry in meters
    pt_x0, pt_w  = cfg["geom"]["pt_x0_cm"]*1e-2, cfg["geom"]["pt_thick_cm"]*1e-2
    pt_y0, pt_y1 = cfg["geom"]["pt_y0_cm"]*1e-2, cfg["geom"]["pt_y1_cm"]*1e-2
    ni_x0, ni_w  = cfg["geom"]["ni_x0_cm"]*1e-2, cfg["geom"]["ni_thick_cm"]*1e-2
    ni_y0, ni_y1 = cfg["geom"]["ni_y0_cm"]*1e-2, cfg["geom"]["ni_y1_cm"]*1e-2

    Nx, Ny = x.size, y.size
    dx, dy = (x[-1]-x[0])/(Nx-1), (y[-1]-y[0])/(Ny-1)
    Ddx2, Ddy2 = D_eff/dx**2, D_eff/dy**2

    # Masks
    pt_mask = rect_mask(x, y, pt_x0, pt_x0+pt_w, pt_y0, pt_y1)
    ni_mask = rect_mask(x, y, ni_x0, ni_x0+ni_w, ni_y0, ni_y1)

    # Migration drift speed and effective velocity
    u_mig = D_eff * F/(R*T) * (abs(j_A_per_m2)/kappa)
    u_x, u_y = u_flow + u_mig, 0.0
    uxdx, uydy = u_x/dx, u_y/dy
    denom = 2*Ddx2 + 2*Ddy2 + max(u_x,0.0)/dx + max(u_y,0.0)/dy
    if denom == 0.0: denom = 1.0

    # Volumetric sources
    r_surf = abs(j_A_per_m2)/F  # mol/m^2/s
    S = np.zeros((Ny, Nx), dtype=float)
    S[pt_mask] += r_surf/pt_w
    S[ni_mask] -= r_surf/ni_w

    # Solve (vectorized Jacobi-like)
    c = np.full((Ny, Nx), c_bulk, float)
    tol, max_iters, omega = cfg["numerics"]["tolerance"], cfg["numerics"]["max_iters"], cfg["numerics"]["omega"]
    for it in range(max_iters):
        # Dirichlet boundaries
        c[0,:] = c[-1,:] = c_bulk
        c[:,0] = c[:,-1] = c_bulk

        C  = c[1:-1,1:-1]
        CE = c[1:-1,2:]
        CW = c[1:-1,0:-2]
        CN = c[2:,1:-1]
        CS = c[0:-2,1:-1]
        SS = S[1:-1,1:-1]

        num = (Ddx2*(CE+CW) + Ddy2*(CN+CS) + uxdx*CW + uydy*CS + SS)
        c_new = num/denom
        diff = c_new - C
        c[1:-1,1:-1] = C + omega*diff
        if np.max(np.abs(diff)) < tol:
            break
    Pe = u_x*(x[-1]-x[0])/D_eff if D_eff>0 else float('inf')
    return c, u_mig, Pe, it+1, S

def main(config_path="oh_config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Grid (meters)
    Lx, Ly = cfg["geom"]["Lx_cm"]*1e-2, cfg["geom"]["Ly_cm"]*1e-2
    Nx, Ny = cfg["grid"]["Nx"], cfg["grid"]["Ny"]
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    out_dir = cfg["output"]["folder"]
    os.makedirs(out_dir, exist_ok=True)

    # For plots: extent in cm
    extent_cm = [0, cfg["geom"]["Lx_cm"], 0, cfg["geom"]["Ly_cm"]]

    # Loop over current densities (A/cm^2)
    results = []
    for j_cm2 in cfg["cases"]["j_list_A_per_cm2"]:
        j = j_cm2*1e4  # A/m^2
        c_map, u_mig, Pe, iters, S = solve_case(cfg, j, x, y)
        results.append((j_cm2, c_map, u_mig, Pe, iters))

        # Plot
        plt.figure(figsize=(6.5,6))
        plt.imshow(c_map, origin="lower", extent=extent_cm, aspect="equal")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title(f"OH$^-$ concentration (mol/m³) | j={j_cm2:.2f} A/cm²\n"
                  f"D_eff={cfg['transport']['D_eff_m2_per_s']:.2e} m²/s, u_mig={u_mig:.2e} m/s, Pe={Pe:.1f}, it={iters}")
        cb = plt.colorbar(); cb.set_label("mol/m³")
        draw_electrodes(plt.gca(), cfg["geom"])
        plt.tight_layout()
        fname_png = os.path.join(out_dir, f"oh_map_{j_cm2:.2f}_Acm2.png".replace("-", "m"))
        plt.savefig(fname_png, dpi=180)
        plt.close()

        # Save CSV
        fname_csv = os.path.join(out_dir, f"oh_map_{j_cm2:.2f}_Acm2.csv".replace("-", "m"))
        np.savetxt(fname_csv, c_map, delimiter=",")

    # Summary
    sum_path = os.path.join(out_dir, "oh_summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("==== OH- 2D Mapping Summary ====\n")
        f.write(json.dumps(cfg, indent=2))
        f.write("\n\nCases:\n")
        for (j_cm2, c_map, u_mig, Pe, it) in results:
            f.write(f"j={j_cm2:.2f} A/cm^2 | u_mig={u_mig:.3e} m/s | Pe={Pe:.1f} | iters={it} | "
                    f"c_min={float(c_map.min()):.2f}, c_max={float(c_map.max()):.2f}\n")

if __name__ == "__main__":
    # If run directly, write a default config beside the script if not present
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "oh_config.json")
    if not os.path.exists(cfg_path):
        default_cfg = {
            "geom": {
                "Lx_cm": 2.5, "Ly_cm": 2.5,
                "pt_x0_cm": 0.25, "pt_thick_cm": 0.10, "pt_y0_cm": 0.75, "pt_y1_cm": 1.75,
                "ni_x0_cm": 2.25, "ni_thick_cm": 0.10, "ni_y0_cm": 0.75, "ni_y1_cm": 1.75
            },
            "electrolyte": {"c_bulk_mol_per_m3": 1000.0},
            "transport": {
                "D_eff_m2_per_s": 1.05e-8,  # D_base*eddy_multiplier
                "kappa_S_per_m": 25.0,
                "u_flow_x_m_per_s": 0.0
            },
            "grid": {"Nx": 161, "Ny": 161},
            "numerics": {"tolerance": 2e-6, "max_iters": 2500, "omega": 0.9},
            "cases": {"j_list_A_per_cm2": [-0.25, -0.50, -0.75, -1.00]},
            "output": {"folder": os.path.join(here, "outputs")}
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            import json; json.dump(default_cfg, f, indent=2)
    main(cfg_path)
