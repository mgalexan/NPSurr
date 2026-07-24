import numpy as np
from scipy.integrate import solve_ivp

from constants import PhysicsConstants, SimulationParameters

"""
Main Simulation Code for the diffusion-reaction model

Uses Finite Differences to solve the PDEs for the concentrations of nanoparticles (CN), free drug (CF), and internalized drug (CI) in a spherical tumor model.
"""

def D_N_from_dim(constants: PhysicsConstants):
    """
    Diffusion coefficient, given by fully physical parameters
    """
    P = vars(constants)
    D0 = P["k_B"] * P["T_K"] / (3.0 * np.pi * P["eta"] * P["d_val_m"])
    Dn = D0 * np.exp(-P["chi_D"] * (P["d_val_m"] / P["xi_ECM"]) ** P["m_exp"])
    return Dn

def P_N_from_dim(constants: PhysicsConstants):
    """
    Permeability, given by fully physical parameters
    """
    P = vars(constants)
    Pn = P["P0"] * np.exp(-P["chi_P"] * (P["d_val_m"] / P["d_pore"]) ** P["n_exp"])
    return Pn

def D_N_from_dimless(constants: PhysicsConstants):
    """
    Diffusion coefficient given by the learned parameter a_D.
    """
    P = vars(constants)
    D0 = P["k_B"] * P["T_K"] / (3.0 * np.pi * P["eta"] * P["d_val_m"])
    Dn = D0 * np.exp(-P["a_D"] * P["d_val_m"] ** P["m_exp"])
    return Dn

def P_N_from_dimless(constants: PhysicsConstants):
    """
    Permeability given by the learned parameter a_P.
    """
    P = vars(constants)
    Pn = P["P0"] * np.exp(-P["a_P"] * P["d_val_m"] ** P["n_exp"])
    return Pn

def forward_solver(Pn: float, Dn: float, constants: PhysicsConstants, params: SimulationParameters, verbose=False):
    """
    Forward solver for the diffusion-reaction model.
    """
    P = {**vars(constants), **vars(params)}
    R_T, t_f = P["R_T"], P["t_f"]
    r = np.linspace(0.0, R_T, P["Nr"])
    dr = r[1] - r[0]
    ratio = Pn / Dn
    inv_r = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)

    CN_g, CF_g = np.empty(P["Nr"]+2), np.empty(P["Nr"]+2)

    def Cp_np(t):
        return P["C_P0"] * np.exp(-np.log(2.0) / P["tau"] * t)

    def rhs(t, y):
        CN, CF, CI = y[:P["Nr"]], y[P["Nr"]:2*P["Nr"]], y[2*P["Nr"]:]
        Cp = Cp_np(t)
        CN_g[0]=CN[1]; CN_g[1:-1]=CN; CN_g[-1]=CN[-2]+2*dr*ratio*(Cp-CN[-1])
        CF_g[0]=CF[1]; CF_g[1:-1]=CF; CF_g[-1]=CF[-2]
        d2CN=(CN_g[2:]-2*CN_g[1:-1]+CN_g[:-2])/dr**2
        d2CF=(CF_g[2:]-2*CF_g[1:-1]+CF_g[:-2])/dr**2
        dCN_dr=(CN_g[2:]-CN_g[:-2])/(2*dr)
        dCF_dr=(CF_g[2:]-CF_g[:-2])/(2*dr)
        lap_CN=Dn*(d2CN+2*inv_r*dCN_dr); lap_CN[0]=3*Dn*d2CN[0]
        lap_CF=P["D_F"]*(d2CF+2*inv_r*dCF_dr); lap_CF[0]=3*P["D_F"]*d2CF[0]
        dCN=lap_CN-(P["k_rel"]+P["k_up"])*CN
        dCF=lap_CF+P["alpha"]*P["k_rel"]*CN-(P["k_int"]+P["k_clr"])*CF
        dCI=P["k_int"]*CF-P["k_deg"]*CI
        return np.concatenate([dCN, dCF, dCI])

    t_out = np.linspace(0., t_f, P["Nt_out"])
    sol = solve_ivp(rhs, [0., t_f], np.zeros(3*P["Nr"]),
                    method="RK45", t_eval=t_out, rtol=1e-6, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")
    CN, CF, CI = sol.y[:P["Nr"], :].T, sol.y[P["Nr"]:2*P["Nr"], :].T, sol.y[2*P["Nr"]:, :].T
    return r, t_out, CN, CF, CI

def forward_solver_free(constants: PhysicsConstants, params: SimulationParameters, verbose=False):
    """
    Forward solver for the diffusion-reaction model with no nanoparticles.
    """
    P = {**vars(constants), **vars(params)}
    R_T, t_f = P["R_T"], P["t_f"]
    r = np.linspace(0.0, R_T, P["Nr"])
    dr = r[1] - r[0]
    inv_r = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)

    CF_g = np.empty(P["Nr"]+2)
    ratio = P["P_F"] / P["D_F"]

    def Cp_np(t):
        return P["C_P0"] * np.exp(-np.log(2.0) / P["tau"] * t)

    def rhs(t, y):
        CF, CI = y[:P["Nr"]], y[P["Nr"]:2*P["Nr"]]
        Cp = Cp_np(t)
        CF_g[0]=CF[1]; CF_g[1:-1]=CF; CF_g[-1]=CF[-2]+2*dr*ratio*(Cp - CF[-1])
        d2CF=(CF_g[2:]-2*CF_g[1:-1]+CF_g[:-2])/dr**2
        dCF_dr=(CF_g[2:]-CF_g[:-2])/(2*dr)
        lap_CF=P["D_F"]*(d2CF+2*inv_r*dCF_dr); lap_CF[0]=3*P["D_F"]*d2CF[0]
        dCF=lap_CF - (P["k_int"]+P["k_clr"])*CF
        dCI=P["k_int"]*CF-P["k_deg"]*CI
        return np.concatenate([dCF, dCI])

    t_out = np.linspace(0., t_f, P["Nt_out"])
    sol = solve_ivp(rhs, [0., t_f], np.zeros(2*P["Nr"]),
                    method="RK45", t_eval=t_out, rtol=1e-6, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")
    CF, CI = sol.y[:P["Nr"], :].T, sol.y[P["Nr"]:2*P["Nr"], :].T
    return r, t_out, CF, CI


if __name__ == "__main__":
    constants = PhysicsConstants()
    params = SimulationParameters()
    DN = D_N_from_dim(constants)
    PN = P_N_from_dim(constants)
    r, t_out, CN, CF, CI = forward_solver(PN, DN, constants, params)
    print("Test completed successfully.")