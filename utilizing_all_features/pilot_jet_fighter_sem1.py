import numpy as np
import matplotlib.pyplot as plt

# Parameters (range 0 to 1)
w_cl1, w_cl2, w_cl3, w_cl4 = 0.3, 0.2, 0.2, 0.3
w_ss1, w_ss2, w_ss3 = 0.4, 0.3, 0.3
w_rc = 0.6
w_ps1, w_ps2, w_ps3 = 0.4, 0.3, 0.3

gamma_cl = 0.1  # from 0.05 to 0.01
alpha_cl = 0.4
beta_el = 0.6
beta_pfs = 0.6
eta_ss = 0.03
alpha_ss = 0.8
beta_sa = 0.6
alpha_sa = 0.4
gamma_rc = 0.01
beta_ps = 0.09
alpha_ps = 0.01

# New changes
lambda_pfl = lambda_ls = lambda_pl = 0.2

# Time vector
T = 50  # duration
delta = 1  # change rate
steps = int(T / delta)  # time steps
time = np.linspace(0, T, steps)

# Arrays to store results
Hr = np.full(steps, 0.45)
Sq = np.full(steps, 0.667)
Mc = np.full(steps, 0.556)
El_base = np.full(steps, 0.6)
Es = np.full(steps, 0.333)

Cl, Pfs, Pfl = np.zeros(steps), np.zeros(steps), np.zeros(steps)
Ss, Ls, Sa = np.zeros(steps), np.zeros(steps), np.zeros(steps)
Rc, Ps, Pl = np.zeros(steps), np.zeros(steps), np.zeros(steps)
dPfldt, dLsdt, dPldt = np.zeros(steps), np.zeros(steps), np.zeros(steps)

# Initial values
Pfl[0] = Ls[0] = Pl[0] = 0.2

Cl[0] = ((w_cl1 * Hr[0] + w_cl2 * Mc[0] + w_cl3 * Es[0] + w_cl4 * Pfl[0]) - gamma_cl * Sq[0]) * (1 - alpha_cl * El_base[0])
El = np.zeros(steps)
El[0] = beta_el * El_base[0] + (1 - beta_el) * Pl[0]
Pfs[0] = beta_pfs * Hr[0] + (1 - beta_pfs) * (1 - Sq[0])
Ss[0] = ((w_ss1 * Hr[0] + w_ss2 * Mc[0] + w_ss3 * Es[0]) - eta_ss * Sq[0]) * (1 - alpha_ss * El[0])
Sa[0] = beta_sa * El[0] + (1 - beta_sa) * (1 - (alpha_sa * Es[0] + (1 - alpha_sa) * Cl[0]))
Rc[0] = gamma_rc * Sq[0] + (1 - gamma_rc) * Sa[0]
Ps[0] = beta_ps * (alpha_ps * Rc[0] - (1 - alpha_ps) * Sa[0]) + (1 - beta_ps) * (1 - (w_ps1 * Pfl[0] + w_ps2 * Ls[0] + w_ps3 * Cl[0]))

# Simulation cycles
for t in range(1, steps):
    Cl[t] = ((w_cl1 * Hr[t] + w_cl2 * Mc[t] + w_cl3 * Es[t] + w_cl4 * Pfl[t - 1]) - gamma_cl * Sq[t]) * (1 - alpha_cl * El[t - 1])
    El[t] = beta_el * El_base[t] + (1 - beta_el) * Pl[t - 1]
    Pfs[t] = beta_pfs * Hr[t] + (1 - beta_pfs) * (1 - Sq[t])
    dPfldt[t] = Pfs[t] - Pfl[t - 1]
    Pfl[t] = Pfl[t - 1] + (lambda_pfl * dPfldt[t]) * delta
    Ss[t] = ((w_ss1 * Hr[t] + w_ss2 * Mc[t] + w_ss3 * Es[t]) - eta_ss * Sq[t]) * (1 - alpha_ss * El[t])
    dLsdt[t] = Ss[t] - Ls[t - 1]
    Ls[t] = Ls[t - 1] + (lambda_ls * dLsdt[t]) * delta
    Sa[t] = beta_sa * El[t] + (1 - beta_sa) * (1 - (alpha_sa * Es[t] + (1 - alpha_sa) * Cl[t]))
    Rc[t] = gamma_rc * Sq[t] + (1 - gamma_rc) * Sa[t]
    Ps[t] = beta_ps * (alpha_ps * Rc[t] - (1 - alpha_ps) * Sa[t]) + (1 - beta_ps) * (1 - (w_ps1 * Pfl[t] + w_ps2 * Ls[t - 1] + w_ps3 * Cl[t]))
    dPldt[t] = Ps[t] - Pl[t - 1]
    Pl[t] = Pl[t - 1] + (lambda_pl * dPldt[t]) * delta

# Plotting
minLimX, maxLimY = 0, 1.2
t = np.arange(steps)

plt.figure(figsize=(12, 10))
plt.subplot(4, 2, (1, 3, 5))
plt.plot(t, Hr, label="HR", linewidth=2)
plt.plot(t, Mc, label="MC", linewidth=2)
plt.plot(t, Sq, label="SQ", linewidth=2)
plt.plot(t, El, label="EL", linewidth=2)
plt.plot(t, Es, label="ES", linewidth=2)
plt.title("Input features : HR, MC, SQ, EL, and ES")
plt.xlabel("Time")
plt.legend()
plt.grid()

plt.subplot(4, 2, 2)
plt.plot(t, Ls, label="LS", linewidth=2)
plt.plot(t, Ss, label="SS", linewidth=2)
plt.title("Long Stress and Short Stress : LS and SS")
plt.legend()
plt.grid()

plt.subplot(4, 2, 4)
plt.plot(t, Sa, label="SA", linewidth=2)
plt.plot(t, Rc, label="RC", linewidth=2)
plt.title("Social Activity and Reaction Coherence : SA and RC")
plt.legend()
plt.grid()

plt.subplot(4, 2, 6)
plt.plot(t, Cl, label="CL", linewidth=2)
plt.plot(t, Pfl, label="PFL", linewidth=2)
plt.title("Cognition Level and Psychological Fatigue Level : CL and PFL")
plt.legend()
plt.grid()

plt.subplot(4, 2, 8)
plt.plot(t, Ps, label="PS", linewidth=2)
plt.plot(t, Pl, label="PL", linewidth=2)
plt.title("Psychological Stability and Psychological Load : PS and PL")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
