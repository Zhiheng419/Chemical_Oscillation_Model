import numpy as np
def approx_model_FTC(t, vars, params):
    alpha, beta, theta, phi, lam, m = params
    C_A2, C_S = vars
    dC_A2dt = 1 - alpha * C_A2 * C_S**m - theta * C_A2
    dC_Sdt = alpha/lam * C_A2 * C_S**m - beta * \
        C_S**(m + 1) + theta/lam * C_A2 - phi * C_S
    return (dC_A2dt, dC_Sdt)

def calc_all_approx_model_FTC(sol, consts, *params):
    lam, m = consts
    C_A2 = sol.y[0]
    C_S = sol.y[1]
    C_M = C_S ** m
    C_A = 2 * (1 - C_A2) - lam * (C_S + C_M)
    C_O = 1 / (2 * (1 - C_A2) - lam * (C_S + C_M)) ** 2
    return np.array([C_A2, C_S + C_M, C_A, C_O])

def full_model_FTC(t, vars, params):
    alpha, beta, theta, phi, lam, m, ep, delta = params
    cA2, cS, cO, cM = vars
    
    cA2 = np.clip(cA2, -1e6, 1e6)
    cS = np.clip(cS, -1e6, 1e6)
    cO = np.clip(cO, -1e6, 1e6)
    cM = np.clip(cM, -1e6, 1e6)

    dcA2dt = cO * (2 * (1 - cA2) - lam * (cS + cM))**2 - alpha * cM * cA2 - theta * cA2
    dcSdt = alpha/lam * cM * cA2 - beta * cM * cS + theta/lam * cA2 - phi * cS - 1/delta * (cS**m - cM)
    dcMdt = 1/delta * (cS**m - cM)
    dcOdt = 1/ep * (1 - cO * (2 * (1 - cA2) - lam * (cS + cM))**2)

    return dcA2dt, dcSdt, dcMdt, dcOdt

def calc_all_full_model_FTC(sol, const, *params):
    lam, m, ep, delta = const
    cA2, cS, cM, cO = sol.y
    cS_sum = cS + cM
    cA = 2 * (1 - cA2) - lam * (cS + cM)
    return cA2, cS_sum, cA, cO

def approx_model_Hill(t, vars, params):
    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    alpha, beta, theta, phi, k, lam, m, cmc = params
    ep = Hill(cmc, m, k)
    C_A2, C_S = vars
    dC_A2dt = 1 - alpha/ep * C_A2 * Hill(C_S, m, k) - theta * C_A2
    dC_Sdt = alpha/(lam * ep) * C_A2 * Hill(C_S, m, k) - beta/ep * Hill(C_S, m, k) + theta/lam * C_A2 - phi * C_S
    return (dC_A2dt, dC_Sdt)

def calc_all_approx_model_Hill(sol, consts, params):
    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    alpha, beta, theta, phi, k = params
    lam, m, cmc = consts
    ep = Hill(cmc, m, k)
    C_A2 = sol.y[0]
    C_S = sol.y[1]
    C_M = 1/ep * Hill(C_S, m, k)
    C_A = 2 * (1 - C_A2) - lam * (C_S + C_M)
    C_O = 1 / (2 * (1 - C_A2) - lam * (C_S + C_M)) ** 2
    return (C_A2, C_S + C_M, C_A, C_O)