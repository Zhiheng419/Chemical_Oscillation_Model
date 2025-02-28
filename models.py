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
    """
    Full model of the FTC's model. The model has 4 parameters: alpha, beta, theta, phi;
    3 constants: lam, m, ep; and 3 variables: cA2, cS, cO.
    In this model quasi-steady state approximation is not used on cO, but it is used on cM.
    """
    alpha, beta, theta, phi, lam, m, ep = params
    cA2, cS, cO = vars

    dcA2dt = cO * (2 * (1 - cA2) - lam * (cS + cS**m))**2 - alpha * cS**m * cA2 - theta * cA2
    dcSdt = alpha/lam * cS**m * cA2 - beta * cS**(m + 1) + theta/lam * cA2 - phi * cS
    dcOdt = 1/ep * (1 - cO * (2 * (1 - cA2) - lam * (cS + cS**m))**2)

    return dcA2dt, dcSdt, dcOdt

def calc_all_full_model_FTC(sol, const, *params):
    """
    Receive the solution of the full model and return the values of cA2, cS_sum, cA, cO.
    """
    lam, m, ep = const
    cA2, cS, cO = sol.y
    cS_sum = cS + cS**m
    cA = 2 * (1 - cA2) - lam * (cS_sum)
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

def delayed_approx_model_FTC(vars, t, td1, td2, params):
    #td1 is shorter delay, td2 is longer delay
    alpha, beta, theta, phi, k, lam, m = params
    cA2, cS = vars(t)
    cA2_td1, cS_td1 = vars(t - td1)
    cA2_td2, cS_td2 = vars(t - td2)

    cS = max(1e-6, cS)

    cT1 = cS_td1**m
    cT2 = cS_td2**m

    dcA2dt = 1 - alpha * cA2 * cT2 - theta * cA2
    dcSdt = alpha/lam * cA2 * cT2 - beta * cT1 * cS + theta/lam * cA2 - phi * cS
    
    return np.array((dcA2dt, dcSdt))

def calc_all_delayed_approx_model_FTC(sol, consts, *params):
    lam, m = consts
    cA2 = sol[:, 0]
    cS = sol[:, 1]
    cM = cS ** m
    cA = 2 * (1 - cA2) - lam * (cS + cM)
    cO = 1 / (2 * (1 - cA2) - lam * (cS + cM)) ** 2
    return np.array([cA2, cS + cM, cA, cO])