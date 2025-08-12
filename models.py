import numpy as np
from jitcdde import jitcdde, y, t
from symengine import symbols

"""Introduction
This .py file defines the reaction kinetic models in the project. The models includes non-delayed and time-delayed models.
The models are encupsulated in dictionaries in the format of: reaction_model = {'model': model_function, 'calc_all': calc_all_function, 'info': info_string}.
The model_function is the kinetic model defined by ODEs. The calc_all_function is used to calculate the concentrations of all species.
The info_string gives the numbers and names of parameters, constants, delays (if needed) and species in the model.
"""

# Non delayed models
# ------------------ Approx FTC model ------------------#


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


approx_FTC = {'model': approx_model_FTC, 'calc_all': calc_all_approx_model_FTC,
              'info': '4 params: alpha, beta, theta, phi. 2 consts: lam, m, 2 vars: cA2, cS'}
# ------------------------------------------------------#

# ------------------ Full FTC model ------------------#


def full_model_FTC(t, vars, params):
    """
    Full model of the FTC's model. The model has 5 parameters: alpha, beta, theta, phi, k;
    3 constants: lam, m, ep; and 3 variables: cA2, cS, cO.
    In this model quasi-steady state approximation is not used on cO, but it is used on cM.
    """
    alpha, beta, theta, phi, lam, m = params
    cA2, cS, cO = vars

    cA = 2 * (1 - cA2) - lam * (cS + cS**m)

    dcA2dt = cO * cA**2 - \
        alpha * cS**m * cA2 - theta * cA2
    dcSdt = alpha/lam * cS**m * cA2 - beta * \
        cS**m + theta/lam * cA2 - phi * cS
    dcOdt = 1e3 * (1 - cO * cA**2)

    return dcA2dt, dcSdt, dcOdt


def calc_all_full_model_FTC(sol, const, *params):
    """
    Receive the solution of the full model and return the values of cA2, cS_sum, cA, cO.
    """
    lam, m = const
    cA2, cS, cO = sol.y
    cS_sum = cS + cS**m
    cA = 2 * (1 - cA2) - lam * (cS_sum)
    return cA2, cS_sum, cA, cO


full_FTC = {'model': full_model_FTC, 'calc_all': calc_all_full_model_FTC,
            'info': '4 params: alpha, beta, theta, phi. 2 consts: lam, m, 3 vars: cA2, cS, cO'}
# ------------------------------------------------------#

# ------------------ Full FTC first order model ------------------#


def full_model_FTC_first_order(t, vars, params):
    alpha, beta, theta, phi, ep, lam, m = params
    cA2, cS, cO = vars

    cA = 2 * (1 - cA2) - lam * (cS + cS**m)

    dcA2dt = cO * cA - \
        alpha * cS**m * cA2 - theta * cA2
    dcSdt = alpha/lam * cS**m * cA2 - beta * \
        cS**m + theta/lam * cA2 - phi * cS
    dcOdt = ep * (1 - cO * cA)

    return dcA2dt, dcSdt, dcOdt


full_FTC_first_order = {'model': full_model_FTC_first_order, 'calc_all': calc_all_full_model_FTC,
                        'info': '5 params: alpha, beta, theta, phi, epslion. 2 consts: lam, m, 3 vars: cA2, cS, cO'}
# ------------------------------------------------------#

# ------------------ Full FTC model with Hill function ------------------#


def approx_model_Hill(t, vars, params):
    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    alpha, beta, theta, phi, k, lam, m = params
    ep = 1 / Hill(1, m, k)
    C_A2, C_S = vars
    C_M = ep * Hill(C_S, m, k)
    dC_A2dt = 1 - alpha * C_A2 * C_M - theta * C_A2
    dC_Sdt = alpha/lam * C_A2 * C_M - beta * C_M + theta/lam * C_A2 - phi * C_S
    return (dC_A2dt, dC_Sdt)


def calc_all_approx_model_Hill(sol, consts, params):
    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    alpha, beta, theta, phi, k = params
    lam, m = consts
    ep = 1 / Hill(1, m, k)
    C_A2 = sol.y[0]
    C_S = sol.y[1]
    C_M = ep * Hill(C_S, m, k)
    C_A = 2 * (1 - C_A2) - lam * (C_S + C_M)
    C_O = 1 / C_A
    return (C_A2, C_S + C_M, C_A, C_O)


approx_Hill = {'model': approx_model_Hill, 'calc_all': calc_all_approx_model_Hill,
               'info': '5 params: alpha, beta, theta, phi, k. 2 consts: lam, m. 2 vars: cA2, cS'}
# ------------------------------------------------------#

# ----------------------Full model with modified micelle changing rate--------------------------------#


def full_model_4vars(t, vars, params):
    alpha, beta, theta, phi, ep, delta, lam, m = params
    cA2, cS, cO, cM = vars

    cA = 2 * (1 - cA2) - lam * (cS + cM)
    dcA2dt = cO * cA - alpha * cM * cA2 - theta * cA2
    dcSdt = alpha/lam * cM * cA2 + theta/lam * \
        cA2 - phi * cS - delta * (cS**m - cM)
    dcOdt = ep * (1 - cO * cA)
    dcMdt = delta * (cS**m - cM) - beta * cM

    return dcA2dt, dcSdt, dcOdt, dcMdt


def calc_all_full_model_4vars(sol, const, *params):
    """
    Receive the solution of the full model and return the values of cA2, cS_sum, cA, cO.
    """
    lam, m = const
    cA2, cS, cO, cM = sol.y
    cS_sum = cS + cM
    cA = 2 * (1 - cA2) - lam * (cS_sum)
    return cA2, cS_sum, cA, cO


full_model_4vars_dict = {'model': full_model_4vars, 'calc_all': calc_all_full_model_4vars,
                         'info': '6 params: alpha, beta, theta, phi, ep, delta. 2 consts: lam, m, 4 vars: cA2, cS, cO, cM'}
# ------------------------------------------------------#

# Delayed models realized with jitcdde

# ------------------ Delayed approx FTC model ------------------#


def delayed_approx_model_FTC(delays):
    td1, td2 = delays
    alpha, beta, theta, phi, lam, m = symbols('alpha beta theta phi lam m')
    dcA2dt = 1 - alpha * y(0) * y(1, t-td2)**m - theta * y(0)
    dcSdt = alpha/lam * y(0) * y(1, t-td2)**m - beta * \
        y(1, t-td1)**m + theta/lam * y(0) - phi * y(1)
    dde = jitcdde([dcA2dt, dcSdt], control_pars=[
                  alpha, beta, theta, phi, lam, m])
    return dde


def calc_all_delayed_approx_model_FTC(sol, consts, *params):
    lam, m = consts
    cA2 = sol[:, 0]
    cS = sol[:, 1]
    cM = cS ** m
    cA = 2 * (1 - cA2) - lam * (cS + cM)
    cO = 1 / (2 * (1 - cA2) - lam * (cS + cM)) ** 2
    return np.array([cA2, cS + cM, cA, cO])


delayed_approx_FTC = {'model': delayed_approx_model_FTC, 'calc_all': calc_all_delayed_approx_model_FTC,
                      'info': '4 params: alpha, beta, theta, phi. 2 consts: lam, m. 2 delays: td1, td2. 2 vars: cA2, cS.'}
# ------------------------------------------------------#

# ------------------ Delayed full FTC model ------------------#


def delayed_full_model(delays):
    td1, td2 = delays
    alpha, beta, theta, phi, lam, m = symbols('alpha beta theta phi lam m')
    cA = 2 * (1 - y(0)) - lam * (y(1) + y(1)**m)
    dcA2dt = cA**2 * y(2) - alpha * y(0) * y(1, t-td2)**m - theta * y(0)
    dcSdt = alpha/lam * y(0) * y(1, t-td2)**m - beta * \
        y(1, t-td1)**m + theta/lam * y(0) - phi * y(1)
    dcOdt = 1e3 * (1 - y(2) * cA**2)
    dde = jitcdde([dcA2dt, dcSdt, dcOdt], control_pars=[
                  alpha, beta, theta, phi, lam, m])
    return dde


def calc_all_delayed_full_model(sol, consts, *params):
    lam, m = consts
    cA2 = sol[:, 0]
    cS = sol[:, 1]
    cO = sol[:, 2]
    cM = cS ** m
    cA = 2 * (1 - cA2) - lam * (cS + cM)
    return np.array([cA2, cS + cM, cA, cO])


delayed_full_FTC = {'model': delayed_full_model, 'calc_all': calc_all_delayed_full_model,
                    'info': '4 params: alpha, beta, theta, phi. 2 consts: lam, m. 2 delays: td1, td2. 3 vars: cA2, cS, cO'}
# ------------------------------------------------------#

# ------------------ Delayed full FTC model first order------------------#


def delayed_full_first_order_model(delays):
    td1, td2 = delays
    alpha, beta, theta, phi, ep, lam, m = symbols(
        'alpha beta theta phi ep lam m')
    cA = 2 * (1 - y(0)) - lam * (y(1) + y(1)**m)
    dcA2dt = cA * y(2) - alpha * y(0) * y(1, t-td2)**m - theta * y(0)
    dcSdt = alpha/lam * y(0) * y(1, t-td2)**m - beta * \
        y(1, t-td1)**m + theta/lam * y(0) - phi * y(1)
    dcOdt = ep * (1 - y(2) * cA)
    dde = jitcdde([dcA2dt, dcSdt, dcOdt], control_pars=[
                  alpha, beta, theta, phi, ep, lam, m])
    return dde


delayed_full_first_order = {'model': delayed_full_first_order_model, 'calc_all': calc_all_delayed_full_model,
                            'info': '5 params: alpha, beta, theta, phi, epsilon. 2 consts: lam, m, 3 vars: cA2, cS, cO'}
# ------------------------------------------------------#

# ------------------ Delayed full FTC model with oxidant consumption ------------------#


def delayed_full_model_consumeO(delays):
    td1, td2 = delays
    alpha, beta, theta, phi, k, lam, m = symbols(
        'alpha beta theta phi k lam m')
    cA = 2 * (1 - y(0)) - lam * (y(1) + y(1)**m)
    dcA2dt = cA**2 * y(2) - alpha * y(0) * y(1, t-td2)**m - theta * y(0)
    dcSdt = alpha/lam * y(0) * y(1, t-td2)**m - beta * \
        y(1, t-td1)**m + theta/lam * y(0) - phi * y(1)
    dcOdt = 1e3 * (1 - y(2) * cA**2 - k * y(2) * y(1, t-td2)**(2*m))
    dde = jitcdde([dcA2dt, dcSdt, dcOdt], control_pars=[
                  alpha, beta, theta, phi, k, lam, m])
    return dde


def calc_all_delayed_full_model_consumeO(sol, consts, *params):
    lam, m = consts
    cA2 = sol[:, 0]
    cS = sol[:, 1]
    cO = sol[:, 2]
    cM = cS ** m
    cA = 2 * (1 - cA2) - lam * (cS + cM)
    return np.array([cA2, cS + cM, cA, cO])


delayed_full_FTC_consumeO = {'model': delayed_full_model_consumeO, 'calc_all': calc_all_delayed_full_model_consumeO,
                             'info': '5 params: alpha, beta, theta, phi, k. 2 consts: lam, m. 2 delays: td1, td2. 3 vars: cA2, cS, cO'}
# ------------------------------------------------------#

# ------------------ Delayed approx FTC model with Hill function ------------------#


def delayed_approx_model_Hill(delays):
    td1, td2 = delays
    alpha, beta, theta, phi, k, lam, m, cmc = symbols(
        'alpha beta theta phi k lam m cmc')

    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    ep = Hill(cmc, m, k)

    dC_A2dt = 1 - alpha/ep * y(0) * Hill(y(1, t-td2), m, k) - theta * y(0)
    dC_Sdt = alpha/(lam * ep) * y(0) * Hill(y(1, t-td2), m, k) - \
        beta/ep * Hill(y(1, t-td1), m, k) + theta/lam * y(0) - phi * y(1)
    dde = jitcdde([dC_A2dt, dC_Sdt], control_pars=[
                  alpha, beta, theta, phi, k, lam, m, cmc])
    return dde


def calc_all_delayed_approx_model_Hill(sol, consts, params):
    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    alpha, beta, theta, phi, k = params
    lam, m, cmc = consts
    ep = Hill(cmc, m, k)
    C_A2 = sol[:, 0]
    C_S = sol[:, 1]
    C_M = 1/ep * Hill(C_S, m, k)
    C_A = 2 * (1 - C_A2) - lam * (C_S + C_M)
    C_O = 1 / (2 * (1 - C_A2) - lam * (C_S + C_M)) ** 2
    return (C_A2, C_S + C_M, C_A, C_O)


delayed_approx_Hill = {'model': delayed_approx_model_Hill, 'calc_all': calc_all_delayed_approx_model_Hill,
                       'info': '5 params: alpha, beta, theta, phi, k. 3 consts: lam, m, cmc. 2 vars: cA2, cS'}
# ------------------------------------------------------#

# ------------------ Delayed full FTC model with Hill function and oxidant consumption ------------------#


def delayed_full_model_Hill_consumeO(delays):
    td1, td2 = delays
    alpha, beta, theta, phi, k, kappa, lam, m, cmc = symbols(
        'alpha beta theta phi k kappa lam m cmc')

    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    ep = Hill(cmc, m, k)

    cA = 2 * (1 - y(0)) - lam * (y(1) + 1/ep * Hill(y(1), m, k))
    dcA2dt = cA**2 * y(2) - alpha/ep * y(0) * \
        Hill(y(1, t-td2), m, k) - theta * y(0)
    dcSdt = alpha/(lam * ep) * y(0) * Hill(y(1, t-td2), m, k) - beta / \
        ep * Hill(y(1, t-td1), m, k) + theta/lam * y(0) - phi * y(1)
    dcOdt = 1e3 * (1 - y(2) * cA**2 - kappa * y(2) *
                   (1/ep * Hill(y(1, t-td2), m, k))**2)
    dde = jitcdde([dcA2dt, dcSdt, dcOdt], control_pars=[
                  alpha, beta, theta, phi, k, kappa, lam, m, cmc])
    return dde


def calc_all_delayed_full_model_Hill_consumeO(sol, consts, params):
    def Hill(x, m, k):
        return x**m/(k**m + x**m)
    alpha, beta, theta, phi, k, kappa = params
    lam, m, cmc = consts
    ep = Hill(cmc, m, k)
    cA2 = sol[:, 0]
    cS = sol[:, 1]
    cO = sol[:, 2]
    cM = 1/ep * Hill(cS, m, k)
    cA = 2 * (1 - cA2) - lam * (cS + cM)
    return np.array([cA2, cS + cM, cA, cO])


delayed_full_FTC_Hill_consumeO = {'model': delayed_full_model_Hill_consumeO, 'calc_all': calc_all_delayed_full_model_Hill_consumeO,
                                  'info': '6 params: alpha, beta, theta, phi, k, kappa. 3 consts: lam, m, cmc. 2 delays: td1, td2. 3 vars: cA2, cS, cO'}
# ------------------------------------------------------#

# ------------------ Delayed full 4 vars model ------------------#


def delayed_full_model_4vars(delays):
    td1, td2 = delays
    alpha, beta, theta, phi, ep, delta, lam, m = symbols(
        'alpha beta theta phi epsilon delta lam m')

    cA = 2 * (1 - y(0)) - lam * (y(1) + y(3))

    dcA2dt = y(2) * cA - alpha * y(0) * y(3, t-td2) - theta * y(0)
    dcSdt = alpha/lam * y(0) * y(3, t-td2) + theta/lam * \
        y(0) - phi * y(1) - delta * (y(1)**m - y(3))
    dcOdt = ep * (1 - y(2) * cA)
    dcMdt = delta * (y(1)**m - y(3)) - beta * y(3, t-td1)

    dde = jitcdde([dcA2dt, dcSdt, dcOdt, dcMdt], control_pars=[
                  alpha, beta, theta, phi, ep, delta, lam, m])
    return dde


def calc_all_delayed_full_model_4vars(sol, consts, *params):
    lam, m = consts
    cA2 = sol[:, 0]
    cS = sol[:, 1]
    cO = sol[:, 2]
    cM = sol[:, 3]
    cA = 2 * (1 - cA2) - lam * (cS + cM)
    return np.array([cA2, cS + cM, cA, cO])


delayed_full_4vars = {'model': delayed_full_model_4vars, 'calc_all': calc_all_delayed_full_model_4vars,
                      'info': '6 params: alpha, beta, theta, phi, ep, delta. 2 consts: lam, m, 4 vars: cA2, cS, cO, cM'}
# ------------------------------------------------------#

#-------------------Delayed mixed models------------------
def delayed_full_two_thiols_model(delays):
    td1, td2 = delays
    alpha1, beta1, theta1, phi1, ep1, delta1, alpha2, beta2, theta2, phi2, ep2, delta2, lam1, m1, lam2, m2 = \
        symbols(
            'alpha1 beta1 theta1 phi1 ep1 delta1 alpha2 beta2 theta2 phi2 ep2 delta2 lam1 m1 lam2 m2')

    r = lam2/lam1
    cA2, cS1, cS2, cM1, cM2, cO = y(0), y(1), y(2), y(3), y(4), y(5)
    cM1_d1, cM1_d2 = y(3, t-td1), y(3, t-td2)
    cM2_d1, cM2_d2 = y(4, t-td1), y(4, t-td2)
    cA = 2 * (1 - cA2) - lam1 * (cS1 + cM1) - lam2 * (cS2 + cM2)

    dcA2dt = cO * cA - 1/2 * alpha1 * (cM1_d2 + r * cM2_d2) * cA2 - 1/2 * alpha2 * (
        1/r * cM1_d2 + cM2_d2) * cA2 - 1/2 * (theta1 + theta2) * cA2
    dcS1dt = 1/2 * theta1/lam1 * cA2 - 1/2 * phi1 * cS1 - 1/4 * \
        (phi1 + phi2) * cS1 + 1/2 * alpha1/lam1 * \
        (cM1_d2 + r * cM2_d2) * cA2 - delta1 * (cS1**m1 - cM1)
    dcS2dt = 1/2 * theta2/lam2 * cA2 - 1/2 * phi2 * cS2 - 1/4 * \
        (phi1 + phi2) * cS2 + 1/2 * alpha2/lam2 * \
        (1/r * cM1_d2 + cM2_d2) * cA2 - delta2 * (cS2**m2 - cM2)
    dcM1dt = delta1 * (cS1**m1 - cM1) - (beta1)/2 * \
        cM1_d1 - 1/4 * (beta1 + beta2) * cM1_d1
    dcM2dt = delta2 * (cS2**m2 - cM2) - (beta2)/2 * \
        cM2_d1 - 1/4 * (beta1 + beta2) * cM2_d1
    dcOdt = ep1 * (1 - cO * cA)
    dde = jitcdde([dcA2dt, dcS1dt, dcS2dt, dcM1dt, dcM2dt, dcOdt], control_pars=[
                  alpha1, beta1, theta1, phi1, ep1, delta1, alpha2, beta2, theta2, phi2, ep2, delta2, lam1, m1, lam2, m2])
    return dde


def calc_all_delayed_full_two_thiols(sol, consts, *params):
    lam1, m1, lam2, m2 = consts
    cA2 = sol[:, 0]
    cS1 = sol[:, 1]
    cS2 = sol[:, 2]
    cM1 = sol[:, 3]
    cM2 = sol[:, 4]
    cO = sol[:, 5]
    cS1_sum = cS1 + cM1
    cS2_sum = cS2 + cM2
    cA = 2 * (1 - cA2) - lam1 * cS1_sum - lam2 * cS2_sum
    return [cA2, cS1_sum, cS2_sum, cA, cO]


delayed_full_two_thiols = {'model': delayed_full_two_thiols_model, 'calc_all': calc_all_delayed_full_two_thiols,
                           'info': '5*2 parameters, 2*2 constants, 4 species: A2, S1, S2, O'}
#----------------------------------------------------------#