# To define an oscillation class, required properties are model (model), parameters (params), constants (consts) and the conservative relations of species (calc_all)
# Format of calc_all: calc_all(sol: solution of solve_ivp, consts: np.array) -> all concentrations: np.array
# Species are strictly in the order of (A2, S, A, O)
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial
import pandas as pd
from IPython.display import display
import ipywidgets as widgets
from jitcdde import jitcdde, y, t, jitcdde_input
from symengine import symbols

class oscillation:
    def __init__(self, model, params, consts, init_cond, calc_all):

        # Model and parameters
        self.__model = model
        self.__params = params
        self.__consts = consts
        self.__calc_all = calc_all
        self.__init_cond = init_cond

        # Essential information
        self.__species = ['A2', 'S_sum', 'A', 'O']
        self.__info = None

        # Experimental data
        self.__exp_data = None

    @property
    def info(self):
        print(
            f'The model includes {len(self.__params)} parameters and {len(self.__consts)} constants. The species are {self.__species}. Initial condition: {self.__init_cond}')
        print(f'Additional information: {self.__info}')

    def add_info(self, info):
        self.__info = info

    def set_params(self, params):
        self.__params = params

    def set_consts(self, consts):
        self.__consts = consts

    def simulate(self, t=10, t_eval='default', init_cond = None, method='RK45'):
        params_pass = np.hstack((self.__params, self.__consts))
        model_partial = partial(self.__model, params=params_pass)
        t_span = (0, t)

        if type(t_eval) == str:
            t_eval = np.linspace(0, t, 500)

        #Default: initial condition is passed by the property. It can also be passed by external input

        if isinstance(init_cond, (np.ndarray, list, tuple)):
            y0 = init_cond
        else:
            y0 = self.__init_cond

        sol = solve_ivp(model_partial, t_span=t_span,
                        y0=y0, method=method, t_eval=t_eval, rtol=1e-6, atol=1e-8)
        return sol

    def plot(self, t=10, exp=False, method='RK45', ylim=None):
        i = 0
        color = ['purple', 'b', 'r', 'g']
        if exp == True:
            y0 = [np.array(self.__exp_data.iloc[0, 1]), np.array(self.__exp_data.iloc[0, 3])]
            sol = self.simulate(t=self.__exp_data.iloc[-1, 0], init_cond=y0, method=method)
            c = self.__calc_all(sol, self.__consts, self.__params)
            fig, axes = plt.subplots(2, 1, figsize=(5, 5))
            for ax in axes:
                ax.plot(
                    self.__exp_data.iloc[:, 2*i], self.__exp_data.iloc[:, 2*i+1], label=f'exp_{self.__species[i]}', color=color[i])
                ax.plot(sol.t, c[i], label=self.__species[i], linestyle='--', color=color[i])
                ax.set_xlabel('Normalized Time')
                ax.set_ylabel('Normalized Concentration')
                ax.legend(loc="upper right")
                i += 1
                plt.tight_layout()
        else:
            sol = self.simulate(t, method=method)
            c = self.__calc_all(sol, self.__consts, self.__params)
            fig, axes = plt.subplots(2, 2, figsize=(7, 5))
            for ax, y in zip(axes.flatten(), c):
                ax.plot(sol.t, y, label=self.__species[i], color=color[i])
                ax.set_xlabel('Normalized Time')
                ax.set_ylabel('Normalized Concentration')
                ax.legend(loc="upper right")
                i += 1
                plt.tight_layout()

        return fig, axes

    def interactive_plot(self, t=10, ran=5, step=0.05, exp=False, ylim=None):

        if len(self.__params) == 4:
            def plot_temp(alpha, beta, theta, phi):
                params = [alpha, beta, theta, phi]
                params_old = self.__params
                self.__params = params
                self.plot(t, exp=exp, ylim=ylim)
                self.__params = params_old
                print(params)
        elif len(self.__params) == 5:
            def plot_temp(alpha, beta, theta, phi, K):
                params = [alpha, beta, theta, phi, K]
                params_old = self.__params
                self.__params = params
                self.plot(t, exp=exp, ylim=ylim)
                self.__params = params_old
                print(params)

        params_list = ['alpha', 'beta', 'theta', 'phi', 'K']
        sliders = []

        for i in range(len(self.__params)):
            slider = widgets.FloatSlider(value=self.__params[i], min=max(
                0, self.__params[i]-ran), max=self.__params[i]+ran, step=step, description=params_list[i])
            sliders.append(slider)
        
        if len(self.__params) == 4:
            interactive_widget = widgets.interactive(plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3])
        elif len(self.__params) == 5:
            interactive_widget = widgets.interactive(plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3], K=sliders[4])
        display(interactive_widget)

    def add_exp_data(self, exp_data):
        print(
            f'The species are {self.__species}. Please check if the data is in the same order and correct format (time, concentration).')
        data = exp_data.clip(lower=0)
        self.__exp_data = data

    def set_params(self, params):
        self.__params = params

    def fit(self, plot=False, overwrite=False):
        tA2 = np.array(self.__exp_data.iloc[:, 0])
        cA2 = np.array(self.__exp_data.iloc[:, 1])
        tS = np.array(self.__exp_data.iloc[:, 2])
        cS = np.array(self.__exp_data.iloc[:, 3])

        if self.__exp_data.shape[1] > 4:
            tA = np.array(self.__exp_data.iloc[:, 4])
            cA = np.array(self.__exp_data.iloc[:, 5])

        t_span_A2 = tA2[-1]
        t_span_S = tS[-1]
        init_cond = [cA2[0], cS[0]]

        def objective(params):
            self.__params = params
            simA2 = self.simulate(t=t_span_A2, t_eval=tA2, init_cond=init_cond)
            simS = self.simulate(t=t_span_S, t_eval=tS, init_cond=init_cond)
            c_all_A2 = self.__calc_all(simA2, self.__consts)
            c_all_S = self.__calc_all(simS, self.__consts)

            penalty = 1e10 * np.sum(np.minimum(self.__params, 0) ** 2)

            obj = np.sum((c_all_A2[0] - cA2)**2 +
                         (c_all_S[1] - cS)**2) + penalty
            return obj

        params_old = self.__params

        opt_result = sp.optimize.minimize(
            objective, self.__params, method='Nelder-Mead', tol=1e-6, options={'maxiter': 1000})
        print(
            f'alpha = {self.__params[0]:.3f}, beta = {self.__params[1]:.3f}, theta = {self.__params[2]:.3f}, phi = {self.__params[3]:.3f}')

        if plot == True:
            self.plot(init_cond, exp=True)

        if overwrite == False:
            self.__params = params_old
        return opt_result
    
class delayed_oscillation(oscillation):
    def __init__(self, model, delay, params, consts, init_cond, calc_all):
        super().__init__(model, params, consts, init_cond, calc_all)
        self.__delay = delay
        self.dde = self._oscillation__model(self.__delay)

    @property
    def info(self):
        print(
            f'Time-delayed model. The model includes {len(self._oscillation__params)} parameters and {len(self._oscillation__consts)} constants. \
            The species are {self._oscillation__species}. Initial condition: {self._oscillation__init_cond}')
        print(f'Additional information: {self._oscillation__info}')

    def set_delay(self, delay):
        self.__delay = delay
        self.dde = self._oscillation__model(self.__delay)

    def simulate(self, t=10, exp=False, nvars=2):
        params_pass = np.hstack(
            (self._oscillation__params, self._oscillation__consts))
        
        if exp == True:
            y0 = np.array([self._oscillation__exp_data.iloc[0, i] for i in range(1, int(nvars*2), 2)])
            t_end = self._oscillation__exp_data.iloc[-1, 0]
            t_eval = np.linspace(0, t_end, int(70*t_end))
            self.dde.constant_past(y0)
        else:
            t_eval = np.linspace(0, t, int(70*t))
            self.dde.constant_past(self._oscillation__init_cond)

        self.dde.reset_integrator()
        self.dde.set_parameters(params_pass)
        self.dde.adjust_diff()

        sol = np.array([self.dde.integrate(time) for time in t_eval])

        return (sol, t_eval)

    def plot(self, t=10, exp=False, ylim=None, nvars=2):
        i = 0
        color = ['purple', 'b', 'r', 'g']
        if exp == True:
            sol, t = self.simulate(exp=exp, nvars=nvars)
            c = self._oscillation__calc_all(
                sol, self._oscillation__consts, self._oscillation__params)
            fig, axes = plt.subplots(nvars, 1, figsize=(5, 2 * nvars))
            for ax in axes:
                #Expereimental data
                ax.plot(
                    self._oscillation__exp_data.iloc[:, 2*i], self._oscillation__exp_data.iloc[:, 2*i+1],
                    label=f'exp-{self._oscillation__species[i]}', color=color[i])
                #Simulation
                ax.plot(t, c[i], label=self._oscillation__species[i],
                        linestyle='--', color=color[i])
                if ylim != None:
                    ax.set_ylim(0, ylim)
                ax.set_xlabel('Normalized Time')
                ax.set_ylabel('Normalized Concentration')
                ax.legend(loc="upper right")
                i += 1
                plt.tight_layout()
        else:
            sol, t = self.simulate(t, exp=exp, nvars=nvars)
            c = self._oscillation__calc_all(
                sol, self._oscillation__consts, self._oscillation__params)
            fig, axes = plt.subplots(2, 2, figsize=(7, 5))
            #Simulations only
            for ax, y in zip(axes.flatten(), c):
                ax.plot(
                    t, y, label=self._oscillation__species[i], color=color[i])
                if ylim != None:
                    ax.set_ylim(0, ylim)
                ax.set_xlabel('Normalized Time')
                ax.set_ylabel('Normalized Concentration')
                ax.legend(loc="upper right")
                i += 1
                plt.tight_layout()
        return fig, axes

    def interactive_plot(self, t=10, ran=5, step=0.05, exp=False, ylim=None, nvars=2):
        if len(self._oscillation__params) == 4:
            def plot_temp(alpha, beta, theta, phi):
                params = [alpha, beta, theta, phi]
                params_old = self._oscillation__params
                self.set_params(params)
                self.plot(t, exp=exp, ylim=ylim, nvars=nvars)
                self.set_params(params_old)
                print(params)
        elif len(self._oscillation__params) == 5:
            def plot_temp(alpha, beta, theta, phi, K):
                params = [alpha, beta, theta, phi, K]
                params_old = self._oscillation__params
                self.set_params(params)
                self.plot(t, exp=exp, ylim=ylim, nvars=nvars)
                self.set_params(params_old)
                print(params)

        params_list = ['alpha', 'beta', 'theta', 'phi', 'K']
        sliders = []

        for i in range(len(self._oscillation__params)):
            slider = widgets.FloatSlider(value=self._oscillation__params[i], min=max(
                0, self._oscillation__params[i]-ran), max=self._oscillation__params[i]+ran, step=step, description=params_list[i])
            sliders.append(slider)

        if len(self._oscillation__params) == 4:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3])
        elif len(self._oscillation__params) == 5:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3], K=sliders[4])

        display(interactive_widget)