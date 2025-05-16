import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial
from IPython.display import display
import ipywidgets as widgets
import pandas as pd

def input_expdata(files: str, params: list|np.ndarray) -> pd.DataFrame :
    """
    Read the experimental data from a csv file and normalize the data. The data should be in the format of [time, conc, time, conc,...].
    The species should be in the order of A2, S, A (if applicable).
    Parameters are CMC and input rate (rext)."""
    df = pd.read_csv(files)
    cmc, rext = params
    col_titles = ['tA2', 'cA2', 'tS', 'cS', 'tA', 'cA']
    df.columns = col_titles[:len(df.columns)]
    
    cA2_tol = df.loc[0, 'cA2'] + df.loc[0, 'cA'] / 2 if 'cA' in df.columns else df.loc[0, 'cA2']
    tau = cA2_tol / rext
    print(f'cA2_tol = {cA2_tol:.3f}, tau = {tau:.3f}')

    df['cA2'] /= cA2_tol
    if 'cA' in df.columns:
        df['cA'] /= cA2_tol
    df['cS'] /= cmc
    df['tA2'] /= tau
    df['tS'] /= tau
    if 'tA' in df.columns:
        df['tA'] /= tau
    
    return df

# -----------Non-delayed oscillation model-----------#


class oscillation:
    # Initializations
    def __init__(self, model: dict, params: list | np.ndarray, consts: list | np.ndarray, init_cond: list | np.ndarray):

        # Model and parameters
        self._model = model['model']
        self._params = params
        self._consts = consts
        self._calc_all = model['calc_all']
        self._init_cond = init_cond

        # Essential information
        self._species = ['$c_{A_2}$', '$c_S$', '$c_A$', '$c_O$']
        self._info = model['info']

        # Experimental data
        self._exp_data = None

    # Modify the properties
    @property
    def info(self):
        print(
            f'The model includes {len(self._params)} parameters and {len(self._consts)} constants. The species are {self._species}. Initial condition: {self._init_cond}')
        print(f'Additional information: {self._info}')

    def add_info(self, info: str):
        self._info = info

    def set_params(self, params: list | np.ndarray):
        self._params = params

    def set_consts(self, consts: list | np.ndarray):
        self._consts = consts

    def set_species(self, species: list[str]):
        self._species = species

    def set_init_cond(self, init_cond: list | np.ndarray):
        self._init_cond = init_cond
        print(f'Initial condition is set as {self._init_cond}')

    def add_exp_data(self, exp_data: pd.DataFrame):
        print(
            f'The species are {self._species}. Please check if the data is in the same order and correct format (time, concentration).')
        self._exp_data = exp_data
        self._init_cond = np.array(
            [self._exp_data.iloc[0, i] for i in range(1, exp_data.shape[1], 2)])
        print(f'Initial condition is set as {self._init_cond}')

    # Simulation and visualization

    def simulate(self, t=10, npoints=500, init_cond=None, method='RK45'):
        """
        Solve the kinetic model, return scipy.integrate.solve_ivp solution
        """
        params_pass = np.hstack((self._params, self._consts))
        model_partial = partial(self._model, params=params_pass)
        t_span = (0, t)
        t_eval = np.linspace(0, t, npoints)

        # Default: initial condition is passed by the property. It can also be passed by external input
        if isinstance(init_cond, (np.ndarray, list, tuple)):
            y0 = init_cond
        else:
            y0 = self._init_cond

        sol = solve_ivp(model_partial, t_span=t_span,
                        y0=y0, method=method, t_eval=t_eval, rtol=1e-6, atol=1e-8)
        return sol

    def plot(self, t=10, exp=False, method='RK45', ylim=None, npoints=500):
        i = 0
        color = ['purple', 'b', 'r', 'g']
        if exp == True:
            sol = self.simulate(
                t=self._exp_data.iloc[-1, 0], init_cond=self._init_cond, method=method, npoints=npoints)
            c = self._calc_all(sol, self._consts, self._params)
            fig, axes = plt.subplots(2, 1, figsize=(5, 5))
            for ax in axes:
                ax.plot(
                    self._exp_data.iloc[:, 2*i], self._exp_data.iloc[:, 2*i+1], label=f'{self._species[i]}(exp)', color=color[i])
                ax.plot(sol.t, c[i], label=f'{self._species[i]}(sim)',
                        linestyle='--', color=color[i])
                if ylim != None:
                    ax.set_ylim(0, ylim)
                ax.legend(loc="upper right")
                i += 1

            fig.supxlabel('Normalized Time')
            fig.supylabel('Normalized Concentration')
            plt.tight_layout()
        else:
            sol = self.simulate(t, method=method, npoints=npoints)
            c = self._calc_all(sol, self._consts, self._params)
            fig, axes = plt.subplots(2, 2, figsize=(7, 5))
            for ax, y in zip(axes.flatten(), c):
                ax.plot(sol.t, y, label=self._species[i], color=color[i])
                if ylim != None:
                    ax.set_ylim(0, ylim)
                ax.legend(loc="upper right")
                i += 1

            fig.supxlabel('Normalized Time')
            fig.supylabel('Normalized Concentration')
            plt.tight_layout()
        return fig, axes

    def interactive_plot(self, t=10, ran=5, step=0.05, exp=False, ylim=None):
        if len(self._params) == 4:
            def plot_temp(alpha, beta, theta, phi):
                params = [alpha, beta, theta, phi]
                params_old = self._params
                self._params = params
                self.plot(t, exp=exp, ylim=ylim)
                self._params = params_old
                print(params)
        elif len(self._params) == 5:
            def plot_temp(alpha, beta, theta, phi, ep):
                params = [alpha, beta, theta, phi, ep]
                params_old = self._params
                self._params = params
                self.plot(t, exp=exp, ylim=ylim)
                self._params = params_old
                print(params)
        elif len(self._params) == 6:
            def plot_temp(alpha, beta, theta, phi, ep, delta):
                params = [alpha, beta, theta, phi, ep, delta]
                params_old = self._params
                self._params = params
                self.plot(t, exp=exp, ylim=ylim)
                self._params = params_old
                print(params)

        params_list = ['alpha', 'beta', 'theta', 'phi', 'ep', 'delta']
        sliders = []

        for i in range(len(self._params)):
            slider = widgets.FloatSlider(value=self._params[i], min=max(
                0, self._params[i]-ran), max=self._params[i]+ran, step=step, description=params_list[i])
            sliders.append(slider)

        if len(self._params) == 4:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3])
        elif len(self._params) == 5:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3], ep=sliders[4])
        elif len(self._params) == 6:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3], ep=sliders[4], delta=sliders[5])
        display(interactive_widget)

    # Fitting
    def fit(self, plot=False, overwrite=False):
        """
        Fit the kinetic parameters to the experimental data by minimum square method. 

        Options: plot: plot the simulation with the experimental data. overwrite: overwrite the parameters
        """
        tA2 = np.array(self._exp_data.iloc[:, 0])
        cA2 = np.array(self._exp_data.iloc[:, 1])
        tS = np.array(self._exp_data.iloc[:, 2])
        cS = np.array(self._exp_data.iloc[:, 3])

        if self._exp_data.shape[1] > 4:
            tA = np.array(self._exp_data.iloc[:, 4])
            cA = np.array(self._exp_data.iloc[:, 5])

        t_span_A2 = tA2[-1]
        t_span_S = tS[-1]
        init_cond = [cA2[0], cS[0]]

        def objective(params):
            self._params = params
            simA2 = self.simulate(t=t_span_A2, t_eval=tA2, init_cond=init_cond)
            simS = self.simulate(t=t_span_S, t_eval=tS, init_cond=init_cond)
            c_all_A2 = self._calc_all(simA2, self._consts)
            c_all_S = self._calc_all(simS, self._consts)

            penalty = 1e10 * np.sum(np.minimum(self._params, 0) ** 2)

            obj = np.sum((c_all_A2[0] - cA2)**2 +
                         (c_all_S[1] - cS)**2) + penalty
            return obj

        params_old = self._params

        opt_result = sp.optimize.minimize(
            objective, self._params, method='Nelder-Mead', tol=1e-6, options={'maxiter': 1000})
        print(
            f'alpha = {self._params[0]:.3f}, beta = {self._params[1]:.3f}, theta = {self._params[2]:.3f}, phi = {self._params[3]:.3f}')

        if plot == True:
            self.plot(init_cond, exp=True)

        if overwrite == False:
            self._params = params_old
        return opt_result
# ---------------------------------------------------#


# -----------Time-delayed oscillation model-----------#


class delayed_oscillation(oscillation):
    def __init__(self, model: dict, delay: list | np.ndarray, params: list | np.ndarray, consts: list | np.ndarray, init_cond: list | np.ndarray):
        super().__init__(model, params, consts, init_cond)
        self._delay = delay
        self.dde = self._model(self._delay)

    @property
    def info(self):
        print(
            f'Time-delayed model. The model includes {len(self._params)} parameters and {len(self._consts)} constants. \
            The species are {self._species}. Initial condition: {self._init_cond}')
        print(f'Additional information: {self._info}')

    def set_delay(self, delay):
        self._delay = delay
        self.dde = self._model(self._delay)

    def simulate(self, t=10, exp=False, nvars=2, acc=80):
        """
        Solve the time-delayed kinetic model, return jitcdde solution and time points in a tuple
        """
        params_pass = np.hstack(
            (self._params, self._consts))

        if exp == True:
            t_end = self._exp_data.iloc[-1, 0]
            t_eval = np.linspace(0, t_end, int(acc*t_end))
            self.dde.constant_past(self._init_cond)
        else:
            t_eval = np.linspace(0, t, int(acc*t))
            self.dde.constant_past(self._init_cond)

        self.dde.reset_integrator()
        self.dde.set_parameters(params_pass)
        self.dde.adjust_diff()

        sol = np.array([self.dde.integrate(time) for time in t_eval])

        return (sol, t_eval)

    def plot(self, t=10, exp=False, ylim=None, nvars=2, acc=80):
        """
        Plot the simulation or the comparison between the simulation and the experimental data.

        The value of nvars is the number of species in the experimental data. Only required when exp=True.
        """
        i = 0
        color = ['purple', 'b', 'r', 'g']
        if exp == True:
            sol, t = self.simulate(exp=exp, nvars=nvars, acc=acc)
            c = self._calc_all(
                sol, self._consts, self._params)
            fig, axes = plt.subplots(nvars, 1, figsize=(5, 2 * nvars))
            for ax in axes:
                # Expereimental data
                ax.plot(
                    self._exp_data.iloc[:, 2 *
                                        i], self._exp_data.iloc[:, 2*i+1],
                    label=f'{self._species[i]}(exp)', color=color[i])
                # Simulation
                ax.plot(t, c[i], label=f'{self._species[i]}(sim)',
                        linestyle='--', color=color[i])
                if ylim != None:
                    ax.set_ylim(0, ylim)
                ax.legend(loc="upper right")
                i += 1
            fig.supxlabel('Normalized Time')
            fig.supylabel('Normalized Conc')
            plt.tight_layout()
        else:
            sol, t = self.simulate(t, exp=exp, nvars=nvars, acc=acc)
            c = self._calc_all(
                sol, self._consts, self._params)
            fig, axes = plt.subplots(nvars, 1, figsize=(5, 2 * nvars))
            # Simulations only
            for ax, y in zip(axes, c):
                ax.plot(
                    t, y, label=self._species[i], color=color[i])
                if ylim != None:
                    ax.set_ylim(0, ylim)
                ax.legend(loc="upper right")
                i += 1
            fig.supxlabel('Normalized Time')
            fig.supylabel('Normalized Conc')
            plt.tight_layout()
        return fig, axes

    def interactive_plot(self, t=10, ran=5, step=0.05, exp=False, ylim=None, nvars=2, acc=80):
        """
        Plot the interactive plot with tunable parameters. Only works for 4, 5 and 6 parameters model.

        The value of nvars is the number of species in the experimental data. Only required when exp=True.
        """
        if len(self._params) == 4:
            def plot_temp(alpha, beta, theta, phi):
                params = [alpha, beta, theta, phi]
                params_old = self._params
                self.set_params(params)
                self.plot(t, exp=exp, ylim=ylim, nvars=nvars, acc=acc)
                self.set_params(params_old)
                print(params)
        elif len(self._params) == 5:
            def plot_temp(alpha, beta, theta, phi, K):
                params = [alpha, beta, theta, phi, K]
                params_old = self._params
                self.set_params(params)
                self.plot(t, exp=exp, ylim=ylim, nvars=nvars, acc=acc)
                self.set_params(params_old)
                print(params)
        elif len(self._params) == 6:
            def plot_temp(alpha, beta, theta, phi, K, kappa):
                params = [alpha, beta, theta, phi, K, kappa]
                params_old = self._params
                self.set_params(params)
                self.plot(t, exp=exp, ylim=ylim, nvars=nvars, acc=acc)
                self.set_params(params_old)
                print(params)

        params_list = ['alpha', 'beta', 'theta', 'phi', 'k', 'kappa']
        sliders = []

        for i in range(len(self._params)):
            slider = widgets.FloatSlider(value=self._params[i], min=max(
                0, self._params[i]-ran), max=self._params[i]+ran, step=step, description=params_list[i])
            sliders.append(slider)

        if len(self._params) == 4:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3])
        elif len(self._params) == 5:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3], K=sliders[4])
        elif len(self._params) == 6:
            interactive_widget = widgets.interactive(
                plot_temp, alpha=sliders[0], beta=sliders[1], theta=sliders[2], phi=sliders[3], K=sliders[4], kappa=sliders[5])
        display(interactive_widget)
# ---------------------------------------------------#

# -----------Mixed oscillation model: 2 surfactants-----------#


class mixed_oscillation(oscillation):
    def __init__(self, model, params1, params2, consts1, consts2, init_cond):

        params1 = list(params1.values()) if isinstance(
            params1, dict) else params1
        params2 = list(params2.values()) if isinstance(
            params2, dict) else params2
        consts1 = list(consts1.values()) if isinstance(
            consts1, dict) else consts1
        consts2 = list(consts2.values()) if isinstance(
            consts2, dict) else consts2

        # Model and parameters
        self._model = model['model']
        self._params = np.concatenate((params1, params2))
        self._consts = np.concatenate((consts1, consts2))
        self._calc_all = model['calc_all']
        self._init_cond = init_cond

        # Essential information
        self._species = ['A2', 'S1', 'S2', 'A', 'O']
        self._info = model['info']

        # Experimental data
        self._exp_data = None

    def plot(self, t=10, exp=False, method='RK45', ylim=None, npoints=500):
        color = ['purple', 'b', 'lightskyblue', 'r', 'g']

        sol = self.simulate(t, method=method, npoints=npoints)
        c = self._calc_all(sol, self._consts, self._params)
        fig, ax = plt.subplots(2, 2, figsize=(7, 5))

        ax[0, 0].plot(sol.t, c[0], label=self._species[0], color=color[0])
        ax[0, 1].plot(sol.t, c[1], label=self._species[1], color=color[1])
        ax[0, 1].plot(sol.t, c[2], label=self._species[2], color=color[2])
        ax[1, 0].plot(sol.t, c[3], label=self._species[3], color=color[3])
        ax[1, 1].plot(sol.t, c[4], label=self._species[4], color=color[4])

        for a in ax.flatten():
            if ylim != None:
                ax.set_ylim(0, ylim)
            a.legend(loc="upper right")

        fig.supxlabel('Normalized Time')
        fig.supylabel('Normalized Concentration')
        plt.tight_layout()
        return fig, ax
    
    def interactive_plot(self):
        pass

    def fit(self):
        pass
# ---------------------------------------------------#