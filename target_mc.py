"""
Tests different minimization targets and uncertainty calculation strategies using a Monte Carlo
framework. Calibrations are required to be polynomials with a relatively low number of coefficients.

"""
import os.path
from collections import defaultdict
from collections.abc import MutableMapping
import warnings
import json
import datetime
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
import lmfit
from uncertainties import ufloat, UFloat, correlation_matrix
from uncertainties.unumpy import nominal_values, std_devs
from alive_progress import alive_bar

from lines import Line, Observation


class Config:

    cals: tuple[NDArray, ...]
    cal_npoly: tuple[int, ...]
    cal_have_ref: list[int]  # whether to include reference lines in that spectrum

    num_ref_lines: int
    num_noref_lines: int
    num_mc_iters: int

    num_pixels: int
    pixels: NDArray[float]
    pixel_uncs: tuple[float, float]
    wl_uncs: tuple[float, float]

    def __init__(self, cals: tuple[NDArray, ...], num_ref_lines: int, num_noref_lines: int,
                 cal_have_ref: list[int] | None = None, num_mc_iters: int = 100, num_pixels: int = 2048, pixel_uncs: tuple[float, float] = (0.05, 0.5),
                 wl_uncs: tuple[float, float] = (0.0001, 0.001), rng_seed: int = 124):

        self.cals = cals
        self.cal_npoly = tuple([len(cal) for cal in cals])
        self.cal_have_ref = cal_have_ref if cal_have_ref is not None else list(range(len(cals)))

        self.num_ref_lines = num_ref_lines
        self.num_noref_lines = num_noref_lines
        self.num_mc_iters = num_mc_iters

        self.num_pixels = num_pixels
        self.pixels = np.linspace(0, self.num_pixels-1, self.num_pixels, dtype='float64')
        self.pixel_uncs = pixel_uncs
        self.wl_uncs = wl_uncs
        self.rng = np.random.default_rng(rng_seed)

        # assumes calibrations are monotonic TODO Check that this is the case.
        self.min_wl = min(poly(0, *coeffs) for coeffs in self.cals)
        self.max_wl = min(poly(self.num_pixels, *coeffs) for coeffs in self.cals)

    def save(self, folder):

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        if not os.path.isdir(folder):
            os.mkdir(folder)

        with open(folder + '/config.json', 'w') as f:
            json.dump({k: v for k, v in self.__dict__.items() if k != 'rng'}, f, cls=NumpyEncoder)


def poly(x, *coeffs) -> float | UFloat:
    return sum([c * x ** i for i, c in enumerate(coeffs)])


def invert_poly(y, *coeffs):
    return np.interp(y, poly(config.pixels, *coeffs), config.pixels)


def generate_observations(cfg: Config) -> tuple[list[Observation], dict[str, float]] | None:
    """Generates observtions according to config. General strategy is to create a list of lines that will appear
    using the range (cfg.min_wl, cfg.max_wl), and then determine the position of those lines in the other spectra.
    We then add noise to the line positions and reference wavelength (for reference lines)."""
    observations = []
    true_wls = {}

    # Generate reference lines (which have literature wavelength measurements) and noreference lines (that do not)
    line_wls = cfg.rng.uniform(cfg.min_wl, cfg.max_wl, cfg.num_ref_lines + cfg.num_noref_lines)
    line_uncs = cfg.rng.uniform(*cfg.wl_uncs, cfg.num_ref_lines + cfg.num_noref_lines)

    lines = []
    for i, (true_wl, wl_unc) in enumerate(zip(line_wls, line_uncs)):
        label = f'ref_line{i}' if i < cfg.num_ref_lines else f'noref_line{i-cfg.num_ref_lines}'
        true_wls[label] = true_wl
        lines.append((true_wl,
                      Line(wavelength=None if 'noref' in label else ufloat(true_wl + cfg.rng.normal(0, wl_unc), wl_unc), notes=label)))

    for i, cal in enumerate(cfg.cals):
        min_wl, max_wl = poly(0, *cal), poly(cfg.num_pixels-1, *cal)

        # Check line is in wavelength range and remove ref lines if i not in config.cal_have_ref
        cal_lines = [(wl, line) for wl, line in lines if min_wl < wl < max_wl and
                     not (line.wavelength is not None and i not in config.cal_have_ref)]
        line_pos = np.array([invert_poly(wl, *cal) for (wl, line) in cal_lines])
        line_uncs = cfg.rng.uniform(*cfg.pixel_uncs, len(cal_lines))

        for (_, line), (x, dx) in zip(cal_lines, zip(line_pos, line_uncs)):
            obs = Observation(line=line, ind_vars={'x': ufloat(x + cfg.rng.normal(0, dx), dx), 'cidx': i})
            observations.append(obs)

    return observations, true_wls


class FitResult:

    obs: list[Observation]
    pred_wls: dict[str, UFloat]
    minres: lmfit.minimizer.MinimizerResult
    true_wls: dict[str, float] | None
    cfg: Config | None

    stats: dict | None

    def __init__(self, obs: list[Observation], pred_wls: dict[str, UFloat], minres: lmfit.minimizer.MinimizerResult):
        self.obs = obs
        self.pred_wls = pred_wls
        self.minres = minres

        self.true_wls, self.cfg, self.stats = None, None, None

    def add_true_vals(self, true_wls, cfg: Config):
        self.true_wls = true_wls
        self.cfg = cfg

    def generate_statistics(self):
        if self.cfg is None or self.true_wls is None:
            raise ValueError("Cannot calculate statistics when self.cals or self.true_wls is None")

        stats = {}

        # Compare the predicted wavelengths to the true_wavelengths
        # noinspection PyTypeChecker
        resids = np.array([self.pred_wls[label] - true_wl for label, true_wl in self.true_wls.items() if label in self.pred_wls])
        stats['wls_RMSE'] = np.sqrt(np.square(nominal_values(resids)).mean())
        stats['wls_MSD'] = nominal_values(resids).mean()
        stats['wls_MNR'] = (np.abs(nominal_values(resids)) / std_devs(resids)).mean()
        
        for category in ['ref', 'noref']:
            # noinspection PyTypeChecker
            resids_cat = np.array([self.pred_wls[label] - true_wl for label, true_wl in self.true_wls.items() if category in label and label in self.pred_wls])
            stats[f'wls_RMSE_{category}'] = np.sqrt(np.square(nominal_values(resids_cat)).mean())
            stats[f'wls_MSD_{category}'] = nominal_values(resids_cat).mean()
            stats[f'wls_MNR_{category}'] = (np.abs(nominal_values(resids_cat)) / std_devs(resids_cat)).mean()
        # Compare the calibration wavelengths to the predicted wavelengths
        for i, cal in enumerate(self.cfg.cals):
            true_all_wls = poly(config.pixels, *cal)
            pred_coeffs = [self.minres.uvars[f'c{i}_{j}'] for j in range(len(cal))]
            pred_all_wls = poly(config.pixels, *pred_coeffs)
            resid_all_wls = pred_all_wls - true_all_wls
            stats[f'cal{i}'] = {}
            stats[f'cal{i}']['RMSE'] = np.sqrt(np.square(nominal_values(resid_all_wls)).mean())
            stats[f'cal{i}']['MSD'] = nominal_values(resid_all_wls).mean()
            stats[f'cal{i}']['MNR'] = (np.abs(nominal_values(resid_all_wls)) / std_devs(resid_all_wls)).mean()

        self.stats = stats

        return stats


def hunter_fitter(obs: list[Observation], cal_npoly: tuple[int, ...]) -> FitResult:
    def target(pars: lmfit.Parameters):
        coeffs_lists = [[pars[f'c{i}_{p}'].value for p in range(n)] for i, n in enumerate(cal_npoly)]
        resids = []
        for lname in line_names:
            match_obs = [ob for ob in obs if ob.line.notes == lname]
            y_pred: list[UFloat] = [poly(ob.ind_vars['x'], *coeffs_lists[int(ob.ind_vars['cidx'])]) for ob in match_obs]

            if (ref_wl := match_obs[0].line.wavelength) is not None:  # If there is a reference wavelength include it now
                y_pred.append(ref_wl)

            y_predmu = nominal_values(y_pred)
            y_predunc = std_devs(y_pred)

            if pars[f'wl{lname}'].vary is True:  # Not jumping to optimal value
                y_opt = pars[f'wl{lname}'].value
            else:  # compute weighted mean
                y_opt = (y_predmu / y_predunc ** 2).sum() / (1 / y_predunc ** 2).sum()
                opt_values[lname] = y_opt

            resids.append((y_predmu - y_opt) / y_predunc)

        return np.hstack(resids)

    opt_values = {}
    guesses = {0: 2, 1: 2.5e-3}
    params = lmfit.Parameters()
    for cidx, num in enumerate(cal_npoly):
        for power in range(num):
            params.add(f'c{cidx}_{power}', guesses.get(power, 0))

    line_names = set(obs.line.notes for obs in obs)
    for line_name in line_names:
        params.add(f'wl{line_name}', 0, vary=False)

    params = lmfit.minimize(target, params, scale_covar=False).params

    for line_name in line_names:
        params[f'wl{line_name}'].vary = True
        params[f'wl{line_name}'].value = opt_values[line_name]

    minres = lmfit.minimize(target, params, scale_covar=False)
    pred_wls = {line_name: minres.uvars[f'wl{line_name}'] for line_name in line_names}

    return FitResult(obs, pred_wls, minres)


def sasha_fitter1(obs: list[Observation], cal_npoly: tuple[int, ...]) -> FitResult:
    def target(pars: lmfit.Parameters):
        coeffs_lists = [[pars[f'c{i}_{p}'].value for p in range(n)] for i, n in enumerate(cal_npoly)]
        resids = []
        for lname in line_names:
            match_obs = [ob for ob in obs if ob.line.notes == lname]
            y_pred: list[UFloat] = [poly(ob.ind_vars['x'], *coeffs_lists[int(ob.ind_vars['cidx'])]) for ob in match_obs]
            y_predmu = nominal_values(y_pred)
            y_predunc = std_devs(y_pred)

            if (ref_wl := match_obs[0].line.wavelength) is not None:  # ref line
                opt_values[lname] = ref_wl
                # y_predunc = np.sqrt(std_devs(y_pred) + ref_wl.s ** 2)  # include uncertainty of calibration line
                resids.append((y_predmu - ref_wl.n) / y_predunc)

            else:  # noref line
                y_wm_uvar = (np.array(y_pred) / y_predunc ** 2).sum() / (
                            1 / y_predunc ** 2).sum()  # calc weighted average
                y_predunc = np.sqrt(std_devs(y_pred) + y_wm_uvar.s ** 2)  # include uncertainty of weighted mean
                opt_values[lname] = y_wm_uvar
                resids.append((y_predmu - y_wm_uvar.n) / y_predunc)

        return np.hstack(resids)

    opt_values = {}
    guesses = {0: 2, 1: 2.5e-3}
    params = lmfit.Parameters()
    for cidx, num in enumerate(cal_npoly):
        for power in range(num):
            params.add(f'c{cidx}_{power}', guesses.get(power, 0))

    line_names = set(obs.line.notes for obs in obs)
    for line_name in line_names:
        params.add(f'wl{line_name}', 0, vary=False)

    minres = lmfit.minimize(target, params, scale_covar=False)
    params = minres.params
    coefficient_lists = [[minres.uvars[f'c{cidx}_{power}'] for power in range(num)] for cidx, num in enumerate(cal_npoly)]
    for line_name in line_names:
        match_observ = [ob for ob in obs if ob.line.notes == line_name]
        y_pred_final = [poly(ob.ind_vars['x'], *coefficient_lists[int(ob.ind_vars['cidx'])]) for ob in match_observ]  # contains uncertainty from x and coefficients

        # Generalized weighted average, considers correlations
        y_pred_cov = correlation_matrix(y_pred_final)
        y_pred_cov_inv = ((1 / np.tile(std_devs(y_pred_final), (len(y_pred_final), 1)).T) * np.linalg.inv(y_pred_cov) *
                          (1 / np.tile(std_devs(y_pred_final), (len(y_pred_final), 1))))
        y_wm_uvar_final = np.ones(len(y_pred_final))[None, :] @ y_pred_cov_inv @ y_pred_final / np.sum(y_pred_cov_inv)
        opt_values[line_name] = y_wm_uvar_final
        params[f'wl{line_name}'].value = y_wm_uvar_final

    return FitResult(obs, opt_values, minres)


def save_stats(fitreses: list[FitResult], fitter_name: str, folder: str):

    def flatten(dictionary, parent_key='', separator='_'):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(flatten(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    if folder is not None:
        outfile = folder + f'/{fitter_name}.csv'
        stats = [flatten(fitres.stats) for fitres in fitreses]
        pd.DataFrame.from_records(stats).to_csv(outfile, index=False)


def summarize_stats(folder: str, fitter_name: str | None = None, axes: list[plt.Axes] | None = None, **plotkwargs):

    stats = pd.read_csv(folder + f'/{fitter_name}.csv', header=0).to_dict(orient='series')
    # noinspection PyTypeChecker
    stats = [(k, stats[k].to_numpy()) for k in sorted(list(stats.keys()))]
    for i, (stat_name, stat_vals) in enumerate(stats):
        stat_vals = stat_vals[(stat_vals > np.percentile(stat_vals, 2.5)) & (stat_vals < np.percentile(stat_vals, 97.5))]

        if axes is None:
            plt.subplots(figsize=(10, 6))
        else:
            plt.sca(axes[i])

        plt.hist(stat_vals, bins=50, label=fitter_name, **plotkwargs)
        plt.title(f'Distribution of {stat_name}')
        plt.xlabel(f'{stat_name}')
        plt.ylabel('count')
        if fitter_name:
            plt.legend()


if __name__ == '__main__':

    plt.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})
    warnings.simplefilter(action='ignore', category=FutureWarning)

    calibrations = np.array([3.0, 2e-3, 1.5e-6]), np.array([2, 2.5e-3, 1.3e-6]), np.array([2.3, 2.2e-3, 1e-6])
    config = Config(calibrations, 20, 25, num_mc_iters=500, cal_have_ref=[0])

    fitters: dict[str, Callable[[list[Observation], tuple[int, ...]], FitResult]] = {
        'hunter_fitter': hunter_fitter,
        'sasha_fitter1': sasha_fitter1
    }

    fitresults = defaultdict(list)
    with alive_bar(config.num_mc_iters, spinner='crab', force_tty=True) as bar:
        niters = 0
        while niters < config.num_mc_iters:
            data, true_wavelengths = generate_observations(config)
            for fittername, fitter in fitters.items():
                try:
                    fitresult = fitter(data, config.cal_npoly)
                except AttributeError as E:
                    print(f'Fit failed (no uncertainties) on iter {niters} with fitter {fittername}')
                    break
                fitresult.add_true_vals(true_wavelengths, config)
                fitresult.generate_statistics()  # calc rmse, mse, msd
                fitresults[fittername].append(fitresult)
            else:
                niters += 1
                bar()

    axs = [plt.subplots()[1] for _ in range(9 + 3 * len(config.cals))]  # TODO: Make this decoupled, maybe class for stats?

    dtstr = datetime.datetime.now().strftime("%H%M%S_%m%d%Y")
    run_folder = f'results/{len(config.cals)}cals_{config.num_ref_lines}ref_{config.num_noref_lines}noref_{config.num_mc_iters}iters_{dtstr}'
    config.save(folder=run_folder)
    save_stats(fitresults['hunter_fitter'], fitter_name='hunter_fitter', folder=run_folder)
    save_stats(fitresults['sasha_fitter1'], fitter_name='sasha_fitter1', folder=run_folder)

    summarize_stats(folder=run_folder, fitter_name='hunter_fitter', axes=axs, color='b', alpha=0.5)
    summarize_stats(folder=run_folder, fitter_name='sasha_fitter1', axes=axs, color='r', alpha=0.5)

    plt.show()
