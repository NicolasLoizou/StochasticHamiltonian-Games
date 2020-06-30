import glob
import os
import json
from collections import defaultdict
import warnings
import types
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 7, 5
list_colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple",
               "tab:brown", "tab:pink", "tab:gray", "tab:cyan", "tab:olive"]


def load_exp(path):
    list_exp = glob.glob(path)
    list_logs = {}
    list_modes = []
    for i, exp in enumerate(list_exp):
        try:
            with open(os.path.join(exp, "config.json")) as f:
                config = json.load(f)
            with open(os.path.join(exp, "results.json")) as f:
                results = defaultdict(list)
                results.update(json.load(f))
            log_id = str(i)
            list_logs[log_id] = dict(name=config["name"], config=config,
                                     results=results)
            if config["mode"] not in list_modes:
                list_modes.append(config["mode"])
        except:
            warnings.warn("Failed to load: %s" % exp)
    return list_logs


def filter_logs(list_logs, list_filters):
    if isinstance(list_filters, dict):
        list_filters = [list_filters, ]
    list_id_grouped = []

    for filter in list_filters:
        list_id = []
        for log_id, log in list_logs.items():
            flag = True
            for key, value in filter.items():
                _flag = False
                if not isinstance(value, (list, tuple)):
                    value = (value,)
                for v in value:
                    if isinstance(v, types.FunctionType):
                        if v(log["config"][key]):
                            _flag = True
                        continue
                    if log["config"][key] == v:
                        _flag = True
                flag = flag and _flag
            if flag:
                list_id.append(log_id)
        list_id_grouped.append(list_id)
    return list_id_grouped


def merge_group(list_logs, list_id_grouped):
    results_grouped = []
    for group in list_id_grouped:
        group_log = dict(results=defaultdict(list), seed=[],
                         num_points=defaultdict(list))
        for log_id in group:
            log = list_logs[log_id]
            if log["config"]["seed"] in group_log["seed"]:
                continue
            for key, results in log["results"].items():
                group_log["results"][key].append(results)
                group_log["num_points"][key].append(len(results))
            group_log["seed"].append(log["config"]["seed"])
            group_log["name"] = log["config"]["name"]
        list_keys = list(group_log["results"].keys())
        for key in list_keys:
            results_list = group_log["results"][key]
            results = []
            num_points = min(group_log["num_points"][key])
            for r in results_list:
                results.append(r[:num_points])
            results = np.array(results)
            if "distance_to_optimum" in key or "grad_norm" in key:
                if results.shape[1] > 0:
                    results = results/results[:, 0].reshape(-1, 1)
                results = np.log10(results)
            group_log["results"][key] = np.mean(results, 0)
            group_log["results"][key + ":std"] = np.std(results, 0)
        results_grouped.append(group_log)
    return results_grouped


def get_logs(list_logs, list_id):
    list_logs_filter = []
    for group in list_id:
        if isinstance(group, str):
            group = [group, ]
        for id in group:
            list_logs_filter.append(list_logs[id])
    return list_logs_filter


def split_list(list_id):
    new_list_id = []
    for group in list_id:
        for id in group:
            new_list_id.append([id])
    return new_list_id


def plot(list_logs, key, averaging=False, scale=1, subsampling=1e4,
         name=None, color=None):
    list_name = []
    subsampling = int(subsampling)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, log in enumerate(list_logs):
        if name is not None:
            _name = name[i]
        else:
            _name = log["name"]
        if _name in list_name:
            continue
        list_name.append(_name)

        _color = None
        if color is not None:
            _color = list_colors[color[i]]

        results = log["results"]
        num_samples = results["num_samples"][::subsampling]
        if averaging and len(results[key+":avg"]) > 0:
            data = results[key+":avg"][::subsampling]
            label = "%s avg" % _name
        else:
            data = results[key][::subsampling]
            label = "%s" % _name
        if key+":std" in results:
            std = results[key+":std"][::subsampling]
            upper = data+scale*std
            lower = data-scale*std
            upper = np.power(10, upper)
            lower = np.power(10, lower)
            ax.fill_between(num_samples, lower, upper, alpha=0.5, color=_color)
        data = np.power(10, data)
        ax.plot(num_samples, data, label=label, color=_color)
    return fig, ax


def plot_grad(game, grid=((-1, 1), (-1, 1)), fig=None):
    if fig is None:
        fig = plt.figure()

    if len(fig.axes) < 1:
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]

    x = torch.linspace(grid[0][0], grid[0][1], 10)
    y = torch.linspace(grid[1][0], grid[1][1], 10)
    grid_x, grid_y = torch.meshgrid(x, y)
    grid_x = grid_x.contiguous().view(-1)
    grid_y = grid_y.contiguous().view(-1)
    grad = []
    for x, y in zip(grid_x, grid_y):
        game.players[0].data = x
        game.players[1].data = y
        g = game.compute_grad(game.sample_batch())
        grad.append((g[0], g[1]))
    grad = -torch.tensor(grad)
    ax.quiver(grid_x, grid_y, grad[:, 0], grad[:, 1])

    return fig


def plot_trajectory(logger, fig=None, subsampling=1, name=None, color=None):
    if fig is None:
        fig = plt.figure()

    if len(fig.axes) < 1:
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]

    if name is None:
        name = logger["config"]["name"]

    if color is not None:
        color = list_colors[color]

    data = []
    for params in logger["params"]:
        data.append((params["players.0"], params["players.1"]))
    data = torch.Tensor(data)
    data = data[::subsampling]
    ax.plot(data[:, 0], data[:, 1], label=name, color=color)

    return fig


def make_spd_matrix(n_samples, dim):
    A = torch.rand(n_samples, dim, dim)
    X = torch.bmm(A.transpose(-1, -2), A)
    U, s, V = torch.svd(X)
    X = torch.bmm(torch.bmm(U, 1.0 + torch.diag_embed(torch.rand(n_samples, dim))), V.transpose(-1, -2))
    return X
