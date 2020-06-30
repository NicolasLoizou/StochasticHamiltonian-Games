from lib.algorithm import run
from lib.game import NonMonotoneGame
from lib import utils
import torch
import itertools
import argparse
import os
import copy

parser = argparse.ArgumentParser()
parser.add_argument("output_path")
parser.add_argument("-m", "--mode", default=None, choices=("shgd-constant",
                                                           "shgd-decreasing",
                                                           "shgd-biased",
                                                           "svrh",
                                                           "svrh-restart",
                                                           "svre"))
parser.add_argument("-d", "--dim", default=100, type=int)
parser.add_argument("-s", "--seed", default=1234, type=int)
parser.add_argument("--num-seeds", default=5, type=int)
parser.add_argument("--scale", default=7, type=float)
parser.add_argument("--num-iter", default=200000, type=int)
args = parser.parse_args()

if args.mode == "shgd-constant":
    mode_id = 0
elif args.mode == "shgd-decreasing":
    mode_id = 1
elif args.mode == "shgd-biased":
    mode_id = 2
elif args.mode == "svrh":
    mode_id = 3
elif args.mode == "svrh-restart":
    mode_id = 4
elif args.mode == "svre":
    mode_id = 5

torch.manual_seed(args.seed)
game = NonMonotoneGame(args.dim, args.scale, bias=True)
output_path = os.path.join(args.output_path, "sufficiently_bilinear")

list_seeds = [i for i in range(args.num_seeds)]
default_hyper = dict(output=output_path, game=game,
                     num_iter=args.num_iter, seed=list_seeds)

list_hyper_all = [dict(mode="shgd", biased=False, lr=2e-2, lr_schedule=None),
                  dict(mode="shgd", biased=False, lr=2e-2, lr_schedule=10000),
                  dict(mode="shgd", biased="copt", lr=1e-2),
                  dict(mode="svrh", prob=1e-2, lr=1e-1),
                  dict(mode="restart-svrh", prob=1e-2, lr=1e-1, restart=1e3),
                  dict(mode="svre", biased=False, lr=5e-2, restart=1e-1)]

if args.mode is not None:
    list_hyper = [list_hyper_all[mode_id], ]
else:
    list_hyper = list_hyper_all


def generate_config(list_hyper, default={}):
    list_hyper = copy.deepcopy(list_hyper)
    list_config = []
    for hyper in list_hyper:
        hyper.update(default_hyper)
        list_values = []
        for key, value in hyper.items():
            if not isinstance(value, (tuple, list)):
                value = [value]
            list_values.append(value)
        p = itertools.product(*list_values)
        for c in p:
            config = default.copy()
            for key, value in zip(hyper.keys(), c):
                config[key] = value
            list_config.append(config)
    return list_config


list_configs = generate_config(list_hyper, default=default_hyper)

for i, config in enumerate(list_configs):
    print("Running job %i out of %i" % (i+1, len(list_configs)))
    run(config)

print("Generating plot...")
list_logs = utils.load_exp(os.path.join(output_path, "*/*"))
list_id = utils.filter_logs(list_logs, list_hyper_all)
results_grouped = utils.merge_group(list_logs, list_id)

color = [2, 3, 1, 4, 6, 5]
name = ["SHGD (constant step-size)", "SHGD (decreasing step-size)",
        "Biased SHGD", "L-SVRHG", "L-SVRHG with restart", "SVRE"]

fig, ax = utils.plot(results_grouped, "grad_norm", averaging=False, scale=2,
                     subsampling=1, name=name, color=color)
ax.set_yscale("log")
ax.legend(loc="lower right")
ax.set_xlabel("Num of samples")
ax.set_ylabel(r"$\frac{H(x_k)}{H(x_0)}$")
ax.ticklabel_format(style='sci', axis='x', scilimits=(2, 2))
ax.set_xlim(left=0, right=4000e2)
ax.set_ylim(top=2, bottom=1e-8)

if not os.path.exists(os.path.join(args.output_path, "plots")):
    os.makedirs(os.path.join(args.output_path, "plots"))
fig.savefig(os.path.join(args.output_path, "plots", "sufficiently_bilinear.pdf"))
