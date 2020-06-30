from lib.algorithm import run
from lib.game import BilinearGame
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
                                                           "svre"))
parser.add_argument("-d", "--dim", default=100, type=int)
parser.add_argument("-s", "--seed", default=1234, type=int)
parser.add_argument("--num-seeds", default=5, type=int)
parser.add_argument("-", "--num-iter", default=50000, type=int)
args = parser.parse_args()

if args.mode == "shgd-constant":
    mode_id = 0
elif args.mode == "shgd-decreasing":
    mode_id = 1
elif args.mode == "shgd-biased":
    mode_id = 2
elif args.mode == "svrh":
    mode_id = 3
elif args.mode == "svre":
    mode_id = 4

torch.manual_seed(args.seed)
game = BilinearGame(args.dim, bias=True)
output_path = os.path.join(args.output_path, "bilinear")

list_seeds = [i for i in range(args.num_seeds)]
default_hyper = dict(output=output_path, game=game,
                     num_iter=args.num_iter, seed=list_seeds)

list_hyper_all = [dict(mode="shgd", lr=5e-1, biased=False, lr_schedule=None),
                  dict(mode="shgd", lr=5e-1, biased=False, lr_schedule=10000),
                  dict(mode="shgd", lr=5e-1, biased="copt", lr_schedule=None),
                  dict(mode="svrh", lr=10, prob=1e-2),
                  dict(mode="svre", lr=3e-1, prob=1e-2, restart=0.1)]

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

color = [2, 3, 1, 4, 5]
name = ["SHGD (constant step-size)", "SHGD (decreasing step-size)",
        "Biased SHGD", "L-SVRHG", "SVRE"]

fig, ax = utils.plot(results_grouped, "distance_to_optimum", averaging=False,
                     scale=1, subsampling=1, name=name, color=color)
ax.set_yscale("log")
ax.legend()
ax.set_xlabel("Num of samples")
ax.set_ylabel(r"$\frac{||x_k - x^*||^2}{||x_0 - x^*||^2}$")
ax.ticklabel_format(style='sci', axis='x', scilimits=(2, 2))
ax.set_xlim(left=0, right=1e5)
ax.set_ylim(bottom=1e-8, top=2)

if not os.path.exists(os.path.join(args.output_path, "plots")):
    os.makedirs(os.path.join(args.output_path, "plots"))
fig.savefig(os.path.join(args.output_path, "plots", "bilinear.pdf"))
