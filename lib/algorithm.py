import torch
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
from collections import defaultdict
from torch import autograd
import copy
from .extragradient import Extragradient
import json
import os
import time
import numpy as np
from .game import compute_hamiltonian
from tqdm import tqdm


class Config(dict):
    defaults = dict(mode=None, num_iter=None, lr=None, seed=None, biased=False,
                    prob=None, restart=None, lr_schedule=None,
                    same_sample=False, hamiltonian_coeff=None,
                    batch_size=None, shuffling=False)

    def __init__(self, config={}):
        defaults = self.defaults.copy()
        self.update(defaults)
        self.update(config)

    @property
    def name(self):
        name = ""
        if self["biased"] is True:
            name += "biased_"
        elif self["biased"] == "copt":
            name += "copt_"
        name += "%s lr=%.1e" % (self["mode"], self["lr"])
        if self["lr_schedule"] is not None:
            name += " lr_schedule=%i" % (self["lr_schedule"])
        if self["prob"] is not None:
            name += " prob=%.1e" % (self["prob"])
        if self["restart"] is not None:
            name += " restart=%.1e" % (self["restart"])
        if self["batch_size"] is not None:
            name += " size=%.1e" % (self["batch_size"])
        if self["hamiltonian_coeff"] is not None:
            name += " coeff=%.1e" % (self["hamiltonian_coeff"])
        if self["shuffling"]:
            name += " with shuffling"
        return name


def run(args):
    for key, value in args.items():
        if isinstance(value, str):
            if value.lower() == 'true':
                args[key] = True
            if value.lower() == 'false':
                args[key] = False

    if args["mode"] == "shgd":
        logger, config = run_SHGD(**args)
    elif args["mode"] == "hgd":
        logger, config = run_HGD(**args)
    elif args["mode"] == "svrh":
        logger, config = run_SVRH(**args)
    elif args["mode"] == "restart-svrh":
        logger, config = run_restartSVRH(**args)
    elif args["mode"] == "semi-svrh":
        logger, config = run_semiSVRH(**args)
    elif args["mode"] == "noisy-svrh":
        logger, config = run_noisySVRH(**args)
    elif args["mode"] == "extragradient":
        logger, config = run_extragradient(**args)
    elif args["mode"] == "svre":
        logger, config = run_SVRE(**args)
    elif args["mode"] == "svrg":
        logger, config = run_SVRG(**args)
    elif args["mode"] == "sgd":
        logger, config = run_SGD(**args)
    elif args["mode"] == "consensus_opt":
        logger, config = run_COPT(**args)
    else:
        raise ValueError("Mode %s is not supported" % args["mode"])

    path = config["path"]

    torch.save(logger["params"], os.path.join(path, "params.json"))

    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump(logger, f, default=lambda o: '<not serializable>')

    logger["config"] = config

    return logger, config


class SchedulerLR():
    def __init__(self, num_start):
        self.num_start = num_start

    def __call__(self, k):
        if k <= self.num_start:
            gamma = 1
        else:
            gamma = (self.num_start/2)*(2*k+1)/(k+1)**2
        return gamma


# Full-Batch Hamiltonian
def run_HGD(game, num_iter=5000, lr=0.5, seed=1234,
            save_params=False, **kwargs):
    config = Config(dict(mode="hgd", num_iter=num_iter, lr=lr, seed=seed))
    torch.manual_seed(seed)
    game.reset()
    sgd = optim.SGD(game.parameters(), lr=lr)
    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    n_samples = 0
    params_history = []
    start_time = time.time()
    for i in tqdm(range(num_iter)):
        index = game.sample_batch()
        hamiltonian = game.compute_hamiltonian(index)
        grad = autograd.grad(hamiltonian, game.parameters())
        for p, g in zip(game.parameters(), grad):
            p.grad = g
        sgd.step()

        n_samples += 2*len(index)

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)
        logger["num_samples"].append(n_samples)
        logger["time"].append(time.time()-start_time)

        if save_params:
            params_history.append(copy.deepcopy(game.state_dict()))

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    logger["params"] = params_history
    return logger, config


# Hamiltonian SGD
def run_SHGD(game, num_iter=5000, lr=None, seed=1234, save_params=False,
             biased=False, shuffling=False, lr_schedule=None, **kwargs):
    if lr is None:
        lr = float(1/(2*game.L))
    if lr_schedule == "optimal":
        lr_schedule = int(4*(game.L/game.mu))

    config = Config(dict(mode="shgd", num_iter=num_iter, lr=lr, seed=seed,
                         biased=biased, lr_schedule=lr_schedule,
                         shuffling=shuffling))
    torch.manual_seed(seed)
    game.reset()

    sgd = optim.SGD(game.parameters(), lr=lr)
    if lr_schedule is not None:
        lr_schedule = SchedulerLR(lr_schedule)
        scheduler = LambdaLR(sgd, lr_schedule)
    else:
        scheduler = LambdaLR(sgd, lambda k: 1.)
    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    if shuffling:
        game.shuffle()

    n_samples = 0
    params_history = []
    start_time = time.time()
    for i in tqdm(range(num_iter)):
        index1 = game.sample()
        index2 = game.sample()
        if biased is True:
            hamiltonian = game.compute_hamiltonian(index1)
            n_samples += 1
        elif biased == "copt":
            hamiltonian = game.compute_hamiltonian(torch.cat([index1, index2]))
            n_samples += 2
        elif biased is False:
            hamiltonian = game.compute_hamiltonian(index1, index2)
            n_samples += 2
        else:
            raise ValueError()
        grad = autograd.grad(hamiltonian, game.parameters())
        for p, g in zip(game.parameters(), grad):
            p.grad = g
        sgd.step()
        scheduler.step()

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)
        #logger["lr"].append(scheduler.get_last_lr())
        logger["num_samples"].append(n_samples)
        logger["time"].append(time.time()-start_time)

        if save_params:
            params_history.append(copy.deepcopy(game.state_dict()))

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    logger["params"] = params_history
    return logger, config


# Hamiltonian L-SVRG
def run_SVRH(game, num_iter=5000, lr=0.2, seed=1234, save_params=False,
             biased=False, shuffling=False, prob=0.01, **kwargs):
    config = Config(dict(mode="svrh", num_iter=num_iter, lr=lr, seed=seed,
                         biased=biased, prob=prob, shuffling=shuffling))
    torch.manual_seed(seed)
    game.reset()
    game_snapshot = copy.deepcopy(game)
    sgd = optim.SGD(game.parameters(), lr=lr)

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    if shuffling:
        game.shuffle()

    params_history = []
    start_time = time.time()
    batch = game.sample_batch()
    hamiltonian = game_snapshot.compute_hamiltonian(batch)
    batch_grad = autograd.grad(hamiltonian, game_snapshot.parameters())

    n_samples = 2*len(batch)
    for i in tqdm(range(num_iter)):
        index1 = game.sample()
        index2 = game.sample()

        if biased is True:
            hamiltonian = game.compute_hamiltonian(index1)
            hamiltonian_snap = game_snapshot.compute_hamiltonian(index1)
            n_samples += 2

        elif biased == "copt":
            hamiltonian = game.compute_hamiltonian(torch.cat([index1, index2]))
            hamiltonian_snap = game_snapshot.compute_hamiltonian(
                               torch.cat([index1, index2]))
            n_samples += 4

        elif biased is False:
            hamiltonian = game.compute_hamiltonian(index1, index2)
            hamiltonian_snap = game_snapshot.compute_hamiltonian(index1,
                                                                 index2)
            n_samples += 4
        else:
            raise ValueError()

        grad = autograd.grad(hamiltonian, game.parameters())
        grad_snapshot = autograd.grad(hamiltonian_snap,
                                      game_snapshot.parameters())

        for p, g, gs, bg in zip(game.parameters(), grad,
                                grad_snapshot, batch_grad):
            p.grad = g - gs + bg
        sgd.step()

        if torch.rand(1) < prob:
            game_snapshot.load_state_dict(game.state_dict())
            batch = game.sample_batch()
            hamiltonian = game_snapshot.compute_hamiltonian(batch)
            batch_grad = autograd.grad(hamiltonian, game_snapshot.parameters())
            n_samples += 2*len(batch)

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)
        logger["num_samples"].append(n_samples)
        logger["time"].append(time.time()-start_time)

        if save_params:
            params_history.append(copy.deepcopy(game.state_dict()))

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    logger["params"] = params_history
    return logger, config


def run_restartSVRH(game, num_iter=5000, lr=0.2, seed=1234, save_params=False,
                    biased=False, shuffling=False, prob=0.01,
                    restart=100, **kwargs):
    config = Config(dict(mode="restart-svrh", num_iter=num_iter, lr=lr,
                         seed=seed, biased=biased, prob=prob,
                         shuffling=shuffling, restart=restart))
    torch.manual_seed(seed)
    game.reset()
    game_snapshot = copy.deepcopy(game)
    sgd = optim.SGD(game.parameters(), lr=lr)

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    if shuffling:
        game.shuffle()

    players_history = [copy.deepcopy(game.state_dict())]
    params_history = []
    n_samples = 0
    for i in tqdm(range(num_iter)):
        if i % restart == 0:
            state_dict = np.random.choice(players_history)
            game.load_state_dict(state_dict)
            game_snapshot.load_state_dict(state_dict)

            batch = game.sample_batch()
            hamiltonian = game_snapshot.compute_hamiltonian(batch)
            batch_grad = autograd.grad(hamiltonian, game_snapshot.parameters())
            n_samples += 2*len(batch)
            players_history = [state_dict]

        index1 = game.sample()
        index2 = game.sample()

        if biased is True:
            hamiltonian = game.compute_hamiltonian(index1)
            hamiltonian_snap = game_snapshot.compute_hamiltonian(index1)
            n_samples += 2

        elif biased == "copt":
            hamiltonian = game.compute_hamiltonian(torch.cat([index1, index2]))
            hamiltonian_snap = game_snapshot.compute_hamiltonian(
                               torch.cat([index1, index2]))
            n_samples += 4

        elif biased is False:
            hamiltonian = game.compute_hamiltonian(index1, index2)
            hamiltonian_snap = game_snapshot.compute_hamiltonian(index1,
                                                                 index2)
            n_samples += 4

        else:
            raise ValueError()

        grad = autograd.grad(hamiltonian, game.parameters())
        grad_snapshot = autograd.grad(hamiltonian_snap,
                                      game_snapshot.parameters())

        for p, g, gs, bg in zip(game.parameters(), grad,
                                grad_snapshot, batch_grad):
            p.grad = g - gs + bg
        sgd.step()

        players_history.append(copy.deepcopy(game.state_dict()))

        if torch.rand(1) < prob:
            game_snapshot.load_state_dict(game.state_dict())
            batch = game.sample_batch()
            hamiltonian = game_snapshot.compute_hamiltonian(batch)
            batch_grad = autograd.grad(hamiltonian, game_snapshot.parameters())
            n_samples += 2*len(batch)

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)
        logger["num_samples"].append(n_samples)

        if save_params:
            params_history.append(copy.deepcopy(game.state_dict()))

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    logger["params"] = params_history
    return logger, config


def run_COPT(game, num_iter=5000, lr=0.5, seed=1234, biased=False,
             shuffling=False, lr_schedule=None,
             hamiltonian_coeff=10, **kwargs):
    config = Config(dict(mode="consensus_opt", num_iter=num_iter, lr=lr,
                         seed=seed, hamiltonian_coeff=hamiltonian_coeff,
                         shuffling=shuffling))
    torch.manual_seed(seed)
    game.reset()
    sgd = optim.SGD(game.parameters(), lr=lr)
    if lr_schedule is not None:
        lr_schedule = SchedulerLR(lr_schedule)
        scheduler = LambdaLR(sgd, lr_schedule)
    else:
        scheduler = LambdaLR(sgd, lambda k: 1.)
    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    if shuffling:
        game.shuffle()

    n_samples = 0
    start_time = time.time()
    for i in tqdm(range(num_iter)):
        index1 = game.sample()
        index2 = game.sample()
        if biased is True:
            grad1 = game.compute_grad(index1)
            grad2 = grad1
            hamiltonian = compute_hamiltonian(grad1)
            n_samples += 1

        elif biased == "copt":
            grad1 = game.compute_grad(torch.cat([index1, index2]))
            grad2 = grad1
            hamiltonian = compute_hamiltonian(grad1)
            n_samples += 2

        elif biased is False:
            grad1 = game.compute_grad(index1)
            grad2 = game.compute_grad(index2)
            hamiltonian = compute_hamiltonian(grad1)
            n_samples += 2

        else:
            raise ValueError()

        grad_H = autograd.grad(hamiltonian, game.parameters())
        for p, g1, g2, gH in zip(game.parameters(), grad1, grad2, grad_H):
            p.grad = 0.5*(g1+g2) + hamiltonian_coeff*gH
        sgd.step()
        scheduler.step()

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)
        logger["lr"].append(scheduler.get_last_lr())
        logger["num_samples"].append(n_samples)
        logger["time"].append(time.time()-start_time)

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    return logger, config


def run_semiSVRH(game, num_iter=5000, lr=3e1, seed=1234, biased=False,
                 prob=0.01, **kwargs):
    config = Config(dict(mode="semi-svrh", num_iter=num_iter, lr=lr,
                         seed=seed, biased=biased, prob=prob))
    torch.manual_seed(seed)
    game.reset()
    game_snapshot = copy.deepcopy(game)
    sgd = optim.SGD(game.parameters(), lr=lr)

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    batch = game_snapshot.sample_batch()
    full_grad = game_snapshot.compute_grad(batch)

    n_samples = len(batch)

    for i in tqdm(range(num_iter)):
        index1 = game.sample()
        index2 = game.sample()

        if biased:
            hamiltonian = game.compute_hamiltonian(index1)
            hamiltonian_snap = game_snapshot.compute_hamiltonian(index1)
            n_samples += 2

        else:
            hamiltonian = game.compute_hamiltonian(index1, index2)
            grad1 = game_snapshot.compute_grad(index1)
            grad2 = game_snapshot.compute_grad(index2)
            hamiltonian_snap = compute_hamiltonian(grad1, grad2)
            n_samples += 4

        grad = autograd.grad(hamiltonian, game.parameters())
        grad_snapshot = autograd.grad(hamiltonian_snap,
                                      game_snapshot.parameters(),
                                      retain_graph=True)

        hamiltonian_batch = compute_hamiltonian(grad1, full_grad)
        batch_grad = autograd.grad(hamiltonian_batch,
                                   game_snapshot.parameters(),
                                   retain_graph=True)

        for p, g, gs, bg in zip(game.parameters(), grad,
                                grad_snapshot, batch_grad):
            p.grad = g - gs + bg
        sgd.step()

        if torch.rand(1) < prob:
            game_snapshot.load_state_dict(game.state_dict())
            batch = game_snapshot.sample_batch()
            full_grad = game_snapshot.compute_grad(batch)
            n_samples += len(batch)

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)
        logger["num_samples"].append(n_samples)

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    return logger, config


def run_noisySVRH(game, num_iter=5000, lr=3e1, seed=1234, biased=False,
                  prob=0.01, batch_size=10, **kwargs):
    config = Config(dict(mode="noisy-svrh", num_iter=num_iter, lr=lr,
                         seed=seed, biased=biased, prob=prob,
                         batch_size=batch_size))
    torch.manual_seed(seed)
    game.reset()
    game_snapshot = copy.deepcopy(game)
    sgd = optim.SGD(game.parameters(), lr=lr)

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    batch1 = game_snapshot.sample_batch(batch_size)
    batch2 = game_snapshot.sample_batch(batch_size)
    hamiltonian = game_snapshot.compute_hamiltonian(batch1, batch2)
    batch_grad = autograd.grad(hamiltonian, game_snapshot.parameters())

    n_samples = 2*batch_size
    for i in tqdm(range(num_iter)):
        index1 = game.sample()
        index2 = game.sample()

        if biased:
            hamiltonian = game.compute_hamiltonian(index1)
            hamiltonian_snap = game_snapshot.compute_hamiltonian(index1)
            n_samples += 2

        else:
            hamiltonian = game.compute_hamiltonian(index1, index2)
            hamiltonian_snap = game_snapshot.compute_hamiltonian(index1,
                                                                 index2)
            n_samples += 4

        grad = autograd.grad(hamiltonian, game.parameters())
        grad_snapshot = autograd.grad(hamiltonian_snap,
                                      game_snapshot.parameters())

        for p, g, gs, bg in zip(game.parameters(), grad,
                                grad_snapshot, batch_grad):
            p.grad = g - gs + bg
        sgd.step()

        if torch.rand(1) < prob:
            game_snapshot.load_state_dict(game.state_dict())
            batch1 = game_snapshot.sample_batch(batch_size)
            batch2 = game_snapshot.sample_batch(batch_size)
            hamiltonian = game_snapshot.compute_hamiltonian(batch1, batch2)
            batch_grad = autograd.grad(hamiltonian, game_snapshot.parameters())
            n_samples += 2*batch_size

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)
        logger["num_samples"].append(n_samples)

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    return logger, config


def run_SGD(game, num_iter=5000, lr=5e-2, seed=1234,
            save_params=False, **kwargs):
    config = Config(dict(mode="sgd", num_iter=num_iter, lr=lr, seed=seed))
    torch.manual_seed(seed)
    game.reset()
    game_avg = copy.deepcopy(game)
    optimizer = optim.SGD(game.parameters(), lr=lr)

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    n_update = 0
    n_samples = 0
    params_history = []
    for i in tqdm(range(num_iter)):
        index = game.sample()
        grad = game.compute_grad(index)
        for p, g in zip(game.parameters(), grad):
            p.grad = g
        optimizer.step()

        n_samples += 1

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)

        n_update += 1
        for p, p_avg in zip(game.parameters(), game_avg.parameters()):
            p_avg.data = p/(n_update+1) + p_avg*n_update/(n_update+1)

        metrics = game_avg.compute_metrics()
        for key, value in metrics.items():
            logger[key+":avg"].append(value)
        logger["num_samples"].append(n_samples)

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

        if save_params:
            params_history.append(copy.deepcopy(game.state_dict()))

    logger["params"] = params_history
    return logger, config


def run_extragradient(game, num_iter=5000, lr=5e-2, seed=1234,
                      same_sample=False, **kwargs):
    config = Config(dict(mode="extragradient", num_iter=num_iter, lr=lr,
                         seed=seed, same_sample=same_sample))
    torch.manual_seed(seed)
    game.reset()
    game_avg = copy.deepcopy(game)
    sgd = optim.SGD(game.parameters(), lr=lr)
    optimizer = Extragradient(sgd, game.parameters())

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    n_update = 0
    n_samples = 0
    for i in tqdm(range(num_iter)):
        index = game.sample()
        grad = game.compute_grad(index)
        for p, g in zip(game.parameters(), grad):
            p.grad = g
        optimizer.step()

        if not same_sample:
            index = game.sample()

        grad = game.compute_grad(index)
        for p, g in zip(game.parameters(), grad):
            p.grad = g
        optimizer.step()

        n_samples += 2

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)

        n_update += 1
        for p, p_avg in zip(game.parameters(), game_avg.parameters()):
            p_avg.data = p/(n_update+1) + p_avg*n_update/(n_update+1)

        metrics = game_avg.compute_metrics()
        for key, value in metrics.items():
            logger[key+":avg"].append(value)
        logger["num_samples"].append(n_samples)

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    return logger, config


def run_SVRG(game, num_iter=5000, lr=5e-2, seed=1234, prob=0.1,
             restart=0.1, **kwargs):
    config = Config(dict(mode="svrg", num_iter=num_iter, lr=lr, seed=seed,
                         prob=prob, restart=restart))
    torch.manual_seed(seed)
    game.reset()
    game_snapshot = copy.deepcopy(game)
    game_avg = copy.deepcopy(game)
    optimizer = optim.SGD(game.parameters(), lr=lr)

    batch = game_snapshot.sample_batch()
    grad_batch = game_snapshot.compute_grad(batch)

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    n_update = 0
    n_samples = len(batch)
    for i in tqdm(range(num_iter)):
        index = game.sample()
        grad = game.compute_grad(index)

        grad_snapshot = game_snapshot.compute_grad(index)

        for p, g, gs, bg in zip(game.players, grad, grad_snapshot, grad_batch):
            p.grad = g - gs + bg

        optimizer.step()

        n_samples += 2

        if torch.rand(1) < prob:
            if torch.rand(1) < restart:
                game.load_state_dict(game_avg.state_dict())
                n_update = 0
            game_snapshot.load_state_dict(game.state_dict())
            batch = game_snapshot.sample_batch()
            grad_batch = game_snapshot.compute_grad(batch)
            n_samples += len(batch)

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)

        n_update += 1
        for p, p_avg in zip(game.parameters(), game_avg.parameters()):
            p_avg.data = p/(n_update+1) + p_avg*n_update/(n_update+1)

        metrics = game_avg.compute_metrics()
        for key, value in metrics.items():
            logger[key+":avg"].append(value)
        logger["num_samples"].append(n_samples)

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

    return logger, config


def run_SVRE(game, num_iter=5000, lr=5e-2, seed=1234, save_params=False,
             prob=0.1, restart=0.1, same_sample=False, **kwargs):
    config = Config(dict(mode="svre", num_iter=num_iter, lr=lr, seed=seed,
                         prob=prob, restart=restart, same_sample=same_sample))
    torch.manual_seed(seed)
    game.reset()
    game_snapshot = copy.deepcopy(game)
    game_avg = copy.deepcopy(game)
    sgd = optim.SGD(game.parameters(), lr=lr)
    optimizer = Extragradient(sgd, game.parameters())

    batch = game_snapshot.sample_batch()
    grad_batch = game_snapshot.compute_grad(batch)

    logger = defaultdict(list)

    if kwargs["output"] is not None:
        path = os.path.join(kwargs["output"], config.name, str(seed))
        config["path"] = path
        if not os.path.exists(path):
            os.makedirs(os.path.join(path, "results"))

        config["name"] = config.name

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, default=lambda x: "non-serializable")

    n_update = 0
    n_samples = len(batch)
    params_history = []
    for i in tqdm(range(num_iter)):
        index = game.sample()

        grad = game.compute_grad(index)
        grad_snapshot = game_snapshot.compute_grad(index)
        for p, g, gs, bg in zip(game.players, grad, grad_snapshot, grad_batch):
            p.grad = g - gs + bg
        optimizer.step()

        if not same_sample:
            index = game.sample()

        grad = game.compute_grad(index)
        grad_snapshot = game_snapshot.compute_grad(index)
        for p, g, gs, bg in zip(game.players, grad, grad_snapshot, grad_batch):
            p.grad = g - gs + bg
        optimizer.step()

        n_samples += 4

        if torch.rand(1) < prob:
            if torch.rand(1) < restart:
                game.load_state_dict(game_avg.state_dict())
                n_update = 0
            game_snapshot.load_state_dict(game.state_dict())
            batch = game_snapshot.sample_batch()
            grad_batch = game_snapshot.compute_grad(batch)
            n_samples += len(batch)

        metrics = game.compute_metrics()
        for key, value in metrics.items():
            logger[key].append(value)

        n_update += 1
        for p, p_avg in zip(game.parameters(), game_avg.parameters()):
            p_avg.data = p/(n_update+1) + p_avg*n_update/(n_update+1)

        metrics = game_avg.compute_metrics()
        for key, value in metrics.items():
            logger[key+":avg"].append(value)
        logger["num_samples"].append(n_samples)

        if i % 10000 == 0:
            with open(os.path.join(path, "results.json"), "w") as f:
                json.dump(logger, f)

        if save_params:
            params_history.append(copy.deepcopy(game.state_dict()))

    logger["params"] = params_history
    return logger, config