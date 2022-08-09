""" PPO Mujoco Config """

env = {
    "render": False,
}

agent = {
    "agent_name": "ppo",
    "action_type": "continuous",
    "hidden_size": 512,  # <- TODO: debug(when it changed)
    "gamma": 0.99,
    "batch_size": 32,  # <- Must be smaller than n_step
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
    "clip_grad_norm": 1.0,
    "use_standardization": 1.0,
    "lr_decay": False,
}

network = {
    "head": "mlp",
    "network_name": "continuous_policy_value",
}

optim_actor = {
    "optim_name": "adam",
    "lr": 3e-4,  # <- alpha
}

optim_meta = {
    "optim_name": "adam",
    "lr": 1e-4,  # <- beta
}

train = {
    "training": True,
    "load_path": None,
    "epoch": 10,
    "run_step": 512,  # 1000000,
    "print_period": 5,  # 10000,
    "save_period": 5,  # 100000,
    "eval_iteration": 10,
    "record": True,
    "record_period": 500000,
}
