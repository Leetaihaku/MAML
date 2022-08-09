import argparse
import ray
import process
import torch

from environment import *
from agent import *
from network import *
from optimizer import *
from copy import deepcopy


if __name__ == '__main__':
    # parameter input
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--config", type=str, default="MAML_test")
    parser.add_argument("--env", type=str, default="ant")
    parser.add_argument("--inst", type=int, default=2)
    args, unknown = parser.parse_known_args()

    # input check
    assert args.agent is not None, "You have to enter agent which you want to be learn"
    assert args.env is not None, "You have to enter envs which you want to learn"

    # call and combine to set up simulation
    config_module = __import__('config.%s.%s' % (str(args.agent).lower(), str(args.config)), fromlist=[args.config])

    # set up env config and init meta_env
    config_env = config_module.env
    meta_env = Env(args.env, **config_env)

    # set up network config and init meta_network
    config_network = config_module.network
    config_network.update({
        "D_in": meta_env.state_size,
        "D_out": meta_env.action_size,
    })
    meta_network = Network(**config_network)
    meta_weight = meta_network.state_dict()

    # set up optim config and init meta_optimizer
    config_optim_actor = config_module.optim_actor
    config_optim_meta = config_module.optim_meta
    meta_optimizer = Optimizer(**config_optim_meta, params=meta_network.parameters())

    # set up train config, agents configs and init meta_agent
    config_train = config_module.train
    additional_info = {
            "env_name": args.env,
            "state_size": meta_env.state_size,
            "action_size": meta_env.action_size,
            "meta_weight": meta_weight,
    }
    tmp_config_agent = {}
    for dictionary in [config_network, config_env, config_train, additional_info]:
        tmp_config_agent.update(dictionary)
    _config_agent = deepcopy(config_module.agent)
    _config_agent.update(tmp_config_agent)
    config_actor, config_meta = deepcopy(_config_agent), deepcopy(_config_agent)
    config_actor.update(config_optim_actor)
    config_meta.update(config_optim_meta)
    meta_agent = Agent(**config_meta)

    # TODO: MAML: sync learn process run
    try:
        ray.init()
        for epoch in range(config_train["epoch"]):
            # run training process [I: weight(meta_network), O: weight(actor)]
            meta_weight = ray.get([
                process.training.remote(config_actor, meta_train=False)
                for task in range(args.inst)
            ])

            # set up new weight from actor for meta learning
            total_config_meta = [config_meta for _ in range(len(meta_weight))]
            print(f'total_config_meta : {total_config_meta}')
            for config_meta, idx in zip(total_config_meta, range(len(meta_weight))):
                config_meta["meta_weight"] = meta_weight[idx]
            print(f'total_config_meta : {total_config_meta}')

            # 업데이트 할 네트워크로 loss 계산해야 backward 가능
            exit()

            # extract loss function for meta per theta [I: weight(actor), O: loss(actor)]
            losses = ray.get([
                process.training.remote(config_meta, meta_train=True)
                for config_meta in total_config_meta
            ])

            # summation loss for meta per theta
            meta_loss = None
            for inst in range(args.inst):
                if meta_loss is None:
                    meta_loss = losses[inst]["total_loss"]
                else:
                    meta_loss = meta_loss + losses[inst]["total_loss"]

            # TODO: learn meta_network(check learning)
            print(f'origin meta : {meta_network.state_dict()["mu.weight"]}')
            meta_optimizer.zero_grad(set_to_none=True)
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_network.parameters(), config_meta["clip_grad_norm"])
            meta_optimizer.step()
            print(f'new meta : {meta_network.state_dict()["mu.weight"]}')
            exit()

            # update actor_setup_info
            meta_network.load_state_dict(torch.load())
            config_meta["meta_weight"] = meta_network.state_dict()

            # TODO: logging
            # TODO: printing
            # TODO: saving

    # TODO: remove memo cmd for exception(at the end)
    # except Exception:
    #     print("occurred exception.")
    finally:
        # TODO: set eval sequence

        # main process end
        print("program terminated.")
