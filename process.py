import ray

from agent import *
from environment import *


@ray.remote
def training(config, meta_train=False):
    try:
        # set agent
        agent = Agent(**config)
        env = Env(config["env_name"], **config)
        result_dict = {}

        # sample trajectory(rule: run_step = frame)
        state = env.reset()
        for step in range(1, config["run_step"]):
            action_dict = agent.act(state, config["training"])
            next_state, reward, done = env.step(action_dict["action"])
            transitions = {
                "state": state,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }
            transitions.update(action_dict)

            # interact_callback
            transitions = agent.interact_callback(transitions)

            # TODO: update network weight of actor(check the mean of sample 'K' in paper)
            if transitions:
                result = agent.process([transitions], step, meta_train)
                # print result(per period)
                if result is not None:
                    print(f'result : {result}')
                    # print(f'keys : {result.keys()}')
                    # print(f'values : {result.values()}')
                    # print(f'requires_grad : {result["total_loss"].requires_grad}')
                    for key, value in result.items():
                        if key not in result_dict.keys():
                            result_dict[key] = result[key]
                        else:
                            result_dict[key] += result[key]

            # state <- next_state
            state = next_state if not done else env.reset()
    except Exception:
        traceback.print_exc()
    finally:
        # return network weight of actor or losses of new thetas
        if not meta_train:
            return agent.network.state_dict()
        else:
            return result_dict
