import torch
import torch.nn.functional as F
import ray


def network():
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=4, out_features=3),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=3, out_features=2),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=2, out_features=1),
        torch.nn.ReLU()
    )


@ray.remote
def actor(iteration, weight):
    net = network()
    net.load_state_dict(weight)
    optim = torch.optim.Adam(params=net.parameters(), lr=1e-4)

    losses = None
    for _ in range(iteration):
        s = torch.tensor([1., 2., 3., 4.])
        y_hat = meta_net(s)
        y = torch.mean(s, dim=0, keepdim=True)
        # print(f'y_hat : {y_hat}')
        # print(f'y : {y}')
        loss = F.mse_loss(y, y_hat)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if losses is None:
            losses = loss
        else:
            losses += loss

    return losses


if __name__ == '__main__':
    meta_net = network()
    meta_optim = torch.optim.Adam(params=meta_net.parameters(), lr=1e-3)
    epoch = 100
    meta_epoch = 10

    for _ in range(meta_epoch):
        weight = meta_net.state_dict()
        losses = ray.get(actor.remote(epoch, weight))

        print(f'origin weight : {weight}')
        meta_optim.zero_grad()
        losses.backward()
        meta_optim.step()
        print(f'new weight : {meta_net.state_dict()}')

