import torch
import torch.nn as nn


def unpack(d):
    return d["data"], d["label"], d["status"]


def pack(data, label, status):
    return {"data": data, "label": label, "status": status}


class FFBlock(nn.Module):
    def __init__(self,
                 upstream,
                 downstream,
                 outs,
                 threshold=.1,
                 optimizer=None,
                 name=''):
        super(FFBlock, self).__init__()
        self.name = name
        self.threshold = threshold * outs
        self.w_up = nn.Parameter(torch.empty(1, upstream))
        self.w_down = nn.Parameter(torch.empty(1, downstream))
        nn.init.kaiming_normal_(self.w_up)
        nn.init.kaiming_normal_(self.w_down)
        self.layer = nn.Linear(upstream + downstream, outs, bias=True)
        self.norm = nn.BatchNorm1d(1)
        self.act = nn.ReLU()
        self.squash = nn.Sigmoid()
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=5e-5)
        else:
            self.optimizer = optimizer

    def squared_sum_loss_positive(self, x):
        return torch.abs(torch.sum(x**2) - self.threshold)

    def squared_sum_loss_negative(self, x):
        return torch.abs(torch.sum(x**2))

    def compute(self, upstream, downstream):
        x = torch.concat((self.w_up * upstream, self.w_down * downstream),
                         dim=-1)
        x = self.layer(x)
        x = self.norm(x.unsqueeze(1)).squeeze(1)
        x = self.act(x)
        x = self.squash(x)
        return x

    def update(self, upstream, downstream):
        # print(f'Updating {self.name}')
        up_x, up_label, up_status = unpack(upstream)
        down_x, _, down_status = unpack(downstream)
        # if both samples were positive => run optimizer
        if up_status == down_status == 1 or up_status == down_status == -1:
            y = self.compute(up_x, down_x)
            if up_status == down_status == 1:
                loss = self.squared_sum_loss_positive(y)
            elif up_status == down_status == -1:
                loss = self.squared_sum_loss_negative(y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        with torch.no_grad():
            y = self.compute(up_x, down_x)
        return pack(y, up_label, up_status)


class Model():
    def __init__(self, device='cpu', inner_size: int = 1000) -> None:
        self.inner_size = inner_size
        self.device = device
        self.l0 = nn.Linear(28 * 28 + 10, inner_size).to(device)
        self.l1 = FFBlock(upstream=inner_size,
                          downstream=inner_size,
                          outs=inner_size,
                          name="L1").to(device)
        self.v1 = self.init_values(inner_size, device)
        self.l2 = FFBlock(upstream=inner_size,
                          downstream=inner_size,
                          outs=inner_size,
                          name="L2").to(device)
        self.v2 = self.init_values(inner_size, device)
        self.l3 = FFBlock(upstream=inner_size,
                          downstream=inner_size,
                          outs=inner_size,
                          name="L3").to(device)
        self.v3 = self.init_values(inner_size, device)
        self.l4 = FFBlock(upstream=inner_size,
                          downstream=inner_size,
                          outs=inner_size,
                          name="L4").to(device)
        self.v4 = self.init_values(inner_size, device)
        self.ll = nn.Linear(inner_size, 10).to(device)

    def intro(self, input):
        x = input['data'].flatten(
            start_dim=-2
        )  # -3 bcs we are including channels which is 1 for MNIST
        x = self.l0(x)
        return x

    def outro(self, x):
        x = self.ll(x['data'])
        return x

    def init_values(self, inner_size, device='cpu'):
        return {
            "data": torch.zeros(1,
                                inner_size).requires_grad_(False).to(device),
            "label": -1,
            "status": 0
        }

    def update(self, x):
        data = self.intro(x)
        t1 = self.l1.update(pack(data, x['label'], x['status']), self.v2)
        t2 = self.l2.update(self.v1, self.v3)
        t3 = self.l3.update(self.v2, self.v4)
        t4 = self.l4.update(self.v3, self.v3)
        output = self.outro(self.v4)
        self.v1, self.v2, self.v3, self.v4 = t1, t2, t3, t4
        return output

    def compute(self, x):
        with torch.no_grad():
            x = self.intro(x)
            x = self.l1.compute(
                x,
                torch.zeros(1, self.inner_size).to(self.device))
            x = self.l2.compute(
                x,
                torch.zeros(1, self.inner_size).to(self.device))
            x = self.l3.compute(
                x,
                torch.zeros(1, self.inner_size).to(self.device))
            x = self.l4.compute(
                x,
                torch.zeros(1, self.inner_size).to(self.device))
            output = self.outro(pack(x, 0, 0))
        return output
