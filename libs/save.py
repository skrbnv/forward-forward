class Classifier(nn.Module):
    def __init__(self,
                 inputs=100,
                 num_classes=10,
                 optimizer=None,
                 device='cpu'):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(inputs, num_classes, bias=True).to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.parameters(), lr=5e-5)

    def update(self, inputs, labels, statuses):
        # print('Updating classifier')
        y = self.layer(inputs)
        loss = torch.nn.functional.cross_entropy(y, labels.flatten())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            y = self.layer(inputs)
        return y, loss

    def forward(self, inputs):
        return self.layer(inputs)