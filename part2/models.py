import torch


# Q-Network FC-128, ReLU
class QNetwork_1(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork_1, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    

    # Q-Network FC-64, ReLU and FC-64, ReLU 
class QNetwork_2(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=64, hidden_dim2=64):
        super(QNetwork_2, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    

    # Q-Network FC-128, ReLU and FC-128, ReLU 
class QNetwork_3(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=128, hidden_dim2=128):
        super(QNetwork_3, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    

    # Q-Network FC-128, ReLU and FC-128, ReLU and FC-128, ReLU
class QNetwork_4(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=128, hidden_dim2=128, hidden_dim3=128):
        super(QNetwork_4, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = torch.nn.Linear(hidden_dim3, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x
    

    # Q-Network FC-256, ReLU and FC-256, ReLU 
class QNetwork_5(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=256, hidden_dim2=256):
        super(QNetwork_5, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x