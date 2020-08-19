import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

HIDDEN1_UNITS = 50
HIDDEN2_UNITS = 30

class Critic(object):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        

        #Now create the model
        self.model = CriticNetwork(state_size, action_size)
        self.target_model = CriticNetwork(state_size, action_size)
        self.initialize_target_network(self.target_model, self.model)
        self.model.cuda()
        self.target_model.cuda()
        self.optim = Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def initialize_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):    
            target_param.data.copy_(target_param.data.copy_(param.data))
    
    def train(self, q_values, y_t):
        loss = self.criterion(q_values, y_t)
        loss.backward()
        self.optim.step()

    def target_train(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1-self.TAU) + param.data * self.TAU
            )

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc21 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)

        self.fc22 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN2_UNITS+HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.fc4 = nn.Linear(HIDDEN2_UNITS, 1)
    
    def forward(self, xs):
        x, a = xs
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc21(x)
        a = self.fc22(a)

        out = self.fc3(torch.cat([x, a], 1))
        out = F.relu(out)
        out = self.fc4(out)
        return out