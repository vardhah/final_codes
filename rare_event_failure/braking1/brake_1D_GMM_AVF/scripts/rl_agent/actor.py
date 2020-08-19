import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

HIDDEN1_UNITS = 50
HIDDEN2_UNITS = 30

class Actor(object):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        #Now create the model
        self.model = ActorNetwork(state_size, action_size)
        self.target_model = ActorNetwork(state_size, action_size)
        self.initialize_target_network(self.target_model, self.model)
        self.model.cuda()
        self.target_model.cuda()
        self.optim = Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def initialize_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):    
            target_param.data.copy_(target_param.data.copy_(param.data))

    def train(self, policy_loss):
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.optim.step()

    def target_train(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1-self.TAU) + param.data * self.TAU
            )

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN2_UNITS, action_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x
