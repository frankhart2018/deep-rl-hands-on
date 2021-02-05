import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

HIDDEN_SIZE = 128

class Net(nn.Module):

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size=obs_size, hidden_size=HIDDEN_SIZE, n_actions=n_actions)
    net.load_state_dict(torch.load("cartpole.pth"))

    obs = env.reset()
    obs_v = torch.FloatTensor(obs)

    total_rewards = 0

    while True:
        action_scores_v = net(obs_v)
        action_probs = F.softmax(action_scores_v, dim=0)
        _, action = torch.max(action_probs, dim=0)

        obs, reward, is_done, _ = env.step(action.item())
        env.render()

        if is_done:
            break

        total_rewards += reward
        obs_v = torch.FloatTensor(obs)

    env.close()
    print(f"Reward: {total_rewards}")