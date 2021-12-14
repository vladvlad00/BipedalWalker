import torch
import torch.nn.functional as func

from actor import Actor
from critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self, lr, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)

        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # actor_target = actor

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update_critic(self, state, action, critic, critic_optimizer, target_q):
        q = critic(state, action)
        loss_q = func.mse_loss(q, target_q)
        critic_optimizer.zero_grad()
        loss_q.backward()
        critic_optimizer.step()

    def update_actor(self, state):
        loss = -self.critic(state, self.actor(state)).mean() # e cu - pentru ca facem ascent in loc de descent
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update_target(self, current_model, target_model, tau):
        for current_params, target_params in zip(current_model.parameters(), target_model.parameters()):
            target_params.data.copy_(tau * current_params.data + (1 - tau) * target_params.data)

    def update(self, replay_buffer, iterations, replay_batch_size=100, gamma=0.99, tau=0.005):
        for _ in range(iterations):
            # doamne ajuta sa mearga :(
            state, action_np, reward, next_state, done = replay_buffer.sample(replay_batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_np).to(device)
            reward = torch.FloatTensor(reward).reshape(replay_batch_size, 1).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape(replay_batch_size, 1).to(device)

            next_action = self.actor_target(next_state)

            target_q = reward + ((1 - done) * gamma * self.critic_target(next_state, next_action).detach())

            self.update_critic(state, action, self.critic, self.critic_optimizer, target_q)

            self.update_actor(state)

            self.update_target(self.actor, self.actor_target, tau)
            self.update_target(self.critic, self.critic_target, tau)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + '_actor.pth')
        torch.save(self.critic.state_dict(), path + '_critic.pth')
        torch.save(self.actor_target.state_dict(), path + '_actor_target.pth')
        torch.save(self.critic_target.state_dict(), path + '_critic_target.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + '_actor.pth', map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load(path + '_actor_target.pth', map_location=lambda storage, loc: storage))

        self.critic.load_state_dict(
            torch.load(path + '_critic.pth', map_location=lambda storage, loc: storage))
        self.critic_target.load_state_dict(
            torch.load(path + '_critic_target.pth', map_location=lambda storage, loc: storage))
