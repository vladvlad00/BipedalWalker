import torch
import torch.nn.functional as func

from actor import Actor
from critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # actor_target = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critics = [
            Critic(state_dim, action_dim).to(device),
            Critic(state_dim, action_dim).to(device)
        ]

        self.critic_targets = [
            Critic(state_dim, action_dim).to(device),
            Critic(state_dim, action_dim).to(device)
        ]
        for i in range(len(self.critic_targets)):
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())

        self.critic_optimizers = [
            torch.optim.Adam(self.critics[0].parameters(), lr=lr),
            torch.optim.Adam(self.critics[1].parameters(), lr=lr)
        ]

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
        loss = -self.critics[0](state, self.actor(state)).mean() # e cu - pentru ca facem ascent in loc de descent
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update_target(self, current_model, target_model, tau):
        for current_params, target_params in zip(current_model.parameters(), target_model.parameters()):
            target_params.data.copy_(tau * current_params.data + (1 - tau) * target_params.data)

    def update(self, replay_buffer, iterations, replay_batch_size=100, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
               policy_freq=2):
        for _ in range(iterations):
            # doamne ajuta sa mearga :(
            state, action_np, reward, next_state, done = replay_buffer.sample(replay_batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_np).to(device)
            reward = torch.FloatTensor(reward).reshape(replay_batch_size, 1).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape(replay_batch_size, 1).to(device)

            # Poate facem o functie care sa returneze next_action
            noise = torch.FloatTensor(action_np).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = (critic_target(next_state, next_action) for critic_target in self.critic_targets)
            target_q = reward + ((1 - done) * gamma * torch.min(target_q1, target_q2)).detach()  # oare de ce detach?

            self.update_critic(state, action, self.critics[0], self.critic_optimizers[0], target_q)
            self.update_critic(state, action, self.critics[1], self.critic_optimizers[1], target_q)

            if _ % policy_freq == 0:
                self.update_actor(state)
                self.update_target(self.actor, self.actor_target, tau)
                self.update_target(self.critics[0], self.critic_targets[0], tau)
                self.update_target(self.critics[1], self.critic_targets[1], tau)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + '_actor.pth')
        torch.save(self.critics[0].state_dict(), path + '_critic1.pth')
        torch.save(self.critics[1].state_dict(), path + '_critic2.pth')
        torch.save(self.actor_target.state_dict(), path + '_actor_target.pth')
        torch.save(self.critic_targets[0].state_dict(), path + '_critic1_target.pth')
        torch.save(self.critic_targets[1].state_dict(), path + '_critic2_target.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + '_actor.pth', map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(
            torch.load(path + '_actor_target.pth', map_location=lambda storage, loc: storage))

        self.critics[0].load_state_dict(
            torch.load(path + '_critic1.pth', map_location=lambda storage, loc: storage))
        self.critic_targets[0].load_state_dict(
            torch.load(path + '_critic1_target.pth', map_location=lambda storage, loc: storage))

        self.critics[1].load_state_dict(
            torch.load(path + '_critic2.pth', map_location=lambda storage, loc: storage))
        self.critic_targets[1].load_state_dict(
            torch.load(path + '_critic2_target.pth', map_location=lambda storage, loc: storage))
