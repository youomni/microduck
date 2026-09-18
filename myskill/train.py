from env import DuckEnv
from agent import Agent
import torch

def train(episodes=500):
    env = DuckEnv()
    agent = Agent()

    for ep in range(episodes):
        obs = env.reset()

        log_probs = []
        rewards = []
        total_reward = 0

        done = False
        while not done:
            action, probs = agent.act(obs)

            obs_t = torch.tensor(obs, dtype=torch.float32)
            logits = agent.model(obs_t)
            log_prob = torch.log_softmax(logits, dim=-1)[action]

            obs, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

        agent.learn(log_probs, rewards)

        if ep % 20 == 0:
            print(f"Episode {ep} | reward: {total_reward:.2f}")

if __name__ == "__main__":
    train()
