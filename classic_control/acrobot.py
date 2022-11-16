import gymnasium as gym
from acrobot_agent import AcrobotAgent

acrobot_name = "acrobot_0.0.1"
nb_steps = 5000
visualize = True
verbose = 2
nb_episodes=5

env = gym.make("Acrobot-v1", render_mode="human")
# observation, info = env.reset(seed=42)
# my_acrobot_agent = AcrobotAgent(acrobot_name, env)
# for _ in range(1000):
#    action = my_acrobot_agent.action(env)
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()

acro_agent = AcrobotAgent(acrobot_name, env)
acro_agent.fit(env, nb_steps, visualize, verbose)
acro_agent.test(env, nb_episodes, visualize)

env.close()