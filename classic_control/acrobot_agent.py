import gymnasium as gym
import numpy as np

# I kinda know what these are but...
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

# ...what are these?
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class AcrobotAgent:
    
    def __init__(self, name, env):
        #self.model = Sequential()
        self.nb_actions = env.action_space.n
        with self.model as m:
            # Most of this I understand generally although I should reread the specifics.
            m = Sequential()
            m.add(Flatten(input_shape=(1,) + env.observation_space.shape))
            m.add(Dense(16))
            m.add(Activation(('relu')))
            m.add(Dense(self.nb_actions))
            m.add(Activation('linear'))
            print(m.summary)
        self.name = name
        # Sort of remember reading this.
        self.policy = EpsGreedyQPolicy()
        # No idea about this. CompSci thing to make memor access faster?
        self.memory = SequentialMemory(limit=50000, window_length=1)
        # This looks important. No clue though.
        self.dqn = DQNAgent(model = self.model, nb_actions=self.nb_actions, memory=self.memory, nb_steps_warmup=10, target_model_update=1e-2, policy=self.policy)
        # I've seen this before. Long ago.
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    def action(self, env):
        print("Action taken.")
        return env.action_space.sample()
    
    def fit(self, env, nb_steps, visualize, verbose):
        self.dqn.fit(env, nb_steps, visualize, verbose)
        
    def test(self, env, nb_episodes, visualize):
        self.dqn.test(env, nb_episodes, visualize)
    
    