from audioop import avg
from random import randint
import gym
import numpy as np

MAX_ITERATIONS = 10000
 
def value_iteration(env, gamma=0.9, theta=1e-10):
    state_value = [0 for i in range(env.nS)]
    new_state_value = state_value.copy()
    for i in range(MAX_ITERATIONS):
        for state in range(env.nS):
            action_values = []
            for action in range(env.nA):
                value = 0
                for i in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][i]
                    state_action_value = prob * (reward + gamma * state_value[next_state])
                    value += state_action_value
                action_values.append(value)
                best_action = np.argmax(np.asarray(action_values))
                new_state_value[state] = action_values[best_action]
        if abs(sum(state_value) - sum(new_state_value)) < theta:
            break
        else:
            state_value = new_state_value.copy()
    return state_value

def get_policy(state_value, env, gamma=0.9):
    pi = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, reward, done = env.P[state][action][i]
                action_value += reward + gamma * prob * state_value[next_state]
            action_values.append(action_value)
        pi[state] = np.argmax(np.asarray(action_values))
    return pi

def get_score(env, policy, episodes = 10000):
    success = 0
    fail = 0
    step_list = []
    for _ in range(episodes):
        state = env.reset()
        step = 0
        while True:
            step += 1
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            if done:
                step_list.append(step)
                if reward == 1:
                    success += 1
                    break
                if reward == 0:
                    fail += 1
                    break
            state = next_state
    print(f'Average steps to the goal: {np.mean(step_list)}')
    print(f'Success: {success}, Failure: {fail}')


env = gym.make("FrozenLake-v1", map_name="4x4")
env.reset()
state_value = value_iteration(env)
policy = get_policy(state_value, env)
random_policy = [ randint(0, env.nA - 1) for _ in range(16)]

print("Value Iteration: ")
get_score(env, policy)