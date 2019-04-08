import sys
import os
import argparse
# temporary fix for path
python_path = os.getcwd() + "/python"
sys.path.insert(0, python_path)

from python.unityagents import UnityEnvironment
import torch
from time import gmtime, strftime
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from ddpg_agent import Agent


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def multi_agent_ddpg(env, brain_name, title, n_episodes, action_size, state_size, num_agents, print_every,
                     n_updates, update_intervals, device):

    # create save dir for this experiment
    if title is None:
        title = "experiment"
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    title = title + "_" + current_time

    # write a new file
    os.makedirs("experiments/{}".format(title), exist_ok=True)
    f = open("experiments/{}/scores.txt".format(title), "w")
    f.close()

    all_agents_statesize = state_size * num_agents

    agent1 = Agent(state_size=all_agents_statesize, action_size=action_size, num_agents=1, random_seed=123, device=device)
    agent2 = Agent(state_size=all_agents_statesize, action_size=action_size, num_agents=1, random_seed=123, device=device)

    scores_deque = deque(maxlen=100)
    mean_scores = []

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]

        states = env_info.vector_observations
        states = np.reshape(states, (1, all_agents_statesize))  # reshape so we can feed both agents states to each agent

        # reset each agent for a new episode
        agent1.reset()
        agent2.reset()

        # set the initial episode score to zero.
        agent_scores = np.zeros(num_agents)
        t = 0
        while True:
            # determine actions for the unity agents from current sate, using noise for exploration
            actions_1 = agent1.act(states, add_noise=True)
            actions_2 = agent2.act(states, add_noise=True)

            # send the actions to the unity agents in the environment and receive resultant environment information
            actions = np.concatenate((actions_1, actions_2), axis=0)
            actions = np.reshape(actions, (1, 4))
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations  # get the next states for each unity agent in the environment
            next_states = np.reshape(next_states, (1, all_agents_statesize))
            rewards = env_info.rewards  # get the rewards for each unity agent in the environment
            dones = env_info.local_done  # see if episode has finished for each unity agent in the environment

            # Send (S, A, R, S') info to the training agent for replay buffer (memory) and network updates
            agent1.step(states, actions_1, rewards[0], next_states, dones[0], n_updates, update_intervals, t)
            agent2.step(states, actions_2, rewards[1], next_states, dones[1], n_updates, update_intervals, t)

            # set new states to current states for determining next actions
            states = next_states
            # print(states)
            # Update episode score for each unity agent
            agent_scores += rewards

            # If any unity agent indicates that the episode is done,
            # then exit episode loop, to begin new episode
            if np.any(dones):
                break
            t += 1

        scores_deque.append(np.max(agent_scores))
        print('\rEpisode {}\tLast 100 average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        # save score and model every print_every
        if i_episode % print_every == 0:
            f = open("experiments/{}/scores.txt".format(title), "a")
            f.write("{},{}\n".format(i_episode, np.mean(scores_deque)))
            f.close()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            mean_scores.append(np.mean(scores_deque))
            # save if best model
            if np.mean(scores_deque) == max(mean_scores):
                torch.save(agent1.actor_local.state_dict(), 'experiments/{}/checkpoint_actor1.pth'.format(title))
                torch.save(agent1.critic_local.state_dict(), 'experiments/{}/checkpoint_critic1.pth'.format(title))
                torch.save(agent2.actor_local.state_dict(), 'experiments/{}/checkpoint_actor2.pth'.format(title))
                torch.save(agent2.critic_local.state_dict(), 'experiments/{}/checkpoint_critic2.pth'.format(title))

            if np.mean(scores_deque) >= 1.0 and i_episode > 100:
                print("\rEnvironment solved with average score of 30")
                break



def main():
    parser = argparse.ArgumentParser(description='Options to train the model.')
    parser.add_argument("--title",                          type=str,       default="experiment")
    parser.add_argument('--n_episodes',                     type=int,       default=2000)
    parser.add_argument("--device",                         type=int,       default=1)
    parser.add_argument('--port',                           type=int,       default=64735)
    parser.add_argument('--print_every',                    type=int,       default=10)
    parser.add_argument('--n_updates',                      type=int,       default=10)
    parser.add_argument('--update_intervals',               type=int,       default=20)


    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda:{}".format(args.device)

    env = UnityEnvironment(file_name="data/Tennis_Linux_NoVis/Tennis", base_port=args.port)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    multi_agent_ddpg(env, brain_name,
                     title=args.title,
                     n_episodes=args.n_episodes,
                     action_size=action_size,
                     state_size=state_size,
                     num_agents=num_agents,
                     print_every=args.print_every,
                     n_updates=args.n_updates,
                     update_intervals=args.update_intervals,
                     device=args.device)

    env.close()


if __name__ == '__main__':
    main()