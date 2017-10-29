import gym
import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers import observation_space
import numpy as np
from matplotlib import pyplot as plt
import pylab
import itertools
import random as rd
import cv2
from doom_py import ScreenResolution
import sklearn
import math

env = gym.make('ppaquette/DoomDefendLine-v0')
# learning parameters for both Q-learning and Sarsa
discount_factor = 0.9
alpha = 0.0001
num_episodes = 800
# problem specific parameters
angle_per_action = 3.10345  # angle for each movement
number_of_actions = 3
set_of_actions = ['right', 'left', 'shoot']  # index 0, 1, and 2


def action_to_angle(action):
    return {
        'right': angle_per_action,
        'left': -angle_per_action,
        'shoot': 0
    }.get(action, 0)  # 0 default if action not found


def get_max_q_for_next_state(next_state, weights):
    """
    This array starts with 1 because of the linear regression w0 coefficient
    and has as second term the angular position variable
    IMPORTANT: we assume that in the next state only the angular position will change
    :type next_state: numpy array
    """
    # estimate the Q for each of the two states with linear regression weights
    Q_max = 0
    for action_index in range(number_of_actions):
        Q1 = np.dot(weights[action_index, :], next_state)
        if Q1 > Q_max:
            Q_max = Q1
    return Q_max


def get_best_action_index(current_state, weights):
    """
    Same as get_max_Q_for_next_state but returns the index for the best action.
    :type current_state: numpy array
    :type weights: numpy array
    """
    # estimate the Q for each of the two states with linear regression weights
    max_action_index = 0
    Q_max = 0
    for action_index in range(number_of_actions):
        Q1 = np.dot(weights[action_index, :], current_state)
        if Q1 > Q_max:
            Q_max = Q1
            max_action_index = action_index
    return max_action_index


# tile maker for angle
def angular_tile_maker(angular_pos):
    n_tiles1 = 10
    tile_size1 = 360/n_tiles1
    offset1 = -1 - 180
    n_tiles2 = 15
    tile_size2 = 360 / n_tiles2
    offset2 = -2 - 180
    features1 = get_tile(angular_pos, tile_size1, offset1, n_tiles1)
    features2 = get_tile(angular_pos, tile_size2, offset2, n_tiles2)
    tile_features = np.hstack((features1, features2))
    return tile_features


# get tile
def get_tile(angular_pos, tile_size, offset, n_tiles):
    features = np.zeros(n_tiles+1)
    for k in range(n_tiles+1):
        if ((tile_size*k + offset) < angular_pos) and (angular_pos <= (tile_size*(k+1) + offset)):
            features[k] = 1
    return features

''' this function is to slice the given frame into 2 frames and will do the tile approximation on those slices
and returns the total number of tiles'''
def tile_maker(mask, x):
    #splitting into two slices horizontally based on frame width and length
    frame1_div = mask[183:240, 0:639]
    frame2_div = mask[241:298, 0:639]
    tile = []
    #number of divisions for the given tile number
    n = np.floor(mask.shape[1] / x)
    n = int(n)
    i = 0
    while i <= mask.shape[1]:
        tile.append(frame1_div[:, i:i + n])
        i = i + n
    i = 0
    while i <= mask.shape[1]:
        tile.append(frame2_div[:, i:i + n])
        i = i + n

    #if sum of divised frame is positive making them as 1, if not 0
    tile_filter = np.where(np.array([np.sum(t) for t in tile]) > 0, 1, 0)
    return tile_filter

''' this function is to locate the demon and monster positions given the
    observation'''
def get_demons_state(observation):
    # getting RGB pixel values
    observation = observation[..., ::-1]
    #RGB to HSV
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    # red color filter
    lower_red = np.array([-10, 100, 100])
    upper_red = np.array([10, 200, 200])
    # brown color filter
    lower_brown = np.array([16, 100, 100])
    upper_brown = np.array([16, 250, 250])
    #red color filtered mask
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # brown color filtered mask
    mask2 = cv2.inRange(hsv, lower_brown, upper_brown)
    mask = mask2 + mask1
    res = cv2.bitwise_and(observation, observation, mask=mask)
    #20 tiles
    features1 = tile_maker(mask, 20)
    # 15 tiles
    features2 = tile_maker(mask, 15)
    # 10 tiles
    features3 = tile_maker(mask, 10)
    # 5 tiles
    features4 = tile_maker(mask, 5)
    #horizontal concatenation of features
    features = np.hstack((features1, features2, features3, features4))
    #     print("lenght:", len(features))
    #     print(features)
    #     print("Finished")
    return (features)


"""
    Q-LEARNING ALGORITHM
"""


# without tile coding for angular position
def q_learning_no_tile(seed):
    epsilon = 0.9
    # set seed
    np.random.seed(seed)
    # this list is going to be returned
    scores_list = []

    for i_episode in range(num_episodes):
        observation = env.reset()
        angular_pos = 0  # start angular position on each episode
        score = 0  # will count score per episode
        # number of features to approximate Q
        number_of_features = (66 + 42) + 1  # (features from image) + features from angular position
        current_state = np.hstack((np.array([1]), np.array(angular_pos), get_demons_state(observation)))
        # weights initialization as zeros, expect for first element of each weight vector
        weights = np.zeros((3, number_of_features + 1), dtype=np.float128) + \
                  np.random.rand(3, number_of_features + 1) * 10

        for t in itertools.count():
            """
                choosing action: epsilon greedy selection policy
            """
            if np.random.random() <= epsilon:
                # select a random action from set of all actions
                action_index = set_of_actions.index(rd.choice(set_of_actions))
                action = set_of_actions[action_index]
            # if the generated num is greater than epsilon, we follow exploitation policy
            else:
                # select an action with highest value for current state
                action_index = get_best_action_index(current_state, weights)
                action = set_of_actions[action_index]

            """
                perform action
            """
            # convert chosen action to a vector suitable for the simulator
            action_vector = [0] * 43  # simulator requires a vector like this
            if action == 'left':
                action_vector[15] = 1
            elif action == 'right':
                action_vector[14] = 1
            elif action == 'shoot':
                action_vector[0] = 1
            else:
                print("error in action")
            # apply selected action, collect values for next_state and reward
            observation, reward, done, info = env.step(action_vector)
            # sum up reward to calculate score for episode
            score += reward
            reward *= 10
            """
                get next state
            """
            # update angular position once the action has been taken
            angular_pos = angular_pos*180.0 + action_to_angle(action)
            # correct it if needed
            if angular_pos > 180:
                angular_pos = (angular_pos - 180) - 180
            if angular_pos < -180:
                angular_pos = (angular_pos + 180) + 180
            # now form next state
            # scale angular position from -1 to 1
            angular_pos = angular_pos / 180.0
            next_state = np.hstack((np.array([1]), np.array(angular_pos), get_demons_state(observation)))

            """
                Q-learning calculations
            """
            # Calculate the Q-learning target value
            Q_target = reward + discount_factor * get_max_q_for_next_state(next_state, weights)
            # Calculate the difference/error between target and current Q
            Q_error = Q_target - np.dot(weights[action_index, :], current_state)
            # Update the Q table, alpha is the learning rate
            weights[action_index, :] = weights[action_index, :] - alpha * Q_error * current_state

            """
                update state
            """
            current_state = next_state

            """
                Check if done
            """
            if done:
                scores_list.append(score)
                if i_episode % np.floor(num_episodes/20) == 0:
                    print("======================")
                    print("episode:", i_episode)
                    print("score:", score)
                    print("State", current_state)
                    print("weights", weights)
                    print("epsilon:", epsilon)
                break

        # gradually decay the epsilon on each episode
        if epsilon > 0.1:
            epsilon -= 1.0 / num_episodes
    return scores_list


# with tile coding for angular position
def q_learning_w_tile(seed):
    epsilon = 0.9
    # set seed
    np.random.seed(seed)
    # this list is going to be returned
    scores_list = []

    for i_episode in range(num_episodes):
        observation = env.reset()
        angular_pos = 0  # start angular position on each episode
        score = 0  # will count score per episode
        # number of features to approximate Q
        number_of_features = (66 + 42) + 27  # (features from image) + features from angular position
        current_state = np.hstack((np.array([1]), get_demons_state(observation), angular_tile_maker(angular_pos)))
        # weights initialization as zeros, expect for first element of each weight vector
        weights = np.zeros((3, number_of_features + 1), dtype=np.float128) + \
                  np.random.rand(3, number_of_features + 1) * 10
        for t in itertools.count():
            """
                choosing action: epsilon greedy selection policy
            """
            if np.random.random() <= epsilon:
                # select a random action from set of all actions
                action_index = set_of_actions.index(rd.choice(set_of_actions))
                action = set_of_actions[action_index]
            # if the generated num is greater than epsilon, we follow exploitation policy
            else:
                # select an action with highest value for current state
                action_index = get_best_action_index(current_state, weights)
                action = set_of_actions[action_index]

            """
                perform action
            """
            # convert chosen action to a vector suitable for the simulator
            action_vector = [0] * 43  # simulator requires a vector like this
            if action == 'left':
                action_vector[15] = 1
            elif action == 'right':
                action_vector[14] = 1
            elif action == 'shoot':
                action_vector[0] = 1
            else:
                print("error in action")
            # apply selected action, collect values for next_state and reward
            observation, reward, done, info = env.step(action_vector)
            # sum up reward to calculate score for episode
            score += reward
            reward *= 10
            """
                get next state
            """
            # update angular position once the action has been taken
            angular_pos = angular_pos + action_to_angle(action)
            # correct it if needed
            if angular_pos > 180:
                angular_pos = (angular_pos - 180) - 180
            if angular_pos < -180:
                angular_pos = (angular_pos + 180) + 180
            # now form next state
            next_state = np.hstack((np.array([1]), get_demons_state(observation), angular_tile_maker(angular_pos)))

            """
                Q-learning calculations
            """
            # Calculate the Q-learning target value
            Q_target = reward + discount_factor * get_max_q_for_next_state(next_state, weights)
            # Calculate the difference/error between target and current Q
            Q_error = Q_target - np.dot(weights[action_index, :], current_state)
            # Update the Q table, alpha is the learning rate
            weights[action_index, :] = weights[action_index, :] - alpha * Q_error * current_state

            """
                update state
            """
            current_state = next_state

            """
                Check if done
            """
            if done:
                scores_list.append(score)
                if i_episode % np.floor(num_episodes/20) == 0:
                    print("======================")
                    print("episode:", i_episode)
                    print("score:", score)
                    print("State", current_state)
                    print("weights", weights)
                    print("epsilon:", epsilon)
                break

        # gradually decay the epsilon on each episode
        if epsilon > 0.1:
            epsilon -= 1.0 / num_episodes
    return scores_list


"""
    Sarsa ALGORITHM
"""


# no tile coding for angular position
def sarsa_no_tile(seed):
    epsilon = 0.9
    # set seed
    np.random.seed(seed)
    scores_list = []
    for i_episode in range(num_episodes):
        observation = env.reset()
        score = 0
        angular_pos = 0
        # number of features to approximate Q
        number_of_features = (66 + 42) + 1  # (features from image) + features from angular position\
        current_state = np.hstack((np.array([1]), np.array(angular_pos), get_demons_state(observation)))
        # current_state = np.hstack((np.array([1]), get_demons_state(observation), angular_tile_maker(angular_pos)))
        # weights initialization as zeros, expect for first element of each weight vector
        weights = np.zeros((3, number_of_features + 1), dtype=np.float128) + \
                  np.random.rand(3, number_of_features + 1) * 10
        """
            choose first action with epsilon greedy selection policy
        """
        if np.random.random() <= epsilon:
            # select a random action from set of all actions
            current_index_action = set_of_actions.index(rd.choice(set_of_actions))
            action = set_of_actions[current_index_action]
        # if the generated num is greater than epsilon, we follow exploitation policy
        else:
            # select an action with highest value for current state
            current_index_action = get_best_action_index(current_state, weights)
            action = set_of_actions[current_index_action]

        for t in itertools.count():
            """
                perform action and get reward
            """
            # convert chosen action to a vector suitable for the simulator
            action_vector = [0] * 43  # simulator requires a vector like this
            if action == 'left':
                action_vector[15] = 1
            elif action == 'right':
                action_vector[14] = 1
            elif action == 'shoot':
                action_vector[0] = 1
            else:
                print("error in action")

            # apply selected action, collect values for next_state and reward
            observation, reward, done, info = env.step(action_vector)
            # sum up reward to calculate score for episode
            score += reward
            reward *= 10

            """
                get next state
            """
            # update angular position once the action has been taken
            angular_pos = angular_pos*180.0 + action_to_angle(action)
            # correct it if needed
            if angular_pos > 180:
                angular_pos = (angular_pos - 180) - 180
            if angular_pos < -180:
                angular_pos = (angular_pos + 180) + 180
            # scale angular position from -1 to 1
            angular_pos = angular_pos/180.0
            next_state = np.hstack((np.array([1]), np.array(angular_pos), get_demons_state(observation)))

            """
                choosing action: epsilon greedy selection policy
            """
            if np.random.random() <= epsilon:
                # select a random action from set of all actions
                next_action_index = set_of_actions.index(rd.choice(set_of_actions))
                next_action = set_of_actions[next_action_index]
            # if the generated num is greater than epsilon, we follow exploitation policy
            else:
                # select an action with highest value for current state
                next_action_index = get_best_action_index(current_state, weights)
                next_action = set_of_actions[next_action_index]

            """
                Sarsa calculations
            """

            # Calculate the Sarsa target value
            Q_target = reward + discount_factor * np.dot(weights[next_action_index, :], next_state)
            # Calculate the difference/error between target and current Q
            Q_error = Q_target - np.dot(weights[current_index_action, :], current_state)
            # Update the Q table, alpha is the learning rate
            weights[current_index_action, :] = weights[current_index_action, :] - alpha * Q_error * current_state

            """
                update state
            """
            current_state = next_state
            action = next_action
            current_index_action = next_action_index

            """
                Check if done
            """
            if done:
                scores_list.append(score)
                if i_episode % np.floor(num_episodes/20) == 0:
                    print("======================")
                    print("episode:", i_episode)
                    print("score:", score)
                    print("State", current_state)
                    print("weights", weights)
                    print("epsilon:", epsilon)
                break

        # gradually decay the epsilon
        if epsilon > 0.1:
            epsilon -= 1.0 / num_episodes
    return scores_list


# with tile coding for angular position
def sarsa_w_tile(seed):
    epsilon = 0.9
    # set seed
    np.random.seed(seed)
    scores_list = []
    for i_episode in range(num_episodes):
        observation = env.reset()
        score = 0
        angular_pos = 0
        # number of features to approximate Q
        number_of_features = (66 + 42) + 27  # (features from image) + features from angular position\
        current_state = np.hstack((np.array([1]), get_demons_state(observation), angular_tile_maker(angular_pos)))
        # weights initialization as zeros, expect for first element of each weight vector
        weights = np.zeros((3, number_of_features + 1), dtype=np.float128) + \
                  np.random.rand(3, number_of_features + 1) * 10
        """
            choose first action with epsilon greedy selection policy
        """
        if np.random.random() <= epsilon:
            # select a random action from set of all actions
            current_index_action = set_of_actions.index(rd.choice(set_of_actions))
            action = set_of_actions[current_index_action]
        # if the generated num is greater than epsilon, we follow exploitation policy
        else:
            # select an action with highest value for current state
            current_index_action = get_best_action_index(current_state, weights)
            action = set_of_actions[current_index_action]

        for t in itertools.count():
            """
                perform action and get reward
            """
            # convert chosen action to a vector suitable for the simulator
            action_vector = [0] * 43  # simulator requires a vector like this
            if action == 'left':
                action_vector[15] = 1
            elif action == 'right':
                action_vector[14] = 1
            elif action == 'shoot':
                action_vector[0] = 1
            else:
                print("error in action")

            # apply selected action, collect values for next_state and reward
            observation, reward, done, info = env.step(action_vector)
            # sum up reward to calculate score for episode
            score += reward
            reward *= 10

            """
                get next state
            """
            # update angular position once the action has been taken
            angular_pos = angular_pos + action_to_angle(action)
            # correct it if needed
            if angular_pos > 180:
                angular_pos = (angular_pos - 180) - 180
            if angular_pos < -180:
                angular_pos = (angular_pos + 180) + 180
            next_state = np.hstack((np.array([1]), get_demons_state(observation), angular_tile_maker(angular_pos)))

            """
                choosing action: epsilon greedy selection policy
            """
            if np.random.random() <= epsilon:
                # select a random action from set of all actions
                next_action_index = set_of_actions.index(rd.choice(set_of_actions))
                next_action = set_of_actions[next_action_index]
            # if the generated num is greater than epsilon, we follow exploitation policy
            else:
                # select an action with highest value for current state
                next_action_index = get_best_action_index(current_state, weights)
                next_action = set_of_actions[next_action_index]

            """
                Sarsa calculations
            """

            # Calculate the Sarsa target value
            Q_target = reward + discount_factor * np.dot(weights[next_action_index, :], next_state)
            # Calculate the difference/error between target and current Q
            Q_error = Q_target - np.dot(weights[current_index_action, :], current_state)
            # Update the Q table, alpha is the learning rate
            weights[current_index_action, :] = weights[current_index_action, :] - alpha * Q_error * current_state

            """
                update state
            """
            current_state = next_state
            action = next_action
            current_index_action = next_action_index

            """
                Check if done
            """
            if done:
                scores_list.append(score)
                if i_episode % np.floor(num_episodes/20) == 0:
                    print("======================")
                    print("episode:", i_episode)
                    print("score:", score)
                    print("State", current_state)
                    print("weights", weights)
                    print("epsilon:", epsilon)
                break

        # gradually decay the epsilon
        if epsilon > 0.1:
            epsilon -= 1.0 / num_episodes
    return scores_list


"""
    Calculate them all
"""

# q-learning
scores_list_q1 = q_learning_no_tile(10)
scores_list_q2 = q_learning_no_tile(20)
scores_list_q3 = q_learning_no_tile(30)
scores_list_q4 = q_learning_w_tile(10)
scores_list_q5 = q_learning_w_tile(20)
scores_list_q6 = q_learning_w_tile(30)
# sarsa
scores_list_s1 = sarsa_no_tile(10)
scores_list_s2 = sarsa_no_tile(20)
scores_list_s3 = sarsa_no_tile(30)
scores_list_s4 = sarsa_w_tile(10)
scores_list_s5 = sarsa_w_tile(20)
scores_list_s6 = sarsa_w_tile(30)

#Q-learning
avg_scores_q1 = [sum(scores_list_q1[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_q1 = [sum(scores_list_q1[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]


avg_scores_q2 = [sum(scores_list_q2[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_q2 = [sum(scores_list_q2[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]


avg_scores_q3 = [sum(scores_list_q3[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_q3 = [sum(scores_list_q3[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

avg_scores_q4 = [sum(scores_list_q4[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_q4 = [sum(scores_list_q4[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

avg_scores_q5 = [sum(scores_list_q5[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_q5 = [sum(scores_list_q5[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

avg_scores_q6 = [sum(scores_list_q6[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_q6 = [sum(scores_list_q6[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

#SARSA-learning
avg_scores_s1 = [sum(scores_list_s1[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_s1 = [sum(scores_list_s1[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]


avg_scores_s2 = [sum(scores_list_s2[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_s2 = [sum(scores_list_s2[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]


avg_scores_s3 = [sum(scores_list_s3[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_s3 = [sum(scores_list_s3[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

avg_scores_s4 = [sum(scores_list_s4[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_s4 = [sum(scores_list_s4[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

avg_scores_s5 = [sum(scores_list_s5[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_s5 = [sum(scores_list_s5[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

avg_scores_s6 = [sum(scores_list_s6[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes_s6 = [sum(scores_list_s6[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]

avg_score_q_w = [(avg_scores_q1[i]+avg_scores_q2[i]+avg_scores_q3[i])/3 for i in range(len(avg_scores_q1))]
avg_score_q_wt = [(avg_scores_q4[i]+avg_scores_q5[i]+avg_scores_q6[i])/3 for i in range(len(avg_scores_q4))]
avg_score_s_w = [(avg_scores_s1[i]+avg_scores_s2[i]+avg_scores_s3[i])/3 for i in range(len(avg_scores_s1))]
avg_score_s_wt = [(avg_scores_s4[i]+avg_scores_s5[i]+avg_scores_s6[i])/3 for i in range(len(avg_scores_s4))]
"""
    Plot them all (TODO)
"""


# without tiles avg learning curve plot
plt.plot(np.arange(num_episodes-1)+1, avg_scores_q1, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_q2, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_q3, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_score_q_w, label = "Q-Learning without tiles", color = 'r',linewidth = 3.0)

plt.xticks(np.arange(0, num_episodes, 100))
plt.yticks(np.arange(0, 25, 2))

plt.xlabel("Episodes", size=13)
plt.ylabel("Average Reward", size=13)
#plt.legend(loc='upper right')
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_q1,avg_scores_q2,alpha = 0.4, color = 'y', label = "average score range for Q learning without tiles")
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_q1,avg_scores_q3, color = 'y', alpha = 0.4)
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_q2,avg_scores_q3, color = 'y', alpha = 0.4)
#plt.show()
#


# with tiles avg learning curve plot
plt.plot(np.arange(num_episodes-1)+1, avg_scores_q4, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_q5, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_q6, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_score_q_wt, label = "Q-Learning with tiles", color = 'b',linewidth = 3.0)

plt.xticks(np.arange(0, num_episodes, 100))
plt.yticks(np.arange(0, 25, 2))

plt.xlabel("Episodes", size=13)
plt.ylabel("Average Reward", size=13)

plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_q4,avg_scores_q5,alpha = 0.5, color = 'c', label = "average score range for Q learning with tiles")
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_q4,avg_scores_q6, color = 'c',alpha = 0.5)
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_q5,avg_scores_q6, color = 'c',alpha = 0.5)

plt.legend(loc='upper right', prop={'size': 12})
plt.suptitle("Q-Learning with Tiles and without Tiles")
plt.show()

#
'''sarsa without tails'''

plt.plot(np.arange(num_episodes-1)+1, avg_scores_s1, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_s2, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_s3, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_score_s_w, label = "SARSA learning without tiles", color = 'r',linewidth = 3.0)

plt.xticks(np.arange(0, num_episodes, 100))
plt.yticks(np.arange(0, 25, 2))

plt.xlabel("Episodes", size=13)
plt.ylabel("Average Reward", size=13)

plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_s1,avg_scores_s2, alpha = 0.5, color = 'y', label = "average score range for SARSA learning without tiles")
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_s1,avg_scores_s3, color = 'y', alpha = 0.5)
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_s2,avg_scores_s3, color = 'y', alpha = 0.5)
#plt.show()

#SARSA learning curve plot with tiles
# with tiles avg learning curve plot
plt.plot(np.arange(num_episodes-1)+1, avg_scores_s4, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_s5, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_scores_s6, linewidth = 0.0)
plt.plot(np.arange(num_episodes-1)+1, avg_score_s_wt, label = "SARSA Learning with tiles", color = 'b',linewidth = 3.0)

plt.xticks(np.arange(0, num_episodes, 100))
plt.yticks(np.arange(0, 25, 2))

plt.xlabel("Episodes", size=13)
plt.ylabel("Average Reward", size=13)

plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_s4,avg_scores_s5, color = 'c', alpha = 0.4, label = "average score range for SARSA learning with tiles")
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_s4,avg_scores_s6, color = 'c', alpha = 0.4)
plt.fill_between(np.arange(num_episodes-1)+1, avg_scores_s5,avg_scores_s6, color = 'c', alpha = 0.4)

plt.legend(loc='upper right', prop={'size': 12})
#plt.rc('legend.fontsize', size = 12)

plt.suptitle("SARSA Learning with Tiles and without Tiles")
plt.show()
#

#plotting avg learning for sarsa and Q-learning

plt.plot(np.arange(num_episodes-1)+1, avg_score_q_wt, label = "Q-Learning with tiles", color = 'b',linewidth = 3.0)
plt.plot(np.arange(num_episodes-1)+1, avg_score_s_wt, label = "SARSA Learning with tiles", color = 'r',linewidth = 3.0)

plt.xticks(np.arange(0, num_episodes, 100))
plt.yticks(np.arange(0, 25, 2))

plt.xlabel("Episodes", size=13)
plt.ylabel("Average Reward", size=13)

plt.legend(loc='upper right', prop={'size': 12})
#plt.rc('legend.fontsize', size = 12)

plt.suptitle("SARSA and Q learning with Tiles")
plt.show()

#without tiles Q-leanrning vs SARSA
plt.plot(np.arange(num_episodes-1)+1, avg_score_q_w, label = "Q-Learning without tiles", color = 'r',linewidth = 3.0)
plt.plot(np.arange(num_episodes-1)+1, avg_score_s_w, label = "SARSA learning without tiles", color = 'b',linewidth = 3.0)
plt.xticks(np.arange(0, num_episodes, 100))
plt.yticks(np.arange(0, 25, 2))

plt.xlabel("Episodes", size=13)
plt.ylabel("Average Reward", size=13)

plt.legend(loc='upper right', prop={'size': 12})
#plt.rc('legend.fontsize', size = 12)

plt.suptitle("SARSA and Q learning without Tiles")
plt.show()