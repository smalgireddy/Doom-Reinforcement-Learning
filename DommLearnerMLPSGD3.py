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
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model

env = gym.make('ppaquette/DoomDefendLine-v0')
# learning parameters for both Q-learning and Sarsa
epsilon = 0.9
discount_factor = 0.9
alpha = 0.0001
num_episodes = 1000
# number of features to approximate Q
number_of_features = 67 + 42  # angular position + features from image
# weights initialization as zeros, expect for first element of each weight vector
weights = np.zeros((3, number_of_features + 1), dtype=np.float128) + np.random.rand(3, number_of_features + 1) * 10

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


def get_max_q_for_next_state(next_state, fit_left, fit_right, fit_shoot):
    """
    This array starts with 1 because of the linear regression w0 coefficient
    and has as second term the angular position variable
    IMPORTANT: we assume that in the next state only the angular position will change
    :type next_state: numpy array
    """
    # estimate the Q for each of the two states with linear regression weights
    Q_max = 0
    Q_left = fit_left.predict(np.array([next_state]))
    Q_right = fit_right.predict(np.array([next_state]))
    Q_shoot = fit_shoot.predict(np.array([next_state]))
    if Q_left > Q_max:
        Q_max = Q_left
    if Q_right > Q_max:
        Q_max = Q_right
    if Q_shoot > Q_max:
        Q_max = Q_shoot
    return Q_max


def get_best_action_index(current_state, fit_left, fit_right, fit_shoot):
    """
    Same as get_max_Q_for_next_state but returns the index for the best action.
    :type current_state: numpy array
    :type weights: numpy array
    """
    # estimate the Q for each of the two states with linear regression weights
    max_action_index = 0
    Q_max = 0
    Q_left = fit_left.predict(np.array([current_state]))
    Q_right = fit_right.predict(np.array([current_state]))
    Q_shoot = fit_shoot.predict(np.array([current_state]))
    if Q_left > Q_max:
        Q_max = Q_left
        max_action_index = 0
    if Q_right > Q_max:
        Q_max = Q_right
        max_action_index = 1
    if Q_shoot > Q_max:
        Q_max = Q_shoot
        max_action_index = 2
    return max_action_index


def tile_maker(mask, x):
    #     frame_crop =  mask[183:298, 0:639]
    frame1_div = mask[183:240, 0:639]  # TO CHECK LATER!!
    frame2_div = mask[241:298, 0:639]

    #     cv2.imshow('a',frame2_div)
    #     cv2.waitKey(0)

    tile = []
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

    tile_filter = np.where(np.array([np.sum(t) for t in tile]) > 0, 1, 0)
    return tile_filter


def get_demons_state(observation):
    #     print("Start")
    observation = observation[..., ::-1]
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
    lower_red = np.array([-10, 100, 100])
    upper_red = np.array([10, 200, 200])
    lower_brown = np.array([16, 100, 100])
    upper_brown = np.array([16, 250, 250])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_brown, upper_brown)
    mask = mask2 + mask1
    res = cv2.bitwise_and(observation, observation, mask=mask)
    features1 = tile_maker(mask, 20)
    features2 = tile_maker(mask, 15)
    features3 = tile_maker(mask, 10)
    features4 = tile_maker(mask, 5)
    features = np.hstack((features1, features2, features3, features4))
    #     print("lenght:", len(features))
    #     print(features)
    #     print("Finished")
    return (features)

"""
    Q-LEARNING ALGORITHM
"""
# this is for Q learning!!!
scores_list = []

#initalize the MLPREgressor models
fit_left = MLPRegressor(learning_rate = 'invscaling')
fit_right = MLPRegressor(learning_rate = 'invscaling')
fit_shoot = MLPRegressor(learning_rate = 'invscaling')


#get the available methods to the context
possibles = globals().copy()
possibles.update(locals())



Q_target = 0
flag_left = True
flag_right = True
flag_shoot = True
current_state = np.array([np.hstack((np.zeros(110)))])
#initialize models training with random choice
fit_left.partial_fit(current_state, np.zeros(1))
fit_right.partial_fit(current_state, np.zeros(1))
fit_shoot.partial_fit(current_state, np.zeros(1))


for i_episode in range(num_episodes):
    observation = env.reset()

    angular_pos = 0  # start angular position on each episode
    score = 0  # will count score per episode

    # at the beginning, do a random action to get the pixels from the image
    # because the only way to get these is by using the "step" function
    # ===observation, _, _, _ = env.step(env.action_space.sample())
    current_state = np.hstack((np.array([1]), np.array(angular_pos), get_demons_state(observation)))

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
        action_index = get_best_action_index(current_state, fit_left, fit_right, fit_shoot)
        action = set_of_actions[action_index]

    for t in itertools.count():

        #if i_episode > math.floor(0.95*num_episodes):
        #env.render()

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
            action_index = get_best_action_index(current_state, fit_left, fit_right, fit_shoot)
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

        reward *= 1

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
            Q-learning calculations
        """
        # Calculate the Q-learning target value
        #decide the possbile fit
        method = possibles.get(''.join(["fit_", action]))

        # initialize Q_target with the reward
        if flag_left & action_index==0:
            Q_target = reward
        elif flag_right & action_index==1:
            Q_target = reward
        elif flag_shoot & action_index==2:
            Q_target = reward
        else:
            Q_target = reward + discount_factor * \
                                get_max_q_for_next_state(next_state, fit_left, fit_right, fit_shoot)

        # now insert new observation to the model and retrain with partial_fit
        train_X = np.array([current_state])
        train_Y = np.array([Q_target])
        fit_q = method.partial_fit(np.array([current_state]), train_Y)



        """
            update state
        """
        current_state = next_state


        """
            Check if done
        """
        if done:
            scores_list.append(score)
            if i_episode % 100 == 0:
                print("======================")
                print("episode:", i_episode)
                print("score:", score)
                print("State", current_state)
                #print("weights", weights)
                print("epsilon:", epsilon)
            break

    # gradually decay the epsilon on each episode
    if epsilon > 0.1:
        epsilon -= 1.0 / num_episodes



# if i_episode % 20 == 0:
plt.figure(i_episode)
avg_scores = [sum(scores_list[0:(i + 1)]) / (i + 1) for i in range(num_episodes - 1)]
avg_100_episodes = [sum(scores_list[0:(i + 1)]) / (i + 1) for i in range(100, num_episodes-1)]
plt.plot(np.arange(num_episodes-1)+1, avg_scores)
plt.xticks(np.arange(0, num_episodes, 100))
plt.yticks(np.arange(0, 25, 2))
plt.xlabel("Episodes", size=14)
plt.ylabel("Average Reward")
plt.show()