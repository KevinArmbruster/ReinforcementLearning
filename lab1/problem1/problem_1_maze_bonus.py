# Kevin Armbruster (930519-T711)
# Mohammed Akif (990123-4493)

import random

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'
GOLD = '#FFD700'


class BonusMaze:
    # Cells
    EMPTY_CELL = 0
    OBSTACLE_CELL = 1
    GOAL_CELL = 2
    KEY_CELL = 3

    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STAY_REWARD = -2
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    DEAD_REWARD = -1000000

    def __init__(self, maze, minotaur_can_stay=False, prob_minotaur_moves_iid=0.65):
        self.maze = maze
        self.minotaur_can_stay = minotaur_can_stay
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.prob_minotaur_moves_iid = prob_minotaur_moves_iid
        # self.transition_probabilities = self.__transitions()
        # self.rewards = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        state_no = 0
        for key in [0, 1]:
            for y in range(self.maze.shape[0]):
                for x in range(self.maze.shape[1]):
                    for yM in range(self.maze.shape[0]):
                        for xM in range(self.maze.shape[1]):
                            if self.maze[y, x] != self.OBSTACLE_CELL:  # exclude states where player is in wall
                                state_tpl = (y, x, yM, xM, key)
                                states[state_no] = state_tpl
                                map[state_tpl] = state_no
                                state_no += 1
        return states, map

    def __move(self, state, action, move_towards_player=False) -> list:
        if self.__check_player_dead(state) or self.check_player_escaped(state):
            return [state]

        # Player
        next_p_s = self.__move_player(state=state, action=action)

        # Minotaur
        if move_towards_player:
            next_states = [self.__get_next_state_when_Minotaur_moves_in_direction_of_player(next_p_s)]
        else:
            next_states = self.__get_possible_states_after_minotaur(next_p_s)
        return next_states

    def __move_player(self, state, action):
        (yP, xP, yM, xM, key) = self.states[state]
        y = yP + self.actions[action][0]
        x = xP + self.actions[action][1]

        if self.__check_state_possible(y, x):
            if self.maze[y, x] == self.KEY_CELL:
                key = 1  # found key
            return self.map[(y, x, yM, xM, key)]
        else:
            return state

    def __get_next_state_when_Minotaur_moves_in_direction_of_player(self, state):
        (yP, xP, yM, xM, key) = self.states[state]
        y = yP - yM
        x = xP - xM

        if abs(y) > abs(x):
            yM_new = yM + np.sign(y)
            return self.map[(yP, xP, yM_new, xM, key)]
        else:
            xM_new = xM + np.sign(x)
            return self.map[(yP, xP, yM, xM_new, key)]

    def __get_possible_states_after_minotaur(self, state):
        (yP, xP, yM, xM, key) = self.states[state]
        possible_next_states = []

        for action in self.actions.keys():
            if not self.minotaur_can_stay and action == self.STAY:
                continue

            (y_action, x_action) = self.actions.get(action)
            yM_new = yM + y_action
            xM_new = xM + x_action

            if self.__check_state_possible(yM_new, xM_new, walls_walkable=True):
                state = self.map[(yP, xP, yM_new, xM_new, key)]
                possible_next_states.append(state)

        return possible_next_states

    def __check_state_possible(self, y, x, walls_walkable=False):
        impossible = (y < 0) or (y >= self.maze.shape[0]) or \
                     (x < 0) or (x >= self.maze.shape[1]) or \
                     (not walls_walkable and self.maze[y, x] == self.OBSTACLE_CELL)
        return not impossible

    def __reward_f(self, s, a, next_s):
        if self.__check_player_dead(s):
            reward = self.DEAD_REWARD
        elif self.check_player_escaped(s):
            reward = self.GOAL_REWARD
        elif s == next_s and a != self.STAY:  # move is programmed to STAY if action is not possible
            reward = self.IMPOSSIBLE_REWARD
        elif self.states[s][-1] != self.states[next_s][-1]:  # if key is acquired, possible only once
            reward = self.GOAL_REWARD
        elif a != self.STAY:
            reward = self.STEP_REWARD
        else:
            reward = self.STAY_REWARD
        return reward

    def __check_player_dead(self, s):
        (y, x, yM, xM, key) = self.states[s]
        return y == yM and x == xM

    def check_player_escaped(self, s):
        (y, x, yM, xM, key) = self.states[s]
        return self.maze[y, x] == self.GOAL_CELL and key == 1

    def __check_game_ended(self, s):
        return self.__check_player_dead(s) or self.check_player_escaped(s)

    def __step(self, s, a):
        move_towards_player = np.random.rand() <= self.prob_minotaur_moves_iid
        next_states = self.__move(s, a, move_towards_player)
        next_s = np.random.choice(next_states, 1)[0]
        reward = self.__reward_f(s, a, next_s)
        return next_s, reward

    def simulate_path(self, start, policy, max_steps=50):
        path = list()
        # Initialize current state, next state and time
        s = self.map[start]
        last_s = -1
        path.append(start)
        i = 0

        while last_s != s and i < max_steps:
            next_s, _ = self.__step(s, policy[s])
            path.append(self.states[next_s])

            last_s = s
            s = next_s
            i += 1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)

    def q_learning(self, start=(0, 0, 6, 5, 0), number_episodes=50000, discount_factor=49 / 50,
                   step_size_exponent=2 / 3, exploration_prob=0.1, exploration_decay=None):
        # init
        Q = np.ones((self.n_states, self.n_actions))  # q values per (s,a)
        N = np.zeros((self.n_states, self.n_actions))  # count visits per (s,a)

        history_V_initial_state = []

        for k in range(1, number_episodes+1):
            t = 0
            s = self.map[start]  # reset env
            last_s = -1
            if exploration_decay:
                exploration_prob = 1/k**exploration_decay
            history_V_initial_state.append(np.max(Q[s, :]))  # V of initial state

            if k % 10000 == 0:
                print(f"Iteration ", k)

            while last_s != s:  # not self.__check_game_ended(s) and
                # select action
                a = self.__epsilon_greedy_policy(Q[s, :], exploration_prob)
                # observe next state and reward
                next_s, reward = self.__step(s, a)
                # update Q with sampled TD error
                N[s, a] += 1
                step_size = 1 / (N[s, a] ** step_size_exponent)
                Q[s, a] += step_size * (reward + discount_factor * np.max(Q[next_s, :]) - Q[s, a])

                # print(s, a, reward, next_s, np.argmax(Q[next_s, :]))

                last_s = s
                s = next_s
                t += 1

        # V = np.max(Q, axis=1)  # not needed
        policy = np.argmax(Q, axis=1)
        return Q, N, policy, history_V_initial_state

    def sarsa(self, start=(0, 0, 6, 5, 0), number_episodes=50000, discount_factor=49 / 50, step_size_exponent=2 / 3,
              exploration_prob=0.1, exploration_decay=None):
        # init
        Q = np.zeros((self.n_states, self.n_actions))  # q values per (s,a)
        N = np.zeros((self.n_states, self.n_actions))  # count visits per (s,a)

        history_V_initial_state = []

        for k in range(1, number_episodes+1):
            t = 0
            s = self.map[start]
            last_s = -1
            if exploration_decay:
                exploration_prob = 1/k**exploration_decay
            a = self.__epsilon_greedy_policy(Q[s, :], exploration_prob)
            history_V_initial_state.append(np.max(Q[s, :]))  # V of initial state

            if k % 10000 == 0:
                print(f"Iteration ", k)

            while last_s != s:
                # observe next state and reward
                next_s, reward = self.__step(s, a)
                next_a = self.__epsilon_greedy_policy(Q[next_s, :], exploration_prob)
                # update Q with sampled TD error
                N[s, a] += 1
                step_size = 1 / (N[s, a] ** step_size_exponent)
                Q[s, a] += step_size * (reward + discount_factor * Q[next_s, next_a] - Q[s, a])

                last_s = s
                s = next_s
                a = next_a
                t += 1

        # V = np.max(Q, axis=1)  # not needed
        policy = np.argmax(Q, axis=1)
        return Q, N, policy, history_V_initial_state

    def __epsilon_greedy_policy(self, Qs, exploration_prob):
        if np.random.rand() <= exploration_prob:
            a = random.randrange(self.n_actions)  # choose iid action
        else:
            a = np.argmax(Qs)  # choose best action
        return a


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: GOLD}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)


def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: GOLD}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Update the color at each frame
    for i in range(len(path)):
        # new cell coloring
        grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0:2])].get_text().set_text('Player')

        grid.get_celld()[(path[i][2:4])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2:4])].get_text().set_text('Minotaur')

        if i > 0:
            if path[i] == path[i - 1]:
                # goal coloring
                grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0:2])].get_text().set_text('Player is out')

            if path[i][0:2] != path[i - 1][0:2]:
                # remove old cell coloring
                grid.get_celld()[(path[i - 1][0:2])].set_facecolor(col_map[maze[path[i - 1][0:2]]])
                grid.get_celld()[(path[i - 1][0:2])].get_text().set_text('')

            if path[i][2:4] != path[i - 1][2:4]:
                # remove old cell coloring
                grid.get_celld()[(path[i - 1][2:4])].set_facecolor(col_map[maze[path[i - 1][2:4]]])
                grid.get_celld()[(path[i - 1][2:4])].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.5)


def visualize_policy(env, policy, minotaur_position=(5, 5), key=0):
    maze = env.maze
    if np.ndim(policy) == 2:
        policy = policy[:, 0]

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: GOLD}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Update the color
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == env.OBSTACLE_CELL:
                continue
            s = env.map[(y, x, minotaur_position[0], minotaur_position[1], key)]
            a = policy[s]

            # https://unicode.org/charts/nameslist/n_2190.html
            if a == 0:
                arrow = 'STAY'
            elif a == 1:
                arrow = '\u2190'  # left
            elif a == 2:
                arrow = '\u2192'  # right
            elif a == 3:
                arrow = '\u2191'  # up
            else:
                arrow = '\u2193'  # down

            grid.get_celld()[(y, x)].get_text().set_text(arrow)

    grid.get_celld()[minotaur_position].set_facecolor(LIGHT_PURPLE)
    grid.get_celld()[minotaur_position].get_text().set_text('Minotaur')

    plt.show()
