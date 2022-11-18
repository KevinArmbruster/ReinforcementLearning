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


class Maze:
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
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    DEAD_REWARD = -1000000

    def __init__(self, maze, minotaur_can_stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.minotaur_can_stay = minotaur_can_stay
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards()

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
        for y in range(self.maze.shape[0]):
            for x in range(self.maze.shape[1]):
                for yM in range(self.maze.shape[0]):
                    for xM in range(self.maze.shape[1]):
                        if self.maze[y, x] != 1:  # exclude states where player is in wall
                            state_tpl = (y, x, yM, xM)
                            states[state_no] = state_tpl
                            map[state_tpl] = state_no
                            state_no += 1
        return states, map

    def __move(self, state, action) -> list:
        if self.__check_player_dead(state) or self.check_player_escaped(state):
            return [state]

        next_p_s = self.__move_player(state=state, action=action)
        next_states = self.__get_possible_states_after_minotaur(state=next_p_s)
        return next_states

    def __move_player(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        y = self.states[state][0] + self.actions[action][0]
        x = self.states[state][1] + self.actions[action][1]

        if self.__check_state_possible(y, x):
            return self.map[(y, x, self.states[state][2], self.states[state][3])]
        else:
            return state

    def __get_possible_states_after_minotaur(self, state):
        (yP, xP, yM, xM) = self.states[state]
        possible_next_states = []

        for action in self.actions.keys():
            if not self.minotaur_can_stay and action == self.STAY:
                continue

            (y_action, x_action) = self.actions.get(action)
            yM_new = yM + y_action
            xM_new = xM + x_action

            if self.__check_state_possible(yM_new, xM_new, walls_walkable=True):
                state = self.map[(yP, xP, yM_new, xM_new)]
                possible_next_states.append(state)

        return possible_next_states

    def __check_state_possible(self, y, x, walls_walkable=False):
        impossible = (y < 0) or (y >= self.maze.shape[0]) or \
                     (x < 0) or (x >= self.maze.shape[1]) or \
                     (not walls_walkable and self.maze[y, x] == 1)
        return not impossible

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probabilities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                possible_next_states = self.__move(s, a)
                for next_s in possible_next_states:
                    transition_probabilities[next_s, s, a] = 1 / len(possible_next_states)

        return transition_probabilities

    def __rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                possible_next_states = self.__move(s, a)

                for next_s in possible_next_states:
                    # with stochastic rewards, use the average
                    rewards[s, a] += self.__reward_f(s, a, next_s)
                rewards[s, a] /= len(possible_next_states)

        return rewards

    def __reward_f(self, s, a, next_s):
        if self.__check_player_dead(s):
            reward = self.DEAD_REWARD
        elif self.check_player_escaped(s):
            reward = self.GOAL_REWARD
        elif s == next_s and a != self.STAY:  # move is programmed to STAY if action is not possible
            reward = self.IMPOSSIBLE_REWARD
        else:
            reward = self.STEP_REWARD

        return reward  # * self.transition_probabilities[next_s, s, a]

    def __check_player_dead(self, s):
        (y, x, yM, xM) = self.states[s]
        return y == yM and x == xM

    def check_player_escaped(self, s):
        return self.maze[self.states[s][0:2]] == 2

    def simulate_DP(self, start, policy):
        path = list()
        # Deduce the horizon from the policy shape
        horizon = policy.shape[1]
        # Initialize current state and time
        t = 0
        s = self.map[start]
        # Add the starting position in the maze to the path
        path.append(start)
        while t < horizon - 1:
            # Move to next state given the policy and the current state
            next_states = self.__move(s, policy[s, t])
            # random iid movement of minotaur
            next_s = np.random.choice(next_states, 1)[0]
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t += 1
            s = next_s
        return path

    def simulate_VI(self, start, policy):
        path = list()
        # Initialize current state, next state and time
        t = 1
        s = self.map[start]
        # Add the starting position in the maze to the path
        path.append(start)
        # Move to next state given the policy and the current state
        next_states = self.__move(s, policy[s])
        # random iid movement of minotaur
        next_s = np.random.choice(next_states, 1)[0]
        # Add the position in the maze corresponding to the next state
        # to the path
        path.append(self.states[next_s])
        # Loop while state is not the goal state
        while s != next_s:
            # Update state
            s = next_s
            # Move to next state given the policy and the current state
            next_states = self.__move(s, policy[s])
            # random iid movement of minotaur
            next_s = np.random.choice(next_states, 1)[0]
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t += 1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming backwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def value_iteration(env, gamma, epsilon, max_iter=200):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV, ord=np.inf) >= tol and n < max_iter:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    print("Needed iterations: ", n, ", Final error: ", np.linalg.norm(V - BV, ord=np.inf))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

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
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

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

        grid.get_celld()[(path[i][2:])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2:])].get_text().set_text('Minotaur')

        if i > 0:
            if path[i] == path[i - 1]:
                # goal coloring
                grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0:2])].get_text().set_text('Player is out')

            if path[i][0:2] != path[i - 1][0:2]:
                # remove old cell coloring
                grid.get_celld()[(path[i - 1][0:2])].set_facecolor(col_map[maze[path[i - 1][0:2]]])
                grid.get_celld()[(path[i - 1][0:2])].get_text().set_text('')

            if path[i][2:] != path[i - 1][2:]:
                # remove old cell coloring
                grid.get_celld()[(path[i - 1][2:])].set_facecolor(col_map[maze[path[i - 1][2:]]])
                grid.get_celld()[(path[i - 1][2:])].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


def visualize_policy(env, policy, minotaur_position=(5, 5)):
    maze = env.maze
    if np.ndim(policy) == 2:
        policy = policy[:, 0]

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

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
            if maze[y, x] == 1:
                continue
            s = env.map[(y, x, minotaur_position[0], minotaur_position[1])]
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


def visualize_probability_over_time_horizons(maze, start=(0, 0, 6, 5), runs=10000, min_H=1, max_H=30):
    ### NO STAY
    env = Maze(maze, minotaur_can_stay=False)
    probabilities_for_plot_no_stay = simulate_DP_over_time_horizons(env, start, runs, max_H, min_H)

    ### STAY
    env = Maze(maze, minotaur_can_stay=True)
    probabilities_for_plot_with_stay = simulate_DP_over_time_horizons(env, start, runs, max_H, min_H)

    ### PLOT
    plt.plot(list(range(1, max_H + 1)), probabilities_for_plot_no_stay, c="blue", label="Minotaur can't stay")
    plt.plot(list(range(1, max_H + 1)), probabilities_for_plot_with_stay, c="red", label="Minotaur can stay")
    plt.title("Probability of Escaping over Time Horizons")
    plt.ylabel("Probability")
    plt.xlabel("Time Horizon")
    plt.legend()
    plt.show()


def simulate_DP_over_time_horizons(env, start=(0, 0, 6, 5), runs=10000, min_H=1, max_H=30):
    probabilities = []
    for h in range(min_H, max_H + 1):
        prob = 0
        V, policy_DP_stay = dynamic_programming(env, h)

        for i in range(runs):
            path = env.simulate_DP(start, policy_DP_stay)
            prob += env.check_player_escaped(env.map[path[-1]])
        prob = prob / runs
        print(h, prob)
        probabilities.append(prob)
    return probabilities

def simulate_VI_for_probability(env, gamma, epsilon, start=(0, 0, 6, 5), runs=10000):
    prob = 0
    V, policy_VI = value_iteration(env, gamma, epsilon)

    for i in range(runs):
        path = env.simulate_VI(start, policy_VI)
        prob += env.check_player_escaped(env.map[path[-1]])
    prob = prob / runs
    return prob
