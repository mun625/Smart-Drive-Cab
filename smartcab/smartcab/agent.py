import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here : states_indexed, Q_matrix, policy
        self.states_indexed = []
        # Q_matrix[state][action] = value. and state is represented by index(number)
        self.Q_matrix = [[0 for x in range(0,4)] for x in range(0,512)]
        # policy[state] = optimal_action. and state is represented by index(number)
        self.policy = []

        # Initialize state (total 512 states : 2 X 4 X 4 X 4 X 4)
        light_states = ['green', 'red']
        oncoming_states = [None, 'forward', 'right', 'left']
        right_states = [None, 'forward', 'right', 'left']
        left_states = [None, 'forward', 'right', 'left']
        nextway_states = [None, 'forward', 'left', 'right']
        for light in light_states:
            for oncoming in oncoming_states:
                for right in right_states:
                    for left in left_states:
                        for nextway in nextway_states:
                            self.states_indexed.append(({'light' : light, 'oncoming' : oncoming, 'right' : right, 'left' : left}, nextway))

        # Initialize Q_matrix Table => Q_matrix[states_indexed][action] = value
        self.action_candidates = ['forward', 'left', 'right', None]
        for i,item in enumerate(self.states_indexed):
            for j,jtem in enumerate(self.action_candidates):
                self.Q_matrix[i][j] = 0

        # Initialize policy => policy[states_indexed] = value
        # First policy is going forward in any situations.
        for i,item in enumerate(self.states_indexed):
            self.policy.append('forward')


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs,self.next_waypoint)
        indexOfstate = self.states_indexed.index(self.state)

        # TODO: Select action according to your policy
        action = self.policy[indexOfstate]
        indexOfaction = self.action_candidates.index(action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # [STEP 1] : Sense new state
        new_state = (self.env.sense(self), self.planner.next_waypoint())
        index_new_state = self.states_indexed.index(new_state)

        # [STEP 2] : Refresh the Q_matrix => Q(State, action) = Reward + Max(Q(new_state),alpha)
        max = 0
        for j,item in enumerate(self.action_candidates):
            if self.Q_matrix[index_new_state][j] > self.Q_matrix[index_new_state][max]:
                max = j
        self.Q_matrix[indexOfstate][indexOfaction] = reward + self.Q_matrix[index_new_state][max]
        # [STEP 3] : Update the policy (we just modify policy[indexOfstate] because other states are not influenced by that action)
        max = 0
        for j,item in enumerate(self.action_candidates):
            if self.Q_matrix[indexOfstate][j] > self.Q_matrix[indexOfstate][max]:
                max = j
        self.policy[indexOfstate] = self.action_candidates[max]

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
