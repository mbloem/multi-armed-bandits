"""
author: Cameron Davidson-Pilon

from: https://gist.github.com/CamDavidsonPilon/4c4561cfd9463fb875b6
downloaded on: 25 February 2019
"""

from pymc import rbeta
import numpy as np

rand = np.random.rand

class Bandits(object):
    """
    This class represents N bandits machines.

    parameters:
        p_array: a (n,) Numpy array of probabilities >0, <1.

    methods:
        pull( i ): return the results, 0 or 1, of pulling 
                   the ith bandit.
    """
    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)
        
    def pull( self, i ):
        #i is which arm to pull
        return rand() < self.p[i]
    
    def __len__(self):
        return len(self.p)

    
class MABPolicy( object ):
    def __init__(self, bandits):
        
        self.bandits = bandits
        n_bandits = len( self.bandits )
        self.wins = np.zeros( n_bandits )
        self.trials = np.zeros(n_bandits )
        self.N = 0
        self.choices = []
        self.score = []

    def sample( self ):
        raise NotImplementedError('MABPolicy must implement the sample method')

class BayesianStrategy( MABPolicy ):
    """
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
    
    methods:
        sample_bandits(n): sample and train on n pulls.

    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        score: the historical score as a (N,) array

    """
    
    def __init__(self, bandits):
        
        MABPolicy.__init__(self, bandits)

    def sample( self, n=1 ):
        
        score = np.zeros( n )
        choices = np.zeros( n )
        
        for k in range(n):
            #sample from the bandits's priors, and select the largest sample
            choice = np.argmax( rbeta( 1 + self.wins, 1 + self.trials - self.wins) )
            
            #sample the chosen bandit
            result = self.bandits.pull( choice )
            
            #update priors and score
            self.wins[ choice ] += result
            self.trials[ choice ] += 1
            score[ k ] = result 
            self.N += 1
            choices[ k ] = choice
            
        self.score = np.r_[ self.score, score ]
        self.choices = np.r_[ self.choices, choices ]
        return 

class epsilonGreedyStrategy( MABPolicy ):
    def __init__(self, bandits, epsilon):
        MABPolicy.__init__(self, bandits)

        self.epsilon = epsilon
        self.Q = np.zeros(len( self.bandits ), dtype=np.float)

    def sample( self, n=1 ):
        pass
    
    # Update Q action-value using:
    # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
    def update_Q(self, action, reward):
        self.k[action] += 1  # update action counter k -> k+1
        self.Q[action] += (1./self.k[action]) * (reward - self.Q[action])

    # Choose action using an epsilon-greedy agent
    def get_action(self, bandit, force_explore=False):
        rand = np.random.random()  # [0.0,1.0)
        if (rand < self.epsilon) or force_explore:
            action_explore = np.random.randint(bandit.N)  # explore random bandit
            return action_explore
        else:
            #action_greedy = np.argmax(self.Q)  # exploit best current bandit
            action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
            return action_greedy