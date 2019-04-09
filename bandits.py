"""
bandits.py

author: Michael Bloem
mbloem@mosaicdatascience.com

Most of this code is based on Cameron Davidson-Pilon's Baysian Bandits code.
https://gist.github.com/CamDavidsonPilon/4c4561cfd9463fb875b6
downloaded on: 25 February 2019

The epsilon-greedy policy is based on the code by Anson Wong
https://github.com/ankonzoid/LearningX/blob/master/classical_RL/MAB/MAB.py
downloaded on: 25 February 2019
"""

from pymc import rbeta
import numpy as np

rand = np.random.rand

class Bandits(object):
    """
    This class represents an N-armed bandits machine.

    author: Cameron Davidson-Pilon

    from: https://gist.github.com/CamDavidsonPilon/4c4561cfd9463fb875b6
    downloaded on: 25 February 2019

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
    """
    MABPolicy
    
    This is a base class for multi-armed bandits policies.

    parameters:
        bandits: a Bandit class with .pull method

    methods:
        sample: sample and train; must be implemented by sub class

    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        score: the historical score as a (N,) array
    """
    def __init__(self, bandits):
        
        self.bandits = bandits
        n_bandits = len( self.bandits )
        self.wins = np.zeros( n_bandits )
        self.trials = np.zeros(n_bandits )
        self.N = 0
        self.choices = []
        self.score = []
        self.regret = []

    def sample( self ):
        raise NotImplementedError('MABPolicy must implement the sample method')

    def update_stats( self , k, choice, result, ):
        self.wins[ choice ] += result
        self.trials[ choice ] += 1
        self.N += 1

class BayesianStrategy( MABPolicy ):
    """
    BayesianStrategy
    
    Implements a online Bayesian Bandits learning strategy to solve
    the Multi-Armed Bandit problem.

    Sub-class of MABPolicy base class.

    author: Cameron Davidson-Pilon

    from: https://gist.github.com/CamDavidsonPilon/4c4561cfd9463fb875b6
    downloaded on: 25 February 2019
    
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
        regret = np.zeros( n )
        
        for k in range(n):
            # sample from the bandits's priors, and select the largest sample
            choice = np.argmax( rbeta( 1 + self.wins, 1 + self.trials - self.wins ) )
            
            # sample the chosen bandit
            result = self.bandits.pull( choice )
            
            # update stats (which updates priors for next pull)
            self.update_stats( k, choice, result )
            score[ k ] = result
            choices[ k ] = choice
            regret[ k ] = self.bandits.p[ self.bandits.optimal ] - self.bandits.p[ choice ]
            
        self.score = np.r_[ self.score, score ]
        self.choices = np.r_[ self.choices, choices ]
        self.regret = np.r_[ self.regret, regret ]
        return 

class epsilonGreedyStrategy( MABPolicy ):
    """
    epsilonGreedyStrategy
    
    Implements a online epsilon-greedy Q-learning strategy to solve
    the Multi-Armed Bandit problem.

    Sub-class of MABPolicy base class.

    Based on https://github.com/ankonzoid/LearningX/blob/master/classical_RL/MAB/MAB.py
    (author: Anson Wong / git: ankonzoid)
    
    parameters:
        bandits: a Bandit class with .pull method
    
    methods:
        sample_bandits(n): sample and train on n pulls.

    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        score: the historical score as a (N,) array

    """
    
    def __init__(self, bandits, epsilon):
        MABPolicy.__init__(self, bandits)

        self.epsilon = epsilon
        self.Q = np.zeros(len( self.bandits ), dtype=np.float)

    def sample( self, n=1 ):
        score = np.zeros( n )
        choices = np.zeros( n )
        regret = np.zeros( n )
        
        for k in range(n):
            # Choose action using an epsilon-greedy agent
            choice = self.get_action()
            
            # sample the chosen bandit
            result = self.bandits.pull( choice )
            
            # update stats
            self.update_stats( k, choice, result )
            score[ k ] = result
            choices[ k ] = choice
            regret[ k ] = self.bandits.p[ self.bandits.optimal ] - self.bandits.p[ choice ]

            # Update Q action-value using:
            # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
            # note that self.trials[ choice ] has been incremented to k+1 in update_stats above
            self.Q[ choice ] += (1./self.trials[ choice ]) * (float(result) - self.Q[ choice ])
            
        self.score = np.r_[ self.score, score ]
        self.choices = np.r_[ self.choices, choices ]
        self.regret = np.r_[ self.regret, regret ]
        return

    # Choose action using an epsilon-greedy agent
    def get_action( self, force_explore=False ):
        rand = np.random.random()  # [0.0,1.0)
        if (rand < self.epsilon) or force_explore:
            action_explore = np.random.randint( len(self.bandits) )  # explore random bandit
            return action_explore
        else:
            # randomly select from among the bandits with the current maximum Q value
            action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
            return action_greedy

class RandomStrategy( MABPolicy ):
    """
    RandomStrategy
    
    Implements a random strategy for
    the Multi-Armed Bandit problem.

    Sub-class of MABPolicy base class.
    
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
        regret = np.zeros( n )
        
        for k in range(n):
            # random action
            choice = np.random.randint( len(self.bandits) )
            
            # sample the chosen bandit
            result = self.bandits.pull( choice )
            
            # update stats
            self.update_stats( k, choice, result )
            score[ k ] = result
            choices[ k ] = choice
            regret[ k ] = self.bandits.p[ self.bandits.optimal ] - self.bandits.p[ choice ]
            
        self.score = np.r_[ self.score, score ]
        self.choices = np.r_[ self.choices, choices ]
        self.regret = np.r_[ self.regret, regret ]
        return 