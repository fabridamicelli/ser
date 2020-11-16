"""
SER class
"""
from typing import Optional
import warnings

import numpy as np
from numba import njit


class SER:
    '''
    Parameters
    ----------
    n_steps: int
        Number of steps of simulated activity each time function self.run() is called.
    prob_spont_act: float 0-1
        Probability of spontaneous activity (turning susceptible -> active).
    prob_recovery: float 0-1
        Probability of recovery (transition refractory to susceptible).
    threshold: float
        If network is binara: number of active (incoming) neighbors necessary to
        activate a node.
        If network is weighted: minimum weighted sum over inputs necessary to
        activate a node.
    n_transient: int
        Number of initial time steps to remove.
    prop_e: float 0-1
        Proportion of nodes in state E(xcited) at start of run.
    prop_s: float 0-1
        Proportion of nodes in state S(usceptible) at start of run.

    Notes
    -----
    The proportion of nodes in state R(efractory) is prop_r = 1 - (prop_e + prop_s).
    '''
    def __init__(
        self, *,
        n_steps: int,
        prob_spont_act: float = 0,
        prob_recovery: float = 1,
        threshold: float = 1,
        n_transient: int = 0,
        prop_e: float,
        prop_s: float,
    ) -> None:
        self.n_steps = n_steps
        self.prob_spont_act = prob_spont_act
        self.prob_recovery = prob_recovery
        self.threshold = threshold
        self.n_transient = n_transient
        self.prop_e = prop_e
        self.prop_s = prop_s

    @staticmethod
    def init_states(*, n_nodes: int, prop_e: float, prop_s: float) -> np.ndarray:
        '''
        Create states vector to start simulation of the time window.
        Node states are respresented with integers (S = 0, E = 1, R = -1)

        Parameters
        ----------
        n_nodes: int
            Number of nodes
        prop_e: float (0-1)
            Proportion of E(xcited) nodes
        prop_s: float (0-1)
            Proportion of S(usceptible) nodes

        Returns
        -------
        states: 1-D array of length n_nodes
        Activity is encoded as follows:
            S(usceptible): 0
            E(xcited): 1
            R(efractory): -1
        '''
        if prop_e is None or prop_s is None:
            raise ValueError("prop_e and prop_s must be defined.")
        if prop_e + prop_s > 1:
            raise ValueError(f'prop_e + prop_s must be <= 1, given = {prop_e + prop_s}')

        # Initialize vector (-1 values mean Refractory)
        states = np.zeros(n_nodes, dtype=np.int) - 1

        # Absolute number of nodes in each state
        n_nodes_e = int(round(n_nodes * prop_e, 2))  # round: avoid float repr.error
        n_nodes_s = int(round(n_nodes * prop_s, 2))

        # Set states
        states[: n_nodes_e] = 1                         # excited
        states[n_nodes_e: (n_nodes_e + n_nodes_s)] = 0  # susceptible

        # Randomize positions
        np.random.shuffle(states)

        if len(set(states)) != 3:
            warnings.warn('WARNING: not all the states are present')

        return states

    def run(self, *,
            adj_mat: np.ndarray,
            states: Optional[np.ndarray] = None
        ) -> np.ndarray:
        '''
        Parameters
        ----------
        adj_mat: 2-D numpy array
            Adjacency matrix
        states: 1-D numpy array of shape (n_nodes, ), optional
            Initial state to begin run.
            If None, random states are generated using the parameters prop_s, prop_e.
            If given, the parameters prop_s, prop_e are ignored.

        Returns
        -------
        act_mat: 2D np.ndarray of shape (n_nodes, n_steps)
            Activity matrix (node vs time).
            Activity is encoded as follows:
                S(usceptible): 0
                E(xcited): 1
                R(efractory): -1
        '''
        # Generate random initial state if no initial state is specified
        if states is None:
            states = SER.init_states(
                n_nodes=len(adj_mat), prop_e=self.prop_e, prop_s=self.prop_s
            )
        states = states.astype(adj_mat.dtype)  # cast for numba
        return _run(
            adj_mat=adj_mat,
            states=states,
            n_steps=self.n_steps,
            prob_spont_act=self.prob_spont_act,
            prob_recovery=self.prob_recovery,
            threshold=self.threshold,
            n_transient=self.n_transient,
        )


@njit(fastmath=True)
def _run(*,
    adj_mat: np.ndarray,
    states: np.ndarray,
    n_steps: int,
    prob_spont_act: float,
    prob_recovery: float,
    threshold: float,
    n_transient: int = 0,
):
    _dtype = adj_mat.dtype
    n_nodes = len(adj_mat)
    # Initialize activity matrix
    act_mat = np.zeros((n_nodes, n_steps), dtype=_dtype)
    act_mat[:, 0] = states

    # Evaluate all the stochastic transition probabilities in advance
    recovered = np.random.random(act_mat.shape) < prob_recovery
    spont_activated = np.random.random(act_mat.shape) < prob_spont_act

    for t in range(n_steps-1):
        # E -> R
        act_mat[act_mat[:, t] == 1, t+1] = -1
        # R --prob_recovery--> S
        refrac = act_mat[:, t] == -1
        act_mat[refrac, t+1] = act_mat[refrac, t] + recovered[refrac, t]
        # S -threshold, prob_spont_act-> E
        susce = act_mat[:, t] == 0
        # Sum of weights over active incoming neighbors
        weighted_input = adj_mat.T @ (act_mat[:, t] == 1).astype(_dtype)
        act_mat[susce, t+1] += np.logical_or(weighted_input[susce] >= threshold,
                                             spont_activated[susce, t])
    return act_mat[:, n_transient:]
