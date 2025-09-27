import pytest
import numpy as np
from ser import SER


def test_init():
    ser = SER(
        n_steps=10,
        prob_spont_act=0.1,
        prob_recovery=0.5,
        threshold=0.8,
        n_transient=2,
        prop_e=0.3,
        prop_s=0.4,
    )

    assert ser.n_steps == 10
    assert ser.prob_spont_act == 0.1
    assert ser.prob_recovery == 0.5
    assert ser.threshold == 0.8
    assert ser.n_transient == 2
    assert ser.prop_e == 0.3
    assert ser.prop_s == 0.4


def test_init_states():
    n_nodes = 100
    prop_e = 0.3
    prop_s = 0.4

    states = SER.init_states(n_nodes=n_nodes, prop_e=prop_e, prop_s=prop_s)

    assert len(states) == n_nodes
    assert np.sum(states == 1) == int(round(n_nodes * prop_e, 2))
    assert np.sum(states == 0) == int(round(n_nodes * prop_s, 2))
    assert np.sum(states == -1) == n_nodes - int(round(n_nodes * prop_e, 2)) - int(
        round(n_nodes * prop_s, 2)
    )


def test_init_states_errors():
    with pytest.raises(ValueError):
        SER.init_states(n_nodes=100, prop_e=None, prop_s=0.4)
    with pytest.raises(ValueError):
        SER.init_states(n_nodes=100, prop_e=0.3, prop_s=None)
    with pytest.raises(ValueError):
        SER.init_states(n_nodes=100, prop_e=0.7, prop_s=0.4)


def test_run():
    """
    Test the run method.
    """
    n_nodes = 10
    n_steps = 5
    adj_mat = np.random.rand(n_nodes, n_nodes)
    ser = SER(n_steps=n_steps, prop_e=0.2, prop_s=0.3)

    act_mat = ser.run(adj_mat=adj_mat)

    assert act_mat.shape == (n_nodes, n_steps)
    assert act_mat.dtype == adj_mat.dtype


def test_run_with_initial_states():
    n_nodes = 10
    n_steps = 5
    adj_mat = np.random.rand(n_nodes, n_nodes)
    initial_states = np.random.choice([-1, 0, 1], size=n_nodes)
    ser = SER(n_steps=n_steps, prop_e=0.2, prop_s=0.3)

    act_mat = ser.run(adj_mat=adj_mat, states=initial_states)

    assert act_mat.shape == (n_nodes, n_steps)
    assert act_mat.dtype == adj_mat.dtype
    assert np.array_equal(act_mat[:, 0], initial_states)


def test_run_with_n_transient():
    n_nodes = 10
    n_steps = 5
    n_transient = 2
    adj_mat = np.random.rand(n_nodes, n_nodes)
    ser = SER(n_steps=n_steps, prop_e=0.2, prop_s=0.3, n_transient=n_transient)

    act_mat = ser.run(adj_mat=adj_mat)

    assert act_mat.shape == (n_nodes, n_steps - n_transient)
