"""Effective sample size and systematic resampling.

Systematic resampling (stochastic universal sampling (SUS)) draws a single uniform and spaces the remaining N-1
positions deterministically. That gives lower variance than drawing N
independent uniforms, and it guarantees a particle holding weight w is
selected either floor(N*w) or ceil(N*w) times.
"""
import numpy as np


def effective_sample_size(weights) -> float:
    """N_eff = 1 / sum(w^2), for NORMALIZED weights.

    Args:
        weights: (N,) array of normalized particle weights.
    Returns:
        float: the effective sample size, between 1 and N.

    """
    N_eff = 1/np.sum(np.power(weights,2))
    return N_eff


def systematic_resample(weights, rng) -> np.ndarray:
    """Systematic resampling of particles based on their weights.

    Args:
        weights: (N,) array of normalized particle weights.
        rng: a random number generator with a `random()` method.
    Returns:
        np.ndarray: (N,) array of ancestor indices.
    """
    F = sum(weights)
    N = len(weights)
    P = F/N



    start = rng.random() * P
    stop = start + P*(N-1)
    pointers = np.linspace(start,stop,N)

    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # protect against floating-point roundoff

    return np.searchsorted(cumulative, pointers, side='right')


if __name__ == '__main__':
    import random as rng
    weights = [0.125,0.375,0.25,0.25]
    weights = np.array(weights)
    keep = systematic_resample(weights,rng)
    print(keep)
