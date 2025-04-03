from typing import Union, Sequence, Protocol, Any
import numpy as np
import scipy.stats as stats
from numpy.typing import ArrayLike, NDArray
import warnings

# --- Type Definitions ---
Scalar = Union[int, float, np.number]
ECDFInput = Any
InputNumeric = Union[Scalar, ArrayLike]
OutputNumeric = Union[Scalar, NDArray]
Size = Union[int, Sequence[int], None]

# --- Protocol Definition ---
class RandomVariable(Protocol):
    """Defines a standard interface for univariate random variables."""
    def cdf(self, x: InputNumeric) -> NDArray: ...
    def ppf(self, q: InputNumeric) -> NDArray: ...
    def mean(self) -> OutputNumeric: ...
    def rvs(self, size: Size = None) -> NDArray: ...

# --- ERV Class ---
class ERV:
    """Adapts empirical distributions (like scipy.stats.ecdf result) to the RandomVariable protocol."""
    _probs: NDArray
    _quants: NDArray
    _prob_jumps: NDArray
    _mean: float

    def __init__(self, ecdf: ECDFInput) -> None:
        """Initializes the ERV adapter from an ECDF-like object."""
        try:
            self._probs = np.asarray(ecdf.cdf.probabilities)
            self._quants = np.asarray(ecdf.cdf.quantiles)
        except AttributeError:
            raise TypeError("Input 'ecdf' object must have '.cdf.probabilities' and '.cdf.quantiles' attributes.")

        if self._probs.shape != self._quants.shape:
            raise ValueError("Probabilities and quantiles must have the same shape.")
        if self._probs.ndim != 1:
             raise ValueError("Probabilities and quantiles must be 1-dimensional.")
        if len(self._probs) == 0:
            raise ValueError("ECDF must contain at least one point.")

        self._prob_jumps = np.diff(self._probs, prepend=0.0)

        self._prob_jumps[self._prob_jumps < 0] = 0.0
        norm_factor: float = np.sum(self._prob_jumps)
        if not np.isclose(norm_factor, 1.0):
             warnings.warn(f"Probability jumps sum to {norm_factor}, normalizing.", UserWarning)
             if norm_factor > np.finfo(float).eps:
                 self._prob_jumps /= norm_factor
             elif len(self._prob_jumps) > 0:
                 self._prob_jumps[:] = 1.0 / len(self._prob_jumps)
             else:
                 raise ValueError("Cannot normalize zero probability jumps for empty ECDF.")

        self._mean = np.sum(self._quants * self._prob_jumps)

    def cdf(self, x: InputNumeric) -> NDArray:
        """Calculates the Cumulative Distribution Function P(X <= x) using searchsorted."""
        x_arr: NDArray = np.asarray(x)
        indices: NDArray = np.searchsorted(self._quants, x_arr, side='right')
        indices = np.clip(indices, 0, len(self._probs))
        result: NDArray = np.zeros_like(x_arr, dtype=float)
        valid_indices_mask: NDArray = indices > 0
        result[valid_indices_mask] = self._probs[indices[valid_indices_mask] - 1]
        return result.item() if x_arr.ndim == 0 else result

    def ppf(self, q: InputNumeric) -> NDArray:
        """Calculates the Percent Point (Quantile) Function using searchsorted."""
        q_arr: NDArray = np.asarray(q)
        q_clipped: NDArray = np.clip(q_arr, 0.0, 1.0)
        indices: NDArray = np.searchsorted(self._probs, q_clipped, side='left')
        indices = np.clip(indices, 0, len(self._quants) - 1)
        result: NDArray = self._quants[indices]
        return result.item() if q_arr.ndim == 0 else result

    def mean(self) -> OutputNumeric:
        """Returns the pre-calculated mean of the empirical distribution."""
        return float(self._mean)

    def rvs(self, size: Size = None) -> NDArray:
        """Generates random variates using efficient numpy.random.choice."""
        samples: NDArray = np.random.choice(
            a=self._quants,
            size=size,
            p=self._prob_jumps,
            replace=True
        )
        if size is None:
             return samples.item() if samples.size == 1 else samples
        return samples
