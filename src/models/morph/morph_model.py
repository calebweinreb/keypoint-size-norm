from typing import NamedTuple, Protocol, Callable
from jaxtyping import Array, Float



class MorphParameters(Protocol):
    """
    Parameter set for a general morph model.
    """
    pass

class MorphHyperparams(Protocol):
    """
    Hyperparameters describing a morph model.

    :param N: Number of subjects.
    :param M: Pose space dimension.
    """
    N: int
    M: int


class MorphModel(NamedTuple):
    sample_parameters: Callable[..., MorphParameters]
    get_transform: Callable[
        [MorphParameters, MorphHyperparams],
        Float[Array, "N KD M"]]