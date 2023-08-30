from dataclasses import dataclass
from enum import Enum

from typing import Dict, Union, Any, List

import numpy.typing as npt

Scalar = Union[bool, bytes, float, int, str]
Config = Dict[str, Scalar]
Metrics = Dict[str, Scalar]

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

class Code(Enum):
    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    TRAIN_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    code: Code
    message: str


@dataclass
class Weights:
    values: List[Union[bytes, float]]
    dtype: str


@dataclass
class GetWeightsInstructions:
    config: Config


@dataclass
class GetWeightsResult:
    status: Status
    weights: Weights

@dataclass
class SetWeightsInstructions:
    weights: Weights

@dataclass
class SetWeightsResult:
    status: Status


@dataclass
class TrainInstructions:
    weights: Weights
    config: Config


@dataclass
class TrainResult:
    status: Status
    weights: Weights
    num_examples: int
    metrics: Metrics


@dataclass
class EvaluateInstructions:
    weights: Weights
    config: Config


@dataclass
class EvaluateResult:
    status: Status
    num_examples: int
    metrics: Metrics

@dataclass
class ClientInitConfig:
    id: str
    config: Config
    init_weights: Union[Weights, None] = None
