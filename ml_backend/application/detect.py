

from typing import Tuple
from ml_backend.domain.entities.rdd_features import RddFeatures
from ml_backend.domain.repositories.rdd_repository_abc import RddRepositoryABC


class Detect:
    rdd_repository: RddRepositoryABC

    def __init__(self, rdd_repository: RddRepositoryABC):
        self.rdd_repository = rdd_repository

    def __call__(self, model_id: str, features: RddFeatures) -> Tuple[list, str, dict]:
        match model_id:
            case 'rdd':
                return self.rdd_repository.detect(features)
            case _:
                raise ValueError(f'Invalid model ID: {model_id}')
