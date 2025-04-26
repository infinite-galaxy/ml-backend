from abc import ABC, abstractmethod
from typing import Tuple
from ml_backend.domain.entities.rdd_features import RddFeatures


class RddRepositoryABC(ABC):
    """
    Abstract base personality repository
    """

    @abstractmethod
    def detect(self, features: RddFeatures) -> Tuple[list, str, dict]:
        """
        Detect road damage from the given features.

        Args:
          features (RddFeatures): Features object containing the image to be detected.

        Returns:
          int: Detected road damage.
        """
        raise NotImplementedError
