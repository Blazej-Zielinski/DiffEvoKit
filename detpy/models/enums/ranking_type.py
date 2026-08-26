from enum import Enum


class RankingType(Enum):
    """
    Enumeration of ranking types used in the FDDE algorithm.

    Attributes:
        FITNESS_ONLY: Ranking based solely on fitness values
        DIVERSITY_ONLY: Ranking based solely on population diversity
        FULL: Ranking based on both fitness and diversity
    """
    FITNESS_ONLY = 1
    DIVERSITY_ONLY = 2
    FULL = 3
