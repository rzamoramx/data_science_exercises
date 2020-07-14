""" Base functions for conditional probability """
import random
from probability.kid_enum import Kid


def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])
