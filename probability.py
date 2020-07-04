""" Exercises about probability """
import enum, random


def main():
    kids()


def kids():
    both_girls = 0
    older_girl = 0
    either_girl = 0

    random.seed(0)

    for _ in range(10000):
        younger = random_kid()
        older = random_kid()

        if older == Kid.GIRL:
            older_girl += 1
        if older == Kid.GIRL and younger == Kid.GIRL:
            both_girls += 1
        if older == Kid.GIRL or younger == Kid.GIRL:
            either_girl += 1

    print(f'P(both | older): {both_girls/older_girl}')
    print(f'P(both | either): {both_girls / either_girl}')


class Kid(enum.Enum):
    BOY = 0
    GIRL = 1


def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])


if __name__ == '__main__':
    main()