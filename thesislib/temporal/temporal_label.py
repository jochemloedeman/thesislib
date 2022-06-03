from enum import Enum
import torch


class TemporalLabel(Enum):
    TEMPORAL = True
    STATIC = False


if __name__ == '__main__':
    index = {0: TemporalLabel.STATIC,
             1: TemporalLabel.TEMPORAL,
             3: TemporalLabel.STATIC}

    a = [e == TemporalLabel.TEMPORAL for e in list(index.values())]
    print(a)
