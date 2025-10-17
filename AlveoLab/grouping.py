import operator
from collections import defaultdict, deque

import numpy as np


def remove_duplicate_sets(arr_of_sets):
    sets = sorted(arr_of_sets, key=min)
    out = []
    i_0 = None
    for i in sets:
        if i != i_0:
            out.append(i)
            i_0 = i
    return np.array(out)


class Grouping(object):
    def __init__(self, mask):
        self.mask = mask | mask.T
        del mask
        self.n = len(self.mask)

        self.group_ids = np.full(self.n, -1)
        self.groups = []

        id = 0
        for i in range(self.n):
            if self.group_ids[i] != -1:
                continue
            to_do = deque([i])
            visited = np.zeros(self.n, bool)
            while to_do:
                row = to_do.pop()
                assert self.group_ids[row] == -1
                visited[row] = True
                args = np.nonzero(self.mask[row])[0]
                to_do.extend(args[~visited[args]])
            self.group_ids[visited] = id
            self.groups.append(set(np.nonzero(visited)[0]))
            id += 1

        assert (self.n == 0) or (self.group_ids.min() == 0)
        self.groups = np.array(self.groups)

    def _get_occurrences(self):
        return np.array([self._count_occurrence(i) for i in range(self.n)])

    def _is_ok(self):
        return (self._get_occurrences() == 1).all()

    def _count_occurrence(self, i):
        out = 0
        for s in remove_duplicate_sets(self.groups):
            if i in s:
                out += 1
        return out


def group_as_dict(to_group_itr, group_getter):

    if isinstance(group_getter, str):
        attr = group_getter
        if attr[-2:] == "()":
            group_getter = operator.methodcaller(attr[:-2])
        else:
            group_getter = operator.attrgetter(attr)

    out = defaultdict(list)

    for i in to_group_itr:
        out[group_getter(i)].append(i)

    return out