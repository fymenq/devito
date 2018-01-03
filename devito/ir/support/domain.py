import abc

import numpy as np

from devito.tools import as_tuple


class AbstractInterval(object):

    """
    A representation of a closed interval on Z.
    """

    __metaclass__ = abc.ABCMeta

    is_Null = False
    is_Defined = False

    def __init__(self, dim):
        self.dim = dim

    @classmethod
    def op(cls, intervals, key):
        intervals = as_tuple(intervals)
        partial = intervals[0]
        for i in intervals[1:]:
            partial = getattr(partial, key)(i)
        return partial

    @abc.abstractmethod
    def rebuild(self):
        return

    @abc.abstractmethod
    def intersection(self, o):
        return

    @abc.abstractmethod
    def union(self, o):
        return

    @abc.abstractmethod
    def negate(self):
        return

    @abc.abstractmethod
    def overlap(self, o):
        return

    def __eq__(self, o):
        return type(self) == type(o) and self.dim == o.dim

    def __hash__(self):
        return hash(self.dim.name)


class NullInterval(AbstractInterval):

    is_Null = True

    def __repr__(self):
        return "%s[Null]" % self.dim

    def rebuild(self):
        return NullInterval(self.dim)

    def intersection(self, o):
        return self.rebuild()

    def negate(self):
        return self.rebuild()

    def union(self, o):
        if self.dim == o.dim:
            return o.rebuild()
        else:
            return Box([self.rebuild(), o.rebuild()])

    def overlap(self, o):
        return False


class Interval(AbstractInterval):

    """
    Interval(dim, lower, upper)

    Create an :class:`Interval` of extent: ::

        dim.extent + abs(upper - lower)
    """

    is_Defined = True

    def __init__(self, dim, lower, upper):
        assert isinstance(lower, int)
        assert isinstance(upper, int)
        super(Interval, self).__init__(dim)
        self.lower = lower
        self.upper = upper
        self.min_extent = abs(upper - lower)
        self.extent = dim.symbolic_size + self.min_extent

    def __repr__(self):
        return "%s[%s, %s]" % (self.dim, self.lower, self.upper)

    def rebuild(self):
        return Interval(self.dim, self.lower, self.upper)

    def intersection(self, o):
        if self.overlap(o):
            return Interval(self.dim, max(self.lower, o.lower),
                            min(self.upper, o.upper))
        else:
            return NullInterval(self.dim)

    def union(self, o):
        if self.overlap(o):
            return Interval(self.dim, min(self.lower, o.lower),
                            max(self.upper, o.upper))
        elif o.is_Null and self.dim == o.dim:
            return self.rebuild()
        else:
            return Box([self.rebuild(), o.rebuild()])

    def negate(self):
        return Interval(self.dim, -self.lower, -self.upper)

    def overlap(self, o):
        if self.dim != o.dim:
            return False
        try:
            # In the "worst case scenario" the dimension extent is 0
            # so we can just neglect it
            min_extent = max(self.min_extent, o.min_extent)
            return (self.lower <= o.lower and o.lower <= self.lower + min_extent) or\
                (self.lower >= o.lower and self.lower <= o.lower + min_extent)
        except AttributeError:
            return False

    def __eq__(self, o):
        return super(Interval, self).__eq__(o) and\
            self.lower == o.lower and self.upper == o.upper

    def __hash__(self):
        return hash((self.dim.name, self.lower, self.upper))


class Box(object):

    """
    A bag of :class:`Interval`s.
    """

    def __init__(self, intervals):
        def key(interval):
            lower = -np.inf if interval.is_Null else interval.lower
            upper = np.inf if interval.is_Null else interval.upper
            return (interval.dim.name, lower, upper)
        self.intervals = sorted(set(as_tuple(intervals)), key=key)

    def __repr__(self):
        return "Box[%s]" % ', '.join([repr(i) for i in self.intervals])

    def __eq__(self, o):
        return len(self.intervals) == len(o.intervals) and\
            all(i == j for i, j in zip(self.intervals, o.intervals))

    def intersection(self, *boxes):
        mapper = {i.dim: [i] for i in self.intervals}
        for i in boxes:
            for interval in i.intervals:
                mapper.get(interval.dim, []).append(interval)
        return Box([Interval.op(v, 'intersection') for v in mapper.values()])

    def negate(self):
        return Box([i.negate() for i in self.intervals])
