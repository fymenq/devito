import abc

import numpy as np

from devito.tools import as_tuple


class AbstractInterval(object):

    """
    A representation of a closed interval on the set of integer numbers Z.
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

    @abc.abstractproperty
    def is_unbounded(self):
        return

    @abc.abstractmethod
    def rebuild(self):
        return

    @abc.abstractmethod
    def intersection(self, other):
        return

    @abc.abstractmethod
    def union(self, other):
        return

    @abc.abstractmethod
    def overlap(self, other):
        return

    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def __hash__(self):
        return hash(self.dim.name)


class NullInterval(AbstractInterval):

    is_Null = True

    def __repr__(self):
        return "NullInterval(%s)" % self.dim

    @property
    def is_unbounded(self):
        return False

    def rebuild(self):
        return NullInterval(self.dim)

    def intersection(self, other):
        return self.rebuild()

    def union(self, other):
        if self.dim == other.dim:
            return other.rebuild()
        else:
            return Box([self.rebuild(), other.rebuild()])

    def union_if_overlap(self, other):
        return self.rebuild()

    def overlap(self, other):
        return False


class Interval(AbstractInterval):

    is_Defined = True

    def __init__(self, dim, lower=None, upper=None):
        super(Interval, self).__init__(dim)
        self.lower = lower or -np.inf
        self.upper = upper or np.inf
        assert self.lower <= self.upper

    def __repr__(self):
        return "Interval(%s <= %s <= %s)" % (self.lower, self.dim, self.upper)

    @property
    def is_unbounded(self):
        return self.lower == -np.inf or self.upper == np.inf

    def rebuild(self):
        return Interval(self.dim, self.lower, self.upper)

    def intersection(self, other):
        if self.overlap(other):
            return Interval(self.dim, max(self.lower, other.lower),
                            min(self.upper, other.upper))
        else:
            return NullInterval(self.dim)

    def union(self, other):
        if self.overlap(other):
            return Interval(self.dim, min(self.lower, other.lower),
                            max(self.upper, other.upper))
        elif other.is_Null and self.dim == other.dim:
            return self.rebuild()
        else:
            return Box([self.rebuild(), other.rebuild()])

    def union_if_overlap(self, other):
        if self.overlap(other):
            return Interval(self.dim, min(self.lower, other.lower),
                            max(self.upper, other.upper))
        else:
            return self.rebuild()

    def overlap(self, other):
        if self.dim != other.dim:
            return False
        try:
            return (self.lower <= other.lower and other.lower <= self.upper) or\
                (self.lower >= other.lower and self.lower <= other.upper)
        except AttributeError:
            return False

    def __eq__(self, other):
        return super(Interval, self).__eq__(other) and\
            self.lower == other.lower and self.upper == other.upper

    def __hash__(self):
        return hash((self.dim.name, self.lower, self.upper))


class Box(object):

    """
    A bag of disjoint Intervals.
    """

    def __init__(self, intervals):
        def key(interval):
            lower = -np.inf if interval.is_Null else interval.lower
            upper = np.inf if interval.is_Null else interval.upper
            return (interval.dim.name, lower, upper)
        self.intervals = sorted(set(as_tuple(intervals)), key=key)

    def __repr__(self):
        return "Box[%s]" % ', '.join([repr(i) for i in self.intervals])

    def __eq__(self, other):
        return len(self.intervals) == len(other.intervals) and\
            all(i == j for i, j in zip(self.intervals, other.intervals))

    @classmethod
    def union(cls, *boxes):
        mapper = {}
        for i in boxes:
            for interval in i.intervals:
                mapper.setdefault(interval.dim, []).append(interval)
        return Box([Interval.op(v, 'union') for v in mapper.values()])

    def union_if_bounded_overlap(self, *boxes):
        union_box = Box.union(*boxes)
        intervals = []
        for i in self.intervals:
            if i.is_unbounded:
                intervals.append(i.rebuild())
                continue
            base = i
            for j in union_box.intervals:
                if j.is_unbounded:
                    continue
                base = base.union_if_overlap(j)
            intervals.append(base)
        return Box(intervals)
