import abc

import numpy as np

from devito.tools import as_tuple

__all__ = ['NullInterval', 'Interval', 'Box', 'Schedule']


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
    def _rebuild(self):
        return

    def intersection(self, o):
        return self._rebuild()

    @abc.abstractmethod
    def union(self, o):
        return self._rebuild()

    def subtract(self, o):
        return self._rebuild()

    def negate(self):
        return self._rebuild()

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

    def _rebuild(self):
        return NullInterval(self.dim)

    def union(self, o):
        if self.dim == o.dim:
            return o._rebuild()
        else:
            return Box([self._rebuild(), o._rebuild()])

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

    def _rebuild(self):
        return Interval(self.dim, self.lower, self.upper)

    def intersection(self, o):
        if self.overlap(o):
            return Interval(self.dim, max(self.lower, o.lower), min(self.upper, o.upper))
        else:
            return NullInterval(self.dim)

    def union(self, o):
        if self.overlap(o):
            return Interval(self.dim, min(self.lower, o.lower), max(self.upper, o.upper))
        elif o.is_Null and self.dim == o.dim:
            return self._rebuild()
        else:
            return Box([self._rebuild(), o._rebuild()])

    def subtract(self, o):
        if self.dim != o.dim or o.is_Null:
            return self._rebuild()
        else:
            return Interval(self.dim, self.lower - o.lower, self.upper - o.upper)

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
        self.intervals = as_tuple(set(intervals))

    def __repr_key__(self, interval):
        lower = -np.inf if interval.is_Null else interval.lower
        upper = np.inf if interval.is_Null else interval.upper
        return (interval.dim.name, lower, upper)

    def __repr__(self):
        intervals = sorted(self.intervals, key=self.__repr_key__)
        return "%s[%s]" % (self.__class__.__name__,
                           ', '.join([repr(i) for i in intervals]))

    def __eq__(self, o):
        return set(self.intervals) == set(o.intervals)

    def intersection(self, *boxes):
        mapper = {i.dim: [i] for i in self.intervals}
        for i in boxes:
            for interval in i.intervals:
                mapper.get(interval.dim, []).append(interval)
        return Box([Interval.op(v, 'intersection') for v in mapper.values()])

    def subtract(self, o):
        mapper = {i.dim: i for i in self.intervals}
        intervals = [mapper[i.dim].subtract(i) for i in o.intervals if i.dim in mapper]
        return Box(intervals)

    def negate(self):
        return Box([i.negate() for i in self.intervals])


class Schedule(Box):

    """
    A special :class:`Box` with ordered intervals. The ordering is established
    through a callable ``key`` which accepts as input a :class:`Interval` and
    returns a value.
    """

    def __init__(self, maybe_box, key):
        assert callable(key)
        intervals = maybe_box.intervals if isinstance(maybe_box, Box) else maybe_box
        self.intervals = as_tuple(sorted(intervals, key=key))
        self.key = key

    def __repr_key__(self, interval):
        return (self.key(interval),) + super(Schedule, self).__repr_key__(interval)

    def intersection(self, *boxes):
        return Schedule(super(Schedule, self).intersection(*boxes), key=self.key)

    def subtract(self, o):
        return Schedule(super(Schedule, self).subtract(o), key=self.key)

    def negate(self):
        return Schedule(super(Schedule, self).negate(), key=self.key)
