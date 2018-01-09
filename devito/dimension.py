from cached_property import cached_property

from devito.arguments import DimensionArgProvider
from devito.types import Scalar, Symbol

__all__ = ['Dimension', 'SpaceDimension', 'TimeDimension', 'SteppingDimension']


class Dimension(Symbol, DimensionArgProvider):

    is_Space = False
    is_Time = False

    is_Derived = False
    is_Stepping = False
    is_Lowered = False

    """Index object that represents a problem dimension and thus
    defines a potential iteration space.

    :param name: Name of the dimension symbol.
    :param reverse: Optional, Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(Dimension, self).__init__(*args, **kwargs)
            self._reverse = kwargs.get('reverse', False)
            self._spacing = kwargs.get('spacing', Scalar(name='h_%s' % self.name))

    def __str__(self):
        return self.name

    @cached_property
    def symbolic_size(self):
        """The symbolic size of this dimension."""
        return Symbol(name=self.size_name)

    @cached_property
    def symbolic_start(self):
        return Symbol(name=self.start_name)

    @cached_property
    def symbolic_end(self):
        return Symbol(name=self.end_name)

    @property
    def symbolic_extent(self):
        """Return the extent of the loop over this dimension.
        Would be the same as size if using default values """
        _, start, end = self.rtargs
        return (self.symbolic_end - self.symbolic_start)

    @property
    def limits(self):
        _, start, end = self.rtargs
        return (self.symbolic_start, self.symbolic_end, 1)

    @property
    def size_name(self):
        return "%s_size" % self.name

    @property
    def start_name(self):
        return "%s_s" % self.name

    @property
    def end_name(self):
        return "%s_e" % self.name

    @property
    def reverse(self):
        return self._reverse

    @property
    def spacing(self):
        return self._spacing

    @reverse.setter
    def reverse(self, val):
        # TODO: this is an outrageous hack. TimeFunctions are updating this value
        # at construction time. This is a symptom we need local and global dimensions
        self._reverse = val

    def _hashable_content(self):
        return super(Dimension, self)._hashable_content() +\
            (self.reverse, self.spacing)


class SpaceDimension(Dimension):

    is_Space = True

    """
    Dimension symbol to represent a space dimension that defines the
    extent of physical grid. :class:`SpaceDimensions` create dedicated
    shortcut notations for spatial derivatives on :class:`Function`
    symbols.

    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class TimeDimension(Dimension):

    is_Time = True

    """
    Dimension symbol to represent a dimension that defines the extent
    of time. As time might be used in different contexts, all derived
    time dimensions should inherit from :class:`TimeDimension`.

    :param name: Name of the dimension symbol.
    :param reverse: Traverse dimension in reverse order (default False)
    :param spacing: Optional, symbol for the spacing along this dimension.
    """


class SteppingDimension(Dimension):

    is_Derived = True
    is_Stepping = True

    """
    Dimension symbol that defines the stepping direction of an
    :class:`Operator` and implies modulo buffered iteration. This is most
    commonly use to represent a timestepping dimension.

    :param parent: Parent dimension over which to loop in modulo fashion.
    """

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(SteppingDimension, self).__init__(*args, **kwargs)
            self._modulo = kwargs.get('modulo', 2)
            self._parent = kwargs['parent']

            # Inherit time/space identifiers
            assert isinstance(self.parent, Dimension)
            self.is_Time = self.parent.is_Time
            self.is_Space = self.parent.is_Space

    @property
    def parent(self):
        return self._parent

    @property
    def modulo(self):
        return self._modulo

    @modulo.setter
    def modulo(self, val):
        # TODO: this is an outrageous hack. TimeFunctions are updating this value
        # at construction time. This is a symptom we need local and global dimensions
        self._modulo = val

    @property
    def reverse(self):
        return self.parent.reverse

    @property
    def spacing(self):
        return self.parent.spacing

    def _hashable_content(self):
        return (self.parent._hashable_content(), self.modulo)


class LoweredDimension(Dimension):

    is_Lowered = True

    """
    Dimension symbol representing a modulo iteration created when
    resolving a :class:`SteppingDimension`.

    :param stepping: :class:`SteppingDimension` from which this
                     :class:`Dimension` originated.
    :param offset: Offset value used in the modulo iteration.
    """

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(LoweredDimension, self).__init__(*args, **kwargs)
            self._origin = kwargs['origin']

    @property
    def origin(self):
        return self._origin

    def _hashable_content(self):
        return Symbol._hashable_content(self) + (self.origin)
