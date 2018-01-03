import sympy

from devito.ir.support import Interval, Box, Stencil
from devito.symbolics import indexify

__all__ = ['Eq']


class Eq(sympy.Eq):

    """
    A new SymPy equation with an associated data space.

    All :class:`Function` objects within ``expr`` get indexified and thus turned
    into objects of type :class:`types.Indexed`.

    A data space is an object of type :class:`Box`. It represents the data points
    accessed by the equation along each iteration :class:`Dimension`. The iteration
    :class:`Dimension`s are extracted from the :class:`Indexed`s of the equation.
    """

    def __new__(cls, expr, dspace=None, subs=None, **kwargs):
        # Sanity check
        assert isinstance(expr, sympy.Eq)

        # Indexification
        expr = indexify(expr)

        # Apply caller-provided substitution
        if subs is not None:
            expr = expr.xreplace(subs)

        expr = super(Eq, cls).__new__(cls, expr.lhs, expr.rhs, **kwargs)

        # Data space derivation
        if dspace is not None:
            expr.dspace = dspace
        else:
            stencil = Stencil(expr)
            stencil = stencil.replace({d.parent: d for d in stencil.dimensions
                                       if d.is_Stepping})
            intervals = [Interval(k, min(v), max(v)) for k, v in stencil.items()]
            expr.dspace = Box(intervals)

        return expr
