import sympy

from devito.ir.support import Interval, Box, Stencil
from devito.symbolics import indexify

__all__ = ['Eq']


class Eq(sympy.Eq):

    """
    A new SymPy equation with an associated domain.

    The domain is an object of type :class:`Box`. If not provided, the domain is
    inferred from the :class:`Stencil` of ``expr``.

    All :class:`Function` objects within ``expr`` get indexified and thus turned
    into objects of type :class:`types.Indexed`.

    .. note::

        With the domain-allocation switch, the domain will be inferred from
        the :class:`Function`s in ``expr`` (ie, from their halo region), rather
        than from its :class:`Stencil`.
    """

    def __new__(cls, expr, domain=None, subs=None, **kwargs):
        # Sanity check
        assert isinstance(expr, sympy.Eq)

        # Indexification
        expr = indexify(expr)

        # Apply caller-provided substitution
        if subs is not None:
            expr = expr.xreplace(subs)

        expr = super(Eq, cls).__new__(cls, expr.lhs, expr.rhs, **kwargs)

        # Domain derivation
        if domain is not None:
            expr.domain = domain
        else:
            stencil = Stencil(expr)
            stencil = stencil.replace({d.parent: d for d in stencil.dimensions
                                       if d.is_Stepping})
            intervals = []
            for k, v in stencil.items():
                lower = min(v) if min(v) < 0 else None
                upper = max(v) if max(v) > 0 else None
                intervals.append(Interval(k, lower, upper))
            expr.domain = Box(intervals)

        return expr
