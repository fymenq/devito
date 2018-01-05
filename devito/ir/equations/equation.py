from operator import attrgetter

import sympy

from devito.dimension import Dimension
from devito.ir.support import Box, Interval, Stencil
from devito.symbolics import indexify, retrieve_indexed
from devito.tools import filter_sorted, partial_order

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

    def __new__(cls, input_expr, subs=None):
        # Sanity check
        assert isinstance(input_expr, sympy.Eq)

        # Indexification
        expr = indexify(input_expr)

        # Apply caller-provided substitution
        if subs is not None:
            expr = expr.xreplace(subs)

        expr = super(Eq, cls).__new__(cls, expr.lhs, expr.rhs, evaluate=False)
        expr.is_Increment = getattr(input_expr, 'is_Increment', False)

        # Data space derivation
        # 1) Detect data accesses
        stencil = Stencil(expr)
        dims = {i for i in expr.free_symbols if isinstance(i, Dimension)}
        free_dims = tuple(filter_sorted(dims - set(stencil), key=attrgetter('name')))
        # 2) Normalized dimension ordering
        # TODO: move to a separate routine
        indexeds = retrieve_indexed(expr, mode='all')
        constraints = [tuple(i.indices) for i in indexeds] + [free_dims]
        for i, constraint in enumerate(list(constraints)):
            normalized = []
            for j in constraint:
                found = [d for d in j.free_symbols if isinstance(d, Dimension)]
                normalized.extend([d for d in found if d not in normalized])
            constraints[i] = normalized
        ordering = sorted(partial_order(constraints), key=lambda i: not i.is_Time)
        # 3) Do not track parent dimensions
        parents = [d.parent for d in ordering if d.is_Stepping]
        ordering = [i for i in ordering if i not in parents]
        # 4) Store the data space as a Box
        expr.dspace = Box([Interval(i, min(stencil.get(i)), max(stencil.get(i)))
                           for i in ordering])

        return expr

    @property
    def is_Scalar(self):
        return self.lhs.is_Symbol

    @property
    def is_Tensor(self):
        return self.lhs.is_Indexed
