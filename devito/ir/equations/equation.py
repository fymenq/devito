import sympy

from devito.ir.support import Box, Interval, Stencil
from devito.symbolics import dimension_sort, indexify

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

        # Well-defined dimension ordering
        ordering = dimension_sort(expr, key=lambda i: not i.is_Time)
        parents = [d.parent for d in ordering if d.is_Stepping]
        ordering = [i for i in ordering if i not in parents]

        # Data space derivation
        stencil = Stencil(expr)
        expr.dspace = Box([Interval(i, min(stencil.get(i)), max(stencil.get(i)))
                           for i in ordering])

        return expr

    @property
    def is_Scalar(self):
        return self.lhs.is_Symbol

    @property
    def is_Tensor(self):
        return self.lhs.is_Indexed
