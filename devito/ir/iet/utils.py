from collections import OrderedDict

import numpy as np

from devito.dimension import LoweredDimension
from devito.ir.iet import (Expression, Iteration, List, UnboundedIndex, FindSections,
                           MergeOuterIterations, Transformer)
from devito.symbolics import Eq
from devito.tools import as_tuple, filter_ordered, flatten
from devito.types import Scalar

__all__ = ['build_iet', 'filter_iterations', 'retrieve_iteration_tree',
           'is_foldable', 'compose_nodes', 'copy_arrays']


def build_iet(clusters, dtype):
    """
    Create an Iteartion/Expression tree (IET) given an iterable of :class:`Cluster`s.

    :param clusters: The iterable :class:`Cluster`s for which the IET is built.
    :param dtype: The data type of the scalar expressions.
    """
    processed = []
    schedule = OrderedDict()
    for cluster in clusters:
        if not cluster.ispace.empty:
            root = None
            intervals = cluster.ispace.intervals

            # Can I reuse any of the previously scheduled Iterations ?
            index = 0
            for i0, i1 in zip(intervals, list(schedule)):
                if i0 != i1 or i0.dim in clusters.atomics[cluster]:
                    break
                root = schedule[i1]
                index += 1
            needed = intervals[index:]

            # Set up substitution rules for derived dimensions
            subs = {}
            for i in intervals:
                for j, offs in cluster.ispace.sub_iterators.get(i.dim, []):
                    for n, o in enumerate(filter_ordered(offs)):
                        name = "%s%d" % (j.name, n)
                        subs[j+o] = LoweredDimension(name=name, stepping=j, offset=o)
                subs[i.dim] = Scalar(name=i.dim.name, dtype=np.int32)

            # Build Iterations, including any necessary unbounded index
            iters = []
            for i in needed:
                uindices = []
                for j, offs in cluster.ispace.sub_iterators.get(i.dim, []):
                    for n, o in enumerate(filter_ordered(offs)):
                        name = "%s%d" % (j.name, n)
                        vname = Scalar(name=name, dtype=np.int32)
                        value = (i.dim + o) % j.modulo
                        uindices.append(UnboundedIndex(vname, value, value))
                iters.append(Iteration([], i.dim, i.dim.limits, offsets=i.limits,
                                       uindices=uindices))

            # Build Expressions
            exprs = []
            for k, v in cluster.trace.items():
                dtype = np.int32 if cluster.trace.is_index(k) else dtype
                exprs.append(Expression(v.xreplace(subs), dtype))

            # Compose Iterations and Expressions
            body, tree = compose_nodes(iters + [exprs], retrieve=True)

            # Update the current scheduling
            scheduling = OrderedDict(zip(needed, tree))
            if root is None:
                processed.append(body)
                schedule = scheduling
            else:
                nodes = list(root.nodes) + [body]
                mapper = {root: root._rebuild(nodes, **root.args_frozen)}
                transformer = Transformer(mapper)
                processed = list(transformer.visit(processed))
                schedule = OrderedDict(list(schedule.items())[:index] +
                                       list(scheduling.items()))
                for k, v in list(schedule.items()):
                    schedule[k] = transformer.rebuilt.get(v, v)
        else:
            # No Iterations are needed
            processed.extend([Expression(e, dtype) for e in cluster.exprs])

    return List(body=processed)


def retrieve_iteration_tree(node, mode='normal'):
    """Return a list of all :class:`Iteration` sub-trees rooted in ``node``.
    For example, given the Iteration tree:

        .. code-block:: c

           Iteration i
             expr0
             Iteration j
               Iteraion k
                 expr1
             Iteration p
               expr2

    Return the list: ::

        [(Iteration i, Iteration j, Iteration k), (Iteration i, Iteration p)]

    :param node: The searched Iteration/Expression tree.
    :param mode: Accepted values are 'normal' (default) and 'superset', in which
                 case iteration trees that are subset of larger iteration trees
                 are dropped.
    """
    assert mode in ('normal', 'superset')

    trees = [i for i in FindSections().visit(node) if i]
    if mode == 'normal':
        return trees
    else:
        match = []
        for i in trees:
            if any(set(i).issubset(set(j)) for j in trees if i != j):
                continue
            match.append(i)
        return match


def filter_iterations(tree, key=lambda i: i, stop=lambda: False):
    """
    Given an iterable of :class:`Iteration` objects, return a new list
    containing all items such that ``key(o)`` is True.

    This function accepts an optional argument ``stop``. This may be either a
    lambda function, specifying a stop criterium, or any of the following
    special keywords: ::

        * 'any': Return as soon as ``key(o)`` is False and at least one
                 item has been collected.
        * 'asap': Return as soon as at least one item has been collected and
                  all items for which ``key(o)`` is False have been encountered.

    It is useful to specify a ``stop`` criterium when one is searching the
    first Iteration in an Iteration/Expression tree for which a given property
    does not hold.
    """
    assert callable(stop) or stop in ['any', 'asap']

    tree = list(tree)
    filtered = []
    off = []

    if stop == 'any':
        stop = lambda: len(filtered) > 0
    elif stop == 'asap':
        hits = [i for i in tree if not key(i)]
        stop = lambda: len(filtered) > 0 and len(off) == len(hits)

    for i in tree:
        if key(i):
            filtered.append(i)
        else:
            off.append(i)
        if stop():
            break

    return filtered


def is_foldable(nodes):
    """
    Return True if the iterable ``nodes`` consists of foldable :class:`Iteration`
    objects, False otherwise.
    """
    nodes = as_tuple(nodes)
    if len(nodes) <= 1 or any(not i.is_Iteration for i in nodes):
        return False
    main = nodes[0]
    return all(i.dim == main.dim and i.limits == main.limits and i.index == main.index
               and i.properties == main.properties for i in nodes)


def compose_nodes(nodes, retrieve=False):
    """
    Build an Iteration/Expression tree by nesting the nodes in ``nodes``.
    """
    l = list(nodes)
    tree = []

    if not isinstance(l[0], Iteration):
        # Nothing to compose
        body = flatten(l)
        body = List(body=body) if len(body) > 1 else body[0]
    else:
        body = l.pop(-1)
        while l:
            handle = l.pop(-1)
            body = handle._rebuild(body, **handle.args_frozen)
            tree.append(body)

    if retrieve is True:
        tree = list(reversed(tree))
        return body, tree
    else:
        return body


def copy_arrays(mapper, reverse=False):
    """
    Build an Iteration/Expression tree performing the copy ``k = v``, or
    ``v = k`` if reverse=True, for each (k, v) in mapper. (k, v) are expected
    to be of type :class:`IndexedData`. The loop bounds are inferred from
    the dimensions used in ``k``.
    """
    if not mapper:
        return ()

    # Build the Iteration tree for the copy
    iterations = []
    for k, v in mapper.items():
        handle = []
        indices = k.function.indices
        for i, j in zip(k.shape, indices):
            handle.append(Iteration([], dimension=j, limits=i))
        lhs, rhs = (v, k) if reverse else (k, v)
        handle.append(Expression(Eq(lhs[indices], rhs[indices]), dtype=k.function.dtype))
        iterations.append(compose_nodes(handle))

    # Maybe some Iterations are mergeable
    iterations = MergeOuterIterations().visit(iterations)

    return iterations
