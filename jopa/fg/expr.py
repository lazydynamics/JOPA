"""Variables and deterministic (delta) nodes.

A `Variable` is a named vector-valued random variable. `Fn(f, *args)` is a
deterministic node ``y = f(args)`` whose arguments are variables, other
deterministic nodes or constant arrays; arithmetic operators build the common
ones, so ``g * Fn(jnp.tanh, v)`` is two delta nodes. Expressions are values only: a
factor (`jopa.fg.factors`) makes them part of a graph, and inference
linearizes each delta node at the current estimate.
"""
from __future__ import annotations

import jax.numpy as jnp


class Expr:
    """Base of every graph expression; supports ``+ - *`` and negation."""

    def __add__(self, other):
        return Fn(jnp.add, self, other)

    def __radd__(self, other):
        return Fn(jnp.add, other, self)

    def __sub__(self, other):
        return Fn(jnp.subtract, self, other)

    def __rsub__(self, other):
        return Fn(jnp.subtract, other, self)

    def __mul__(self, other):
        return Fn(jnp.multiply, self, other)

    def __rmul__(self, other):
        return Fn(jnp.multiply, other, self)

    def __neg__(self):
        return Fn(jnp.negative, self)


class Variable(Expr):
    """A latent vector of size ``dim``, identified by ``name``."""

    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = int(dim)

    def __repr__(self):
        return f"Variable({self.name!r}, {self.dim})"


class Fn(Expr):
    """Deterministic node ``f(*args)``; ``f`` must be a JAX-traceable function."""

    def __init__(self, f, *args):
        self.f = f
        self.args = args

    def __repr__(self):
        name = getattr(self.f, "__name__", type(self.f).__name__)
        return f"Fn({name}, {len(self.args)} args)"


def variables(*exprs) -> tuple[Variable, ...]:
    """Distinct variables reachable from ``exprs``, in first-seen order."""
    seen: dict[str, Variable] = {}

    def visit(e):
        if isinstance(e, Variable):
            if e.name in seen and seen[e.name].dim != e.dim:
                raise ValueError(f"variable {e.name!r} used with dims "
                                 f"{seen[e.name].dim} and {e.dim}")
            seen.setdefault(e.name, e)
        elif isinstance(e, Fn):
            for a in e.args:
                visit(a)

    for e in exprs:
        visit(e)
    return tuple(seen.values())


def evaluate(e, values: dict, cache: dict | None = None):
    """Value of ``e`` given ``values[name]`` for its variables. ``cache``
    (keyed by node identity) shares sub-expressions within one evaluation."""
    if isinstance(e, Variable):
        return values[e.name]
    if not isinstance(e, Fn):
        return e
    cache = {} if cache is None else cache
    key = id(e)
    if key not in cache:
        cache[key] = e.f(*(evaluate(a, values, cache) for a in e.args))
    return cache[key]
