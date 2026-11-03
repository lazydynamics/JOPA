"""Factor graphs with Gaussian message passing.

Build a graph from variables, deterministic nodes and `Normal` nodes, then
infer::

    from jopa import fg

    u = fg.Variable("u", 2)
    x = fg.Fn(dynamics, x0, u)                     # deterministic node
    graph = fg.Graph([fg.Normal(u, 0.0, 1e-2),     # action prior
                      fg.Box(u, -1.0, 1.0),        # actuator limits
                      fg.Normal(x, goal, 1.0)])    # goal
    post = graph.infer({"u": jnp.zeros(2)})        # post.mean["u"], post.cov["u"]

See `jopa.fg.graph` for the inference schedule.
"""
from .expr import Expr, Fn, Variable, evaluate, variables
from .factors import Box, Normal, Potential
from .graph import Graph, Marginals, eliminate, eliminate_chain

__all__ = [
    "Box", "Expr", "Fn", "Graph", "Marginals", "Normal", "Potential", "Variable",
    "eliminate", "eliminate_chain", "evaluate", "variables",
]
