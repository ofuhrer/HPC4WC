from typing import Any

from dsl.ir.ir import Node

class IRNodeVisitor:
    """
    Basic visitor for all the nodes of the IR

    A NodeVisitor instance walks a node tree and calls a visitor
    function for every item found. This class is meant to be subclassed,
    with the subclass adding visitor methods.

    """

    def visit(self, node: Node, **kwargs: Any) -> Any:
        return self._visit(node, **kwargs)

    def _visit(self, node: Node, **kwargs: Any) -> Any:
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(node, **kwargs)


    def generic_visit(self, node: Node, **kwargs: Any) -> Any:
        raise NotImplementedError