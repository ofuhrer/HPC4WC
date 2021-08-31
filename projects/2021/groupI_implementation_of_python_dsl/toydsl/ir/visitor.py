from typing import Any

from toydsl.ir.ir import Node


class IRNodeVisitor:
    """
    Basic visitor for all the nodes of the IR

    A NodeVisitor instance walks a node tree and calls a visitor
    function for every item found. This class is meant to be subclassed,
    with the subclass adding visitor methods.

    """

    def visit(self, node: Any, **kwargs: Any) -> Any:
        visitor = self.generic_visit

        if isinstance(node, list):
            if not node:
                # note(pascal): skip empty lists. This might not always be
                # what we want, but for now I don't see a better solution.
                pass
            else:
                element = node[0]
                for node_class in element.__class__.__mro__:
                    method_name = "visit_list_of_" + node_class.__name__
                    if hasattr(self, method_name):
                        visitor = getattr(self, method_name)
                        break
        elif isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
