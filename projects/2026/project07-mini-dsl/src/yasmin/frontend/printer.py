from typing import Any

from yasmin.frontend import Operator
from yasmin.lowering.stencil_to_loop import lower


def print_loop_ir(node: Operator, indent: int = 3) -> None:
    if not isinstance(node, Operator):
        raise TypeError(f"Expected type Operator, got {type(node).__name__}")

    indent_unit = " " * indent

    if hasattr(node, "_as_ir"):
        print(
            _rec_loop_ir_to_string(
                lower(node._as_ir()),
                0,
                indent_unit,
            )
        )
    else:
        raise TypeError("Invalid input type. Function _as_ir is missing")


def _rec_loop_ir_to_string(
    node: Any,
    level: int = 0,
    indent_unit: str = " ",
) -> str:
    cls_name = type(node).__name__
    curr_ind = indent_unit * level
    next_ind = indent_unit * (level + 1)

    if cls_name == "Dimension":
        return f'Dimension(name="{node.name}")'

    if cls_name == "Index":
        return f'Index(name="{node.name}")'

    if cls_name == "Scalar":
        dtype_str = getattr(node.dtype, "name", str(node.dtype))
        return f'Scalar(name="{node.name}", dtype={dtype_str})'

    if cls_name == "Literal":
        return f"Literal({node.value})"

    if cls_name == "Field":
        dims_code = ", ".join(
            _rec_loop_ir_to_string(
                dimension,
                level + 1,
                indent_unit,
            )
            for dimension in node.dims
        )
        comma = "," if len(node.dims) == 1 else ""
        dtype_str = getattr(node.dtype, "name", str(node.dtype))

        return (
            "Field(\n"
            f"{next_ind}name='{node.name}',\n"
            f"{next_ind}dims=({dims_code}{comma}),\n"
            f"{next_ind}dtype={dtype_str},\n"
            f"{curr_ind})"
        )

    if cls_name == "Extent":
        field_code = _rec_loop_ir_to_string(
            node.field,
            level + 1,
            indent_unit,
        )

        return (
            "Extent(\n"
            f"{next_ind}field={field_code},\n"
            f"{next_ind}dim={node.dim},\n"
            f"{curr_ind})"
        )

    if cls_name == "BinaryExpr":
        op_name = node.op.name if hasattr(node.op, "name") else str(node.op)
        lhs_code = _rec_loop_ir_to_string(
            node.lhs,
            level + 1,
            indent_unit,
        )
        rhs_code = _rec_loop_ir_to_string(
            node.rhs,
            level + 1,
            indent_unit,
        )

        return (
            "BinaryExpr(\n"
            f"{next_ind}BinaryOp.{op_name},\n"
            f"{next_ind}{lhs_code},\n"
            f"{next_ind}{rhs_code},\n"
            f"{curr_ind})"
        )

    if cls_name == "Load":
        field_code = _rec_loop_ir_to_string(
            node.field,
            level + 1,
            indent_unit,
        )
        index_strings = [
            _rec_loop_ir_to_string(
                index,
                level + 2,
                indent_unit,
            )
            for index in node.indices
        ]
        comma = "," if len(node.indices) == 1 else ""

        if index_strings and not any(
            "\n" in index_string for index_string in index_strings
        ):
            indices_string = f"({', '.join(index_strings)}{comma})"
        else:
            joined_indices = (",\n" + indent_unit * (level + 2)).join(index_strings)

            indices_string = (
                f"(\n{indent_unit * (level + 2)}{joined_indices}{comma}\n{next_ind})"
            )

        return (
            "Load(\n"
            f"{next_ind}field={field_code},\n"
            f"{next_ind}indices={indices_string},\n"
            f"{curr_ind})"
        )

    if cls_name == "Store":
        field_code = _rec_loop_ir_to_string(
            node.field,
            level + 1,
            indent_unit,
        )
        index_strings = [
            _rec_loop_ir_to_string(
                index,
                level + 2,
                indent_unit,
            )
            for index in node.indices
        ]
        comma = "," if len(node.indices) == 1 else ""

        if index_strings and not any(
            "\n" in index_string for index_string in index_strings
        ):
            indices_string = f"({', '.join(index_strings)}{comma})"
        else:
            joined_indices = (",\n" + indent_unit * (level + 2)).join(index_strings)

            indices_string = (
                f"(\n{indent_unit * (level + 2)}{joined_indices}{comma}\n{next_ind})"
            )

        value_code = _rec_loop_ir_to_string(
            node.value,
            level + 1,
            indent_unit,
        )

        return (
            "Store(\n"
            f"{next_ind}field={field_code},\n"
            f"{next_ind}indices={indices_string},\n"
            f"{next_ind}value={value_code},\n"
            f"{curr_ind})"
        )

    if cls_name == "For":
        index_code = _rec_loop_ir_to_string(
            node.index,
            level + 1,
            indent_unit,
        )
        lower_code = _rec_loop_ir_to_string(
            node.lower,
            level + 1,
            indent_unit,
        )
        upper_code = _rec_loop_ir_to_string(
            node.upper,
            level + 1,
            indent_unit,
        )

        body_statements = getattr(node, "body", ())
        body_lines = [
            _rec_loop_ir_to_string(
                statement,
                level + 2,
                indent_unit,
            )
            for statement in body_statements
        ]
        comma = "," if len(body_statements) == 1 else ""

        if body_lines:
            joined_body = (",\n" + indent_unit * (level + 2)).join(body_lines)

            body_string = (
                f"(\n{indent_unit * (level + 2)}{joined_body}{comma}\n{next_ind})"
            )
        else:
            body_string = "()"

        return (
            "For(\n"
            f"{next_ind}index={index_code},\n"
            f"{next_ind}lower={lower_code},\n"
            f"{next_ind}upper={upper_code},\n"
            f"{next_ind}body={body_string},\n"
            f"{curr_ind})"
        )

    if cls_name == "Function":
        fields = getattr(node, "fields", ())
        scalars = getattr(node, "scalars", ())
        body_statements = getattr(node, "body", ())

        field_lines = [
            _rec_loop_ir_to_string(
                field,
                level + 2,
                indent_unit,
            )
            for field in fields
        ]
        field_comma = "," if len(fields) == 1 else ""
        joined_fields = (",\n" + indent_unit * (level + 2)).join(field_lines)

        fields_string = (
            (f"(\n{indent_unit * (level + 2)}{joined_fields}{field_comma}\n{next_ind})")
            if fields
            else "()"
        )

        scalar_lines = [
            _rec_loop_ir_to_string(
                scalar,
                level + 2,
                indent_unit,
            )
            for scalar in scalars
        ]
        scalar_comma = "," if len(scalars) == 1 else ""
        joined_scalars = (",\n" + indent_unit * (level + 2)).join(scalar_lines)

        scalars_string = (
            (
                f"(\n"
                f"{indent_unit * (level + 2)}"
                f"{joined_scalars}{scalar_comma}\n"
                f"{next_ind})"
            )
            if scalars
            else "()"
        )

        body_lines = [
            _rec_loop_ir_to_string(
                statement,
                level + 2,
                indent_unit,
            )
            for statement in body_statements
        ]
        body_comma = "," if len(body_statements) == 1 else ""
        joined_body = (",\n" + indent_unit * (level + 2)).join(body_lines)

        body_string = (
            (f"(\n{indent_unit * (level + 2)}{joined_body}{body_comma}\n{next_ind})")
            if body_lines
            else "()"
        )

        return (
            "Function(\n"
            f"{next_ind}name='{node.name}',\n"
            f"{next_ind}fields={fields_string},\n"
            f"{next_ind}scalars={scalars_string},\n"
            f"{next_ind}body={body_string},\n"
            f"{curr_ind})"
        )

    return str(node)


def print_stencil_ir(
    operator: Any,
    indent: int = 3,
) -> None:
    indent_unit = " " * indent

    if hasattr(operator, "_as_ir"):
        print(
            _rec_stencil_ir_to_string(
                operator._as_ir(),
                0,
                indent_unit,
            )
        )
    else:
        raise TypeError("Invalid input type. Function _as_ir is missing.")


def _rec_stencil_ir_to_string(
    node: Any,
    level: int = 0,
    indent_unit: str = " ",
) -> str:
    cls_name = type(node).__name__
    curr_ind = indent_unit * level
    next_ind = indent_unit * (level + 1)

    if cls_name in ("Dimension", "Field"):
        return str(node.name)

    if cls_name == "FieldAccess":
        field_ref = node.field.name if hasattr(node.field, "name") else str(node.field)
        return f"FieldAccess({field_ref}, {tuple(node.offsets)})"

    if cls_name == "Literal":
        return f"Literal({node.value})"

    if cls_name == "BinaryExpr":
        op_repr = (
            f"BinaryOp.{node.op.name}" if hasattr(node.op, "name") else str(node.op)
        )
        lhs_code = _rec_stencil_ir_to_string(
            node.lhs,
            level + 1,
            indent_unit,
        )
        rhs_code = _rec_stencil_ir_to_string(
            node.rhs,
            level + 1,
            indent_unit,
        )

        return (
            "BinaryExpr(\n"
            f"{next_ind}{op_repr},\n"
            f"{next_ind}{lhs_code},\n"
            f"{next_ind}{rhs_code},\n"
            f"{curr_ind})"
        )

    if cls_name == "Assign":
        target_code = _rec_stencil_ir_to_string(
            node.target,
            level + 1,
            indent_unit,
        )
        value_code = _rec_stencil_ir_to_string(
            node.value,
            level + 1,
            indent_unit,
        )

        return (
            "Assign(\n"
            f"{next_ind}target={target_code},\n"
            f"{next_ind}value={value_code},\n"
            f"{curr_ind})"
        )

    if cls_name == "Operator":
        statements = getattr(node, "statements", ())
        statement_lines = [
            _rec_stencil_ir_to_string(
                statement,
                level + 2,
                indent_unit,
            )
            for statement in statements
        ]
        joined_statements = (f",\n{indent_unit * (level + 2)}").join(statement_lines)

        return (
            "Operator(\n"
            f"{next_ind}statements=(\n"
            f"{indent_unit * (level + 2)}"
            f"{joined_statements},\n"
            f"{next_ind}),\n"
            f"{curr_ind})"
        )

    return str(node)
