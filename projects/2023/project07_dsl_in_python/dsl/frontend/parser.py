import ast, inspect
from dsl.frontend.frontend import LanguageParser
from dsl.backend.codegen import CodeGen


def parse_function(function) -> None:
    print("Parsing the function...")
    tree = ast.parse(inspect.getsource(function))

    print("Generating IR...")
    lp = LanguageParser()
    lp.visit(tree.body[0])
    ir = lp._IR

    print("Generating code...")
    cg = CodeGen()
    cg.apply(ir)

    print("Done!")
