from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, List

import toydsl.ir.ir as ir
from toydsl.ir.visitor import IRNodeVisitor

def load_cpp_module(so_filename: Path):
    """
    Load the python module from the .so file.

    https://stackoverflow.com/a/67692
    """

    spec = importlib.util.spec_from_file_location("dslgen", so_filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def format_cpp(cpp_filename: Path, cmake_dir: Path):
    """
    Format the generated C++ source code to make it prettier to look at.

    This doesn't change anything about the generated module, but it's
    useful to have readable code when debugging the code generator.
    """

    clang_format_installed = shutil.which("clang-format") is not None
    if clang_format_installed:
        clang_format_config = open(cmake_dir / '.clang-format').read()

        subprocess.call([
            "clang-format",
            "-style={{{}}}".format(', '.join(clang_format_config.splitlines())),
            "-i",
            cpp_filename.resolve(),
        ])

def compile_cpp(code_dir: Path, cmake_dir: Path, build_type: str = "Release"):
    """
    Compile the generated C++ code using CMake.
    """

    build_dir = code_dir / "build"
    os.makedirs(build_dir, exist_ok=True)

    ret = subprocess.call([
        "cmake",
        cmake_dir.resolve(),
        "-DCMAKE_BUILD_TYPE={}".format(build_type),
        "-DSOURCE_DIR={}".format(code_dir.resolve()),
        "-B{}".format(build_dir.resolve())
    ], cwd=build_dir)
    if ret != 0:
        raise Exception("CMake failed. build directory: {dir}. return code: {ret}. build type: {build_type}".format(dir=build_dir, ret=ret, build_type=build_type))

    ret = subprocess.call(["make", "-j", "VERBOSE=1"], cwd=build_dir)
    if ret != 0:
        raise Exception("make failed. build directory: {dir}. return code: {ret}".format(dir=build_dir, ret=ret))

def offset_to_string(offset: ir.AccessOffset, unroll_offset: int = 0) -> str:
    """
    Converts the offset of a FieldAccess to a 1-dimensional array access with the proper indexing
    """
    return "[(idx_i + {i}) + (idx_j + {j})*dim2 + (idx_k + {k})*dim3 + {unroll_offset}]".format(
        i=offset.offsets[0],
        j=offset.offsets[1],
        k=offset.offsets[2],
        unroll_offset=unroll_offset
    )

def create_loop_header(loop_variable: str, extents: List[str], stride: int = 1) -> str:
    assert loop_variable in ["i", "j", "k"]

    return "for (std::size_t {var} = {start}; {var} <= {end} - {stride}; {var} += {stride})".format(
        start=extents[0],
        end=extents[1],
        var="idx_{}".format(loop_variable),
        stride=stride
    )

def create_extents(extents: ir.AxisInterval, loop_variable: str) -> List[str]:
    def create_offset(offset: ir.Offset):
        side = "start" if offset.level == ir.LevelMarker.START else "end"
        return "{}_{} + {}".format(side, loop_variable, offset.offset)

    return [
        create_offset(extents.start),
        create_offset(extents.end)
    ]

def generate_converter(arg_name: str):
    return "auto {a} = reinterpret_cast<scalar_t*>({a}_np.get_data());".format(a=arg_name)

def check_openmp_private(node: ir.IR):
    written = set()
    read = set()
    for vert in node.body:
        if isinstance(vert, ir.VerticalDomain):
            for hori in vert.body:
                if isinstance(hori, ir.HorizontalDomain):
                    for stmt in hori.body:
                        if isinstance(stmt, ir.AssignmentStmt):
                            if isinstance(stmt.left,ir.FieldAccessExpr):
                                written.add(stmt.left.name)
                            if isinstance(stmt.left,ir.BinaryOp):
                                written = written.union(check_binop(stmt.left))
                            if isinstance(stmt.right,ir.FieldAccessExpr):
                                read.add(stmt.right.name)
                            if isinstance(stmt.right,ir.BinaryOp):
                                read = read.union(check_binop(stmt.right))

    global private_var
    private_var = (written&read)
    global public_var
    public_var = set(node.api_signature) - private_var

def check_binop(node: ir.BinaryOp):
    names = set()
    if isinstance(node.left,ir.FieldAccessExpr):
        names.add(node.left.name)
    if isinstance(node.right,ir.FieldAccessExpr):
        names.add(node.right.name)
    if isinstance(node.left,ir.BinaryOp):
        names = names.union(check_binop(node.left))
    if isinstance(node.right,ir.BinaryOp):
        names = names.union(check_binop(node.right))
    return names

class CodeGenCpp(IRNodeVisitor):
    """
    The code-generation module that traverses the IR and generates code form it.
    """
    def __init__(self):
        # The private variables here are properties that count for certain subtrees of the AST.
        # Any visitor can modify them to influence all the visitors in the subtree below
        # itself, but note that the setter of the variable is responsible to return it to
        # its previous value when the subtree has been processed.

        self._repetitions = 1 # how many times should statements be executed
        self._unroll_offset = 0 # indexes the repeated statements in an unrolled loop
        self._vectorize = True # use avx2 instructions
        self._openmp = True # use openmp

    @classmethod
    def apply(cls: CodeGenCpp, ir: ir.IR) -> str:
        """
        Entrypoint for the code generation, applying this to an IR returns a formatted function for that IR
        """
        codegen = cls()
        return codegen.visit(ir)

    # ---- Visitor handlers ----
    def generic_visit(self, node: Any, **kwargs) -> None:
        """
        Each visit needs to do something in code-generation, there can't be a default visit
        """
        raise RuntimeError("Invalid IR node: {}".format(node))

    def visit_LiteralExpr(self, node: ir.LiteralExpr) -> str:
        # instruction: __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
        if self._vectorize:
            return "_mm256_set1_pd({d})".format(d=node.value)
        else:
            return node.value

    def visit_FieldAccessExpr(self, node: ir.FieldAccessExpr) -> str:
        array_access = node.name + offset_to_string(node.offset, self._unroll_offset)
        if self._vectorize:
            # instruction: __m256d _mm256_loadu_pd (double const * mem_addr)
            return "_mm256_loadu_pd(&{})".format(array_access)
        else:
            return array_access

    def visit_AssignmentStmt(self, node: ir.AssignmentStmt) -> str:
        right = self.visit(node.right)

        if self._vectorize:
            self._vectorize = False
            # On the left side we only want to generate the normal array access
            # so that we can then take the address of it when using it as the
            # destination in the stream function. We thus turn off `_vectorize`
            # in order to not generate a load instruction.
            left = self.visit(node.left)
            self._vectorize = True

            # instruction: void _mm256_storeu_pd (double * mem_addr, __m256d a)
            return "_mm256_storeu_pd(&{}, {});".format(left, right)
        else:
            left = self.visit(node.left)
        return "{} = {};".format(left, right)

    def visit_BinaryOp(self, node: ir.BinaryOp) -> str: # TODO : Do not strip out the brackets
        # We actually don't have to add the avx2 intrinsics
        # for binary operators here because the arithmetic
        # operators are apparently overloaded for `__m256d`.

        assert(node.operator),"Unknown operator"
        # Keep the commented lines bellow, it might be useful later
        # if node.operator == "+":
        #     op_string = "+"
        # elif node.operator == "-":
        #     op_string = "-"
        # elif node.operator == "*":
        #     op_string = "*"
        # elif node.operator == "/":
        #     op_string = "/"
        # elif node.operator == "**":
        #     op_string = "**"
        # elif node.operator == "%":
        #     op_string = "%"
        # else:
        #     assert(False),"Operator has been defined in frontend.py only"
        if node.operator == "**":
            binaryOp_str = "pow("+self.visit(node.left) + "," + self.visit(node.right) + ")"
        else:
            binaryOp_str = "(" + self.visit(node.left) + node.operator + self.visit(node.right) + ")"
        return binaryOp_str

    def visit_VerticalDomain(self, node: ir.VerticalDomain) -> List[str]:
        if self._openmp:
            if(len(private_var)!=0 and len(public_var)!=0):
                pragma_string = "#pragma omp parallel for default(none) shared(dim2,dim3,start_k,start_j,start_i,end_k,end_j,end_i,{public}) firstprivate({private})".format(public=", ".join(public_var),private=", ".join(private_var))
            elif (len(private_var)==0):
                pragma_string = "#pragma omp parallel for default(none) shared(dim2,dim3,start_k,start_j,start_i,end_k,end_j,end_i,{public})".format(public=", ".join(public_var))
            elif (len(public_var)==0):
                pragma_string = "#pragma omp parallel for default(none) shared(dim2,dim3,start_k,start_j,start_i,end_k,end_j,end_i) firstprivate({private})".format(private=", ".join(private_var))
            else:
                pragma_string = "#pragma omp parallel for default(none) shared(dim2,dim3,start_k,start_j,start_i,end_k,end_j,end_i)"

            vertical_loop = [pragma_string]
        else:
            vertical_loop = []
        vertical_loop.append(create_loop_header("k", create_extents(node.extents, "k")))
        vertical_loop.append("{")
        for stmt in node.body:
            lines_of_code = self.visit(stmt)
            for line in lines_of_code:
                vertical_loop.append(line)
        vertical_loop.append("}")

        return vertical_loop

    def visit_HorizontalDomain(self, node: ir.HorizontalDomain) -> List[str]:
        unroll_factor = 4

        inner_extents = create_extents(node.extents[0], "i")
        inner_loop = []

        self._repetitions *= unroll_factor
        inner_loop.append(create_loop_header("i", inner_extents, self._repetitions))
        inner_loop.append("{")
        inner_loop.extend(self.visit(node.body))
        inner_loop.append("}")
        self._repetitions //= unroll_factor

        # Generate instructions for the rest that was not evenly divisible by the unroll factor
        if unroll_factor > 1:
            inner_extents[0] = "{e} - ({e} - ({s})) % {r}".format(
                s=inner_extents[0],
                e=inner_extents[1],
                r=unroll_factor
            )

            inner_loop.append(create_loop_header("i", inner_extents, self._repetitions))
            inner_loop.append("{")
            inner_loop.extend(self.visit(node.body))
            inner_loop.append("}")

        outer_loop = [create_loop_header("j", create_extents(node.extents[1], "j"))]
        outer_loop.append("{")
        for line in inner_loop:
            outer_loop.append(line)
        outer_loop.append("}")

        return outer_loop

    def visit_list_of_Stmt(self, nodes: List[ir.Stmt]) -> List[str]:
        res = []

        # Make sure to adjust this if we change the instrucions or
        # the datatype used.
        vectorize_width = 4

        # The `previous_*` variables are used to remember the value
        # of some subtree properties so that we can change the
        # property for the current subtree and then change it back
        # to the previous value.

        previous_unroll_offset = self._unroll_offset
        previous_repetitions = self._repetitions
        self._repetitions = 1

        if self._vectorize and previous_repetitions % vectorize_width == 0:
            for i in range(previous_repetitions // vectorize_width):
                self._unroll_offset = i * vectorize_width
                for stmt in nodes:
                    res.append(self.visit(stmt))
        else:
            previous_vectorize = self._vectorize
            self._vectorize = False

            for i in range(previous_repetitions):
                self._unroll_offset = i
            for stmt in nodes:
                res.append(self.visit(stmt))

            self._vectorize = previous_vectorize

        self._repetitions = previous_repetitions
        self._unroll_offset = previous_unroll_offset

        return res

    def visit_IR(self, node: ir.IR) -> str:
        if self._openmp:
            check_openmp_private(node)

        scope = [""" #include <common_python.hpp>
            #include <immintrin.h>

            void {name}({array_args}, {bounds}) {{

                const auto [start_i, end_i, start_j, end_j, start_k, end_k] = get_bounds(i, j, k);
                const std::size_t dim2 = (end_i - start_i);
                const std::size_t dim3 = dim2 * (end_j - start_j);

                    {converters}

        """.format(
            name=node.name,
            array_args=", ".join(["array_t &{}_np".format(arg) for arg in node.api_signature]),
            bounds=", ".join(["const bounds_t &{}".format(axis) for axis in ["k", "j", "i"]]),
            converters="\n".join(map(generate_converter, node.api_signature))
        )]

        for stmt in node.body:
            scope.extend(self.visit(stmt))

        scope.append("""

                return;
            }}

            BOOST_PYTHON_MODULE(dslgen) {{
                Py_Initialize();
                np::initialize();
                boost::python::def("{name}", {name});
            }}
        """.format(name=node.name))

        return "\n".join(scope)
