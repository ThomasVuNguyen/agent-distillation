#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import builtins
import difflib
import inspect
import logging
import math
import re
from collections.abc import Mapping
from functools import wraps
from importlib import import_module
from concurrent.futures import TimeoutError
from timeout_decorator import timeout

from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .tools import Tool
from .utils import BASE_BUILTIN_MODULES, truncate_content


logger = logging.getLogger(__name__)


class InterpreterError(ValueError):
    """
    An error raised when the interpreter cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """

    pass


ERRORS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
}

DEFAULT_MAX_LEN_OUTPUT = 50000
MAX_OPERATIONS = 10000000
MAX_WHILE_ITERATIONS = 1000000


def custom_print(*args):
    return None


BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
    "complex": complex,
}

DANGEROUS_FUNCTIONS = [
    "builtins.compile",
    "builtins.eval",
    "builtins.exec",
    "builtins.globals",
    "builtins.locals",
    "builtins.__import__",
    "os.popen",
    "os.system",
    "posix.system",
]


class PrintContainer:
    def __init__(self):
        self.value = ""

    def append(self, text):
        self.value += text
        return self

    def __iadd__(self, other):
        """Implements the += operator"""
        self.value += str(other)
        return self

    def __str__(self):
        """String representation"""
        return self.value

    def __repr__(self):
        """Representation for debugging"""
        return f"PrintContainer({self.value})"

    def __len__(self):
        """Implements len() function support"""
        return len(self.value)


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


def get_iterable(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, "__iter__"):
        return list(obj)
    else:
        raise InterpreterError("Object is not iterable")


# ✅ 중첩 루프 깊이 계산 함수
def count_loop_depth(node: ast.AST, current_depth=0, max_depth=[0]):
    """
    Recursively count the depth of nested loops in AST.
    Includes for-loops, while-loops, async-for, and comprehensions.
    """
    new_depth = current_depth

    if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
        new_depth += 1
        if new_depth > max_depth[0]:
            max_depth[0] = new_depth

    # Handle comprehensions (list, set, dict, generator)
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        comp_depth = current_depth + len(node.generators)
        if comp_depth > max_depth[0]:
            max_depth[0] = comp_depth

    for child in ast.iter_child_nodes(node):
        count_loop_depth(child, new_depth, max_depth)

def is_static_int(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, int)

def estimate_len(node, state):
    # 간단한 경우만 정적 추정: set([...]), list([...]), range(...)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == 'range' and len(node.args) == 1 and is_static_int(node.args[0]):
            return node.args[0].value
    elif isinstance(node, (ast.List, ast.Set)):
        return len(node.elts)
    return None  # unknown

DANGEROUS_CALLS = {
    ("itertools", "product"),
    ("itertools", "permutations"),
    ("itertools", "combinations"),
    ("itertools", "combinations_with_replacement"),
}


def detect_dangerous_calls(expression: ast.AST, state=None):
    # 1) Build a map of local names → real module/function names
    import_map = {}  # e.g. { "product": "itertools.product", "np": "numpy" }
    for node in ast.walk(expression):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                import_map[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                import_map[local] = f"{module}.{alias.name}"

    # 2) Walk all calls and resolve their “module”+“attr” properly
    for node in ast.walk(expression):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        module, attr = None, None

        # case A: module.func(…)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            raw_mod = func.value.id
            module = import_map.get(raw_mod, raw_mod)
            attr   = func.attr

        # case B: directly imported func(…)
        elif isinstance(func, ast.Name):
            full = import_map.get(func.id)
            if full and "." in full:
                module, attr = full.rsplit(".", 1)
            else:
                # e.g. builtin eval; you could also handle bare names here if you
                # choose to block eval, exec, etc.
                module = ""
                attr   = func.id

        # otherwise we don’t know what this is → skip
        if module is None or attr is None:
            continue

        # only care about the ones on our “block list”
        if (module, attr) not in DANGEROUS_CALLS:
            continue

        name = f"{module}.{attr}"

        # --- itertools.product case ---
        if attr == "product":
            # must have at least one iterable
            if not node.args:
                raise InterpreterError(f"{name} missing iterable — blocking for safety")
            base = estimate_len(node.args[0])
            repeat = None

            # pick up repeat=… if present
            for kw in node.keywords:
                if kw.arg == "repeat":
                    if isinstance(kw.value, ast.Constant):
                        repeat = kw.value.value
                    else:
                        raise InterpreterError(f"{name} repeat is dynamic — blocking for safety")
                    break

            # default repeat=1 if the user omitted it
            if repeat is None:
                repeat = 1

            if base is not None:
                total_size = base ** repeat
                if total_size > 1_000_000:
                    raise InterpreterError(f"{name} will generate {total_size} items — too large")
            else:
                raise InterpreterError(f"{name} size estimation failed — blocking for safety")

        # --- itertools.permutations case ---
        elif attr == "permutations":
            if not node.args:
                raise InterpreterError(f"{name} missing iterable — blocking for safety")
            base = estimate_len(node.args[0])
            r = None

            # positional r
            if len(node.args) == 2 and isinstance(node.args[1], ast.Constant):
                r = node.args[1].value

            # keyword r
            else:
                for kw in node.keywords:
                    if kw.arg == "r":
                        if isinstance(kw.value, ast.Constant):
                            r = kw.value.value
                        else:
                            raise InterpreterError(f"{name} r is dynamic — blocking for safety")
                        break

            # default r = len(data) if still None
            if base is not None and r is None:
                r = base

            if base is None or r is None:
                raise InterpreterError(f"{name} cannot estimate size — blocking for safety")

            try:
                size = factorial(base) // factorial(base - r)
            except ValueError:
                raise InterpreterError(f"{name} invalid r value — blocking for safety")

            if size > 1_000_000:
                raise InterpreterError(f"{name} will generate {size} items — too large")

        # --- anything else is outright forbidden ---
        else:
            raise InterpreterError(f"Usage of {name} is blocked as potentially dangerous")


def fix_final_answer_code(code: str) -> str:
    """
    Sometimes an LLM can try to assign a variable to final_answer, which would break the final_answer() tool.
    This function fixes this behaviour by replacing variable assignments to final_answer with final_answer_variable,
    while preserving function calls to final_answer().
    """
    # First, find if there's a direct assignment to final_answer
    # Use word boundary and negative lookbehind to ensure it's not an object attribute
    assignment_pattern = r"(?<!\.)(?<!\w)\bfinal_answer\s*="
    if "final_answer(" not in code or not re.search(assignment_pattern, code):
        # If final_answer tool is not called in this blob, then doing the replacement is hazardous because it could false the model's memory for next steps.
        # Let's not modify the code and leave the subsequent assignment error happen.
        return code

    # Pattern for replacing variable assignments
    # Looks for 'final_answer' followed by '=' with optional whitespace
    # Negative lookbehind ensures we don't match object attributes
    assignment_regex = r"(?<!\.)(?<!\w)(\bfinal_answer)(\s*=)"
    code = re.sub(assignment_regex, r"final_answer_variable\2", code)

    # Pattern for replacing variable usage but not function calls
    # Negative lookahead (?!\s*\() ensures we don't match function calls
    # Negative lookbehind (?<!\.|\w) ensures we don't match object methods or other variables
    variable_regex = r"(?<!\.)(?<!\w)(\bfinal_answer\b)(?!\s*\()"
    code = re.sub(variable_regex, "final_answer_variable", code)
    return code


def safer_eval(func: Callable):
    """
    Decorator to make the evaluation of a function safer by checking its return value.

    Args:
        func: Function to make safer.

    Returns:
        Callable: Safer function with return value check.
    """

    @wraps(func)
    def _check_return(
        expression,
        state,
        static_tools,
        custom_tools,
        authorized_imports=BASE_BUILTIN_MODULES,
    ):
        result = func(expression, state, static_tools, custom_tools, authorized_imports=authorized_imports)
        if "*" not in authorized_imports:
            if isinstance(result, ModuleType):
                if result.__name__ not in authorized_imports:
                    raise InterpreterError(f"Forbidden access to module: {result.__name__}")
            elif isinstance(result, dict) and result.get("__spec__"):
                if result["__name__"] not in authorized_imports:
                    raise InterpreterError(f"Forbidden access to module: {result['__name__']}")
            elif isinstance(result, (FunctionType, BuiltinFunctionType)):
                for qualified_function_name in DANGEROUS_FUNCTIONS:
                    module_name, function_name = qualified_function_name.rsplit(".", 1)
                    if (
                        function_name not in static_tools
                        and result.__name__ == function_name
                        and result.__module__ == module_name
                    ):
                        raise InterpreterError(f"Forbidden access to function: {function_name}")
        return result

    return _check_return


def evaluate_attribute(
    expression: ast.Attribute,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if expression.attr.startswith("__") and expression.attr.endswith("__"):
        raise InterpreterError(f"Forbidden access to dunder attribute: {expression.attr}")
    value = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    return getattr(value, expression.attr)


def evaluate_unaryop(
    expression: ast.UnaryOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    operand = evaluate_ast(expression.operand, state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression.op, ast.USub):
        return -operand
    elif isinstance(expression.op, ast.UAdd):
        return operand
    elif isinstance(expression.op, ast.Not):
        return not operand
    elif isinstance(expression.op, ast.Invert):
        return ~operand
    else:
        raise InterpreterError(f"Unary operation {expression.op.__class__.__name__} is not supported.")


def evaluate_lambda(
    lambda_expression: ast.Lambda,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    args = [arg.arg for arg in lambda_expression.args.args]

    def lambda_func(*values: Any) -> Any:
        new_state = state.copy()
        for arg, value in zip(args, values):
            new_state[arg] = value
        return evaluate_ast(
            lambda_expression.body,
            new_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )

    return lambda_func


def evaluate_while(
    while_loop: ast.While,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    iterations = 0
    while evaluate_ast(while_loop.test, state, static_tools, custom_tools, authorized_imports):
        for node in while_loop.body:
            try:
                evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            except BreakException:
                return None
            except ContinueException:
                break
        iterations += 1
        if iterations > MAX_WHILE_ITERATIONS:
            raise InterpreterError(f"Maximum number of {MAX_WHILE_ITERATIONS} iterations in While loop exceeded")
    return None


def create_function(
    func_def: ast.FunctionDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    source_code = ast.unparse(func_def)

    def new_func(*args: Any, **kwargs: Any) -> Any:
        func_state = state.copy()
        arg_names = [arg.arg for arg in func_def.args.args]
        default_values = [
            evaluate_ast(d, state, static_tools, custom_tools, authorized_imports) for d in func_def.args.defaults
        ]

        # Apply default values
        defaults = dict(zip(arg_names[-len(default_values) :], default_values))

        # Set positional arguments
        for name, value in zip(arg_names, args):
            func_state[name] = value

        # Set keyword arguments
        for name, value in kwargs.items():
            func_state[name] = value

        # Handle variable arguments
        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args

        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            func_state[kwarg_name] = kwargs

        # Set default values for arguments that were not provided
        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        # Update function state with self and __class__
        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        result = None
        try:
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, static_tools, custom_tools, authorized_imports)
        except ReturnException as e:
            result = e.value

        if func_def.name == "__init__":
            return None

        return result

    # Store original AST, source code, and name
    new_func.__ast__ = func_def
    new_func.__source__ = source_code
    new_func.__name__ = func_def.name

    return new_func


def evaluate_function_def(
    func_def: ast.FunctionDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Callable:
    custom_tools[func_def.name] = create_function(func_def, state, static_tools, custom_tools, authorized_imports)
    return custom_tools[func_def.name]


def evaluate_class_def(
    class_def: ast.ClassDef,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> type:
    class_name = class_def.name
    bases = [evaluate_ast(base, state, static_tools, custom_tools, authorized_imports) for base in class_def.bases]
    class_dict = {}

    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            class_dict[stmt.name] = evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = evaluate_ast(
                        stmt.value,
                        state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )
                elif isinstance(target, ast.Attribute):
                    class_dict[target.attr] = evaluate_ast(
                        stmt.value,
                        state,
                        static_tools,
                        custom_tools,
                        authorized_imports,
                    )
        else:
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    new_class = type(class_name, tuple(bases), class_dict)
    state[class_name] = new_class
    return new_class


def evaluate_augassign(
    expression: ast.AugAssign,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    def get_current_value(target: ast.AST) -> Any:
        if isinstance(target, ast.Name):
            return state.get(target.id, 0)
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            return obj[key]
        elif isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            return getattr(obj, target.attr)
        elif isinstance(target, ast.Tuple):
            return tuple(get_current_value(elt) for elt in target.elts)
        elif isinstance(target, ast.List):
            return [get_current_value(elt) for elt in target.elts]
        else:
            raise InterpreterError("AugAssign not supported for {type(target)} targets.")

    current_value = get_current_value(expression.target)
    value_to_add = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)

    if isinstance(expression.op, ast.Add):
        if isinstance(current_value, list):
            if not isinstance(value_to_add, list):
                raise InterpreterError(f"Cannot add non-list value {value_to_add} to a list.")
            current_value += value_to_add
        else:
            current_value += value_to_add
    elif isinstance(expression.op, ast.Sub):
        current_value -= value_to_add
    elif isinstance(expression.op, ast.Mult):
        current_value *= value_to_add
    elif isinstance(expression.op, ast.Div):
        current_value /= value_to_add
    elif isinstance(expression.op, ast.Mod):
        current_value %= value_to_add
    elif isinstance(expression.op, ast.Pow):
        current_value **= value_to_add
    elif isinstance(expression.op, ast.FloorDiv):
        current_value //= value_to_add
    elif isinstance(expression.op, ast.BitAnd):
        current_value &= value_to_add
    elif isinstance(expression.op, ast.BitOr):
        current_value |= value_to_add
    elif isinstance(expression.op, ast.BitXor):
        current_value ^= value_to_add
    elif isinstance(expression.op, ast.LShift):
        current_value <<= value_to_add
    elif isinstance(expression.op, ast.RShift):
        current_value >>= value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    # Update the state: current_value has been updated in-place
    set_value(
        expression.target,
        current_value,
        state,
        static_tools,
        custom_tools,
        authorized_imports,
    )

    return current_value


def evaluate_boolop(
    node: ast.BoolOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> bool:
    if isinstance(node.op, ast.And):
        for value in node.values:
            if not evaluate_ast(value, state, static_tools, custom_tools, authorized_imports):
                return False
        return True
    elif isinstance(node.op, ast.Or):
        for value in node.values:
            if evaluate_ast(value, state, static_tools, custom_tools, authorized_imports):
                return True
        return False


def evaluate_binop(
    binop: ast.BinOp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    # Recursively evaluate the left and right operands
    left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
    right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)

    # Determine the operation based on the type of the operator in the BinOp
    if isinstance(binop.op, ast.Add):
        return left_val + right_val
    elif isinstance(binop.op, ast.Sub):
        return left_val - right_val
    elif isinstance(binop.op, ast.Mult):
        return left_val * right_val
    elif isinstance(binop.op, ast.Div):
        return left_val / right_val
    elif isinstance(binop.op, ast.Mod):
        return left_val % right_val
    elif isinstance(binop.op, ast.Pow):
        return left_val**right_val
    elif isinstance(binop.op, ast.FloorDiv):
        return left_val // right_val
    elif isinstance(binop.op, ast.BitAnd):
        return left_val & right_val
    elif isinstance(binop.op, ast.BitOr):
        return left_val | right_val
    elif isinstance(binop.op, ast.BitXor):
        return left_val ^ right_val
    elif isinstance(binop.op, ast.LShift):
        return left_val << right_val
    elif isinstance(binop.op, ast.RShift):
        return left_val >> right_val
    else:
        raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(
    assign: ast.Assign,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = evaluate_ast(assign.value, state, static_tools, custom_tools, authorized_imports)
    if len(assign.targets) == 1:
        target = assign.targets[0]
        set_value(target, result, state, static_tools, custom_tools, authorized_imports)
    else:
        expanded_values = []
        for tgt in assign.targets:
            if isinstance(tgt, ast.Starred):
                expanded_values.extend(result)
            else:
                expanded_values.append(result)

        for tgt, val in zip(assign.targets, expanded_values):
            set_value(tgt, val, state, static_tools, custom_tools, authorized_imports)
    return result


def set_value(
    target: ast.AST,
    value: Any,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    if isinstance(target, ast.Name):
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase the existing tool!")
        state[target.id] = value
    elif isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                value = tuple(value)
            else:
                raise InterpreterError("Cannot unpack non-tuple value")
        if len(target.elts) != len(value):
            raise InterpreterError("Cannot unpack tuple of wrong size")
        for i, elem in enumerate(target.elts):
            set_value(elem, value[i], state, static_tools, custom_tools, authorized_imports)
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        setattr(obj, target.attr, value)


def evaluate_call(
    call: ast.Call,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if not isinstance(call.func, (ast.Call, ast.Lambda, ast.Attribute, ast.Name, ast.Subscript)):
        raise InterpreterError(f"This is not a correct function: {call.func}).")

    func, func_name = None, None

    if isinstance(call.func, ast.Call):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Lambda):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute {func_name}")
        func = getattr(obj, func_name)
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in state:
            func = state[func_name]
        elif func_name in static_tools:
            func = static_tools[func_name]
        elif func_name in custom_tools:
            func = custom_tools[func_name]
        elif func_name in ERRORS:
            func = ERRORS[func_name]
        else:
            raise InterpreterError(
                f"It is not permitted to evaluate other functions than the provided tools or functions defined/imported in previous code (tried to execute {call.func.id})."
            )
    elif isinstance(call.func, ast.Subscript):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
        if not callable(func):
            raise InterpreterError(f"This is not a correct function: {call.func}).")
        func_name = None

    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            args.extend(evaluate_ast(arg.value, state, static_tools, custom_tools, authorized_imports))
        else:
            args.append(evaluate_ast(arg, state, static_tools, custom_tools, authorized_imports))

    kwargs = {
        keyword.arg: evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)
        for keyword in call.keywords
    }

    if func_name == "super":
        if not args:
            if "__class__" in state and "self" in state:
                return super(state["__class__"], state["self"])
            else:
                raise InterpreterError("super() needs at least one argument")
        cls = args[0]
        if not isinstance(cls, type):
            raise InterpreterError("super() argument 1 must be type")
        if len(args) == 1:
            return super(cls)
        elif len(args) == 2:
            instance = args[1]
            return super(cls, instance)
        else:
            raise InterpreterError("super() takes at most 2 arguments")
    elif func_name == "print":
        state["_print_outputs"] += " ".join(map(str, args)) + "\n"
        return None
    else:  # Assume it's a callable object
        if (inspect.getmodule(func) == builtins) and inspect.isbuiltin(func) and (func not in static_tools.values()):
            raise InterpreterError(
                f"Invoking a builtin function that has not been explicitly added as a tool is not allowed ({func_name})."
            )
        return func(*args, **kwargs)


def evaluate_subscript(
    subscript: ast.Subscript,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    index = evaluate_ast(subscript.slice, state, static_tools, custom_tools, authorized_imports)
    value = evaluate_ast(subscript.value, state, static_tools, custom_tools, authorized_imports)
    try:
        return value[index]
    except (KeyError, IndexError, TypeError) as e:
        error_message = f"Could not index {value} with '{index}': {type(e).__name__}: {e}"
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index, list(value.keys()))
            if len(close_matches) > 0:
                error_message += f". Maybe you meant one of these indexes instead: {str(close_matches)}"
        raise InterpreterError(error_message) from e


def evaluate_name(
    name: ast.Name,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    if name.id in state:
        return state[name.id]
    elif name.id in static_tools:
        return static_tools[name.id]
    elif name.id in custom_tools:
        return custom_tools[name.id]
    elif name.id in ERRORS:
        return ERRORS[name.id]
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    raise InterpreterError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(
    condition: ast.Compare,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> bool | object:
    result = True
    left = evaluate_ast(condition.left, state, static_tools, custom_tools, authorized_imports)
    for i, (op, comparator) in enumerate(zip(condition.ops, condition.comparators)):
        op = type(op)
        right = evaluate_ast(comparator, state, static_tools, custom_tools, authorized_imports)
        if op == ast.Eq:
            current_result = left == right
        elif op == ast.NotEq:
            current_result = left != right
        elif op == ast.Lt:
            current_result = left < right
        elif op == ast.LtE:
            current_result = left <= right
        elif op == ast.Gt:
            current_result = left > right
        elif op == ast.GtE:
            current_result = left >= right
        elif op == ast.Is:
            current_result = left is right
        elif op == ast.IsNot:
            current_result = left is not right
        elif op == ast.In:
            current_result = left in right
        elif op == ast.NotIn:
            current_result = left not in right
        else:
            raise InterpreterError(f"Unsupported comparison operator: {op}")

        if current_result is False:
            return False
        result = current_result if i == 0 else (result and current_result)
        left = right
    return result


def evaluate_if(
    if_statement: ast.If,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = None
    test_result = evaluate_ast(if_statement.test, state, static_tools, custom_tools, authorized_imports)
    if test_result:
        for line in if_statement.body:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    else:
        for line in if_statement.orelse:
            line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
            if line_result is not None:
                result = line_result
    return result


def evaluate_for(
    for_loop: ast.For,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Any:
    result = None
    iterator = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    for counter in iterator:
        set_value(
            for_loop.target,
            counter,
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                break
            except ContinueException:
                continue
        else:
            continue
        break
    return result


def evaluate_listcomp(
    listcomp: ast.ListComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> List[Any]:
    def inner_evaluate(generators: List[ast.comprehension], index: int, current_state: Dict[str, Any]) -> List[Any]:
        if index >= len(generators):
            return [
                evaluate_ast(
                    listcomp.elt,
                    current_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
            ]
        generator = generators[index]
        iter_value = evaluate_ast(
            generator.iter,
            current_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
        result = []
        for value in iter_value:
            new_state = current_state.copy()
            if isinstance(generator.target, ast.Tuple):
                for idx, elem in enumerate(generator.target.elts):
                    new_state[elem.id] = value[idx]
            else:
                new_state[generator.target.id] = value
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in generator.ifs
            ):
                result.extend(inner_evaluate(generators, index + 1, new_state))
        return result

    return inner_evaluate(listcomp.generators, 0, state)


def evaluate_setcomp(
    setcomp: ast.SetComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Set[Any]:
    result = set()
    for gen in setcomp.generators:
        iter_value = evaluate_ast(gen.iter, state, static_tools, custom_tools, authorized_imports)
        for value in iter_value:
            new_state = state.copy()
            set_value(
                gen.target,
                value,
                new_state,
                static_tools,
                custom_tools,
                authorized_imports,
            )
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in gen.ifs
            ):
                element = evaluate_ast(
                    setcomp.elt,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                result.add(element)
    return result


def evaluate_try(
    try_node: ast.Try,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    try:
        for stmt in try_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        matched = False
        for handler in try_node.handlers:
            if handler.type is None or isinstance(
                e,
                evaluate_ast(handler.type, state, static_tools, custom_tools, authorized_imports),
            ):
                matched = True
                if handler.name:
                    state[handler.name] = e
                for stmt in handler.body:
                    evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
                break
        if not matched:
            raise e
    else:
        if try_node.orelse:
            for stmt in try_node.orelse:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    finally:
        if try_node.finalbody:
            for stmt in try_node.finalbody:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)


def evaluate_raise(
    raise_node: ast.Raise,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    if raise_node.exc is not None:
        exc = evaluate_ast(raise_node.exc, state, static_tools, custom_tools, authorized_imports)
    else:
        exc = None
    if raise_node.cause is not None:
        cause = evaluate_ast(raise_node.cause, state, static_tools, custom_tools, authorized_imports)
    else:
        cause = None
    if exc is not None:
        if cause is not None:
            raise exc from cause
        else:
            raise exc
    else:
        raise InterpreterError("Re-raise is not supported without an active exception")


def evaluate_assert(
    assert_node: ast.Assert,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    test_result = evaluate_ast(assert_node.test, state, static_tools, custom_tools, authorized_imports)
    if not test_result:
        if assert_node.msg:
            msg = evaluate_ast(assert_node.msg, state, static_tools, custom_tools, authorized_imports)
            raise AssertionError(msg)
        else:
            # Include the failing condition in the assertion message
            test_code = ast.unparse(assert_node.test)
            raise AssertionError(f"Assertion failed: {test_code}")


def evaluate_with(
    with_node: ast.With,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, static_tools, custom_tools, authorized_imports)
        if item.optional_vars:
            state[item.optional_vars.id] = context_expr.__enter__()
            contexts.append(state[item.optional_vars.id])
        else:
            context_var = context_expr.__enter__()
            contexts.append(context_var)

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        for context in reversed(contexts):
            context.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def get_safe_module(raw_module, authorized_imports, visited=None):
    """Creates a safe copy of a module or returns the original if it's a function"""
    # If it's a function or non-module object, return it directly
    if not isinstance(raw_module, ModuleType):
        return raw_module

    # Handle circular references: Initialize visited set for the first call
    if visited is None:
        visited = set()

    module_id = id(raw_module)
    if module_id in visited:
        return raw_module  # Return original for circular refs

    visited.add(module_id)

    # Create new module for actual modules
    safe_module = ModuleType(raw_module.__name__)

    # Copy all attributes by reference, recursively checking modules
    for attr_name in dir(raw_module):
        try:
            attr_value = getattr(raw_module, attr_name)
        except (ImportError, AttributeError) as e:
            # lazy / dynamic loading module -> INFO log and skip
            logger.info(
                f"Skipping import error while copying {raw_module.__name__}.{attr_name}: {type(e).__name__} - {e}"
            )
            continue
        # Recursively process nested modules, passing visited set
        if isinstance(attr_value, ModuleType):
            attr_value = get_safe_module(attr_value, authorized_imports, visited=visited)

        setattr(safe_module, attr_name, attr_value)

    return safe_module


def check_module_authorized(module_name, authorized_imports):
    if "*" in authorized_imports:
        return True
    else:
        module_path = module_name.split(".")
        # ["A", "B", "C"] -> ["A", "A.B", "A.B.C"]
        module_subpaths = [".".join(module_path[:i]) for i in range(1, len(module_path) + 1)]
        return any(subpath in authorized_imports for subpath in module_subpaths)


def evaluate_import(expression, state, authorized_imports):
    if isinstance(expression, ast.Import):
        for alias in expression.names:
            if check_module_authorized(alias.name, authorized_imports):
                raw_module = import_module(alias.name)
                state[alias.asname or alias.name] = get_safe_module(raw_module, authorized_imports)
            else:
                raise InterpreterError(
                    f"Import of {alias.name} is not allowed. Authorized imports are: {str(authorized_imports)}"
                )
        return None
    elif isinstance(expression, ast.ImportFrom):
        if check_module_authorized(expression.module, authorized_imports):
            raw_module = __import__(expression.module, fromlist=[alias.name for alias in expression.names])
            module = get_safe_module(raw_module, authorized_imports)
            if expression.names[0].name == "*":  # Handle "from module import *"
                if hasattr(module, "__all__"):  # If module has __all__, import only those names
                    for name in module.__all__:
                        state[name] = getattr(module, name)
                else:  # If no __all__, import all public names (those not starting with '_')
                    for name in dir(module):
                        if not name.startswith("_"):
                            state[name] = getattr(module, name)
            else:  # regular from imports
                for alias in expression.names:
                    if hasattr(module, alias.name):
                        state[alias.asname or alias.name] = getattr(module, alias.name)
                    else:
                        raise InterpreterError(f"Module {expression.module} has no attribute {alias.name}")
        else:
            raise InterpreterError(
                f"Import from {expression.module} is not allowed. Authorized imports are: {str(authorized_imports)}"
            )
        return None


def evaluate_dictcomp(
    dictcomp: ast.DictComp,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> Dict[Any, Any]:
    result = {}
    for gen in dictcomp.generators:
        iter_value = evaluate_ast(gen.iter, state, static_tools, custom_tools, authorized_imports)
        for value in iter_value:
            new_state = state.copy()
            set_value(
                gen.target,
                value,
                new_state,
                static_tools,
                custom_tools,
                authorized_imports,
            )
            if all(
                evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports)
                for if_clause in gen.ifs
            ):
                key = evaluate_ast(
                    dictcomp.key,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                val = evaluate_ast(
                    dictcomp.value,
                    new_state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
                result[key] = val
    return result


def evaluate_delete(
    delete_node: ast.Delete,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str],
) -> None:
    """
    Evaluate a delete statement (del x, del x[y]).

    Args:
        delete_node: The AST Delete node to evaluate
        state: The current state dictionary
        static_tools: Dictionary of static tools
        custom_tools: Dictionary of custom tools
        authorized_imports: List of authorized imports
    """
    for target in delete_node.targets:
        if isinstance(target, ast.Name):
            # Handle simple variable deletion (del x)
            if target.id in state:
                del state[target.id]
            else:
                raise InterpreterError(f"Cannot delete name '{target.id}': name is not defined")
        elif isinstance(target, ast.Subscript):
            # Handle index/key deletion (del x[y])
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            try:
                del obj[index]
            except (TypeError, KeyError, IndexError) as e:
                raise InterpreterError(f"Cannot delete index/key: {str(e)}")
        else:
            raise InterpreterError(f"Deletion of {type(target).__name__} targets is not supported")


@safer_eval
def evaluate_ast(
    expression: ast.AST,
    state: Dict[str, Any],
    static_tools: Dict[str, Callable],
    custom_tools: Dict[str, Callable],
    authorized_imports: List[str] = BASE_BUILTIN_MODULES,
):
    """
    Evaluate an abstract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abstract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignments.
        static_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. Trying to change one of these static_tools will raise an error.
        custom_tools (`Dict[str, Callable]`):
            Functions that may be called during the evaluation. These static_tools can be overwritten.
        authorized_imports (`List[str]`):
            The list of modules that can be imported by the code. By default, only a few safe modules are allowed.
            If it contains "*", it will authorize any import. Use this at your own risk!
    """
    if state.setdefault("_operations_count", {"counter": 0})["counter"] >= MAX_OPERATIONS:
        raise InterpreterError(
            f"Reached the max number of operations of {MAX_OPERATIONS}. Maybe there is an infinite loop somewhere in the code, or you're just asking too many calculations."
        )
    state["_operations_count"]["counter"] += 1
    common_params = (state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression, ast.Assign):
        # Assignment -> we evaluate the assignment which should update the state
        # We return the variable assigned as it may be used to determine the final result.
        return evaluate_assign(expression, *common_params)
    elif isinstance(expression, ast.AugAssign):
        return evaluate_augassign(expression, *common_params)
    elif isinstance(expression, ast.Call):
        # Function call -> we return the value of the function call
        return evaluate_call(expression, *common_params)
    elif isinstance(expression, ast.Constant):
        # Constant -> just return the value
        return expression.value
    elif isinstance(expression, ast.Tuple):
        return tuple((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, (ast.ListComp, ast.GeneratorExp)):
        return evaluate_listcomp(expression, *common_params)
    elif isinstance(expression, ast.DictComp):
        return evaluate_dictcomp(expression, *common_params)
    elif isinstance(expression, ast.SetComp):
        return evaluate_setcomp(expression, *common_params)
    elif isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, *common_params)
    elif isinstance(expression, ast.Starred):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.BoolOp):
        # Boolean operation -> evaluate the operation
        return evaluate_boolop(expression, *common_params)
    elif isinstance(expression, ast.Break):
        raise BreakException()
    elif isinstance(expression, ast.Continue):
        raise ContinueException()
    elif isinstance(expression, ast.BinOp):
        # Binary operation -> execute operation
        return evaluate_binop(expression, *common_params)
    elif isinstance(expression, ast.Compare):
        # Comparison -> evaluate the comparison
        return evaluate_condition(expression, *common_params)
    elif isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, *common_params)
    elif isinstance(expression, ast.FunctionDef):
        return evaluate_function_def(expression, *common_params)
    elif isinstance(expression, ast.Dict):
        # Dict -> evaluate all keys and values
        keys = (evaluate_ast(k, *common_params) for k in expression.keys)
        values = (evaluate_ast(v, *common_params) for v in expression.values)
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # Expression -> evaluate the content
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.For):
        # For loop -> execute the loop
        return evaluate_for(expression, *common_params)
    elif isinstance(expression, ast.FormattedValue):
        # Formatted value (part of f-string) -> evaluate the content and format it
        value = evaluate_ast(expression.value, *common_params)
        # Early return if no format spec
        if not expression.format_spec:
            return value
        # Apply format specification
        format_spec = evaluate_ast(expression.format_spec, *common_params)
        return format(value, format_spec)
    elif isinstance(expression, ast.If):
        # If -> execute the right branch
        return evaluate_if(expression, *common_params)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        return evaluate_ast(expression.value, *common_params)
    elif isinstance(expression, ast.JoinedStr):
        return "".join([str(evaluate_ast(v, *common_params)) for v in expression.values])
    elif isinstance(expression, ast.List):
        # List -> evaluate all elements
        return [evaluate_ast(elt, *common_params) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # Name -> pick up the value in the state
        return evaluate_name(expression, *common_params)
    elif isinstance(expression, ast.Subscript):
        # Subscript -> return the value of the indexing
        return evaluate_subscript(expression, *common_params)
    elif isinstance(expression, ast.IfExp):
        test_val = evaluate_ast(expression.test, *common_params)
        if test_val:
            return evaluate_ast(expression.body, *common_params)
        else:
            return evaluate_ast(expression.orelse, *common_params)
    elif isinstance(expression, ast.Attribute):
        return evaluate_attribute(expression, *common_params)
    elif isinstance(expression, ast.Slice):
        return slice(
            evaluate_ast(expression.lower, *common_params) if expression.lower is not None else None,
            evaluate_ast(expression.upper, *common_params) if expression.upper is not None else None,
            evaluate_ast(expression.step, *common_params) if expression.step is not None else None,
        )
    elif isinstance(expression, ast.While):
        return evaluate_while(expression, *common_params)
    elif isinstance(expression, (ast.Import, ast.ImportFrom)):
        return evaluate_import(expression, state, authorized_imports)
    elif isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, *common_params)
    elif isinstance(expression, ast.Try):
        return evaluate_try(expression, *common_params)
    elif isinstance(expression, ast.Raise):
        return evaluate_raise(expression, *common_params)
    elif isinstance(expression, ast.Assert):
        return evaluate_assert(expression, *common_params)
    elif isinstance(expression, ast.With):
        return evaluate_with(expression, *common_params)
    elif isinstance(expression, ast.Set):
        return set((evaluate_ast(elt, *common_params) for elt in expression.elts))
    elif isinstance(expression, ast.Return):
        raise ReturnException(evaluate_ast(expression.value, *common_params) if expression.value else None)
    elif isinstance(expression, ast.Pass):
        return None
    elif isinstance(expression, ast.Delete):
        return evaluate_delete(expression, *common_params)
    else:
        # For now we refuse anything else. Let's add things as we need them.
        raise InterpreterError(f"{expression.__class__.__name__} is not supported.")


class FinalAnswerException(Exception):
    def __init__(self, value):
        self.value = value


def evaluate_python_code(
    code: str,
    static_tools: Optional[Dict[str, Callable]] = None,
    custom_tools: Optional[Dict[str, Callable]] = None,
    state: Optional[Dict[str, Any]] = None,
    authorized_imports: List[str] = BASE_BUILTIN_MODULES,
    max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
    timeout_length: float = 10.0,
    set_timeout: bool = False,
):
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        static_tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. These can also be agents in a multiagent setting.
            These tools cannot be overwritten in the code: any assignment to their name will raise an error.
        custom_tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation.
            These tools can be overwritten in the code: any assignment to their name will overwrite them.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be
            updated by this function to contain all variables as they are evaluated.
            The print outputs will be stored in the state under the key "_print_outputs".
    """
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^\n"
            f"Error: {str(e)}"
        )

    # # Prevent running OOM-sensitive codes
    max_loop_depth = [0]
    count_loop_depth(expression, current_depth=0, max_depth=max_loop_depth)
    if max_loop_depth[0] > 1:
        raise InterpreterError("too many for loops - not running the code to prevent out-of-memory error")
    detect_dangerous_calls(expression, state)

    if state is None:
        state = {}
    static_tools = static_tools.copy() if static_tools is not None else {}
    custom_tools = custom_tools if custom_tools is not None else {}
    result = None
    state["_print_outputs"] = PrintContainer()
    state["_operations_count"] = {"counter": 0}

    if "final_answer" in static_tools:
        previous_final_answer = static_tools["final_answer"]

        def final_answer(answer):  # Using 'answer' as the argument like in the original function
            raise FinalAnswerException(previous_final_answer(answer))

        static_tools["final_answer"] = final_answer

    try:
        for node in expression.body:
            # timeout does not work in multi-threading...
            if set_timeout:
                result = timeout(timeout_length)(evaluate_ast)(node, state, static_tools, custom_tools, authorized_imports)
            else:
                result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]), max_length=max_print_outputs_length
        )
        is_final_answer = False
        return result, is_final_answer
    except FinalAnswerException as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]), max_length=max_print_outputs_length
        )
        is_final_answer = True
        return e.value, is_final_answer
    except Exception as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]), max_length=max_print_outputs_length
        )
        raise InterpreterError(
            f"Code execution failed at line '{ast.get_source_segment(code, node)}' due to: {type(e).__name__}: {e}"
        )


class PythonExecutor:
    pass


class LocalPythonExecutor(PythonExecutor):
    def __init__(
        self,
        additional_authorized_imports: List[str],
        max_print_outputs_length: Optional[int] = None,
        set_timeout: bool = False
    ):
        self.custom_tools = {}
        self.state = {}
        self.max_print_outputs_length = max_print_outputs_length
        if max_print_outputs_length is None:
            self.max_print_outputs_length = DEFAULT_MAX_LEN_OUTPUT
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        # TODO: assert self.authorized imports are all installed locally
        self.static_tools = None
        self.set_timeout = set_timeout

    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        output, is_final_answer = evaluate_python_code(
            code_action,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,
            state=self.state,
            authorized_imports=self.authorized_imports,
            max_print_outputs_length=self.max_print_outputs_length,
            set_timeout=self.set_timeout,
        )
        logs = str(self.state["_print_outputs"])
        return output, logs, is_final_answer

    def send_variables(self, variables: dict):
        self.state.update(variables)

    def send_tools(self, tools: Dict[str, Tool]):
        self.static_tools = {**tools, **BASE_PYTHON_TOOLS.copy()}


__all__ = ["evaluate_python_code", "LocalPythonExecutor"]
