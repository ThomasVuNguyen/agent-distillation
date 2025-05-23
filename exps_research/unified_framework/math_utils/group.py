import math
import random
import signal
from collections import defaultdict
from multiprocessing import Manager
from typing import Any, Dict, List, Literal

import numpy as np
from latex2sympy2 import latex2sympy
from sympy import latex, simplify

from .qwen_math_parser import extract_answer, strip_string


# Timeout exception
class TimeoutException(Exception):
    pass


# Signal handler for timeout
def timeout_handler(signum, frame):
    raise TimeoutException


manager = Manager()
shared_cache = manager.dict()

def memoized_canonical_form(expression: str, timeout_seconds: int = 3) -> str:
    """
    Compute a canonical form for a mathematical expression using sympy.
    Uses a shared cache across processes for memoization.

    Args:
        expression (str): A LaTeX-formatted mathematical expression.
        timeout_seconds (int): Timeout duration in seconds.

    Returns:
        str: The canonical form of the expression or the original expression as fallback.
    """
    # Check if the result is already cached
    if expression in shared_cache:
        return shared_cache[expression]

    try:
        # Set up the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        # Parse and simplify the mathematical expression
        parsed_expr = latex2sympy(expression)
        simplified_expr = simplify(parsed_expr)

        # Reset the alarm
        signal.alarm(0)

        canonical_form = latex(simplified_expr)  # Convert back to a string
        shared_cache[expression] = canonical_form  # Cache the result
        return canonical_form
    except TimeoutException:
        # Fallback: Use a stripped version of the input on timeout
        fallback = strip_string(expression)
        shared_cache[expression] = fallback  # Cache the fallback result
        return fallback
    except Exception:
        # Fallback: Use a stripped version of the input on other errors
        fallback = strip_string(expression)
        shared_cache[expression] = fallback  # Cache the fallback result
        return fallback
    finally:
        # Ensure the alarm is turned off
        signal.alarm(0)

def find_answer_with_largest_sum(answers: List[str], scores: List[float]) -> str:
    """
    Groups answers based on their canonical forms and finds the group with the largest sum of scores.

    Args:
        answers (list of str): A list of strings to be grouped.
        scores (list of float): A list of scores corresponding to each string.

    Returns:
        str: The string representing the group with the largest sum of scores.
    """
    if len(answers) == 0 or len(scores) == 0:
        raise ValueError("answers and scores cannot be empty")

    # Grouping using canonical forms
    canonical_groups = defaultdict(
        float
    )  # Stores cumulative scores for each canonical group
    canonical_to_original = {}  # Maps canonical form back to an original answer

    for answer, score in zip(answers, scores):
        # Compute the canonical form
        canonical_form = memoized_canonical_form(answer)

        # Aggregate scores and track the original answer
        canonical_groups[canonical_form] += score
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # # Find the canonical form with the largest cumulative score
    # max_canonical = max(canonical_groups, key=canonical_groups.get)
    # return canonical_to_original[max_canonical]

    max_sum = max(canonical_groups.values())
    max_canonical_forms = [
        cf for cf, total in canonical_groups.items() if total == max_sum
    ]
    # Sort to ensure deterministic tie-breaking
    max_canonical_forms.sort()
    max_canonical = max_canonical_forms[0]  # Choose the lexicographically smallest
    return canonical_to_original[max_canonical]


def find_majority_answer(answers: List[str]) -> str:
    """
    Groups answers based on their canonical forms and finds the group with the largest number of elements.
    In case of a tie, returns the first occurring group with the largest size.

    Args:
        answers (list of str): A list of strings to be grouped.

    Returns:
        str: The string representing the group with the largest number of elements.

    Example:
        answers = ["a", "b", "a", "c"]
        result = find_majority_answer(answers)
        # result would be "a" since "a" appears most frequently.
    """
    if len(answers) == 0:
        raise ValueError("answers cannot be empty")

    # Group answers using canonical forms
    canonical_groups = defaultdict(int)  # Count occurrences for each canonical form
    canonical_to_original = {}  # Map canonical form back to an original answer

    for answer in answers:
        # Compute the canonical form
        canonical_form = memoized_canonical_form(answer)

        # Increment count for the canonical form
        canonical_groups[canonical_form] += 1

        # Track the original answer for this canonical form
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    # # Find the canonical form with the largest count
    # max_count = max(canonical_groups.values())
    # for canonical_form, count in canonical_groups.items():
    #     if count == max_count:
    #         # Return the first occurring group in case of a tie
    #         return canonical_to_original[canonical_form]

    max_count = max(canonical_groups.values())
    max_canonical_forms = [
        cf for cf, count in canonical_groups.items() if count == max_count
    ]
    # Sort to ensure deterministic tie-breaking
    max_canonical_forms.sort()
    max_canonical = max_canonical_forms[0]  # Choose the lexicographically smallest
    return canonical_to_original[max_canonical]