# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Functions that check types."""

from typing import Any


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_float(x):
    return isinstance(x, float)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_positive_float(x):
    return is_float(x) and x > 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


def validate_bool(x: Any, field_name: str) -> None:
    """Validate x is of type boolean."""

    if not is_bool(x):
        raise TypeError(
            f"Expected type of boolean for {field_name} but got type={type(x).__name__}"
        )


def validate_optional(x: Any, field_name: str, *expected_type: Any) -> None:
    """Validate type of x is the same as the type passed in expected_type."""

    if x is not None and not isinstance(x, expected_type):
        expected_type_name = ','.join(x.__name__ for x in expected_type)

        raise TypeError(
            f"Expected None or type(s) of {expected_type_name} for"
            f" {field_name} but got type={type(x).__name__}"
        )


def validate_positive_int(x: Any, field_name: str) -> None:
    """Validate x is a positive integer."""

    if not is_int(x):
        raise TypeError(
            f"Expected a positive integer for {field_name} but got"
            f" type {type(x).__name__} with value {x}."
        )
    if x <= 0:
        raise ValueError(f"Expected positive integer for {field_name} but got {x}.")


def validate_positive_float(x: Any, field_name: str) -> None:
    """Validate x is a positive floating point number."""

    if not is_float(x) and not is_int(x):
        raise TypeError(
            f"Expected a positive float for {field_name} but got"
            f" type {type(x).__name__} with value {x}.",
        )
    if x <= 0:
        raise ValueError(f"Expected positive float for {field_name} but got {x}")


def validate_float_range(
    x: Any, field_name: str, min_val: float, max_val: float, range_inclusive: bool
) -> None:
    """Validate that x is a float within [min_val, max_val] or (min_val, max_val)."""

    if not isinstance(x, (float, int)):
        raise TypeError(f"{field_name} must be a float, but got {type(x).__name__}")

    if range_inclusive:
        if not (min_val <= x <= max_val):
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val}, inclusive."
            )
    else:
        if not (min_val < x < max_val):
            raise ValueError(
                f"{field_name} must be strictly between {min_val} and {max_val}."
            )
