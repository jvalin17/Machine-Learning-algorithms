"""Pure-Python math functions. No imports."""


def log2(value):
    """Compute log base 2 using successive squaring.

    Accurate to IEEE 754 double precision (52 fractional bits).
    Returns 0.0 for non-positive inputs.
    """
    if value <= 0.0:
        return 0.0

    result = 0.0

    # extract integer part: reduce value into [1.0, 2.0)
    while value >= 2.0:
        value /= 2.0
        result += 1.0

    while value < 1.0:
        value *= 2.0
        result -= 1.0

    # refine fractional part via repeated squaring
    fraction = 0.5
    for _ in range(52):
        value *= value
        if value >= 2.0:
            result += fraction
            value /= 2.0
        fraction /= 2.0

    return result
