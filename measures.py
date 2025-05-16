import numpy as np


def apply_smoothing(n, x, y, z, smooth_counts):
    n += 4 * smooth_counts
    x += 2 * smooth_counts
    y += 2 * smooth_counts
    z += smooth_counts
    return n, x, y, z


# Confidence
def confidence(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    value = z / x
    if np.any((value < 0) | (value > 1)):
        print(f"Illegal confidence values detected.")
    return value


# Lift
def lift(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    value = n * z / (x * y)
    if np.any(value < 0):
        print(f"Illegal lift values detected.")
    return value


# Cosine
def cosine(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    value = z / np.sqrt(x * y)
    if np.any((value < 0) | (value > 1)):
        print(f"Illegal cosine values detected.")
    return value


# Phi
def phi(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    x_not = n - x
    y_not = n - y
    z_not = x_not - (y - z)
    value = (n * z - x * y) / np.sqrt(x * y * x_not * y_not)
    if np.any((value < -1) | (value > 1)):
        print(f"Illegal phi values detected.")
    return value


# Kruskal
def kruskal(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    x_not = n - x
    y_not = n - y
    max_y = np.maximum(y, y_not)
    value = (np.maximum(z, x - z) + np.maximum(y - z, x_not - (y - z)) - max_y) / (
        n - max_y
    )
    if np.any((value < 0) | (value > 1)):
        print(f"Illegal kruskal values detected.")
    return value


# YuleQ
def yuleQ(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    x_not = n - x
    y_not = n - y
    z_not = x_not - (y - z)
    OR = z * z_not / ((x - z) * (y - z))
    value = (OR - 1) / (OR + 1)
    if np.any((value < -1) | (value > 1)):
        print(f"Illegal YuleQ values detected.")
    return value


# Added Value
def added_value(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    value = z / x - y / n
    return value


# Certainty
def certainty(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    value1 = (z / x - y / n) / (1 - y / n)
    value2 = (z / y - x / n) / (1 - x / n)
    value = np.maximum(value1, value2)
    if np.any((value < -1) | (value > 1)):
        print(f"Illegal certainty values detected.")
    return value


# Support
def support(x, y, z, n, smooth_counts=0.0):
    if smooth_counts > 0:
        n, x, y, z = apply_smoothing(n, x, y, z, smooth_counts)
    value = z / n
    if np.any((value < 0) | (value > 1)):
        print(f"Illegal support values detected.")
    return value


# Reverse Support
def revsupport(x, y, z, n, smooth_counts=0.0):
    value = 1 - support(x, y, z, n, smooth_counts)
    if np.any((value < 0) | (value > 1)):
        print(f"Illegal reverse support values detected.")
    return value
