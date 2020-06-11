# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

"""Functions that check types."""


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False
