"""
Utitlity functions for optimal asset allocation
"""
import itertools
import time
import datetime as dt
from functools import wraps

TimeFormat = "%Y-%m-%d %H:%M:%S"


def now_str() -> str:
    return f'{dt.datetime.now().strftime(TimeFormat)}'


def make_pct_tuple_general(incrmnt: float, size: int) -> [tuple]:
    """ Generate tuples of percent values at fixed increments
    @input: increment - increment between percent values
    @input: size - number of values in the tuple
    @return: list of tuples

    FYI: IMPORTANT Operations are done on integers to avoid rounding errors (e.g. >= 1.0)
    """
    incrment_int = int(incrmnt)  # convert to integer
    nb_pct = int(100 / incrment_int) - 1
    # Because we have size items in the tuple and each of them is at least increment, then no item can be greater than
    # 100 - (size - 1) * incrment_int
    nb_pct -= size - 1
    pct_val_lst = [incrment_int * (1 + x) for x in range(nb_pct + 1)]
    # print(f"pct_val_lst: {printable_float_list(pct_val_lst)}")
    # get the list, since this is an iterator, it returns empty second call
    step1_lst = list(itertools.product(pct_val_lst, repeat=size - 1))
    pct_lst = []
    # Add the last value to each tuple so that the sum of the tuple is 1.0
    for tpl in list(step1_lst):
        tpl_sum = sum(tpl)
        if tpl_sum >= 100:
            continue
        else:
            # print(printable_float_list([*tpl, 1.0 - tpl_sum]))
            full_tpl = [*tpl, 100 - tpl_sum]
            pct_lst.append([x * 0.01 for x in full_tpl])  # get back to float values
    return pct_lst


def is_pct_ok(x: float) -> bool:
    """ Return True if x is a valid percentage """
    return 0.0 <= x <= 100.0


def make_pct_permutations(allocation_lst: [float], new_pct_increment: float) -> [tuple]:
    """ generate a new list of allocations by generating all permuations where new_pct_increment is added to one of
    the elements in allocation_lst and subtracted from another
    @input: allocation_lst - list of allocations
    @input: new_pct_increment - new increment to add to one element and subtract from another
    @return: list of allocations tuples

    Note each allocation must be between 0.0 and 100
    Note that the result list includes the original allocation - so that it can be compared
    """
    result_lst = [tuple(allocation_lst)]
    for i in range(len(allocation_lst)):
        for j in range(i+1, len(allocation_lst)):
            new_lst = allocation_lst.copy()
            new_lst[i] += new_pct_increment
            new_lst[j] -= new_pct_increment
            if is_pct_ok(new_lst[i]) and is_pct_ok(new_lst[j]):
                result_lst.append(tuple(new_lst))
            new_lst = allocation_lst.copy()
            new_lst[i] -= new_pct_increment
            new_lst[j] += new_pct_increment
            if is_pct_ok(new_lst[i]) and is_pct_ok(new_lst[j]):
                result_lst.append(tuple(new_lst))
    return result_lst


def make_allocation_str(in_lst: [float]) -> str:
    """ Return a string representation of a list of percentages with 2 decimal -separated by '-' """
    return "-".join([f"{100 * x:.2f}" for x in in_lst])


def timeit(func):
    """ Decorator to time a function """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        # start the timer
        start_time = time.time()
        # call the decorated function
        result = func(*args, **kwargs)
        # remeasure the time
        end_time = time.time()
        # compute the elapsed time and print it
        execution_time = end_time - start_time
        exec_time = str(dt.timedelta(seconds=execution_time))
        # print(f'Function {func.__name__} with args:{args} kwargs:{kwargs} -> execution time {execution_time:,
        # .3f} seconds')
        print(f'Function {func.__name__} -> execution time(H:M:S): {exec_time} at {now_str()}')
        # return the result of the decorated function execution
        return result

    # return reference to the wrapper function
    return timeit_wrapper


def sum_list_multiply(lst1: [float], lst2: [float]) -> [float]:
    """ Multiply 2 lists element by element and return the sum of the resulting list
    @deprecated: use np.dot instead"""
    assert len(lst1) == len(
            lst2), f"Lists must have the same length {len(lst1)} != {len(lst2)}\nlst1: {lst1}\nlst2: {lst2}"
    return sum(x * y for x, y in zip(lst1, lst2))
