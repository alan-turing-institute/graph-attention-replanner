import itertools
import numpy as np


def generate_bool_table(rows, cols, collab=False, num_task=4, discretize_level=3):
    if cols > rows:
        tables = generate_bool_table_col_filter_first(
            rows,
            cols,
            collab=collab,
            num_task=num_task,
            discretize_level=discretize_level,
        )
    else:
        # TODO: Implement row first filter
        tables = generate_bool_table_row_filter_first(
            rows,
            cols,
            collab=collab,
            num_task=num_task,
            discretize_level=discretize_level,
        )
    return tables


def generate_bool_table_col_filter_first(
    rows, cols, collab=False, num_task=4, discretize_level=3
):
    def check_row(seq):
        # Check each row has at least one a True value
        if any(seq):
            return True
        return False

    def check_row_table_format(arr):
        # Use any() along axis 1 (rows) to check if there's at least one True in each row
        rows_with_true = np.any(arr, axis=1)
        # Check if all rows have at least one True
        all_rows_have_true = np.all(rows_with_true)
        return all_rows_have_true

    def check_each_task_is_attempted(arr, num_task=4, discretize_level=3):
        for t in range(num_task):
            count = 0
            for i in range(discretize_level):
                count += np.sum(arr[i, :])
            if count != discretize_level:
                return False
        return True

    def check_col(seq, collab=False):
        # Check if each column has one True value = No Collab Case
        if not collab:
            return sum(seq) == 1
        # Check if each column at least one True value = Collab Case
        return sum(seq) >= 1

    # Generate all possible combinations
    possible_cols = []
    for seq in itertools.product([True, False], repeat=rows):
        if check_col(seq, collab=collab):
            possible_cols.append(list(seq))

    indices = list(range(len(possible_cols)))
    all_possible_arrangement = list(itertools.product(indices, repeat=cols))
    # print(len(all_possible_arrangement))

    tables = []
    for arrangement in all_possible_arrangement:
        table = []
        for i in arrangement:
            table.append(possible_cols[i])
        table = np.array(table).transpose()
        tables.append(table)

    tables = np.array(tables, dtype=bool)
    return tables


def generate_bool_table_row_filter_first(
    rows, cols, collab=False, num_task=4, discretize_level=3
):
    return


def convert_bool_to_time_table(boolean_table, discretize_level=None):
    rows = len(boolean_table)
    cols = len(boolean_table[0])

    def generate_row_possibilities(row):
        true_count = sum(row)
        if true_count == 0:
            return [list(0 for _ in range(cols))]

        possibilities = []
        for perm in list(itertools.permutations(range(1, true_count + 1))):
            perm = list(perm)
            new_row = []
            perm_index = 0
            for cell in row:
                if cell:
                    new_row.append(perm[perm_index])
                    perm_index += 1
                else:
                    new_row.append(0)
            possibilities.append(list(new_row))
        return possibilities

    row_possibilities = [generate_row_possibilities(row) for row in boolean_table]

    def generate_tables(current_table, row_index):
        if row_index == rows:
            yield current_table
            return

        for row in row_possibilities[row_index]:
            yield from generate_tables(current_table + [row], row_index + 1)

    r = np.array(list(generate_tables([], 0)))
    return r


def generate_bool_row_possibilities(col):
    return list(itertools.product([True, False], repeat=col))


def generate_int_row_possibilities(row, col):
    true_count = sum(row)
    if true_count == 0:
        return [list(0 for _ in range(col))]

    possibilities = []
    for perm in list(itertools.permutations(range(1, true_count + 1))):
        perm = list(perm)
        new_row = []
        perm_index = 0
        for cell in row:
            if cell:
                new_row.append(perm[perm_index])
                perm_index += 1
            else:
                new_row.append(0)
        possibilities.append(list(new_row))
    return possibilities
