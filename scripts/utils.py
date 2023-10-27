from prettytable import PrettyTable
def pretty_table(dct, name):
    print(F'\n--- Results for {name}')
    table = PrettyTable(dct.keys())
    for row in zip(*[dct[k] for k in dct]):
      table.add_row(row)
    print(table)
