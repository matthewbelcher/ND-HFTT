import sys
from collections import deque
from copy import deepcopy
import math

MINLENGTH = 2
MAXLENGTH = 5
# CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CNH", "AUD", "CAD", "CHF", "HKD", "SGD"]
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CNH"]

def nCycles(num_currencies: int, minlength: int, maxlength: int) -> int:
    total = 0
    for i in range(minlength, min(num_currencies, maxlength) + 1):
        total += (math.factorial(i - 1) * math.factorial(num_currencies) // (math.factorial(num_currencies - i) * math.factorial(i)))
    return total

def reorder(cycle: list, pairs: dict) -> str:
    idx = 0
    id = len(pairs)
    for i, pair in enumerate(cycle):
        if pair < id:
            id = pair
            idx = i
    
    cycle = cycle[idx:] + cycle[:idx]
    return ','.join([str(i) for i in cycle])

def getCycles(currencies, pairs, minlength: int, maxlength: int) -> list[str]:
    cycles = deque([[i] for i in pairs])
    complete_cycles = set()

    while cycles:
        cycle = cycles.popleft()
        if len(cycle) > maxlength:
            continue
        elif len(cycle) >= minlength and pairs[cycle[0]][0] == pairs[cycle[-1]][1]:
            complete_cycles.add(reorder(cycle, pairs))
            continue
        else:
            for id, pair in pairs.items():
                if pair[0] != pairs[cycle[-1]][1] or id in cycle:
                    continue
                skip = False
                for pair_id in cycle:
                    if pair[0] == pairs[pair_id][0]:
                        skip = True
                        break
                if skip:
                    continue
                new_cycle = deepcopy(cycle)
                new_cycle.append(id)
                cycles.append(new_cycle)

    csv_cycles = []
    for cycle in complete_cycles:
        ids = [int(i) for i in cycle.split(",")]
        cycle_pairs = [pairs[i] for i in ids]
        pair_names = [currencies[i[0]] for i in cycle_pairs]
        pair_names.append(currencies[cycle_pairs[-1][1]])
        csv_cycles.append("/".join(pair_names))
    
    return csv_cycles


def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".csv"):
        print("USAGE: python3 generate_cycles.py <output.csv>")
        return 1
    csv = sys.argv[1]
    currency_map = {i: v for i, v in enumerate(CURRENCIES)}
    pairs = [(i, j) for j in currency_map for i in currency_map if j != i]
    pairs = {i: v for i, v in enumerate(pairs)}
    cycles = getCycles(currency_map, pairs, MINLENGTH, MAXLENGTH)
    print(f"Expected {nCycles(len(CURRENCIES), MINLENGTH, MAXLENGTH)} cycles. Found {len(cycles)} cycles.")
    with open(csv, "w") as f:
        for cycle in cycles:
            f.write(cycle + "\n")
    return 0

if __name__ == "__main__":
    main()
