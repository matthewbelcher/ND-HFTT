import sys
import math

MINLENGTH = 2
MAXLENGTH = 5
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CNH", "AUD", "CAD", "CHF", "HKD", "SGD"]
# CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CNH"]

def nCycles(num_currencies: int, minlength: int, maxlength: int) -> int:
    total = 0
    for i in range(minlength, min(num_currencies, maxlength) + 1):
        total += (math.factorial(i - 1) * math.factorial(num_currencies) //
                  (math.factorial(num_currencies - i) * math.factorial(i)))
    return total

def getCycles(currency_list, minlength, maxlength):
    num_currencies = len(currency_list)
    currency_ids = list(range(num_currencies))
    id_to_name = {i: name for i, name in enumerate(currency_list)}

    # Build all directed edges
    edges = {i: [j for j in currency_ids if j != i] for i in currency_ids}

    result_set = set()

    def dfs(path, visited_from):
        current = path[-1]
        if len(path) > maxlength:
            return

        for neighbor in edges[current]:
            if neighbor == path[0] and minlength <= len(path) <= maxlength:
                # Found a valid cycle
                # Normalize to avoid duplicates (start at lowest ID)
                min_idx = path.index(min(path))
                normalized = tuple(path[min_idx:] + path[:min_idx])
                result_set.add(normalized)
                continue

            if neighbor in path:
                continue  # No node reuse

            if neighbor in visited_from:
                continue  # Reuse of source currency not allowed

            path.append(neighbor)
            visited_from.add(current)
            dfs(path, visited_from)
            visited_from.remove(current)
            path.pop()

    for start in currency_ids:
        dfs([start], set())

    # Convert to currency strings
    final_cycles = []
    for cycle in result_set:
        names = [id_to_name[i] for i in cycle] + [id_to_name[cycle[0]]]
        final_cycles.append("/".join(names))

    return final_cycles

def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".csv"):
        print("USAGE: python3 generate_cycles.py <output.csv>")
        return 1

    out_csv = sys.argv[1]
    cycles = getCycles(CURRENCIES, MINLENGTH, MAXLENGTH)
    print(f"Expected ~{nCycles(len(CURRENCIES), MINLENGTH, MAXLENGTH)} cycles.")
    print(f"Found {len(cycles)} unique cycles.")
    
    with open(out_csv, "w") as f:
        for cycle in cycles:
            f.write(cycle + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

