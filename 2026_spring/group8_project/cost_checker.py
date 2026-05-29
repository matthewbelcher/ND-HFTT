from data_loader import get_client

START = "2026-01-26T14:30:00"
END   = "2026-01-30T21:00:00"

client = get_client()

for schema, label in [("mbp-10", "Book"), ("trades", "Trades")]:
    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        schema=schema,
        symbols=["ES.c.0"],
        stype_in="continuous",
        start=START,
        end=END,
    )
    print(f"{label}: ${cost:.2f}")