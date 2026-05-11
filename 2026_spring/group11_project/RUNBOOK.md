Project Proposal Link

Make sure to download the necessary packages to run. I'll make a requirements.txt file if it gets too complicated LOL. 

https://docs.google.com/document/d/1fv4EwgSGJBVYm2eQQ-imvPDGo6K_oJuhD2e9cFw-FEM/edit?tab=t.adubfw6c1j5a

## Build

```bash
cd collector && make
```

## Run (foreground)

```bash
# Basic — data goes to ../data/, results to ../results/
./build/collector <kalshi_key_id> <kalshi_pem_path> <coinbase_json_path> [rawdata_dir] [qty] [obi_thr] [max_hold_s] [btc_cancel]

# Example with defaults (qty=10, obi_thr=0.05, max_hold=30s, btc_cancel=0.0001)
cd collector
./build/collector <KALSHI_KEY_ID> ../secrets/TestExample1.pem ../secrets/cdp_api_key.json ../data
```

## Run (background / overnight collection)

```bash
cd collector
nohup ./build/collector <KALSHI_KEY_ID> ../secrets/TestExample1.pem ../secrets/cdp_api_key.json ../data \
  > ../collector.log 2>&1 & echo $! > ../collector.pid

# Stop it later:
kill $(cat ../collector.pid)
```

## Market maker parameters

| Arg | Default | Meaning |
|-----|---------|---------|
| qty | 10 | Contracts per quote |
| obi_thr | 0.05 | OBI threshold to post (range −1..1) |
| max_hold_s | 30 | Seconds before forced taker exit |
| btc_cancel | 0.0001 | \|btc_mom_10s\| above which to cancel quote |

Example with custom params:
```bash
./build/collector <key_id> <pem> <cb_json> ../data 10 0.05 30 0.0001
```

## Market maker results

After each 15-min session ends the MM writes a JSON summary to:

```
results/mm_<ticker>_<unix_timestamp>.json
```

The `results/` directory is created automatically relative to wherever the binary is run from (i.e. `collector/results/` when run from `collector/`). Each file contains per-trade PnL, fill prices, hold times, and aggregate stats (win rate, gross/net PnL, fees paid).

Drive where I will be uploading all of the completed full data eventually:
https://drive.google.com/drive/u/0/folders/0AFg1fGw1tSsRUk9PVA 

I'll probably still leave some in the github folder though for ease of access

environment_clean is my full like "trading" environment

environment.yml is the core essentials to run the C++ part of the system. It's able to be run on like another server. We can use pandas on our own computer. 


I explored a bit on the effects of the microprice and I don't think it's particularly releveant here because there is too much noise and each tick is too big relative to the total possible price levels. Can't find deep signals. 


4/7/2026: made some more edits in the file structure





<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/bc9eb6c0-ac1a-4ba3-a2fc-ad8ed8d11184" />


<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/4cbeb46f-78d1-4530-ac13-31432c4a0fde" />




