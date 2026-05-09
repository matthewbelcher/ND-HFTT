# Database File:  
**kalshi_orderbooks.db**: A standard SQLite database file storing all the collected data.  
# Tables:  
## markets  
**Purpose**: Stores metadata about each market you've encountered or tracked.  

**Columns**:  
* market_ticker (TEXT, PRIMARY KEY): The unique identifier for the market (e.g., KXBTC-25APR1617-B94500). Links this table to the others.  
* asset (TEXT): The underlying asset (e.g., "BTC", "ETH").  
* series_ticker (TEXT): The series the market belongs to (e.g., "KXBTC").  
* event_ticker (TEXT): The specific event the market is part of (e.g., "KXBTC-25APR1617").  
* description (TEXT): A human-readable description or title of the market.  
* last_updated (TIMESTAMP): When this market's information was last inserted or updated in this table.

**Role in Reconstruction**: Primarily provides context. You use the market_ticker from here to filter data in the other tables.  

## orderbook_snapshots
**Purpose**: Stores a complete snapshot of the order book at a specific moment. This happens when you first subscribe to a market or if the tracker resubscribes (e.g., due to a detected sequence gap).  

**Columns**:  
* id (INTEGER, PRIMARY KEY): Unique ID for the snapshot record.  
* market_ticker (TEXT, FOREIGN KEY): Links to the markets table.  
* timestamp (TIMESTAMP): The exact UTC time when the tracker script received and saved this snapshot.  
* snapshot_data (TEXT): A JSON string containing the full order book state at that timestamp. This JSON typically has keys like "yes" and "no", each mapping to a list of [price, size] pairs. Example snapshot_data content (simplified): {"yes": [[95, 100], [94, 50]], "no": [[5, 200], [6, 150]]}

**Role in Reconstruction**: This is your starting point. To reconstruct the book at time T, you need the most recent snapshot for your target market_ticker that has a timestamp less than or equal to T. You parse the snapshot_data JSON to get the initial state of 'yes' and 'no' orders.  
(Potential Improvement: The current schema doesn't store the seq number associated with the snapshot itself, which could be useful for verifying the first delta applied.)  

## orderbook_deltas
**Purpose**: Stores individual changes (deltas) to the order book after an initial snapshot has been received. Each row represents a single modification. 

**Columns**:  
* id (INTEGER, PRIMARY KEY): Unique ID for the delta record.  
* market_ticker (TEXT, FOREIGN KEY): Links to the markets table.  
* timestamp (TIMESTAMP): The exact UTC time when the tracker script received and saved this delta.  
* price (INTEGER): The price level (in cents) where the change occurred.  
* delta (INTEGER): The change in size at that price level. Positive means contracts were added, negative means contracts were removed. This is NOT the absolute size.  
* side (TEXT): Indicates which side of the book changed ('yes' or 'no').  
* seq (INTEGER): The sequence number provided by the Kalshi API for this delta message. Essential for ordering updates correctly and detecting gaps.  

**Role in Reconstruction**: These are the updates applied sequentially to the initial snapshot. After getting the starting book from the snapshot, you gather all deltas for the market_ticker that have a timestamp greater than the snapshot's timestamp and less than or equal to your target time T. You must apply these deltas in strict seq order.  

# How to Recreate the Order Book at Time T:  
* **Select Market**: Choose the market_ticker you want to reconstruct.  
* **Find Base Snapshot**: Query orderbook_snapshots to find the single row for your market_ticker with the latest timestamp <= T.  
* **Load Initial Book**: If a snapshot is found, parse its snapshot_data JSON into an in-memory representation (e.g., two dictionaries, one for 'yes' bids, one for 'no' bids, mapping price to size). Let the snapshot's timestamp be snapshot_timestamp. If no snapshot is found before T, you cannot reconstruct the book accurately for time T.  
* **Gather Deltas**: Query orderbook_deltas to select all rows for your market_ticker where timestamp > snapshot_timestamp AND timestamp <= T. Order these results strictly by seq ASC.  
* **Apply Deltas Sequentially**: Iterate through the ordered deltas:  
* **For each delta (price, delta, side, seq)**:  
  * Check if the seq is the expected next number (previous seq + 1). If not, log a warning about a potential data gap.  
  * Look up the price in the appropriate side ('yes' or 'no') of your in-memory book.  
  * If the price exists, add the delta to its current size.  
  * If the price doesn't exist and delta is positive, add the price level with the delta as its size.  
  * If a price level's size becomes <= 0 after applying the delta, remove that price level from your in-memory book. 
* **Final State**: After applying all the selected deltas, your in-memory representation holds the reconstructed order book state for the market_ticker at time T.  
