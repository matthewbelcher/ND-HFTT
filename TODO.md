Source Data
- Tick by Tick Market Data for Common Indexes (VOO, SPY, VFH, etc.) (Zach) [x] 
    - Re-activating my WRDS account, but should get access soon and be able to pull this data
- Historical Fed Funds Futures Data (Zach) [x]
    - Might be on WRDS, but will need to find an alternative source if not
- Historical FOMC Press Releases (Zach) [x]
    - Downloaded Jan 2006 - Present (Can go back further if necessary)

Data Analysis
- Determine the statistically significant features of the FOMC Press Releases []
- Formulate a signal based on market context, rate change expectations, and press release content []
- Determine the minimum execution time to profit []
    - Find the expected return of various reaction times
- Determine the exit of the position []

Execution
- Continually poll FOMC website for press release (Zach) [x]
    - While loop that begins making network calls 5 seconds before release and continues until download is succesful. Otherwise, short wait. 
    - Use libcurl to download raw HTML content
- Extract relevant variables from the raw report (Zach) [x]
    - Linear scan of content to find actually relevant paragraph
    - Parse target rates at a minimum, but research to find other common impactful phrases regarding inflation, jobs, future changes, etc.
- Transform press release content into normalized variables []
- Calculate signal []
- Mock order entry system for timing []
    - Determine the time from when the press release becomes available till the order is sent using the existing system
    - Compare this time to the minimum execution time determined above