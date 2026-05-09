#!/bin/bash

# Configurable timeout in seconds (default: 60 seconds)
RUN_TIME=${1:-400}

# Array to store PIDs
declare -a PIDS

# Function to clean data files and directories
clean_data() {
    echo "Cleaning up data files and directories..."
    
    # Remove CSV files
    rm -f cash_carry_opportunities.csv coinbase_trades.csv profits_by_contract.csv profits_by_date.csv
    
    # Remove directories
    rm -rf trade_data raw_responses binance_orderbook_data
    
    echo "Clean up completed."
}

# Function to clean up processes when the script is interrupted
cleanup() {
    echo -e "\nInterrupted! Cleaning up processes..."
    # Kill all processes in our PID array
    for pid in "${PIDS[@]}"; do
        if ps -p $pid > /dev/null; then
            kill $pid 2>/dev/null
        fi
    done
    exit 1
}

# Set up trap for CTRL+C (SIGINT)
trap cleanup SIGINT

# Check if clean argument was passed
if [ "$1" == "clean" ]; then
    clean_data
    exit 0
fi

echo "Starting data generation and binance order scripts in parallel..."
echo "They will run for $RUN_TIME seconds (or press CTRL+C to stop)..."

# Run the first script with error handling
python data_generation_futures.py &
PID1=$?
if [ $PID1 -eq 0 ]; then
    PID1=$!
    PIDS+=($PID1)
    echo "Started data_generation_futures.py with PID $PID1"
else
    echo "ERROR: Failed to start data_generation_futures.py"
fi

# Run binance order script with automatic input
# Create a named pipe for input
PIPE_NAME="/tmp/binance_input_pipe"
mkfifo $PIPE_NAME 2>/dev/null

# Start a background process to feed the input
(
    # Just press Enter to use the default BTCUSD
    echo ""
    # Keep the pipe open
    sleep $RUN_TIME
) > $PIPE_NAME &
INPUT_PID=$!

# Run the script with input from the pipe
cat $PIPE_NAME | python hftbinanceorder.py &
BINANCE_PID=$!
PIDS+=($BINANCE_PID)
echo "Started hftbinanceorder.py with PID $BINANCE_PID"

# Check if at least one script started successfully
if [ ${#PIDS[@]} -eq 0 ]; then
    echo "ERROR: No scripts were started successfully. Exiting."
    # Clean up the named pipe
    rm -f $PIPE_NAME
    if ps -p $INPUT_PID > /dev/null; then
        kill $INPUT_PID 2>/dev/null
    fi
    exit 1
fi

# Sleep for the specified amount of time
echo "Running for $RUN_TIME seconds..."
sleep $RUN_TIME

# Kill the processes after the time is up
echo "Time's up! Terminating parallel processes..."
for pid in "${PIDS[@]}"; do
    if ps -p $pid > /dev/null; then
        echo "Terminating process with PID $pid"
        kill $pid 2>/dev/null
    fi
done

# Kill the input process if it's still running
if ps -p $INPUT_PID > /dev/null; then
    kill $INPUT_PID 2>/dev/null
fi

# Clean up the named pipe
rm -f $PIPE_NAME

# Wait for the processes to be killed properly
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null
done

echo "Parallel execution completed after $RUN_TIME seconds."
echo "Starting process.py..."

# Run process.py after the time-limited parallel execution
python process.py
PROCESS_RESULT=$?

# Check if process.py completed successfully
if [ $PROCESS_RESULT -eq 0 ]; then
    echo "process.py completed successfully."
    echo "Starting arb.py..."
    
    # Run arb.py only after process.py completes
    python arb.py
    ARB_RESULT=$?
    
    if [ $ARB_RESULT -eq 0 ]; then
        echo "arb.py completed successfully."
        echo "All tasks completed."
    else
        echo "ERROR: arb.py failed with exit code $ARB_RESULT."
        exit 1
    fi
else
    echo "ERROR: process.py failed with exit code $PROCESS_RESULT. Not proceeding to arb.py."
    exit 1
fi