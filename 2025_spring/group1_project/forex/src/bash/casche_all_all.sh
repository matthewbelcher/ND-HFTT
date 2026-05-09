#!/bin/bash

CYCLE_FILE="artifacts/cycles/cycles_2_5_usd_eur_jpy_gbp_cnh_aud_cad_chf_hkd_sgd.csv"
INPUT_DIR="data/filtered"
OUTPUT_PREFIX="casche_all_arbitrages_cycles_2_5_usd_eur_jpy_gbp_cnh_aud_cad_chf_hkd_sgd"

echo "Starting batch processing of arbitrage files..."
echo "Cycle file: $CYCLE_FILE"
echo "Input directory: $INPUT_DIR"
echo

for INPUT_FILE in "$INPUT_DIR"/*_filtered.csv; do
    BASENAME=$(basename "$INPUT_FILE")
    DATE_PART=${BASENAME%%_filtered.csv}
    OUTPUT_FILE="${OUTPUT_PREFIX}_${DATE_PART}.csv"

    echo "Processing: $BASENAME"
    echo " -> Output: $OUTPUT_FILE"
    ./build/bin/casche_all_arbitrages "$CYCLE_FILE" "$INPUT_FILE" "$OUTPUT_FILE"

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo " ✅ Success"
    else
        echo " ❌ Failed with exit code $STATUS"
    fi
    echo
done

echo "All done."
