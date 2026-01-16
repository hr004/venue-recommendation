#!/bin/bash

echo "Waiting for OpenSearch to start (waiting 10 seconds)"
sleep 10
INDEX_NAME="venue_event_history"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if index already exists
if curl -s -o /dev/null -w "%{http_code}" "localhost:9200/$INDEX_NAME" | grep -q "200"; then
    echo "Index $INDEX_NAME already exists. Skipping creation."
    exit 0
fi

echo "Creating index $INDEX_NAME"
SETTINGS_AND_MAPPINGS=$(cat "$SCRIPT_DIR/mappings.json")
RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "localhost:9200/$INDEX_NAME" -H 'Content-Type: application/json' -d"$SETTINGS_AND_MAPPINGS")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 201 ]; then
    echo "Index $INDEX_NAME created successfully"
else
    echo "Failed to create index $INDEX_NAME. HTTP code: $HTTP_CODE"
    echo "$RESPONSE" | head -n-1
    exit 1
fi