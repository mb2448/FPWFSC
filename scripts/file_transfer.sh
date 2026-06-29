#!/bin/bash

WATCH_DIR="/Users/kaladin/Downloads/temp_files"
REMOTE="jarenashcraft@dalinar.farrealmfarm.org:/home/jarenashcraft/Downloads/temp_remote"
SENT_LOG="/tmp/sent_files.txt"
ERROR_LOG="/tmp/transfer_errors.txt"
MAX_RETRIES=3
EXPECT_SCRIPT="$(dirname "$0")/transfer.exp"

# Check expect is available
if ! command -v expect &> /dev/null; then
    echo "expect is not installed. Exiting."
    exit 1
fi

# Check expect script exists
if [ ! -f "$EXPECT_SCRIPT" ]; then
    echo "transfer.exp not found at $EXPECT_SCRIPT. Exiting."
    exit 1
fi

read -rsp "Enter SSH password: " SSH_PASS
echo

touch "$SENT_LOG"

expect transfer.exp "$SSH_PASS" "$filepath" "$REMOTE"

while true; do
    find "$WATCH_DIR" -maxdepth 1 -type f | while read -r filepath; do
        filename=$(basename "$filepath")

        if ! grep -qF "$filename" "$SENT_LOG"; then
            sleep 2

            success=false
            for attempt in $(seq 1 $MAX_RETRIES); do
                if expect "$EXPECT_SCRIPT" "$SSH_PASS" "$filepath" "$REMOTE"; then
                    echo "$filename" >> "$SENT_LOG"
                    echo "$(date): Sent $filename on attempt $attempt"
                    success=true
                    break
                else
                    echo "$(date): Attempt $attempt failed for $filename" >&2
                    sleep 5
                fi
            done

            if [ "$success" = false ]; then
                echo "$(date): FAILED after $MAX_RETRIES attempts: $filename" | tee -a "$ERROR_LOG"
            fi
        fi
    done

    sleep 2
done
