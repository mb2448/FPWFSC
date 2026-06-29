#!/bin/bash

WATCH_DIR="~/Downloads/temp_files"
REMOTE="jarenashcraft@dalinar.farrealmfarm.org:/home/jarenashcraft/Downloads/"

inotifywait -m -e close_write --format '%f' "$WATCH_DIR" | while read -r filename; do
    echo "New file detected: $filename"
    scp "${WATCH_DIR}/${filename}" "$REMOTE" && echo "Transferred: $filename" || echo "Transfer failed: $filename"
done
