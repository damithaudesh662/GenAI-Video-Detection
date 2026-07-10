#!/bin/bash

# Initialize counter
count=1

# Loop through all .mp4 files in alphabetical order
for file in *.mp4; do
    mv "$file" "${count}.mp4"
    echo "Renamed $file -> ${count}.mp4"
    count=$((count + 1))
done

echo "Renaming complete! Total files renamed: $((count-1))"