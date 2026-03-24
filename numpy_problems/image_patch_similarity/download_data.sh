#!/bin/bash
DATA_DIR="misc_data"
file="$DATA_DIR/images_path.txt"

while IFS= read -r url || [[ -n "$url" ]]; do
    url="${url//$'\r'/}"   
    # wget -c -P "$DATA_DIR" "$url" 
    curl -L $url -o "$DATA_DIR/$(basename "$url")"        
done < $file