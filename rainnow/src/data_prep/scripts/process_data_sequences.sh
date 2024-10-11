#!/bin/bash

# ** run this from <data_prep> root DIR, 1 up from this directory.
# example cmd: ./scripts/process_data_sequences.sh '2020-01-01 00:00:00' '2024-01-01 00:00:00' scripts/sequence_grid_ij_config.txt


# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <start_date> <end_date> <config_file>"
    exit 1
fi

# Assign arguments to variables
start_date=$1
end_date=$2
config_file=$3

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Config file not found: $config_file"
    exit 1
fi

echo -e "***** .sh INFO *****"
echo -e "--> Reading config file and processing (i,j) pairs."

# Read the config file and loop over each line
IFS=$'\n' read -d '' -r -a lines < "$config_file"
for line in "${lines[@]}"; do
    # Extract i and j from the line
    IFS=',' read -r i j <<< "$line"
    # Trim leading/trailing whitespace from i and j
    i=$(echo "$i" | xargs)
    j=$(echo "$j" | xargs)
    echo -e "\n*** Running for i=$i, j=$j ***"
    python process_data_sequences.py "$start_date" "$end_date" "$i" "$j"
done

echo -e "Processing sequences completed."
