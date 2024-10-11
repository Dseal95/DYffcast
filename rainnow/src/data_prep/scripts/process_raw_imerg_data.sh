#!/bin/bash

# example cmd: ./scripts/process_raw_imerg_data.sh "2020-01-01 00:00:00" "2023-12-31 23:59:59" False "logs/corrupt_files.txt" True

# check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <start_date> <end_date> <overwrite_flag> <corrupt_files_path> <delete_flag>"
    exit 1
fi

# Assign arguments to variables
start_date=$1
end_date=$2
overwrite_flag=$3
corrupt_files_path=$4
delete_flag=$5


echo -e "***** .sh INFO *****"
echo -e "--> Running <process_raw_imerg_data.py> then <process_corrupted_files.py> recursively."
echo -e "--> process_raw_imerg_data.py: start_date=$start_date, end_date=$end_date, overwrite=$overwrite_flag"
echo -e "--> process_corrupted_files.py: file_path=$corrupt_files_path, delete_files=$delete_flag\n"

# loop until the corrupt_files.txt file is empty.
while true; do
    python process_raw_imerg_data.py "$start_date" "$end_date" "$overwrite_flag"

    # check if corrupt_files.txt is empty.
    if [ ! -s "$corrupt_files_path" ]; then
        echo "No corrupt files found. Exiting."
        break
    fi
    python process_corrupted_files.py "$corrupt_files_path" "$delete_flag"
    echo -e "Deleted and redownloaded files"
done

echo -e "Processing $start_date to $end_date completed."
