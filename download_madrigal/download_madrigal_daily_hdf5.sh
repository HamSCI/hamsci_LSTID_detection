#!/bin/bash

# Define the start and end datesc (month - day - year)
startDate="12/24/2019"
endDate="12/28/2019"

# Convert dates to a format suitable for looping
currentDate=$(date -d "$startDate" +%Y%m%d)
endDateFormatted=$(date -d "$endDate" +%Y%m%d)

# Loop through each day
while [[ "$currentDate" -le "$endDateFormatted" ]]; do
    # Format current date for output
    formattedStartDate=$(date -d "$currentDate" +"%m/%d/%Y")

    # The end date will be the same as the start date for daily intervals
    formattedEndDate=$formattedStartDate

    echo "Running globalDownload.py for dates: $formattedStartDate to $formattedEndDate"

    globalDownload.py --verbose \
                      --url=http://cedar.openmadrigal.org \
                      --outputDir=../data/madrigal \
                      --user_fullname="" \
                      --user_email=example@email.com \
                      --user_affiliation="" \
                      --format="hdf5" \
                      --startDate="$formattedStartDate" \
                      --endDate="$formattedEndDate" \
                      --inst=8308

    echo "Completed download for date: $formattedStartDate"

    # Move to the next day
    currentDate=$(date -d "$currentDate + 1 day" +%Y%m%d)
done
