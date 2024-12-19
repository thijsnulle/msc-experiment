#!/bin/bash

# Set the number of retries
retries=3

# Loop until we succeed or reach the maximum retries
for (( i=1; i<=$retries; i++ )); do
  # Run the script
  python3 monte_carlo_simulation.py

  # Check if the script exited successfully (exit code 0)
  if [[ $? -eq 0 ]]; then
    echo "Script execution successful."
    break;  # Exit the loop on success
  else
    echo "Script failed on attempt $i. Retrying..."
  fi
done

# If all retries fail, print an error message
if [[ $i -eq $((retries + 1)) ]]; then
  echo "Script failed after $retries retries."
fi
