import time
from datetime import datetime

while True:
    try:
        # Write to a heartbeat file every 5 seconds
        with open("/mnt/data/heartbeat.txt", "a") as heartbeat_file:
            heartbeat_file.write(f"Heartbeat at {datetime.now()}\n")
        print("Heartbeat written.")
    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
    
    time.sleep(5)  # Sleep for 5 seconds