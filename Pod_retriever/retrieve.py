import time
from datetime import datetime
import os

# Retrieve environment variables else default value
data_path = os.getenv('DATA_PATH', '/mnt/data')
model_path = os.getenv('MODEL_PATH', '/mnt/save-model')
volume_mount_path = os.getenv('VOLUME_MOUNT_PATH', '/mnt')

heartbeat_file_path = os.path.join(volume_mount_path, 'test.txt')


while True:
    try:
        # Write to a heartbeat file every 5 seconds
        with open(heartbeat_file_path, "a") as heartbeat_file:
            heartbeat_file.write(f"Heartbeat at {datetime.now()}\n")
        print("Heartbeat written.")
    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
    
    time.sleep(5)  # Sleep for 5 seconds