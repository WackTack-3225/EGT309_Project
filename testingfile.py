import os

# Ensure the directory exists
os.makedirs("/mnt/data", exist_ok=True)

# Write data to the file
data = "This is some datass to store in the PV."

with open("/mnt/data/data.txt", "w") as file:
    file.write(data)

print("Data stored successfully.")