# retrieve_data.py
try:
    with open("/mnt/data/data.txt", "r") as file:
        data = file.read()
    print(f"Retrieved data: {data}")
except FileNotFoundError:
    print("File not found. Please ensure the file exists in the specified path.")