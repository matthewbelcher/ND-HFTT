import os

folder_path = "../../bigdata"
print("Scanning:", folder_path)
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv"):
            path = os.path.join(root, file)
            print("Found file:", path)

            try:
                with open(path, 'r') as f:
                    line_count = sum(1 for _ in f)
                
                if line_count < 4:
                    print(f"Deleting {path} (only {line_count} lines)")
                    os.remove(path)
                    
            except Exception as e:
                print(f"Error reading {path}: {e}")