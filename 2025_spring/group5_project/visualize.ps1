# This script will run each Python file in the .\scripts\viz directory

# Get the path to the target directory
$scriptDir = ".\scripts\viz"

# Check if the directory exists
if (-Not (Test-Path -Path $scriptDir)) {
    Write-Host "Directory $scriptDir does not exist."
    exit 1
}

# Get all Python files in the directory
$pythonFiles = Get-ChildItem -Path $scriptDir -Filter *.py

# Check if there are any Python files
if ($pythonFiles.Count -eq 0) {
    Write-Host "No Python files found in $scriptDir."
    exit 1
}

# Run each Python file
foreach ($file in $pythonFiles) {
    echo "Running $($file.FullName)..."
    if ($file.FullName -eq "C:\Users\12625\Desktop\Code\hft\fed-market-impact\scripts\viz\error_plot.py") {
        Write-Host "Skipping $($file.FullName)..."
        continue
    }
    python $file.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error occurred while running $($file.FullName)."
        exit 1
    }
}

Write-Host "All Python files executed successfully."