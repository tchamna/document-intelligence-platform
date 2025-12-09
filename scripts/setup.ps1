# Healthcare IDP System - Setup (Windows PowerShell)
# Run this script to set up the development environment

Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Healthcare IDP System - Setup Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion"

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Download spaCy model
Write-Host "`nDownloading spaCy language model..." -ForegroundColor Yellow
python -m spacy download en_core_web_lg

# Create necessary directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\samples" | Out-Null
New-Item -ItemType Directory -Force -Path "models\classification" | Out-Null
New-Item -ItemType Directory -Force -Path "models\ner" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
Write-Host "Directories created." -ForegroundColor Green

# Copy environment file
Write-Host "`nSetting up environment file..." -ForegroundColor Yellow
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ".env file created from template. Please update with your credentials." -ForegroundColor Yellow
} else {
    Write-Host ".env file already exists." -ForegroundColor Green
}

# Run tests
Write-Host "`nRunning tests..." -ForegroundColor Yellow
pytest tests/ -v --tb=short

# Summary
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Update .env with your AWS credentials"
Write-Host "2. Run the pipeline: python -m src.pipeline"
Write-Host "3. Start the API: uvicorn api.main:app --reload"
Write-Host "4. Access API docs: http://localhost:8000/docs"
Write-Host ""
