# Document Intelligence Platform - Windows Deployment Script
# Run this from PowerShell to deploy to EC2

param(
    [Parameter(Mandatory=$true)]
    [string]$KeyPath,
    
    [string]$EC2Host = "ec2-18-208-117-82.compute-1.amazonaws.com",
    [string]$EC2User = "ec2-user",
    [int]$AppPort = 8500
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Document Intelligence Platform Deployment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Target: $EC2User@$EC2Host" -ForegroundColor Yellow
Write-Host "Port: $AppPort" -ForegroundColor Yellow
Write-Host ""

# Verify key file exists
if (-not (Test-Path $KeyPath)) {
    Write-Host "ERROR: SSH key not found at $KeyPath" -ForegroundColor Red
    exit 1
}

# Create deployment package
Write-Host "[1/5] Creating deployment package..." -ForegroundColor Green
$TempDir = "$env:TEMP\doc-intel-deploy"
$ZipFile = "$env:TEMP\document-intelligence.tar.gz"

# Clean up old temp files
Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
Remove-Item -Force $ZipFile -ErrorAction SilentlyContinue

# Create temp directory with required files
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# Copy required files (excluding venv, __pycache__, etc.)
$FilesToCopy = @(
    "api",
    "src", 
    "config",
    "static",
    "requirements.txt",
    "deployment/deploy_ec2.sh"
)

foreach ($item in $FilesToCopy) {
    $source = Join-Path $ProjectDir $item
    $dest = Join-Path $TempDir $item
    if (Test-Path $source) {
        if ((Get-Item $source).PSIsContainer) {
            Copy-Item -Recurse $source $dest -Exclude "__pycache__","*.pyc","venv"
        } else {
            $destDir = Split-Path $dest -Parent
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item $source $dest
        }
    }
}

# Create tarball using tar (available in Windows 10+)
Write-Host "[2/5] Compressing files..." -ForegroundColor Green
Push-Location $TempDir
tar -czvf $ZipFile *
Pop-Location

# Check ports on EC2
Write-Host "[3/5] Checking EC2 instance..." -ForegroundColor Green
$sshCmd = "ssh -i `"$KeyPath`" -o StrictHostKeyChecking=no $EC2User@$EC2Host"

# Transfer files
Write-Host "[4/5] Transferring files to EC2..." -ForegroundColor Green
& scp -i $KeyPath -o StrictHostKeyChecking=no $ZipFile "${EC2User}@${EC2Host}:/tmp/"

# Deploy on EC2
Write-Host "[5/5] Running deployment on EC2..." -ForegroundColor Green
$deployScript = @"
set -e
cd /tmp
sudo mkdir -p /opt/document-intelligence
sudo tar -xzf document-intelligence.tar.gz -C /opt/document-intelligence
cd /opt/document-intelligence
sudo bash deployment/deploy_ec2.sh
"@

# Execute deployment
Invoke-Expression "$sshCmd `"$deployScript`""

# Cleanup
Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
Remove-Item -Force $ZipFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access your application at:" -ForegroundColor Cyan
Write-Host "  Main UI: http://${EC2Host}:${AppPort}/batch" -ForegroundColor White
Write-Host "  API Docs: http://${EC2Host}:${AppPort}/docs" -ForegroundColor White
Write-Host ""
