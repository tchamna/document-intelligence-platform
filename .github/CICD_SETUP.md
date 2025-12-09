# Document Intelligence Platform - CI/CD Setup

## GitHub Actions Pipeline

This project uses GitHub Actions for continuous integration and deployment.

### Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    TEST     │ -> │    BUILD    │ -> │   DEPLOY    │
│  - Lint     │    │  - Docker   │    │  - EC2      │
│  - pytest   │    │    build    │    │  - Health   │
│  - Coverage │    │  - Artifact │    │    check    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Triggers

- **Push to main/master**: Full pipeline (test → build → deploy)
- **Pull Request**: Test only (no deployment)

### Setup Instructions

#### 1. Add GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secret:

| Secret Name | Description |
|-------------|-------------|
| `EC2_SSH_KEY` | Your EC2 private key (contents of `test-rag.pem`) |

To add the SSH key:
1. Open `C:\Users\tcham\Downloads\test-rag.pem` in a text editor
2. Copy the entire contents (including `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----`)
3. Create a new secret named `EC2_SSH_KEY` and paste the key

#### 2. Create GitHub Repository

```bash
# Initialize and push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/document-intelligence-platform.git
git branch -M main
git push -u origin main
```

#### 3. Environment Protection (Optional)

For additional security:
1. Go to Settings → Environments
2. Create environment: `production`
3. Add required reviewers for manual approval before deploy

### Pipeline Jobs

#### Test Job
- Runs on every push and PR
- Sets up Python 3.11
- Installs dependencies
- Runs flake8 linting
- Runs pytest with coverage

#### Build Job
- Runs after successful tests
- Builds Docker image
- Tags with commit SHA and `latest`
- Uploads as artifact

#### Deploy Job
- Runs only on main/master branch
- Downloads Docker image
- Transfers to EC2 via SCP
- Loads and runs new container
- Verifies health check
- Cleans up old images

### Manual Deployment

If you need to deploy manually:

```bash
# Build locally
docker build -t document-intelligence:latest -f deployment/Dockerfile .

# Save and transfer
docker save document-intelligence:latest | gzip > docker-image.tar.gz
scp -i path/to/key.pem docker-image.tar.gz ec2-user@ec2-18-208-117-82.compute-1.amazonaws.com:/tmp/

# SSH and deploy
ssh -i path/to/key.pem ec2-user@ec2-18-208-117-82.compute-1.amazonaws.com
docker load < /tmp/docker-image.tar.gz
docker stop document-intelligence && docker rm document-intelligence
docker run -d --name document-intelligence --restart unless-stopped -p 8500:8000 document-intelligence:latest
```

### Monitoring

- **Application**: https://idp.tchamna.com/health
- **API Docs**: https://idp.tchamna.com/docs
- **Web UI**: https://idp.tchamna.com/ui

### Troubleshooting

**Pipeline fails at test:**
- Check if tests pass locally: `pytest tests/ -v`
- Review test output in GitHub Actions logs

**Pipeline fails at deploy:**
- Verify EC2_SSH_KEY secret is set correctly
- Check EC2 security group allows SSH (port 22)
- Verify EC2 instance is running

**Container not starting:**
```bash
# Check container logs
ssh -i key.pem ec2-user@ec2-18-208-117-82.compute-1.amazonaws.com "docker logs document-intelligence"
```
