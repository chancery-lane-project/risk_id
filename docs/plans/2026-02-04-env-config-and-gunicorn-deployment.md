# Environment Configuration and Gunicorn Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Configure the FastAPI app to use environment variables from .env.template and prepare for production deployment with gunicorn.

**Architecture:** Update FastAPI app to read all configuration from environment variables (BASE_PATH, OPENROUTER_API_KEY, OPENROUTER_MODEL, TOKENIZERS_PARALLELISM), add BASE_PATH prefix to all routes and static file mounts, and create systemd service configuration for production deployment.

**Tech Stack:** FastAPI, Gunicorn, systemd, python-dotenv

---

## Task 1: Update Environment Variable Configuration

**Files:**
- Modify: `tclp/app.py:8-11` (environment variable configuration)
- Modify: `tclp/app.py:95-101` (OpenRouter configuration)
- Reference: `.env.template`

**Step 1: Update environment variable loading at the top of app.py**

Currently line 8 hardcodes the TOKENIZERS_PARALLELISM setting. Update to read from environment:

```python
import hashlib
import os
import pickle
import shutil
import time

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Suppress tokenizers parallelism warning - read from env
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
```

**Step 2: Verify OpenRouter configuration is already using environment variables**

Lines 95-101 already read OPENROUTER_API_KEY and OPENROUTER_MODEL from environment - confirm this is working correctly.

**Step 3: Test environment variable loading**

Run: `poetry run python -c "from tclp.app import OPENROUTER_API_KEY; print('API key loaded:', bool(OPENROUTER_API_KEY))"`
Expected: Should print "API key loaded: True" if .env is configured

**Step 4: Commit**

```bash
git add tclp/app.py
git commit -m "feat: read TOKENIZERS_PARALLELISM from environment variables"
```

---

## Task 2: Add BASE_PATH Configuration

**Files:**
- Modify: `tclp/app.py:23-25` (after load_dotenv)
- Modify: `tclp/app.py:25-40` (FastAPI app initialization)
- Modify: `tclp/app.py:72-79` (static file mounts)

**Step 1: Add BASE_PATH configuration variable**

After the existing `load_dotenv()` call (around line 23), add:

```python
load_dotenv()

# Configuration from environment
BASE_PATH = os.getenv("BASE_PATH", "").rstrip("/")  # Remove trailing slash if present
```

**Step 2: Update FastAPI app initialization to use root_path**

Modify the FastAPI initialization (currently line 25):

```python
app = FastAPI(
    root_path=BASE_PATH,
    title="TCLP Risk ID API",
    description="Contract Climate Risk Identification API"
)
```

**Step 3: Update static file mounts to use BASE_PATH**

Modify lines 72-79 to include BASE_PATH in mount paths:

```python
app.mount(
    f"{BASE_PATH}/assets" if BASE_PATH else "/assets",
    StaticFiles(directory=os.path.join(BASE_DIR, "provocotype-1", "assets")),
    name="assets",
)

os.makedirs(output_dir, exist_ok=True)
app.mount(
    f"{BASE_PATH}/output" if BASE_PATH else "/output",
    StaticFiles(directory=output_dir),
    name="output"
)
```

**Step 4: Test BASE_PATH configuration**

Run:
```bash
BASE_PATH=/risk-id poetry run python -c "from tclp.app import app, BASE_PATH; print(f'BASE_PATH: {BASE_PATH}'); print(f'root_path: {app.root_path}')"
```
Expected: Should print BASE_PATH and root_path as "/risk-id"

**Step 5: Commit**

```bash
git add tclp/app.py
git commit -m "feat: add BASE_PATH configuration for subfolder deployment"
```

---

## Task 3: Update Frontend API Calls to Use Relative Paths

**Files:**
- Modify: `tclp/provocotype-1/index.htm:315` (process endpoint)
- Modify: `tclp/provocotype-1/index.htm:327` (find_clauses endpoint)

**Step 1: Verify current API endpoint paths**

Read the current implementation to understand the fetch calls.

**Step 2: Document that relative paths will work with FastAPI root_path**

FastAPI's `root_path` parameter automatically handles the BASE_PATH prefix for:
- All route decorators (@app.get, @app.post)
- Static file mounts
- OpenAPI/docs endpoints

The frontend's relative paths like `/process/` and `/find_clauses/` will automatically be prefixed when the app is deployed with BASE_PATH set.

**Step 3: Update README to document BASE_PATH usage**

Note: Frontend uses relative paths which work automatically with FastAPI root_path. No changes needed to frontend code.

**Step 4: Test locally with BASE_PATH**

Run:
```bash
BASE_PATH=/risk-id poetry run uvicorn tclp.app:app --host 127.0.0.1 --port 8000
```

Then test:
```bash
curl http://127.0.0.1:8000/risk-id/
```

Expected: Should return 401 (auth required) or serve the index page

**Step 5: Commit**

```bash
git add docs/plans/2026-02-04-env-config-and-gunicorn-deployment.md
git commit -m "docs: document BASE_PATH configuration and relative path behavior"
```

---

## Task 4: Add Authentication Environment Variables

**Files:**
- Modify: `tclp/app.py:43-45` (authentication credentials)
- Modify: `.env.template` (add auth variables)

**Step 1: Add authentication variables to .env.template**

Add after the BASE_PATH line:

```bash
# Authentication (Optional - defaults to demo credentials)
# IMPORTANT: Change these for production deployment
AUTH_USERNAME=father
AUTH_PASSWORD=christmas
```

**Step 2: Update app.py to read credentials from environment**

Replace lines 43-45:

```python
# Authentication credentials - read from environment for security
USERNAME = os.getenv("AUTH_USERNAME", "father")
PASSWORD = os.getenv("AUTH_PASSWORD", "christmas")
```

**Step 3: Test authentication configuration**

Run:
```bash
AUTH_USERNAME=testuser AUTH_PASSWORD=testpass poetry run python -c "from tclp.app import USERNAME, PASSWORD; print(f'Username: {USERNAME}, Password length: {len(PASSWORD)}')"
```
Expected: Should print the custom username and password length

**Step 4: Commit**

```bash
git add tclp/app.py .env.template
git commit -m "feat: make authentication credentials configurable via environment variables"
```

---

## Task 5: Create Gunicorn Configuration

**Files:**
- Create: `gunicorn.conf.py`

**Step 1: Create gunicorn configuration file**

```python
"""
Gunicorn configuration for TCLP Risk ID application.

This configuration is optimized for production deployment with:
- Multiple worker processes for handling concurrent requests
- Longer timeouts for ML model processing
- Access logging for monitoring
"""

import multiprocessing
import os

# Read configuration from environment
BASE_PATH = os.getenv("BASE_PATH", "").rstrip("/")

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300  # 5 minutes - ML processing can be slow
keepalive = 5

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "tclp-risk-id"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (configure if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
```

**Step 2: Add gunicorn to pyproject.toml dependencies**

Check if gunicorn is already installed, if not add it to pyproject.toml:

```toml
gunicorn = "^24.0.0"
```

**Step 3: Test gunicorn configuration**

Run:
```bash
poetry run gunicorn -c gunicorn.conf.py tclp.app:app --workers 2 --timeout 60
```

Expected: Server should start without errors

**Step 4: Stop the test server and commit**

```bash
git add gunicorn.conf.py pyproject.toml poetry.lock
git commit -m "feat: add gunicorn configuration for production deployment"
```

---

## Task 6: Create systemd Service Configuration

**Files:**
- Create: `deployment/tclp-risk-id.service`

**Step 1: Create deployment directory**

```bash
mkdir -p deployment
```

**Step 2: Create systemd service file**

```ini
[Unit]
Description=TCLP Risk ID - Contract Climate Risk Identification Service
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/app
EnvironmentFile=/app/.env

# Install dependencies and start gunicorn
ExecStartPre=/usr/local/bin/poetry install --no-dev
ExecStart=/usr/local/bin/poetry run gunicorn -c /app/gunicorn.conf.py tclp.app:app

# Restart policy
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=65536

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tclp-risk-id

[Install]
WantedBy=multi-user.target
```

**Step 3: Create deployment documentation**

Create `deployment/README.md`:

```markdown
# TCLP Risk ID Deployment Guide

## Prerequisites

- Ubuntu/Debian server
- Python 3.10.15
- Poetry installed
- Nginx (for reverse proxy)

## Installation Steps

### 1. Clone the repository

\`\`\`bash
sudo mkdir -p /app
sudo chown $USER:$USER /app
cd /app
git clone <repository-url> .
\`\`\`

### 2. Configure environment variables

\`\`\`bash
cp .env.template .env
nano .env
\`\`\`

Update the following variables:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_MODEL`: The LLM model to use
- `BASE_PATH`: The subfolder path (e.g., `/risk-id`)
- `AUTH_USERNAME`: Change from default
- `AUTH_PASSWORD`: Change from default

### 3. Install dependencies

\`\`\`bash
poetry install --no-dev
\`\`\`

### 4. Download required models and data

Follow the instructions in the main README.md to download:
- CC_BERT model
- Data files (clause library, tags, etc.)

### 5. Install systemd service

\`\`\`bash
sudo cp deployment/tclp-risk-id.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tclp-risk-id
sudo systemctl start tclp-risk-id
\`\`\`

### 6. Check service status

\`\`\`bash
sudo systemctl status tclp-risk-id
sudo journalctl -u tclp-risk-id -f
\`\`\`

### 7. Configure Nginx reverse proxy

Create `/etc/nginx/sites-available/tclp-risk-id`:

\`\`\`nginx
server {
    listen 80;
    server_name your-domain.com;

    # Optional: redirect to HTTPS
    # return 301 https://$server_name$request_uri;

    location /risk-id/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Important: preserve the /risk-id prefix
        proxy_redirect off;

        # Longer timeout for ML processing
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
\`\`\`

Enable the site:

\`\`\`bash
sudo ln -s /etc/nginx/sites-available/tclp-risk-id /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
\`\`\`

## Maintenance

### Update the application

\`\`\`bash
cd /app
git pull
poetry install --no-dev
sudo systemctl restart tclp-risk-id
\`\`\`

### View logs

\`\`\`bash
sudo journalctl -u tclp-risk-id -n 100
sudo journalctl -u tclp-risk-id -f
\`\`\`

### Restart service

\`\`\`bash
sudo systemctl restart tclp-risk-id
\`\`\`

## Security Notes

1. **Change default credentials**: Update AUTH_USERNAME and AUTH_PASSWORD in .env
2. **Restrict file permissions**: Ensure .env is only readable by the service user
3. **Use HTTPS**: Configure SSL/TLS certificates (Let's Encrypt recommended)
4. **Firewall**: Only allow necessary ports (80, 443)
5. **API keys**: Keep OPENROUTER_API_KEY secure and rotate regularly
\`\`\`

**Step 4: Commit deployment files**

```bash
git add deployment/
git commit -m "feat: add systemd service configuration and deployment documentation"
```

---

## Task 7: Update Main README with Deployment Information

**Files:**
- Modify: `README.md` (add deployment section)

**Step 1: Add production deployment section to README**

Add before the "License" section:

```markdown
---

## Production Deployment

For production deployment with gunicorn and systemd, see [deployment/README.md](deployment/README.md).

### Quick Start (Production)

\`\`\`bash
# Configure environment
cp .env.template .env
nano .env  # Update all variables

# Install dependencies
poetry install --no-dev

# Run with gunicorn
poetry run gunicorn -c gunicorn.conf.py tclp.app:app
\`\`\`

### Environment Variables

All configuration is managed through environment variables (see `.env.template`):

- `OPENROUTER_API_KEY`: Required for clause recommendations
- `OPENROUTER_MODEL`: LLM model to use (default: tngtech/deepseek-r1t2-chimera:free)
- `BASE_PATH`: Subfolder deployment path (e.g., `/risk-id`)
- `TOKENIZERS_PARALLELISM`: Suppress tokenizer warnings (default: false)
- `AUTH_USERNAME`: Basic auth username (change for production!)
- `AUTH_PASSWORD`: Basic auth password (change for production!)

### Systemd Service

See [deployment/README.md](deployment/README.md) for systemd service installation and configuration.
```

**Step 2: Update the "Usage" section to mention environment variables**

Update the "Running the Application" section to reference .env:

```markdown
### Running the Application

1. **Configure environment:**

\`\`\`bash
cp .env.template .env
nano .env  # Update OPENROUTER_API_KEY and other variables
\`\`\`

2. **Start the FastAPI Backend:**

\`\`\`bash
poetry run uvicorn tclp.app:app --host 0.0.0.0 --port 8000
\`\`\`
```

**Step 3: Commit README updates**

```bash
git add README.md
git commit -m "docs: add production deployment and environment configuration sections"
```

---

## Task 8: Testing and Verification

**Files:**
- Test all modified files

**Step 1: Test local development mode**

```bash
cp .env.template .env
poetry run uvicorn tclp.app:app --host 127.0.0.1 --port 8000
```

Open browser to `http://127.0.0.1:8000/` and verify:
- Basic auth prompt appears
- After login, index page loads
- Assets load correctly

**Step 2: Test with BASE_PATH set**

```bash
BASE_PATH=/risk-id poetry run uvicorn tclp.app:app --host 127.0.0.1 --port 8000
```

Test:
```bash
curl -u father:christmas http://127.0.0.1:8000/risk-id/
```

Expected: Should return HTML content (the index page)

**Step 3: Test with gunicorn**

```bash
poetry run gunicorn -c gunicorn.conf.py tclp.app:app --workers 2
```

Test:
```bash
curl -u father:christmas http://127.0.0.1:8000/
```

Expected: Should return HTML content

**Step 4: Test with custom auth credentials**

```bash
AUTH_USERNAME=testuser AUTH_PASSWORD=testpass123 poetry run uvicorn tclp.app:app --host 127.0.0.1 --port 8000
```

Test:
```bash
curl -u testuser:testpass123 http://127.0.0.1:8000/
```

Expected: Should return HTML content

**Step 5: Document test results**

If all tests pass, create a commit noting successful verification:

```bash
git add .
git commit -m "test: verify environment configuration and gunicorn deployment"
```

---

## Summary

This plan implements:

1. ✅ Environment-based configuration for all settings
2. ✅ BASE_PATH support for subfolder deployment (e.g., `/risk-id`)
3. ✅ Gunicorn configuration optimized for ML workloads
4. ✅ systemd service configuration for production
5. ✅ Deployment documentation with nginx configuration
6. ✅ Security improvements (configurable auth credentials)
7. ✅ Updated README with deployment instructions

## Notes for Implementer

- FastAPI's `root_path` parameter handles BASE_PATH automatically - no need to modify route decorators
- Frontend uses relative paths which work seamlessly with root_path
- Gunicorn timeout is set to 300s (5 minutes) to accommodate ML model processing
- The systemd service runs as www-data - adjust user/group as needed for your server
- Remember to change AUTH_USERNAME and AUTH_PASSWORD in production!
- The nginx configuration preserves the BASE_PATH prefix for correct routing
