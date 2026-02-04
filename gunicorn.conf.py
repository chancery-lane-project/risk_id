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
