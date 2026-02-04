# TCLP Risk ID Deployment Guide

## Prerequisites

- Ubuntu/Debian server
- Python 3.10.15
- Poetry installed
- Nginx (for reverse proxy)

## Installation Steps

### 1. Clone the repository

```bash
sudo mkdir -p /var/www/labs-apps/risk_id
sudo chown $USER:$USER /var/www/labs-apps/risk_id
cd /var/www/labs-apps/risk_id
git clone <repository-url> .
```

### 2. Configure environment variables

```bash
cp .env.template .env
nano .env
```

Update the following variables:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_MODEL`: The LLM model to use
- `BASE_PATH`: The subfolder path (e.g., `/risk-id`)
- `TOKENIZERS_PARALLELISM`: Keep as `false` to suppress warnings

### 3. Install dependencies

```bash
poetry install --no-dev
```

### 4. Download required models and data

Follow the instructions in the main README.md to download:
- CC_BERT model
- Data files (clause library, tags, etc.)

### 5. Install systemd service

```bash
sudo cp deployment/tclp-risk-id.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tclp-risk-id
sudo systemctl start tclp-risk-id
```

### 6. Check service status

```bash
sudo systemctl status tclp-risk-id
sudo journalctl -u tclp-risk-id -f
```

### 7. Configure Nginx reverse proxy

Create `/etc/nginx/sites-available/tclp-risk-id`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Optional: redirect to HTTPS
    # return 301 https://$server_name$request_uri;

    location /risk-id/ {
        proxy_pass http://127.0.0.1:3003;
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
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/tclp-risk-id /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Maintenance

### Update the application

```bash
cd /var/www/labs-apps/risk_id
git pull
poetry install --no-dev
sudo systemctl restart tclp-risk-id
```

### View logs

```bash
sudo journalctl -u tclp-risk-id -n 100
sudo journalctl -u tclp-risk-id -f
```

### Restart service

```bash
sudo systemctl restart tclp-risk-id
```

## Security Notes

1. **Restrict file permissions**: Ensure .env is only readable by the service user
2. **Use HTTPS**: Configure SSL/TLS certificates (Let's Encrypt recommended)
3. **Firewall**: Only allow necessary ports (80, 443)
4. **API keys**: Keep OPENROUTER_API_KEY secure and rotate regularly
5. **Keep updated**: Regularly update dependencies and system packages
