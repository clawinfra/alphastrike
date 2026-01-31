# AlphaStrike Trading Bot

Autonomous ML-based crypto trading system for WEEX perpetual futures.

## Features

- **ML Ensemble**: 4-model ensemble (XGBoost, LightGBM, LSTM, Random Forest)
- **Regime Adaptation**: Automatic strategy adjustment based on market conditions
- **Adaptive Risk**: Multi-layer risk management with position scaling
- **Self-Healing**: Automatic model health detection and retraining

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your WEEX API credentials

# Run the bot
uv run python main.py
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System architecture details
- [PRD](docs/PRD.md) - Product requirements document

## Development

```bash
# Run type checking
uv run pyright src/

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check src/
```

## Production Deployment

### Prerequisites

- **Operating System**: Ubuntu 22.04+ (or other Linux with systemd)
- **Python**: 3.12+
- **uv**: Package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **systemd**: For service management (included in most Linux distributions)
- **Git**: For version control

### Installation Steps

1. **Create dedicated user**:
   ```bash
   sudo useradd -r -s /bin/false alphastrike
   sudo mkdir -p /opt/alphastrike
   sudo chown alphastrike:alphastrike /opt/alphastrike
   ```

2. **Clone repository**:
   ```bash
   cd /opt/alphastrike
   sudo -u alphastrike git clone https://github.com/yourusername/alphastrike_ai.git .
   ```

3. **Install dependencies**:
   ```bash
   sudo -u alphastrike uv sync
   ```

4. **Create required directories**:
   ```bash
   sudo -u alphastrike mkdir -p data logs ai_logs models backups
   ```

### Configuration

1. **Copy and configure environment file**:
   ```bash
   sudo -u alphastrike cp .env.example .env
   sudo -u alphastrike chmod 600 .env
   sudo nano .env
   ```

2. **Required environment variables**:
   ```bash
   # WEEX Exchange API Credentials
   WEEX_API_KEY=your_api_key_here
   WEEX_API_SECRET=your_api_secret_here
   WEEX_API_PASSPHRASE=your_passphrase_here

   # Trading Mode
   TRADING_ENABLED=true
   PAPER_TRADING=false  # Set to true for testing

   # Logging
   LOG_LEVEL=INFO
   ```

3. **Optional configuration** (defaults in `src/core/config.py`):
   - Risk limits: `RISK_MAX_LEVERAGE`, `RISK_DAILY_DRAWDOWN_LIMIT`
   - Position sizing: `POSITION_PER_TRADE_EXPOSURE`, `POSITION_TOTAL_EXPOSURE`
   - ML weights: `ML_XGBOOST_WEIGHT`, `ML_LIGHTGBM_WEIGHT`, etc.

### Installing the Systemd Service

1. **Install service file**:
   ```bash
   sudo cp scripts/alphastrike.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. **Enable service** (start on boot):
   ```bash
   sudo systemctl enable alphastrike
   ```

### Starting and Stopping

```bash
# Start the service
sudo systemctl start alphastrike

# Stop the service
sudo systemctl stop alphastrike

# Restart the service
sudo systemctl restart alphastrike

# Check status
sudo systemctl status alphastrike
```

### Automated Deployment

Use the deployment script for updates:

```bash
# Full deployment (recommended)
./scripts/deploy.sh

# Skip tests for faster deployment
./scripts/deploy.sh --skip-tests

# Preview changes without executing
./scripts/deploy.sh --dry-run
```

The deployment script will:
1. Pull latest code from git
2. Update dependencies
3. Run type checking and tests
4. Backup database and models
5. Restart the service
6. Perform health check

### Monitoring and Logs

#### View Logs

```bash
# Follow live logs
journalctl -u alphastrike -f

# View last 100 lines
journalctl -u alphastrike -n 100

# View logs from today
journalctl -u alphastrike --since today

# View error logs only
journalctl -u alphastrike -p err
```

#### AI Decision Logs

Trading decisions are logged to `ai_logs/` directory:
```bash
# View today's decisions
cat ai_logs/$(date +%Y%m%d)_decisions.jsonl | jq .
```

#### Health Monitoring

```bash
# Run health check
./scripts/monitor.sh

# Verbose output
./scripts/monitor.sh --verbose

# JSON output (for monitoring systems)
./scripts/monitor.sh --json
```

The monitor script checks:
- Service status
- Memory and CPU usage
- Recent log errors
- Database health
- Disk space
- Network connectivity

Exit codes are compatible with Nagios/monitoring systems:
- 0 = OK
- 1 = WARNING
- 2 = CRITICAL
- 3 = UNKNOWN

#### Setting Up Monitoring Cron

```bash
# Add to crontab for regular monitoring
crontab -e

# Check every 5 minutes, alert on issues
*/5 * * * * /opt/alphastrike/scripts/monitor.sh --json >> /var/log/alphastrike-health.log 2>&1
```

### Troubleshooting

#### Service Won't Start

1. Check logs for errors:
   ```bash
   journalctl -u alphastrike -n 50 --no-pager
   ```

2. Verify configuration:
   ```bash
   # Check .env exists and has correct permissions
   ls -la /opt/alphastrike/.env

   # Validate Python can import
   cd /opt/alphastrike && uv run python -c "from src.core.config import get_settings; print(get_settings())"
   ```

3. Check file permissions:
   ```bash
   # Ensure alphastrike user owns all files
   sudo chown -R alphastrike:alphastrike /opt/alphastrike
   ```

#### High Memory Usage

The service has memory limits configured (4GB max). If exceeded:
1. Check for memory leaks in logs
2. Reduce candle buffer size in configuration
3. Monitor model sizes

#### Connection Issues

1. Check network connectivity:
   ```bash
   curl -s https://api.weex.com/api/v1/ping
   ```

2. Verify API credentials:
   ```bash
   # Check .env file has correct API keys
   grep WEEX_ /opt/alphastrike/.env
   ```

#### Database Issues

1. Check database integrity:
   ```bash
   sqlite3 /opt/alphastrike/data/alphastrike.db "PRAGMA integrity_check;"
   ```

2. Restore from backup if needed:
   ```bash
   # List backups
   ls -la /opt/alphastrike/backups/

   # Restore database
   cp /opt/alphastrike/backups/backup-YYYYMMDD-HHMMSS/alphastrike.db /opt/alphastrike/data/
   ```

### Resource Limits

The systemd service is configured with these limits:
- **Memory**: 4GB max, 3GB high watermark
- **CPU**: 200% (2 cores equivalent)
- **Tasks**: 100 max processes

To adjust, edit `/etc/systemd/system/alphastrike.service`:
```ini
MemoryMax=4G
CPUQuota=200%
```

Then reload: `sudo systemctl daemon-reload && sudo systemctl restart alphastrike`

## License

MIT
