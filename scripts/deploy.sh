#!/usr/bin/env bash
#
# AlphaStrike Trading Bot - Deployment Script
#
# This script automates deployment of the AlphaStrike trading bot.
# It performs the following steps:
#   1. Pull latest code from git
#   2. Install/update dependencies with uv
#   3. Run pre-flight checks (pyright, pytest)
#   4. Backup current state
#   5. Restart service
#   6. Health check after deploy
#
# Usage: ./scripts/deploy.sh [--skip-tests] [--skip-backup]
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
readonly SERVICE_NAME="alphastrike"
readonly BACKUP_DIR="${PROJECT_DIR}/backups"
readonly LOG_FILE="${PROJECT_DIR}/logs/deploy.log"
readonly MAX_BACKUPS=5

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Flags
SKIP_TESTS=false
SKIP_BACKUP=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-tests] [--skip-backup] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --skip-tests   Skip pytest and pyright checks"
            echo "  --skip-backup  Skip database and model backup"
            echo "  --dry-run      Show what would be done without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

info() {
    log "INFO" "${BLUE}$1${NC}"
}

success() {
    log "INFO" "${GREEN}$1${NC}"
}

warn() {
    log "WARN" "${YELLOW}$1${NC}"
}

error() {
    log "ERROR" "${RED}$1${NC}"
}

die() {
    error "$1"
    exit 1
}

# Ensure required directories exist
ensure_directories() {
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    # Check for uv
    if ! command -v uv &> /dev/null; then
        die "uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi

    # Check for git
    if ! command -v git &> /dev/null; then
        die "git is not installed"
    fi

    # Check for systemctl (if not dry-run)
    if [[ "$DRY_RUN" == "false" ]] && ! command -v systemctl &> /dev/null; then
        warn "systemctl not found - service management will be skipped"
    fi

    # Check we're in the right directory
    if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
        die "pyproject.toml not found. Are you in the project directory?"
    fi

    # Check .env exists
    if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
        die ".env file not found. Copy .env.example and configure it."
    fi

    success "Prerequisites check passed"
}

# Pull latest code
pull_latest() {
    info "Pulling latest code from git..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would run: git pull --rebase"
        return 0
    fi

    cd "$PROJECT_DIR"

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        warn "Uncommitted changes detected. Stashing..."
        git stash push -m "deploy-$(date '+%Y%m%d-%H%M%S')"
    fi

    # Pull latest
    if ! git pull --rebase; then
        die "Failed to pull latest code"
    fi

    success "Code updated successfully"
}

# Install/update dependencies
install_dependencies() {
    info "Installing/updating dependencies with uv..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would run: uv sync"
        return 0
    fi

    cd "$PROJECT_DIR"

    if ! uv sync; then
        die "Failed to install dependencies"
    fi

    success "Dependencies installed successfully"
}

# Run pre-flight checks
run_preflight_checks() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping pre-flight checks (--skip-tests)"
        return 0
    fi

    info "Running pre-flight checks..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would run: uv run pyright src/"
        echo "  [DRY-RUN] Would run: uv run pytest tests/ -x -q"
        return 0
    fi

    cd "$PROJECT_DIR"

    # Type checking
    info "Running type checker (pyright)..."
    if ! uv run pyright src/; then
        die "Type checking failed"
    fi
    success "Type checking passed"

    # Run tests
    info "Running tests..."
    if ! uv run pytest tests/ -x -q --tb=short; then
        die "Tests failed"
    fi
    success "Tests passed"

    success "Pre-flight checks completed"
}

# Backup current state
backup_current_state() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        warn "Skipping backup (--skip-backup)"
        return 0
    fi

    info "Backing up current state..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would backup database and models"
        return 0
    fi

    local backup_name="backup-$(date '+%Y%m%d-%H%M%S')"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    mkdir -p "$backup_path"

    # Backup database
    if [[ -f "${PROJECT_DIR}/data/alphastrike.db" ]]; then
        cp "${PROJECT_DIR}/data/alphastrike.db" "${backup_path}/"
        info "  Database backed up"
    fi

    # Backup models
    if [[ -d "${PROJECT_DIR}/models" ]] && [[ -n "$(ls -A "${PROJECT_DIR}/models" 2>/dev/null)" ]]; then
        cp -r "${PROJECT_DIR}/models" "${backup_path}/"
        info "  Models backed up"
    fi

    # Backup .env (without secrets in filename)
    if [[ -f "${PROJECT_DIR}/.env" ]]; then
        cp "${PROJECT_DIR}/.env" "${backup_path}/.env"
        info "  Configuration backed up"
    fi

    # Cleanup old backups (keep only MAX_BACKUPS)
    local backup_count
    backup_count=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "backup-*" | wc -l)
    if [[ $backup_count -gt $MAX_BACKUPS ]]; then
        info "Cleaning up old backups (keeping last $MAX_BACKUPS)..."
        find "$BACKUP_DIR" -maxdepth 1 -type d -name "backup-*" -printf '%T+ %p\n' | \
            sort | head -n -$MAX_BACKUPS | cut -d' ' -f2 | xargs -r rm -rf
    fi

    success "Backup completed: $backup_name"
}

# Stop the service
stop_service() {
    info "Stopping ${SERVICE_NAME} service..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would run: systemctl stop $SERVICE_NAME"
        return 0
    fi

    if ! command -v systemctl &> /dev/null; then
        warn "systemctl not available - skipping service stop"
        return 0
    fi

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        if ! sudo systemctl stop "$SERVICE_NAME"; then
            die "Failed to stop service"
        fi
        success "Service stopped"
    else
        info "Service was not running"
    fi
}

# Start the service
start_service() {
    info "Starting ${SERVICE_NAME} service..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would run: systemctl start $SERVICE_NAME"
        return 0
    fi

    if ! command -v systemctl &> /dev/null; then
        warn "systemctl not available - skipping service start"
        return 0
    fi

    if ! sudo systemctl start "$SERVICE_NAME"; then
        die "Failed to start service"
    fi

    success "Service started"
}

# Health check after deploy
health_check() {
    info "Running post-deploy health check..."

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would check service health"
        return 0
    fi

    if ! command -v systemctl &> /dev/null; then
        warn "systemctl not available - skipping health check"
        return 0
    fi

    # Wait a moment for service to start
    sleep 3

    # Check if service is running
    if ! systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        error "Service failed to start!"
        echo ""
        echo "Recent logs:"
        journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
        die "Health check failed"
    fi

    # Check for recent errors in logs
    local error_count
    error_count=$(journalctl -u "$SERVICE_NAME" --since "30 seconds ago" -p err --no-pager 2>/dev/null | wc -l || echo "0")
    if [[ $error_count -gt 0 ]]; then
        warn "Found $error_count error(s) in recent logs"
        journalctl -u "$SERVICE_NAME" --since "30 seconds ago" -p err --no-pager || true
    fi

    success "Health check passed - service is running"
}

# Print deployment summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "  AlphaStrike Deployment Summary"
    echo "=========================================="
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  Mode: DRY-RUN (no changes made)"
    else
        echo "  Mode: LIVE DEPLOYMENT"
    fi

    echo "  Skip Tests: $SKIP_TESTS"
    echo "  Skip Backup: $SKIP_BACKUP"
    echo ""

    if [[ "$DRY_RUN" == "false" ]] && command -v systemctl &> /dev/null; then
        echo "  Service Status:"
        systemctl status "$SERVICE_NAME" --no-pager -l 2>/dev/null | head -5 || echo "    Service not installed"
    fi

    echo ""
    echo "  Useful commands:"
    echo "    View logs:    journalctl -u $SERVICE_NAME -f"
    echo "    Restart:      sudo systemctl restart $SERVICE_NAME"
    echo "    Stop:         sudo systemctl stop $SERVICE_NAME"
    echo "    Status:       sudo systemctl status $SERVICE_NAME"
    echo ""
}

# Main deployment function
main() {
    echo ""
    echo "=========================================="
    echo "  AlphaStrike Deployment"
    echo "  $(date)"
    echo "=========================================="
    echo ""

    ensure_directories
    check_prerequisites
    pull_latest
    install_dependencies
    run_preflight_checks
    backup_current_state
    stop_service
    start_service
    health_check
    print_summary

    success "Deployment completed successfully!"
}

# Run main
main
