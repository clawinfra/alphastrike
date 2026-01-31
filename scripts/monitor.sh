#!/usr/bin/env bash
#
# AlphaStrike Trading Bot - Health Monitoring Script
#
# This script monitors the health of the AlphaStrike trading bot.
# It checks:
#   1. Service status
#   2. Memory and CPU usage
#   3. Recent errors in logs
#   4. Database health
#   5. Network connectivity
#
# Exit codes (compatible with monitoring systems like Nagios):
#   0 = OK
#   1 = WARNING
#   2 = CRITICAL
#   3 = UNKNOWN
#
# Usage: ./scripts/monitor.sh [--verbose] [--json]
#

set -uo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
readonly SERVICE_NAME="alphastrike"
readonly DB_PATH="${PROJECT_DIR}/data/alphastrike.db"
readonly LOG_DIR="${PROJECT_DIR}/logs"

# Thresholds
readonly MEMORY_WARNING_MB=3072    # 3GB
readonly MEMORY_CRITICAL_MB=3840   # 3.75GB
readonly CPU_WARNING_PERCENT=150   # 150%
readonly CPU_CRITICAL_PERCENT=190  # 190%
readonly ERROR_WARNING_COUNT=5     # Errors in last hour
readonly ERROR_CRITICAL_COUNT=20   # Errors in last hour
readonly DB_SIZE_WARNING_MB=1024   # 1GB
readonly DB_SIZE_CRITICAL_MB=4096  # 4GB

# Exit codes
readonly EXIT_OK=0
readonly EXIT_WARNING=1
readonly EXIT_CRITICAL=2
readonly EXIT_UNKNOWN=3

# Flags
VERBOSE=false
JSON_OUTPUT=false

# Global status tracking
OVERALL_STATUS=$EXIT_OK
declare -a STATUS_MESSAGES=()
declare -a WARNING_MESSAGES=()
declare -a CRITICAL_MESSAGES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--json)
            JSON_OUTPUT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--verbose] [--json]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose  Show detailed output"
            echo "  -j, --json     Output in JSON format"
            echo ""
            echo "Exit codes:"
            echo "  0 = OK"
            echo "  1 = WARNING"
            echo "  2 = CRITICAL"
            echo "  3 = UNKNOWN"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit $EXIT_UNKNOWN
            ;;
    esac
done

# Update overall status (only escalate, never downgrade)
update_status() {
    local new_status=$1
    local message=$2

    if [[ $new_status -eq $EXIT_WARNING ]]; then
        WARNING_MESSAGES+=("$message")
        if [[ $OVERALL_STATUS -lt $EXIT_WARNING ]]; then
            OVERALL_STATUS=$EXIT_WARNING
        fi
    elif [[ $new_status -eq $EXIT_CRITICAL ]]; then
        CRITICAL_MESSAGES+=("$message")
        OVERALL_STATUS=$EXIT_CRITICAL
    else
        STATUS_MESSAGES+=("$message")
    fi
}

# Verbose logging
log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "  [DEBUG] $1"
    fi
}

# Check if service is running
check_service_status() {
    log_verbose "Checking service status..."

    if ! command -v systemctl &> /dev/null; then
        update_status $EXIT_UNKNOWN "systemctl not available"
        return
    fi

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        local uptime
        uptime=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value 2>/dev/null | xargs -I {} date -d {} '+%Y-%m-%d %H:%M' 2>/dev/null || echo "unknown")
        update_status $EXIT_OK "Service running since $uptime"
    else
        local state
        state=$(systemctl is-failed "$SERVICE_NAME" 2>/dev/null || echo "inactive")
        if [[ "$state" == "failed" ]]; then
            update_status $EXIT_CRITICAL "Service FAILED"
        else
            update_status $EXIT_CRITICAL "Service not running (state: $state)"
        fi
    fi
}

# Check memory usage
check_memory_usage() {
    log_verbose "Checking memory usage..."

    local pid
    pid=$(pgrep -f "python.*main.py" 2>/dev/null | head -1)

    if [[ -z "$pid" ]]; then
        log_verbose "Process not found"
        return
    fi

    local memory_kb
    memory_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')

    if [[ -z "$memory_kb" ]]; then
        log_verbose "Could not read memory usage"
        return
    fi

    local memory_mb=$((memory_kb / 1024))

    if [[ $memory_mb -ge $MEMORY_CRITICAL_MB ]]; then
        update_status $EXIT_CRITICAL "Memory usage CRITICAL: ${memory_mb}MB (threshold: ${MEMORY_CRITICAL_MB}MB)"
    elif [[ $memory_mb -ge $MEMORY_WARNING_MB ]]; then
        update_status $EXIT_WARNING "Memory usage HIGH: ${memory_mb}MB (threshold: ${MEMORY_WARNING_MB}MB)"
    else
        update_status $EXIT_OK "Memory usage: ${memory_mb}MB"
    fi
}

# Check CPU usage
check_cpu_usage() {
    log_verbose "Checking CPU usage..."

    local pid
    pid=$(pgrep -f "python.*main.py" 2>/dev/null | head -1)

    if [[ -z "$pid" ]]; then
        log_verbose "Process not found"
        return
    fi

    # Get CPU usage (this is a snapshot, not average)
    local cpu_percent
    cpu_percent=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ' | cut -d. -f1)

    if [[ -z "$cpu_percent" ]]; then
        log_verbose "Could not read CPU usage"
        return
    fi

    if [[ $cpu_percent -ge $CPU_CRITICAL_PERCENT ]]; then
        update_status $EXIT_CRITICAL "CPU usage CRITICAL: ${cpu_percent}% (threshold: ${CPU_CRITICAL_PERCENT}%)"
    elif [[ $cpu_percent -ge $CPU_WARNING_PERCENT ]]; then
        update_status $EXIT_WARNING "CPU usage HIGH: ${cpu_percent}% (threshold: ${CPU_WARNING_PERCENT}%)"
    else
        update_status $EXIT_OK "CPU usage: ${cpu_percent}%"
    fi
}

# Check for errors in logs
check_log_errors() {
    log_verbose "Checking log errors..."

    if ! command -v journalctl &> /dev/null; then
        log_verbose "journalctl not available"
        return
    fi

    # Count errors in the last hour
    local error_count
    error_count=$(journalctl -u "$SERVICE_NAME" --since "1 hour ago" -p err --no-pager 2>/dev/null | wc -l)

    if [[ $error_count -ge $ERROR_CRITICAL_COUNT ]]; then
        update_status $EXIT_CRITICAL "Log errors CRITICAL: $error_count errors in last hour"
    elif [[ $error_count -ge $ERROR_WARNING_COUNT ]]; then
        update_status $EXIT_WARNING "Log errors WARNING: $error_count errors in last hour"
    else
        update_status $EXIT_OK "Log errors: $error_count in last hour"
    fi

    # Check for specific critical patterns
    local critical_patterns
    critical_patterns=$(journalctl -u "$SERVICE_NAME" --since "15 minutes ago" --no-pager 2>/dev/null | \
        grep -iE "(fatal|panic|shutdown|out.of.memory|killed)" | wc -l)

    if [[ $critical_patterns -gt 0 ]]; then
        update_status $EXIT_CRITICAL "Critical log patterns detected in last 15 minutes"
    fi
}

# Check database health
check_database() {
    log_verbose "Checking database..."

    if [[ ! -f "$DB_PATH" ]]; then
        update_status $EXIT_WARNING "Database file not found"
        return
    fi

    # Check database size
    local db_size_bytes
    db_size_bytes=$(stat -f%z "$DB_PATH" 2>/dev/null || stat --printf="%s" "$DB_PATH" 2>/dev/null)

    if [[ -z "$db_size_bytes" ]]; then
        log_verbose "Could not read database size"
        return
    fi

    local db_size_mb=$((db_size_bytes / 1024 / 1024))

    if [[ $db_size_mb -ge $DB_SIZE_CRITICAL_MB ]]; then
        update_status $EXIT_CRITICAL "Database size CRITICAL: ${db_size_mb}MB"
    elif [[ $db_size_mb -ge $DB_SIZE_WARNING_MB ]]; then
        update_status $EXIT_WARNING "Database size WARNING: ${db_size_mb}MB"
    else
        update_status $EXIT_OK "Database size: ${db_size_mb}MB"
    fi

    # Check database integrity (if sqlite3 available)
    if command -v sqlite3 &> /dev/null; then
        local integrity
        integrity=$(sqlite3 "$DB_PATH" "PRAGMA integrity_check;" 2>/dev/null | head -1)
        if [[ "$integrity" != "ok" ]]; then
            update_status $EXIT_CRITICAL "Database integrity check FAILED"
        else
            log_verbose "Database integrity OK"
        fi
    fi
}

# Check disk space
check_disk_space() {
    log_verbose "Checking disk space..."

    local disk_usage
    disk_usage=$(df -h "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')

    if [[ -z "$disk_usage" ]]; then
        log_verbose "Could not read disk usage"
        return
    fi

    if [[ $disk_usage -ge 95 ]]; then
        update_status $EXIT_CRITICAL "Disk usage CRITICAL: ${disk_usage}%"
    elif [[ $disk_usage -ge 85 ]]; then
        update_status $EXIT_WARNING "Disk usage WARNING: ${disk_usage}%"
    else
        update_status $EXIT_OK "Disk usage: ${disk_usage}%"
    fi
}

# Check network connectivity to exchange
check_network() {
    log_verbose "Checking network connectivity..."

    # Check connectivity to WEEX API
    if command -v curl &> /dev/null; then
        if curl -s --max-time 5 "https://api.weex.com/api/v1/ping" > /dev/null 2>&1; then
            update_status $EXIT_OK "Exchange API reachable"
        else
            update_status $EXIT_WARNING "Exchange API unreachable"
        fi
    fi
}

# Send alert (placeholder for integration with alerting systems)
send_alert() {
    local severity=$1
    local message=$2

    # Placeholder for alerting integrations:
    # - Slack: curl -X POST -H 'Content-type: application/json' --data '{"text":"'"$message"'"}' "$SLACK_WEBHOOK_URL"
    # - Discord: curl -X POST -H 'Content-type: application/json' --data '{"content":"'"$message"'"}' "$DISCORD_WEBHOOK_URL"
    # - PagerDuty: Use pd-send or PagerDuty API
    # - Email: sendmail or mail command

    log_verbose "Alert would be sent: [$severity] $message"

    # Write to alert log file
    local alert_file="${LOG_DIR}/alerts.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$severity] $message" >> "$alert_file" 2>/dev/null || true
}

# Output results in JSON format
output_json() {
    local status_text
    case $OVERALL_STATUS in
        0) status_text="OK" ;;
        1) status_text="WARNING" ;;
        2) status_text="CRITICAL" ;;
        *) status_text="UNKNOWN" ;;
    esac

    # Build JSON manually for compatibility
    echo "{"
    echo "  \"status\": \"$status_text\","
    echo "  \"exit_code\": $OVERALL_STATUS,"
    echo "  \"timestamp\": \"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\","
    echo "  \"service\": \"$SERVICE_NAME\","
    echo "  \"checks\": {"
    echo "    \"ok\": ${#STATUS_MESSAGES[@]},"
    echo "    \"warnings\": ${#WARNING_MESSAGES[@]},"
    echo "    \"critical\": ${#CRITICAL_MESSAGES[@]}"
    echo "  },"
    echo "  \"messages\": ["

    local first=true
    for msg in "${STATUS_MESSAGES[@]}" "${WARNING_MESSAGES[@]}" "${CRITICAL_MESSAGES[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo ","
        fi
        echo -n "    \"$msg\""
    done
    echo ""

    echo "  ]"
    echo "}"
}

# Output results in text format
output_text() {
    local status_text
    case $OVERALL_STATUS in
        0) status_text="OK" ;;
        1) status_text="WARNING" ;;
        2) status_text="CRITICAL" ;;
        *) status_text="UNKNOWN" ;;
    esac

    echo "AlphaStrike Health Check: $status_text"
    echo "========================================"

    if [[ ${#CRITICAL_MESSAGES[@]} -gt 0 ]]; then
        echo ""
        echo "CRITICAL:"
        for msg in "${CRITICAL_MESSAGES[@]}"; do
            echo "  - $msg"
        done
    fi

    if [[ ${#WARNING_MESSAGES[@]} -gt 0 ]]; then
        echo ""
        echo "WARNINGS:"
        for msg in "${WARNING_MESSAGES[@]}"; do
            echo "  - $msg"
        done
    fi

    if [[ "$VERBOSE" == "true" ]] && [[ ${#STATUS_MESSAGES[@]} -gt 0 ]]; then
        echo ""
        echo "OK:"
        for msg in "${STATUS_MESSAGES[@]}"; do
            echo "  - $msg"
        done
    fi

    echo ""
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
}

# Main function
main() {
    # Run all checks
    check_service_status
    check_memory_usage
    check_cpu_usage
    check_log_errors
    check_database
    check_disk_space
    check_network

    # Send alerts for critical/warning states
    if [[ $OVERALL_STATUS -eq $EXIT_CRITICAL ]]; then
        for msg in "${CRITICAL_MESSAGES[@]}"; do
            send_alert "CRITICAL" "$msg"
        done
    elif [[ $OVERALL_STATUS -eq $EXIT_WARNING ]]; then
        for msg in "${WARNING_MESSAGES[@]}"; do
            send_alert "WARNING" "$msg"
        done
    fi

    # Output results
    if [[ "$JSON_OUTPUT" == "true" ]]; then
        output_json
    else
        output_text
    fi

    exit $OVERALL_STATUS
}

# Run main
main
