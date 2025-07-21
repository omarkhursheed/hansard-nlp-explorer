#!/bin/bash
# Tmux launcher for full Hansard dataset processing

SESSION_NAME="hansard-processing"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../data/processed/processing.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏛️  Hansard Dataset Processing Launcher${NC}"
echo "=================================================="

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Session '$SESSION_NAME' already exists${NC}"
    echo "Options:"
    echo "1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create directories
mkdir -p "$(dirname "$LOG_FILE")"

# Create tmux session
echo -e "${GREEN}🚀 Creating tmux session: $SESSION_NAME${NC}"

tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"

# Setup the session
tmux send-keys -t "$SESSION_NAME" "echo '🏛️  Hansard Processing Session Started'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Time: $(date)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Log file: $LOG_FILE'" C-m
tmux send-keys -t "$SESSION_NAME" "echo '========================================'" C-m
tmux send-keys -t "$SESSION_NAME" "" C-m

# Start the processing with logging
tmux send-keys -t "$SESSION_NAME" "python process_full_dataset.py 2>&1 | tee '$LOG_FILE'" C-m

echo -e "${GREEN}✅ Session created successfully!${NC}"
echo ""
echo "📋 Quick Commands:"
echo "  Attach to session:    tmux attach -t $SESSION_NAME"
echo "  Detach from session:  Ctrl+b, then d"
echo "  Kill session:         tmux kill-session -t $SESSION_NAME"
echo "  View log in real-time: tail -f '$LOG_FILE'"
echo ""
echo "📊 Monitoring:"
echo "  Session status:       tmux list-sessions | grep $SESSION_NAME"
echo "  Process status:       ps aux | grep process_full_dataset"
echo "  Disk usage:           du -sh $(dirname '$LOG_FILE')"
echo ""
echo -e "${BLUE}🔗 Attaching to session now...${NC}"

# Auto-attach to the session
tmux attach -t "$SESSION_NAME"