#!/bin/bash

# Parallel T3D Training Script
# Launches 1 GUI instance + 3 headless workers

echo "Starting Parallel T3D Training..."
echo "=================================="

# Shared buffer and model paths
BUFFER="shared_buffer.pkl"
MODEL="t3d_model.pth"
MAP="city_map.png"

# Clean old buffer (optional - comment out to resume training)
# if [ -f "$BUFFER" ]; then
#     echo "Removing old shared buffer..."
#     rm "$BUFFER"
# fi

# Launch headless workers in background
echo "Launching 3 headless workers..."
python citymap_t3d.py --headless --instance-id 1 --shared-buffer "$BUFFER" --model "$MODEL" --map "$MAP" --episodes 10000 > worker1.log 2>&1 &
WORKER1_PID=$!
echo "  Worker 1 (PID: $WORKER1_PID) started"

python citymap_t3d.py --headless --instance-id 2 --shared-buffer "$BUFFER" --model "$MODEL" --map "$MAP" --episodes 10000 > worker2.log 2>&1 &
WORKER2_PID=$!
echo "  Worker 2 (PID: $WORKER2_PID) started"

python citymap_t3d.py --headless --instance-id 3 --shared-buffer "$BUFFER" --model "$MODEL" --map "$MAP" --episodes 10000 > worker3.log 2>&1 &
WORKER3_PID=$!
echo "  Worker 3 (PID: $WORKER3_PID) started"

# Wait a moment for workers to initialize
sleep 2

# Launch GUI instance (foreground)
echo ""
echo "Launching GUI instance..."
echo "Press Ctrl+C to stop all instances"
echo "=================================="
python citymap_t3d.py --shared-buffer "$BUFFER" --map "$MAP"

# Cleanup: kill background workers when GUI exits
echo ""
echo "Stopping workers..."
kill $WORKER1_PID $WORKER2_PID $WORKER3_PID 2>/dev/null
echo "All instances stopped."
