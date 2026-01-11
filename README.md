# 🚗 T3D Autonomous Car Navigation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete implementation of **Twin Delayed Deep Deterministic Policy Gradient (T3D)** for autonomous car navigation with continuous control, featuring distributed parallel training and real-time visualization.

![Demo](demo.gif)

## 🎯 Overview

This project implements a reinforcement learning agent that learns to navigate a 2D city map autonomously. The agent uses continuous control for steering and speed, learns from visual sensors, and can complete multi-target navigation tasks.

### Key Features

- **🧠 T3D Algorithm**: State-of-the-art continuous control RL
- **⚡ Parallel Training**: 4x faster learning with distributed workers
- **📊 Real-time Visualization**: PyQt6 interface with debug tools
- **🎮 Continuous Control**: Smooth steering (-8° to +8°) and speed (1.5-2.5 px/s)
- **🔍 Binary Sensors**: 7 ray-casting sensors for obstacle detection
- **📈 Exploration Decay**: Automatic noise reduction for stable late-stage performance
- **💾 Model Checkpointing**: Save/load trained models and replay buffers

## 🏗️ Architecture

### T3D Algorithm Components

1. **Actor Network** (State → Action)
   - Input: 9D state vector (7 sensors + angle + distance)
   - Hidden: [400, 300] with ReLU
   - Output: 2D continuous action [steering, speed]
   - Activation: Tanh scaled to action bounds

2. **Twin Critic Networks** (State + Action → Q-value)
   - Two independent Q-networks (reduces overestimation)
   - Input: 11D (9 state + 2 action)
   - Hidden: [400, 300] with ReLU
   - Output: Single Q-value

3. **Target Networks**
   - Soft-updated copies (τ=0.005)
   - Stabilizes training

4. **Replay Buffer**
   - Stores experiences: (state, action, reward, next_state, done)
   - Shared across parallel instances
   - Off-policy learning

### Distributed Training System

```
┌─────────────────┐
│  GUI Instance   │ ← Visualization + Training
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Shared Replay Buffer   │ ← All experiences (pickle)
│  Shared Model Weights   │ ← Synchronized policy (PyTorch)
└─────────┬───────────────┘
          │
    ┌─────┴─────┬─────────┬─────────┐
    ▼           ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│ ← Headless training
└────────┘ └────────┘ └────────┘
```

## 📁 Project Structure

```
RL-t3d/
├── citymap_t3d.py          # Main application (GUI + headless mode)
├── t3d.py                   # T3D algorithm implementation
├── run_parallel.sh          # Launch script for parallel training
├── city_map.png             # Default navigation map
├── car.png                  # Car sprite (28x16 pixels)
├── README.md                # This file
├── shared_buffer.pkl        # Shared replay buffer (generated)
├── t3d_model.pth           # Model checkpoint (generated)
└── worker*.log             # Training logs (generated)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PyQt6
- NumPy

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd RL-t3d

# Install dependencies
pip install torch numpy PyQt6
```

### Single Instance Training (GUI)

```bash
python citymap_t3d.py
```

**Setup:**
1. Click on map to place car (starting position)
2. Click again to add target(s) - supports multiple waypoints
3. Right-click when done
4. Press **SPACE** or click **START** to begin training

**Controls:**
- `SPACE` - Start/Pause training
- `RESET ALL` - Clear and restart
- `LOAD MAP` - Load custom map image
- `SAVE MODEL` - Save trained networks
- `LOAD MODEL` - Load pre-trained model

### Parallel Training (Recommended)

**4x faster training with distributed workers:**

```bash
./run_parallel.sh
```

This launches:
- 3 headless workers (background training)
- 1 GUI instance (visualization)
- All sharing the same replay buffer and model weights

**Manual parallel training:**

```bash
# Terminal 1: Worker 1
python citymap_t3d.py --headless --instance-id 1 --shared-buffer shared_buffer.pkl --model t3d_model.pth

# Terminal 2: Worker 2
python citymap_t3d.py --headless --instance-id 2 --shared-buffer shared_buffer.pkl --model t3d_model.pth

# Terminal 3: Worker 3
python citymap_t3d.py --headless --instance-id 3 --shared-buffer shared_buffer.pkl --model t3d_model.pth

# Terminal 4: GUI (optional)
python citymap_t3d.py --shared-buffer shared_buffer.pkl --model t3d_model.pth
```

**Monitor workers:**
```bash
tail -f worker1.log  # Watch worker 1 progress
tail -f worker2.log  # Watch worker 2 progress
tail -f worker3.log  # Watch worker 3 progress
```

## ⚙️ Configuration

### Hyperparameters

```python
# Physics
CAR_WIDTH = 28              # Car width (pixels)
CAR_HEIGHT = 16             # Car height (pixels)
MAX_STEERING = 8.0          # Maximum steering angle (degrees/step)
MAX_SPEED = 2.5             # Maximum speed (pixels/step)
MIN_SPEED = 1.5             # Minimum speed (pixels/step)
SENSOR_DIST = 20            # Sensor range (pixels)
TARGET_RADIUS = 20          # Target reach threshold (pixels)

# T3D Algorithm
BATCH_SIZE = 256            # Training batch size (increased for stability)
GAMMA = 0.99                # Discount factor
TAU = 0.005                 # Soft update rate for target networks
POLICY_NOISE = 0.2          # Target policy smoothing noise
NOISE_CLIP = 0.5            # Noise clipping range
POLICY_FREQ = 2             # Delayed policy update frequency
LR = 5e-4                   # Learning rate (increased for faster convergence)

# Exploration Strategy
START_TIMESTEPS = 10000     # Random exploration steps (warm-up)
EXPL_NOISE_START = 0.2      # Initial exploration noise (20%)
EXPL_NOISE_END = 0.05       # Final exploration noise (5%)
EXPL_DECAY_STEPS = 50000    # Decay over 50k steps
```

### State Space (9D)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0-6 | Sensors | [0, 1] | 7 ray-casting sensors at [-45°, -30°, -15°, 0°, 15°, 30°, 45°] |
| 7 | Angle to target | [-1, 1] | Normalized angle difference |
| 8 | Distance to target | [0, 1] | Normalized distance |

**Sensor Implementation:**
- Binary detection: 1.0 = road (white pixels), 0.0 = obstacle
- White detection: RGB values all >200 with variance <30
- Ray-casting at 20 pixels distance

### Action Space (2D)

| Index | Action | Range | Description |
|-------|--------|-------|-------------|
| 0 | Steering | [-8.0, 8.0] | Degrees per timestep |
| 1 | Speed | [1.5, 2.5] | Pixels per timestep |

**Action Generation:**
```python
action = actor(state)  # Tanh output
action = max_action * action  # Scale to bounds
action += noise  # Add exploration noise (decaying)
```

### Reward Function

**Balanced 3-component reward:**

```python
# Base time penalty
reward = -0.1

# 1. Crash detection
if car_on_obstacle:
    reward = -100
    done = True

# 2. Target reached
elif distance < TARGET_RADIUS:
    reward = +100
    # Continue to next target if available

# 3. Normal navigation
else:
    # Clear sensors bonus
    reward += num_clear_sensors * 2.5
    
    # Approaching target
    reward += distance_improvement * 15
    
    # Alignment with target
    reward += (1.0 - abs(angle_to_target)) * 3
```

**Design Principles:**
- Simple and balanced (no over-penalization)
- Encourages safety (clear sensors)
- Rewards progress (distance improvement)
- Guides direction (alignment bonus)

## 🎓 Training Strategy

### Three-Phase Learning

**Phase 1: Random Exploration (0-10k steps)**
- Pure random actions
- Populate replay buffer with diverse experiences
- No policy learning

**Phase 2: Exploration with Decay (10k-60k steps)**
- Policy-guided actions with decaying noise
- Noise: 20% → 5% (linear decay)
- Active learning and improvement

**Phase 3: Exploitation (60k+ steps)**
- Mostly policy-driven (5% noise)
- Stable, consistent performance
- Rare crashes from policy mistakes only

### Exploration Decay Schedule

```python
progress = (timesteps - 10000) / 50000
current_noise = 0.2 + (0.05 - 0.2) * progress

# Timeline:
# 10k steps:  20% noise (aggressive exploration)
# 35k steps:  12.5% noise (balanced)
# 60k+ steps: 5% noise (mostly exploitation)
```

**Why decay matters:**
- Early: High exploration finds diverse strategies
- Mid: Balanced exploration/exploitation
- Late: Low noise prevents random crashes after learning

### Parallel Training Benefits

**4x Training Speed:**
- 1 GUI + 3 workers = 4x more experiences per minute
- Shared buffer: all instances learn from all experiences
- Shared weights: synchronized policy across instances

**Better Generalization:**
- Each instance explores different scenarios
- Random start positions and targets per instance
- More diverse data = better policy

**Synchronization:**
- Every 10 episodes: Load shared buffer + model
- Every 100 episodes: Save buffer + model
- File-based (pickle + PyTorch) - simple and effective

## 📊 Monitoring Training

### GUI Indicators

- **Reward Chart**: Should trend upward over time
- **Episode Length**: Increases as agent learns to avoid crashes
- **Debug Panel**: Shows sensor values and reward components
- **Sensor Lines**: Color-coded visualization
  - 🟢 Green = road detected
  - 🟡 Yellow = partial detection
  - 🔴 Red = obstacle

### Performance Metrics

| Metric | Early (0-10k) | Mid (10k-60k) | Late (60k+) |
|--------|---------------|---------------|-------------|
| Crash Rate | Very High | Decreasing | Low |
| Episode Length | Short | Increasing | Long |
| Target Completion | Rare | Occasional | Frequent |
| Exploration Noise | Random/20% | 20%→5% | 5% |

### Expected Timeline

- **10 min**: Random exploration complete, policy starts learning
- **30 min**: Noticeable improvement, occasional target reaching
- **60 min**: Consistent navigation, multi-target completion
- **2+ hours**: Stable performance, high success rate

## 🔧 Troubleshooting

### Common Issues

**Car doesn't move**
- Check if training started (timesteps should increase)
- Verify targets are placed
- Press SPACE to start

**Constant crashes**
- Wait for warm-up phase (10k steps)
- Check sensor visualization (should detect obstacles)
- Try simpler map first
- Verify car is placed on white road

**No learning progress**
- Ensure replay buffer has >256 samples
- Check reward chart for trends
- Increase training time (T3D needs 50k+ steps)
- Verify exploration noise is decaying

**Random behavior after training**
- Check total timesteps (should be >60k for stable policy)
- Verify model was saved/loaded correctly
- Ensure exploration noise has decayed (<10%)

**Parallel training not working**
- Check worker logs for errors: `tail -f worker1.log`
- Verify shared files exist: `shared_buffer.pkl`, `t3d_model.pth`
- Ensure all instances use same file paths
- Kill old processes: `pkill -f citymap_t3d.py`

## 🎨 Advanced Usage

### Custom Maps

**Requirements:**
- PNG/JPG format
- White/bright areas = road (RGB >200, low variance)
- Dark/colored areas = obstacles
- Recommended: 800x600 to 1200x900 pixels

**Load custom map:**
1. Click `LOAD MAP` button
2. Select your image file
3. Place car and targets on white areas

### Multi-Target Navigation

- Click multiple times to create waypoint sequence
- Car navigates to each target in order
- Useful for testing path planning
- Targets cycle if all completed

### Model Persistence

**Save model:**
```python
# Saves to t3d_model.pth
- Actor network weights
- Critic network weights
- Target network weights
- Training timesteps
```

**Load model:**
- Continue training from checkpoint
- Use for inference/demonstration
- Compatible across sessions
- Share trained models with others

## 📈 Performance Comparison

### T3D vs DQN

| Aspect | DQN | T3D (This Project) |
|--------|-----|--------------------|
| Action Space | Discrete (5 actions) | Continuous (2D) |
| Control | Jerky, grid-like | Smooth, natural |
| Exploration | Epsilon-greedy | Gaussian noise (decaying) |
| Q-Networks | Single | Twin (reduces overestimation) |
| Policy | Implicit (argmax Q) | Explicit (Actor network) |
| Updates | Every step | Delayed (stability) |
| Warm-up | None | 10k random steps |
| Parallel Training | Difficult | Native support |

### Key Improvements

✅ **Smooth control** - Continuous actions vs discrete  
✅ **Faster learning** - Parallel training (4x speedup)  
✅ **Better stability** - Twin critics + delayed updates  
✅ **Smarter exploration** - Decaying noise prevents late crashes  
✅ **Visual debugging** - Real-time sensor and reward visualization

## 🔬 Technical Details

### Key Design Decisions

**Binary Sensors (vs Brightness)**
- Clear distinction: road vs obstacle
- More robust than brightness averaging
- Prevents misclassification of colored areas
- Matches human intuition

**Exploration Decay**
- Constant noise causes late-stage crashes
- Linear decay (20%→5%) balances exploration/exploitation
- Standard practice in RL (like epsilon decay in DQN)

**Parallel Training**
- Off-policy algorithm enables experience sharing
- File-based sync is simple and effective
- No network overhead for single-machine setup
- Each instance explores different scenarios

**Reward Simplification**
- Simpler rewards often work better
- Avoid over-penalization (causes edge cases)
- Balance safety and progress
- 3 components: safety, progress, alignment

## 📚 References

- **T3D Paper**: ["Addressing Function Approximation Error in Actor-Critic Methods"](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018)
- **DDPG**: ["Continuous control with deep reinforcement learning"](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2015)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Prioritized Experience Replay
- Curriculum Learning
- Recurrent networks for memory
- Multi-GPU training
- Additional maps and scenarios

## 📝 License

MIT License - Educational implementation for TSAI ERA course.

## 🙏 Acknowledgments

- TSAI ERA course for the foundation
- PyTorch team for the framework
- Fujimoto et al. for the T3D algorithm

---

**Happy Training! 🚗💨**
