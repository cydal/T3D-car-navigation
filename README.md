# T3D Car Navigation - Continuous Control

Implementation of **Twin Delayed Deep Deterministic Policy Gradient (T3D)** for autonomous car navigation with continuous action space.

## Overview

This implementation replaces the discrete Q-Learning approach with T3D, enabling:
- **Continuous steering control**: -30° to +30°
- **Continuous speed control**: 0 to 3 pixels/step
- **Twin critic networks** to reduce overestimation bias
- **Delayed policy updates** for stability
- **Target policy smoothing** for robustness

## Key Features

### T3D Algorithm Components
1. **Actor Network**: Maps states → continuous actions [steering, speed]
2. **Twin Critics**: Two Q-networks (Q1, Q2) to estimate action values
3. **Target Networks**: Delayed copies updated via Polyak averaging
4. **Replay Buffer**: Experience replay with 1M capacity
5. **Exploration**: Gaussian noise added to actions

### Simplified Reward Function (2 Components)
1. **Road adherence**: Bonus for staying on road (center sensor)
2. **Distance penalty**: Penalty for moving away from target

### Training Strategy
- **Warm-up phase**: 10,000 random exploration steps
- **Training phase**: Policy-guided actions with Gaussian noise
- **Batch size**: 100 transitions per update
- **Policy frequency**: Actor updated every 2 critic updates

## Files

- `t3d.py`: Core T3D implementation (Actor, Critic, ReplayBuffer, T3D agent)
- `citymap_t3d.py`: Main application with CarBrain and PyQt6 visualization
- `city_map.png`: Default navigation map
- `README.md`: This file

## Requirements

```bash
pip install torch numpy PyQt6
```

## Usage

### 1. Start the Application
```bash
cd RL-t3d
python citymap_t3d.py
```

### 2. Setup Navigation
1. **Click on map** to place the car (starting position)
2. **Click on map** to add target(s) - supports multiple sequential targets
3. **Right-click** when done adding targets
4. **Press SPACE** or click START to begin training

### 3. Training Phases

**Phase 1: Exploration (Steps 0-10,000)**
- Random actions to populate replay buffer
- No policy learning yet

**Phase 2: Training (Steps 10,000+)**
- Policy-guided actions with exploration noise
- Continuous learning from experience

### 4. Controls

- **START/PAUSE**: Toggle training (or press SPACE)
- **RESET ALL**: Clear everything and start over
- **LOAD MAP**: Load custom map image
- **SAVE MODEL**: Save trained actor and critic networks
- **LOAD MODEL**: Load pre-trained model

## Hyperparameters

```python
# Physics
MAX_STEERING = 30.0     # Maximum steering angle (degrees)
MAX_SPEED = 3.0         # Maximum speed (pixels/step)
SENSOR_DIST = 16        # Sensor range (pixels)
TARGET_RADIUS = 20      # Target reach threshold (pixels)

# T3D
BATCH_SIZE = 100        # Training batch size
GAMMA = 0.99            # Discount factor
TAU = 0.005             # Soft update rate
POLICY_NOISE = 0.2      # Target policy smoothing noise
NOISE_CLIP = 0.5        # Noise clipping range
POLICY_FREQ = 2         # Delayed policy update frequency
LR = 3e-4               # Learning rate
EXPL_NOISE = 0.1        # Exploration noise std
START_TIMESTEPS = 10000 # Random exploration steps
```

## Action Space

**Continuous 2D vector**: `[steering_angle, speed]`

- **Steering**: -30° to +30° (left to right)
- **Speed**: 0 to 3 pixels/step

Actions are output by the Actor network using `tanh` activation scaled by max values.

## State Space

**9-dimensional vector**:
1. Sensor 1 (left -45°): Road brightness [0-1]
2. Sensor 2 (-30°): Road brightness [0-1]
3. Sensor 3 (-15°): Road brightness [0-1]
4. Sensor 4 (center 0°): Road brightness [0-1]
5. Sensor 5 (+15°): Road brightness [0-1]
6. Sensor 6 (+30°): Road brightness [0-1]
7. Sensor 7 (right +45°): Road brightness [0-1]
8. Angle to target: Normalized [-1, 1]
9. Distance to target: Normalized [0, 1]

## Reward Structure

```python
# Base time penalty
reward = -0.1

# Crash (off road)
if brightness < 0.4:
    reward = -100
    done = True

# Target reached
elif distance < 20:
    reward = 100
    # Switch to next target if available

# Otherwise
else:
    # Component 1: Road adherence (center sensor)
    reward += (1.0 - center_sensor) * 20
    
    # Component 2: Distance penalty
    if distance > previous_distance:
        reward -= 10
```

## Network Architectures

### Actor (State → Action)
```
Input (9) → FC(400) → ReLU → FC(300) → ReLU → FC(2) → Tanh → Scale
```

### Critic (State + Action → Q-value)
```
Twin networks:
Input (9+2) → FC(400) → ReLU → FC(300) → ReLU → FC(1)
```

## Training Tips

1. **Start simple**: Use single target on simple map first
2. **Monitor exploration**: First 10k steps should show random behavior
3. **Watch reward chart**: Look for upward trend after warm-up
4. **Be patient**: T3D needs ~50k-100k steps to show good performance
5. **Save frequently**: Use SAVE MODEL button to checkpoint progress

## Comparison with DQN

| Aspect | DQN (Previous) | T3D (Current) |
|--------|----------------|---------------|
| Action Space | Discrete (5 actions) | Continuous (2D) |
| Exploration | Epsilon-greedy | Gaussian noise |
| Q-Networks | Single | Twin (Q1, Q2) |
| Policy | Implicit (argmax Q) | Explicit (Actor) |
| Updates | Every step | Delayed (actor/2) |
| Warm-up | None | 10k random steps |

## Troubleshooting

**Car doesn't move**: Check if training started (should see timesteps increasing)

**Always crashes**: 
- Ensure warm-up phase completed (10k steps)
- Try simpler map first
- Check sensor visualization (should turn red near obstacles)

**No learning progress**:
- Verify replay buffer has enough samples (>100)
- Check reward chart for trends
- Increase training time (T3D is slower than DQN)

**Actions seem random after training**:
- Ensure you're past warm-up phase
- Check if model was saved/loaded correctly
- Verify exploration noise isn't too high

## Advanced Usage

### Custom Maps
- Load any PNG/JPG image
- White/bright areas = road
- Dark areas = obstacles
- Recommended: 800x600 to 1200x900 pixels

### Multi-Target Navigation
- Click multiple times to create waypoint sequence
- Car will navigate to each target in order
- Useful for testing path planning capabilities

### Model Persistence
- Models saved as `{name}_actor.pth` and `{name}_critic.pth`
- Load to continue training or for inference
- Compatible across sessions

## References

- **T3D Paper**: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)
- **DDPG**: "Continuous control with deep reinforcement learning" (Lillicrap et al., 2015)

## License

Educational implementation for TSAI ERA course.
