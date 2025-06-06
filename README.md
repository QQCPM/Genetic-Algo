m# Genetic Algorithm Car Simulation

A comprehensive genetic algorithm implementation for evolving neural networks that control autonomous cars navigating complex tracks. This project demonstrates machine learning, evolutionary computation, and neural network visualization techniques.

## 🚗 Project Overview

This simulation evolves populations of cars that learn to navigate tracks using neural networks. Each car has sensors to detect obstacles and uses a neural network brain to make driving decisions. Through genetic algorithms, successful cars pass their traits to the next generation, gradually improving performance.

## 📁 Project Structure

```
Genetic Algo/
├── main.py                 # Main simulation (Pygame implementation)
├── car.html               # Web-based simulation (JavaScript/HTML)
├── config.py              # Configuration management system
├── analytics.py           # Performance tracking and analysis
├── track_generator.py     # Dynamic track generation
├── neural_visualizer.py   # Neural network visualization tools
├── experiment_runner.py   # Automated experiment runner
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Features

### Core Simulation
- **Pygame-based simulation** with real-time visualization
- **Web-based version** for browser compatibility
- **Neural network-controlled cars** with customizable architectures
- **Multiple sensor types** for environment perception
- **Physics simulation** with realistic car movement

### Track System
- **Dynamic track generation** with multiple track types:
  - Oval tracks
  - Figure-eight layouts
  - Maze environments
  - Spiral tracks
  - Complex racing circuits
  - Obstacle courses
- **Checkpoint systems** for progress tracking
- **Collision detection** with walls and obstacles

### Genetic Algorithm
- **Population-based evolution** with configurable parameters
- **Multiple selection methods** (tournament, elitism)
- **Crossover and mutation** operators
- **Fitness evaluation** based on multiple criteria
- **Real-time performance tracking**

### Neural Networks
- **Configurable architectures** (input → hidden layers → output)
- **Multiple activation functions** (tanh, sigmoid, ReLU)
- **Weight visualization** and analysis
- **Real-time network activity display**

### Analysis & Visualization
- **Performance tracking** across generations
- **Statistical analysis** with matplotlib visualizations
- **Experiment comparison** tools
- **Neural network structure visualization**
- **Evolution progress animations**

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Run the main simulation:**
```bash
python main.py
```

2. **Open the web version:**
Open `car.html` in your browser for the JavaScript implementation.

3. **Run automated experiments:**
```bash
python experiment_runner.py
```

## ⚙️ Configuration

The simulation is highly configurable through the `config.py` system:

```python
from config import Config

# Create custom configuration
config = Config()
config.genetic.POPULATION_SIZE = 50
config.genetic.MUTATION_RATE = 0.1
config.neural_network.LAYER_SIZES = [5, 8, 6, 2]

# Save configuration
config.save_to_file("my_config.json")
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `POPULATION_SIZE` | Number of cars per generation | 50 |
| `MUTATION_RATE` | Probability of weight mutation | 0.1 |
| `LAYER_SIZES` | Neural network architecture | [5, 8, 6, 2] |
| `MAX_SPEED` | Maximum car velocity | 5.0 |
| `NUM_SENSORS` | Number of distance sensors | 5 |

## 🧠 Neural Network Architecture

Cars use feedforward neural networks with:
- **Input layer**: Sensor readings (distance to obstacles)
- **Hidden layers**: Configurable depth and width
- **Output layer**: Steering and throttle controls

Example architecture:
```
Sensors (5) → Hidden (8) → Hidden (6) → Controls (2)
     ↓              ↓           ↓            ↓
[s1,s2,s3,s4,s5] → [...] → [...] → [steering, throttle]
```

## 📊 Experiment System

### Running Experiments

```python
from experiment_runner import ExperimentRunner, ExperimentConfig
from config import Config
from track_generator import TrackType

# Create experiment
config = Config()
config.genetic.POPULATION_SIZE = 30

experiment = ExperimentConfig(
    name="fast_evolution",
    description="Quick evolution experiment",
    config=config,
    track_type=TrackType.OVAL,
    max_generations=50
)

# Run experiment
runner = ExperimentRunner()
runner.add_experiment(experiment)
runner.run_all_experiments()
```

### Preset Experiments

The system includes several preset experiments:
- **Small Fast**: Small population, quick evolution
- **Large Slow**: Large population, thorough exploration
- **Complex Network**: Deep neural architecture
- **High Mutation**: Aggressive exploration strategy
- **Complex Track**: Challenging environment testing

## 🏁 Track Generation

Generate custom tracks programmatically:

```python
from track_generator import TrackGenerator, TrackType

generator = TrackGenerator(1400, 800)

# Generate different track types
oval_track = generator.generate_track(TrackType.OVAL)
maze_track = generator.generate_track(TrackType.MAZE, grid_size=15)
circuit_track = generator.generate_track(TrackType.CIRCUIT)

# Save tracks
generator.save_track("my_track.json", oval_track)
```

## 📈 Analytics & Visualization

### Performance Tracking

```python
from analytics import PerformanceTracker

tracker = PerformanceTracker("my_experiment")
# ... during simulation ...
tracker.record_generation(generation, population, generation_time)

# Generate reports
tracker.plot_fitness_evolution()
tracker.generate_report()
```

### Neural Network Visualization

```python
from neural_visualizer import NeuralNetworkVisualizer

visualizer = NeuralNetworkVisualizer()
network_surface = visualizer.draw_network(car.brain, sensor_inputs)
```

## 🎮 Controls

### Simulation Controls
- **Space**: Pause/Resume simulation
- **S**: Toggle sensor visualization
- **↑/↓**: Adjust simulation speed
- **R**: Reset current generation
- **N**: Force next generation

### Web Version Controls
- Interactive buttons for all functions
- Real-time neural network display
- Adjustable simulation parameters

## 🔬 Genetic Algorithm Details

### Selection Methods
- **Tournament Selection**: Best performers from random groups
- **Elitism**: Preserve top performers across generations
- **Fitness Proportionate**: Selection based on relative fitness

### Fitness Function
Cars are evaluated on multiple criteria:
- Distance traveled
- Checkpoints reached
- Survival time
- Speed maintenance
- Collision avoidance

### Mutation Operators
- **Weight Perturbation**: Small random changes to network weights
- **Bias Adjustment**: Modification of neuron bias values
- **Architecture Mutation**: (Optional) Structural network changes

## 📊 Performance Metrics

The system tracks comprehensive metrics:
- **Fitness Evolution**: Best/average/worst per generation
- **Distance Metrics**: Total and average distance traveled
- **Checkpoint Progress**: Navigation milestones
- **Survival Rates**: Population longevity statistics
- **Network Analysis**: Weight distributions and activations

## 🛠️ Advanced Features

### Custom Fitness Functions
Define specialized evaluation criteria:

```python
def custom_fitness(car):
    base_fitness = car.distance_traveled * 0.5
    checkpoint_bonus = car.checkpoints_passed * 1000
    speed_bonus = car.average_speed * 10
    return base_fitness + checkpoint_bonus + speed_bonus
```

### Network Architecture Search
Automatically find optimal network structures:

```python
architectures = [
    [5, 4, 2],      # Simple
    [5, 8, 6, 2],   # Medium
    [5, 12, 8, 4, 2] # Complex
]

for arch in architectures:
    config.neural_network.LAYER_SIZES = arch
    # Run experiment and compare results
```

## 🐛 Troubleshooting

### Common Issues

1. **Slow Performance**
   - Reduce population size
   - Simplify neural network
   - Lower simulation FPS

2. **Poor Learning**
   - Increase mutation rate
   - Adjust fitness function
   - Try different network architectures

3. **Crashes/Instability**
   - Check sensor configurations
   - Validate track generation
   - Review mutation parameters

## 📚 Educational Value

This project demonstrates:
- **Genetic Algorithms**: Population-based optimization
- **Neural Networks**: Feedforward architectures and training
- **Reinforcement Learning**: Environment-agent interaction
- **Computer Graphics**: Real-time visualization with Pygame
- **Software Architecture**: Modular, configurable systems
- **Data Analysis**: Performance tracking and visualization

## 🤝 Contributing

1. Fork the repository
2. Create feature branches
3. Add comprehensive tests
4. Document new features
5. Submit pull requests

## 📜 License

This project is open source and available under the MIT License.

## 🔗 References

- Genetic Algorithms: Holland, J.H. (1992)
- Neural Networks: Goodfellow, I. et al. (2016)
- Evolutionary Computation: Eiben, A.E. & Smith, J.E. (2015)
- Game AI: Millington, I. & Funge, J. (2009)

---

## 🎯 Future Enhancements

- **3D Visualization**: Three-dimensional track environments
- **Multi-objective Optimization**: Pareto-optimal solutions
- **Online Learning**: Real-time network adaptation
- **Distributed Computing**: Parallel population evolution
- **Advanced Physics**: More realistic vehicle dynamics
- **Machine Learning Integration**: Hybrid learning approaches

Happy evolving! 🧬🚗