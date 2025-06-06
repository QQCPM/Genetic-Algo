"""
Configuration file for Genetic Algorithm Car Simulation
Easily adjust parameters to experiment with different evolutionary strategies
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


@dataclass
class DisplayConfig:
    """Display and rendering settings"""
    SCREEN_WIDTH: int = 1400
    SCREEN_HEIGHT: int = 800
    FPS: int = 60
    SHOW_SENSORS: bool = False
    SHOW_TRAILS: bool = True
    TRAIL_LENGTH: int = 50
    
    # Colors (RGB tuples)
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    WALL_COLOR: Tuple[int, int, int] = (255, 255, 255)
    BEST_CAR_COLOR: Tuple[int, int, int] = (0, 255, 0)
    CHECKPOINT_COLOR: Tuple[int, int, int] = (0, 255, 0)


@dataclass
class GeneticConfig:
    """Genetic algorithm parameters"""
    POPULATION_SIZE: int = 50
    MUTATION_RATE: float = 0.1
    CROSSOVER_RATE: float = 0.7
    ELITE_COUNT: int = 5  # Number of best cars to preserve
    TOURNAMENT_SIZE: int = 5
    MUTATION_STRENGTH: float = 0.2
    
    # Evolution conditions
    MAX_GENERATION_TIME: float = 30.0  # seconds
    MAX_STUCK_TIME: float = 5.0  # seconds before car is considered stuck


@dataclass
class NeuralNetworkConfig:
    """Neural network architecture settings"""
    LAYER_SIZES: List[int] = None
    ACTIVATION_FUNCTION: str = "tanh"  # "tanh", "sigmoid", "relu"
    WEIGHT_INIT_RANGE: float = 0.5
    BIAS_INIT_RANGE: float = 0.5
    
    def __post_init__(self):
        if self.LAYER_SIZES is None:
            self.LAYER_SIZES = [5, 8, 6, 2]  # 5 sensors -> 2 outputs


@dataclass
class CarConfig:
    """Car physics and sensor settings"""
    MAX_SPEED: float = 5.0
    ACCELERATION: float = 0.2
    FRICTION: float = 0.95
    TURN_SPEED: float = 0.1
    
    # Sensors
    NUM_SENSORS: int = 5
    SENSOR_LENGTH: float = 100.0
    SENSOR_ANGLES: List[float] = None  # Will be auto-generated if None
    
    # Physics
    CAR_WIDTH: int = 20
    CAR_HEIGHT: int = 10
    COLLISION_RADIUS: float = 10.0
    
    def __post_init__(self):
        if self.SENSOR_ANGLES is None:
            # Generate evenly spaced sensor angles
            angle_step = 90 / (self.NUM_SENSORS - 1) if self.NUM_SENSORS > 1 else 0
            self.SENSOR_ANGLES = [
                -45 + i * angle_step for i in range(self.NUM_SENSORS)
            ]


@dataclass
class FitnessConfig:
    """Fitness function weights and parameters"""
    DISTANCE_WEIGHT: float = 1.0
    TIME_WEIGHT: float = 10.0
    CHECKPOINT_WEIGHT: float = 1000.0
    CRASH_PENALTY: float = -500.0
    SPEED_BONUS_WEIGHT: float = 5.0
    PROGRESS_WEIGHT: float = 50.0


@dataclass
class TrackConfig:
    """Track generation and checkpoint settings"""
    TRACK_TYPE: str = "figure_eight"  # "figure_eight", "oval", "complex", "random"
    WALL_THICKNESS: int = 3
    CHECKPOINT_RADIUS: int = 30
    MIN_CHECKPOINT_DISTANCE: float = 100.0
    
    # Track bounds
    TRACK_MARGIN: int = 50
    
    # Custom track points (for manual track design)
    CUSTOM_WALLS: Optional[List[Tuple[int, int, int, int]]] = None
    CUSTOM_CHECKPOINTS: Optional[List[Tuple[int, int]]] = None


@dataclass
class ExperimentConfig:
    """Experiment tracking and data collection"""
    SAVE_BEST_NETWORKS: bool = True
    SAVE_FREQUENCY: int = 10  # Save every N generations
    LOG_LEVEL: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    METRICS_TO_TRACK: List[str] = None
    
    def __post_init__(self):
        if self.METRICS_TO_TRACK is None:
            self.METRICS_TO_TRACK = [
                "best_fitness", "avg_fitness", "best_distance", 
                "avg_distance", "checkpoints_passed", "generation_time"
            ]


class Config:
    """Main configuration class combining all settings"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.display = DisplayConfig()
        self.genetic = GeneticConfig()
        self.neural_network = NeuralNetworkConfig()
        self.car = CarConfig()
        self.fitness = FitnessConfig()
        self.track = TrackConfig()
        self.experiment = ExperimentConfig()
        
        if config_file:
            self.load_from_file(config_file)
    
    def save_to_file(self, filename: str):
        """Save configuration to JSON file"""
        config_dict = {
            "display": asdict(self.display),
            "genetic": asdict(self.genetic),
            "neural_network": asdict(self.neural_network),
            "car": asdict(self.car),
            "fitness": asdict(self.fitness),
            "track": asdict(self.track),
            "experiment": asdict(self.experiment)
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filename}")
    
    def load_from_file(self, filename: str):
        """Load configuration from JSON file"""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            
            # Update configurations
            for section_name, section_data in config_dict.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            print(f"Configuration loaded from {filename}")
        except FileNotFoundError:
            print(f"Config file {filename} not found, using defaults")
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
    
    def create_preset(self, preset_name: str):
        """Create predefined configuration presets"""
        if preset_name == "fast_evolution":
            self.genetic.POPULATION_SIZE = 30
            self.genetic.MAX_GENERATION_TIME = 15.0
            self.genetic.MUTATION_RATE = 0.15
            self.display.FPS = 120
            
        elif preset_name == "high_quality":
            self.genetic.POPULATION_SIZE = 100
            self.genetic.MAX_GENERATION_TIME = 60.0
            self.genetic.MUTATION_RATE = 0.05
            self.neural_network.LAYER_SIZES = [5, 12, 8, 4, 2]
            
        elif preset_name == "simple_experiment":
            self.genetic.POPULATION_SIZE = 20
            self.neural_network.LAYER_SIZES = [5, 4, 2]
            self.car.MAX_SPEED = 3.0
            
        elif preset_name == "complex_track":
            self.track.TRACK_TYPE = "complex"
            self.car.NUM_SENSORS = 7
            self.neural_network.LAYER_SIZES = [7, 10, 6, 2]
            
        else:
            print(f"Unknown preset: {preset_name}")
    
    def validate(self):
        """Validate configuration parameters"""
        errors = []
        
        # Validate neural network architecture
        if len(self.neural_network.LAYER_SIZES) < 2:
            errors.append("Neural network must have at least 2 layers")
        
        if self.neural_network.LAYER_SIZES[0] != self.car.NUM_SENSORS:
            errors.append(f"First layer size ({self.neural_network.LAYER_SIZES[0]}) must match number of sensors ({self.car.NUM_SENSORS})")
        
        if self.neural_network.LAYER_SIZES[-1] != 2:
            errors.append("Output layer must have 2 neurons (steering, throttle)")
        
        # Validate genetic algorithm parameters
        if self.genetic.POPULATION_SIZE < 2:
            errors.append("Population size must be at least 2")
        
        if not 0 <= self.genetic.MUTATION_RATE <= 1:
            errors.append("Mutation rate must be between 0 and 1")
        
        if not 0 <= self.genetic.CROSSOVER_RATE <= 1:
            errors.append("Crossover rate must be between 0 and 1")
        
        if self.genetic.ELITE_COUNT >= self.genetic.POPULATION_SIZE:
            errors.append("Elite count must be less than population size")
        
        # Validate car parameters
        if self.car.MAX_SPEED <= 0:
            errors.append("Max speed must be positive")
        
        if self.car.NUM_SENSORS != len(self.car.SENSOR_ANGLES):
            errors.append("Number of sensors must match sensor angles list length")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed")
        return True
    
    def print_summary(self):
        """Print a summary of current configuration"""
        print("\n" + "="*50)
        print("GENETIC ALGORITHM CAR SIMULATION CONFIG")
        print("="*50)
        print(f"Population Size: {self.genetic.POPULATION_SIZE}")
        print(f"Neural Network: {' -> '.join(map(str, self.neural_network.LAYER_SIZES))}")
        print(f"Mutation Rate: {self.genetic.MUTATION_RATE}")
        print(f"Max Speed: {self.car.MAX_SPEED}")
        print(f"Sensors: {self.car.NUM_SENSORS}")
        print(f"Track Type: {self.track.TRACK_TYPE}")
        print(f"Screen Size: {self.display.SCREEN_WIDTH}x{self.display.SCREEN_HEIGHT}")
        print("="*50)


# Create default configuration instance
default_config = Config()

# Example usage and preset configurations
if __name__ == "__main__":
    # Create and save default configuration
    config = Config()
    config.save_to_file("default_config.json")
    
    # Create and save preset configurations
    fast_config = Config()
    fast_config.create_preset("fast_evolution")
    fast_config.save_to_file("fast_evolution_config.json")
    
    high_quality_config = Config()
    high_quality_config.create_preset("high_quality")
    high_quality_config.save_to_file("high_quality_config.json")
    
    simple_config = Config()
    simple_config.create_preset("simple_experiment")
    simple_config.save_to_file("simple_experiment_config.json")
    
    print("Configuration files created successfully!")
    print("Available configs: default_config.json, fast_evolution_config.json, high_quality_config.json, simple_experiment_config.json")