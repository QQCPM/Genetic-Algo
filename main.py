import pygame
import numpy as np
import random
import math
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 600  # 10x faster
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
MAX_GENERATION_TIME = 13  # seconds - original 3 + 10 more
SENSOR_LENGTH = 100
NUM_SENSORS = 5
NUM_GOAL_INPUTS = 4  # distance, angle, velocity_toward_goal, progress_rate

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
ORANGE = (255, 165, 0)
DARK_GRAY = (64, 64, 64)


class CarState(Enum):
    ALIVE = 1
    CRASHED = 2
    FINISHED = 3


@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class LSTMCell:
    """Simple LSTM cell for sequential memory"""
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM weights (forget, input, candidate, output gates)
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_f = np.ones(hidden_size) * 0.5  # Forget gate bias (start forgetting slowly)
        
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_i = np.zeros(hidden_size)
        
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_c = np.zeros(hidden_size)
        
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.b_o = np.zeros(hidden_size)
        
        # States
        self.h = np.zeros(hidden_size)  # Hidden state
        self.c = np.zeros(hidden_size)  # Cell state
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, self.h])
        
        # Forget gate
        f = self._sigmoid(np.dot(self.W_f, combined) + self.b_f)
        
        # Input gate
        i = self._sigmoid(np.dot(self.W_i, combined) + self.b_i)
        
        # Candidate values
        c_tilde = np.tanh(np.dot(self.W_c, combined) + self.b_c)
        
        # Update cell state
        self.c = f * self.c + i * c_tilde
        
        # Output gate
        o = self._sigmoid(np.dot(self.W_o, combined) + self.b_o)
        
        # Update hidden state
        self.h = o * np.tanh(self.c)
        
        return self.h
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def reset_state(self):
        """Reset LSTM state for new sequence"""
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)
    
    def copy(self) -> 'LSTMCell':
        new_lstm = LSTMCell(self.input_size, self.hidden_size)
        new_lstm.W_f = self.W_f.copy()
        new_lstm.b_f = self.b_f.copy()
        new_lstm.W_i = self.W_i.copy()
        new_lstm.b_i = self.b_i.copy()
        new_lstm.W_c = self.W_c.copy()
        new_lstm.b_c = self.b_c.copy()
        new_lstm.W_o = self.W_o.copy()
        new_lstm.b_o = self.b_o.copy()
        new_lstm.h = self.h.copy()
        new_lstm.c = self.c.copy()
        return new_lstm
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """Mutate LSTM parameters"""
        for param in [self.W_f, self.W_i, self.W_c, self.W_o]:
            mask = np.random.random(param.shape) < mutation_rate
            param += mask * np.random.randn(*param.shape) * mutation_strength
        
        for param in [self.b_f, self.b_i, self.b_c, self.b_o]:
            mask = np.random.random(param.shape) < mutation_rate
            param += mask * np.random.randn(*param.shape) * mutation_strength


class MemoryNeuralNetwork:
    """Enhanced neural network with LSTM memory and attention"""
    def __init__(self, input_size: int, lstm_hidden_size: int = 16, output_size: int = 2):
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size
        
        # LSTM for memory
        self.lstm = LSTMCell(input_size, lstm_hidden_size)
        
        # Attention mechanism weights
        self.attention_weights = np.random.randn(lstm_hidden_size, input_size) * 0.1
        self.attention_bias = np.zeros(input_size)
        
        # Output layers
        hidden_size = lstm_hidden_size + input_size  # LSTM output + attended input
        self.output_weights = np.random.randn(output_size, hidden_size) * 0.3
        self.output_bias = np.zeros(output_size)
        
        # Memory storage for path learning
        self.memory_buffer = []  # Store recent state-action pairs
        self.max_memory = 20
        
        # Behavioral patterns learned
        self.learned_patterns = {}
        self.pattern_usage_count = {}
        
        # Layer outputs for visualization
        self.layerOutputs = []
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layerOutputs = [inputs]
        
        # LSTM forward pass
        lstm_output = self.lstm.forward(inputs)
        self.layerOutputs.append(lstm_output)
        
        # Attention mechanism
        attention_scores = np.dot(lstm_output, self.attention_weights) + self.attention_bias
        attention_weights = self._softmax(attention_scores)
        attended_input = inputs * attention_weights
        
        # Combine LSTM output with attended input
        combined_features = np.concatenate([lstm_output, attended_input])
        self.layerOutputs.append(combined_features)
        
        # Output layer
        output = np.dot(self.output_weights, combined_features) + self.output_bias
        output = np.tanh(output)  # Keep output in [-1, 1] range
        self.layerOutputs.append(output)
        
        # Store in memory for learning
        self._update_memory(inputs, output)
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)
    
    def _update_memory(self, inputs: np.ndarray, output: np.ndarray):
        """Update memory buffer with recent experiences"""
        memory_entry = {
            'input': inputs.copy(),
            'output': output.copy(),
            'lstm_state': self.lstm.h.copy()
        }
        
        self.memory_buffer.append(memory_entry)
        if len(self.memory_buffer) > self.max_memory:
            self.memory_buffer.pop(0)
    
    def recall_similar_situation(self, current_input: np.ndarray, threshold: float = 0.8) -> Optional[np.ndarray]:
        """Recall action from similar past situation"""
        if not self.memory_buffer:
            return None
        
        best_similarity = 0
        best_action = None
        
        for memory in self.memory_buffer:
            # Calculate similarity (normalized dot product)
            similarity = np.dot(current_input, memory['input']) / (
                np.linalg.norm(current_input) * np.linalg.norm(memory['input']) + 1e-8
            )
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_action = memory['output']
        
        return best_action
    
    def reset_memory(self):
        """Reset memory for new episode"""
        self.lstm.reset_state()
        self.memory_buffer = []
    
    def copy(self) -> 'MemoryNeuralNetwork':
        new_nn = MemoryNeuralNetwork(self.input_size, self.lstm_hidden_size, self.output_size)
        new_nn.lstm = self.lstm.copy()
        new_nn.attention_weights = self.attention_weights.copy()
        new_nn.attention_bias = self.attention_bias.copy()
        new_nn.output_weights = self.output_weights.copy()
        new_nn.output_bias = self.output_bias.copy()
        new_nn.learned_patterns = self.learned_patterns.copy()
        new_nn.pattern_usage_count = self.pattern_usage_count.copy()
        return new_nn
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.2):
        # Mutate LSTM
        self.lstm.mutate(mutation_rate, mutation_strength * 0.5)  # Gentler mutation for LSTM
        
        # Mutate attention weights
        mask = np.random.random(self.attention_weights.shape) < mutation_rate
        self.attention_weights += mask * np.random.randn(*self.attention_weights.shape) * mutation_strength
        
        mask = np.random.random(self.attention_bias.shape) < mutation_rate
        self.attention_bias += mask * np.random.randn(*self.attention_bias.shape) * mutation_strength
        
        # Mutate output weights
        mask = np.random.random(self.output_weights.shape) < mutation_rate
        self.output_weights += mask * np.random.randn(*self.output_weights.shape) * mutation_strength
        
        mask = np.random.random(self.output_bias.shape) < mutation_rate
        self.output_bias += mask * np.random.randn(*self.output_bias.shape) * mutation_strength
    
    # Compatibility properties for existing code
    @property
    def layer_sizes(self):
        return [self.input_size, self.lstm_hidden_size, self.lstm_hidden_size + self.input_size, self.output_size]


class NeuralNetwork:
    """Legacy neural network class (kept for compatibility)"""
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.5
            bias_vector = np.random.randn(layer_sizes[i + 1]) * 0.5
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(weight, activation) + bias
            # Use tanh activation for all layers
            activation = np.tanh(z)
        return activation

    def copy(self) -> 'NeuralNetwork':
        new_nn = NeuralNetwork(self.layer_sizes)
        new_nn.weights = [w.copy() for w in self.weights]
        new_nn.biases = [b.copy() for b in self.biases]
        return new_nn

    def mutate(self, mutation_rate: float, mutation_strength: float = 0.2):
        for i in range(len(self.weights)):
            # Mutate weights
            mask = np.random.random(self.weights[i].shape) < mutation_rate
            self.weights[i] += mask * np.random.randn(*self.weights[i].shape) * mutation_strength

            # Mutate biases
            mask = np.random.random(self.biases[i].shape) < mutation_rate
            self.biases[i] += mask * np.random.randn(*self.biases[i].shape) * mutation_strength


def crossover(parent1, parent2):
    """Enhanced crossover for both legacy and memory neural networks"""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Handle MemoryNeuralNetwork crossover
    if isinstance(parent1, MemoryNeuralNetwork) and isinstance(parent2, MemoryNeuralNetwork):
        # LSTM crossover
        if random.random() < CROSSOVER_RATE:
            # Crossover LSTM weights
            for attr in ['W_f', 'W_i', 'W_c', 'W_o']:
                param1 = getattr(child1.lstm, attr)
                param2 = getattr(child2.lstm, attr)
                mask = np.random.random(param1.shape) < 0.5
                temp = param1[mask].copy()
                param1[mask] = param2[mask]
                param2[mask] = temp
        
        # Attention weights crossover
        if random.random() < CROSSOVER_RATE:
            mask = np.random.random(child1.attention_weights.shape) < 0.5
            temp = child1.attention_weights[mask].copy()
            child1.attention_weights[mask] = child2.attention_weights[mask]
            child2.attention_weights[mask] = temp
        
        # Output weights crossover
        if random.random() < CROSSOVER_RATE:
            mask = np.random.random(child1.output_weights.shape) < 0.5
            temp = child1.output_weights[mask].copy()
            child1.output_weights[mask] = child2.output_weights[mask]
            child2.output_weights[mask] = temp
        
        # Share successful patterns
        if random.random() < 0.3:  # 30% chance to share learned patterns
            combined_patterns = {**parent1.learned_patterns, **parent2.learned_patterns}
            child1.learned_patterns = combined_patterns.copy()
            child2.learned_patterns = combined_patterns.copy()
    
    # Handle legacy NeuralNetwork crossover
    elif hasattr(parent1, 'weights') and hasattr(parent2, 'weights'):
        for i in range(len(parent1.weights)):
            # Crossover weights
            if random.random() < CROSSOVER_RATE:
                mask = np.random.random(parent1.weights[i].shape) < 0.5
                temp = child1.weights[i][mask].copy()
                child1.weights[i][mask] = child2.weights[i][mask]
                child2.weights[i][mask] = temp

            # Crossover biases
            if random.random() < CROSSOVER_RATE:
                mask = np.random.random(parent1.biases[i].shape) < 0.5
                temp = child1.biases[i][mask].copy()
                child1.biases[i][mask] = child2.biases[i][mask]
                child2.biases[i][mask] = temp

    return child1, child2


class Track:
    def __init__(self):
        self.walls = []
        self.checkpoints = []
        self.goal = Point(1200, 400)  # Goal position
        self.start_pos = Point(100, 400)
        self.start_angle = 0
        self._create_track()

    def _create_track(self):
        # Simple track - just boundary walls
        outer_walls = [
            (50, 50, 1350, 50),  # Top wall
            (1350, 50, 1350, 750),  # Right wall
            (1350, 750, 50, 750),  # Bottom wall
            (50, 750, 50, 50),  # Left wall
        ]
        
        # Add a few simple obstacles
        simple_obstacles = [
            (400, 200, 500, 300),  # Small rectangle
            (700, 500, 800, 600),  # Another rectangle
            (900, 200, 1000, 250),  # Top obstacle
        ]

        self.walls = outer_walls + simple_obstacles

    def draw(self, screen):
        for wall in self.walls:
            pygame.draw.line(screen, WHITE, (wall[0], wall[1]), (wall[2], wall[3]), 3)

        # Draw goal
        pygame.draw.circle(screen, GREEN, (int(self.goal.x), int(self.goal.y)), 40, 3)
        pygame.draw.circle(screen, GREEN, (int(self.goal.x), int(self.goal.y)), 5)

        # Draw start position
        pygame.draw.circle(screen, YELLOW, (int(self.start_pos.x), int(self.start_pos.y)), 10)

    def check_collision(self, point: Point) -> bool:
        x, y = point.x, point.y

        # Check boundary collision
        if x < 50 or x > 1350 or y < 50 or y > 750:
            return True

        # Check wall collision using point-to-line distance
        for wall in self.walls:
            if self._point_to_line_distance(point, wall) < 10:  # Car radius
                return True

        return False

    def _point_to_line_distance(self, point: Point, line: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = line
        x0, y0 = point.x, point.y

        # Calculate distance from point to line segment
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            return math.sqrt(A * A + B * B)

        param = dot / len_sq

        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        dx = x0 - xx
        dy = y0 - yy
        return math.sqrt(dx * dx + dy * dy)


class Car:
    def __init__(self, track: Track, brain: Optional[NeuralNetwork] = None):
        self.track = track
        self.position = Point(track.start_pos.x, track.start_pos.y)
        self.angle = track.start_angle
        self.velocity = Point(0, 0)
        self.state = CarState.ALIVE
        self.fitness = 0
        self.distance_traveled = 0
        self.time_alive = 0
        self.checkpoints_passed = 0
        self.last_checkpoint = -1

        # Memory-enhanced neural network with goal-oriented inputs
        total_inputs = NUM_SENSORS + NUM_GOAL_INPUTS  # sensors + goal info
        if brain is None:
            # Use new memory neural network with LSTM
            self.brain = MemoryNeuralNetwork(total_inputs, lstm_hidden_size=20, output_size=2)
        else:
            self.brain = brain

        # Car properties
        self.max_speed = 5
        self.acceleration = 0.2
        self.friction = 0.95
        self.turn_speed = 0.1
        
        # Goal tracking
        self.reached_goal = False
        
        # Random movement patterns
        self.movement_pattern = random.choice(['explorer', 'zigzag', 'spiral', 'random_turn'])
        self.pattern_timer = 0
        self.turn_direction = random.choice([-1, 1])  # For consistent turning
        self.zigzag_phase = 0
        
        # Reward/Punishment tracking
        self.wall_hits = 0
        self.wall_hit_penalty = 0
        self.goal_reward = 0
        self.last_distance_to_goal = float('inf')
        self.progress_reward = 0
        
        # Advanced tracking for enhanced fitness
        self.path_efficiency = 0
        self.stagnation_time = 0
        self.last_position = Point(self.position.x, self.position.y)
        self.momentum_toward_goal = 0
        self.total_path_length = 0
        self.straight_line_distance = 0
        self.speed_consistency_score = 0
        self.exploration_bonus = 0
        self.visited_areas = set()  # Track explored grid cells
        self.smooth_movement_score = 0
        self.direction_changes = 0
        self.last_angle = 0
        self.velocity_history = []
        self.goal_approach_efficiency = 0

        # Sensors
        self.sensor_readings = [0] * NUM_SENSORS
        self.sensor_lines = []

        # Visual properties
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def update(self, dt: float):
        if self.state != CarState.ALIVE:
            return

        self.time_alive += dt

        # Update sensors
        self._update_sensors()

        # Get neural network decision with goal-oriented inputs
        goal_inputs = self._get_goal_oriented_inputs()
        all_inputs = self.sensor_readings + goal_inputs
        inputs = np.array(all_inputs)
        
        # Memory-enhanced decision making
        outputs = self._make_memory_enhanced_decision(inputs)

        # Extract steering and throttle from outputs
        steering = outputs[0]  # -1 to 1
        throttle = max(0, outputs[1])  # 0 to 1
        
        # Apply random movement patterns
        self.pattern_timer += dt
        pattern_steering = self._get_pattern_steering()
        
        # Combine neural network output with random pattern
        steering = np.clip(steering * 0.3 + pattern_steering * 0.7, -1, 1)



        # Apply steering
        self.angle += steering * self.turn_speed

        # Apply throttle
        acceleration_x = math.cos(self.angle) * throttle * self.acceleration
        acceleration_y = math.sin(self.angle) * throttle * self.acceleration

        self.velocity.x += acceleration_x
        self.velocity.y += acceleration_y

        # Apply friction
        self.velocity.x *= self.friction
        self.velocity.y *= self.friction

        # Limit speed
        speed = math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2)
        if speed > self.max_speed:
            self.velocity.x = (self.velocity.x / speed) * self.max_speed
            self.velocity.y = (self.velocity.y / speed) * self.max_speed

        # Update position
        old_pos = Point(self.position.x, self.position.y)
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y

        # Update distance traveled and path tracking
        step_distance = old_pos.distance_to(self.position)
        self.distance_traveled += step_distance
        self.total_path_length += step_distance
        
        # Update movement quality tracking
        self._update_movement_tracking(dt, old_pos)
        
        # Update exploration tracking
        self._update_exploration_tracking()
        
        # Check collision with punishment
        if self.track.check_collision(self.position):
            self.wall_hits += 1
            self.wall_hit_penalty += 100  # Immediate penalty for hitting wall
            self.state = CarState.CRASHED
            return

        # Check if reached goal
        goal_reached_this_step = self._check_goal()
        
        # If goal was just reached, store this as a successful memory pattern
        if goal_reached_this_step and isinstance(self.brain, MemoryNeuralNetwork):
            self._store_successful_pattern()

        # Calculate fitness
        self._calculate_fitness()

    def _update_sensors(self):
        self.sensor_lines = []
        self.sensor_readings = []

        # Cast rays in different directions
        sensor_angles = []
        for i in range(NUM_SENSORS):
            angle_offset = (i - NUM_SENSORS // 2) * (math.pi / 4) / (NUM_SENSORS // 2)
            sensor_angles.append(self.angle + angle_offset)

        for angle in sensor_angles:
            distance = self._cast_ray(angle)
            normalized_distance = min(distance / SENSOR_LENGTH, 1.0)
            self.sensor_readings.append(normalized_distance)

    def _cast_ray(self, angle: float) -> float:
        start_x, start_y = self.position.x, self.position.y
        end_x = start_x + math.cos(angle) * SENSOR_LENGTH
        end_y = start_y + math.sin(angle) * SENSOR_LENGTH

        min_distance = SENSOR_LENGTH

        # Check collision with walls
        for wall in self.track.walls:
            intersection = self._line_intersection(
                (start_x, start_y, end_x, end_y),
                wall
            )
            if intersection:
                distance = math.sqrt((intersection[0] - start_x) ** 2 + (intersection[1] - start_y) ** 2)
                min_distance = min(min_distance, distance)

        # Store sensor line for visualization
        actual_end_x = start_x + math.cos(angle) * min_distance
        actual_end_y = start_y + math.sin(angle) * min_distance
        self.sensor_lines.append(((start_x, start_y), (actual_end_x, actual_end_y)))

        return min_distance

    def _line_intersection(self, line1: Tuple[float, float, float, float],
                           line2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)

        return None

    def _get_goal_oriented_inputs(self) -> List[float]:
        """Generate goal-oriented inputs for neural network"""
        # 1. Normalized distance to goal (0-1)
        distance_to_goal = self.position.distance_to(self.track.goal)
        max_distance = math.sqrt((1350-100)**2 + (750-400)**2)
        normalized_distance = min(distance_to_goal / max_distance, 1.0)
        
        # 2. Angle to goal relative to car heading (-1 to 1)
        goal_dx = self.track.goal.x - self.position.x
        goal_dy = self.track.goal.y - self.position.y
        goal_angle = math.atan2(goal_dy, goal_dx)
        relative_angle = goal_angle - self.angle
        
        # Normalize to -1 to 1
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        normalized_angle = relative_angle / math.pi
        
        # 3. Velocity toward goal (-1 to 1)
        velocity_magnitude = math.sqrt(self.velocity.x**2 + self.velocity.y**2)
        if velocity_magnitude > 0:
            # Dot product of velocity and goal direction
            goal_direction_x = goal_dx / distance_to_goal if distance_to_goal > 0 else 0
            goal_direction_y = goal_dy / distance_to_goal if distance_to_goal > 0 else 0
            velocity_toward_goal = (self.velocity.x * goal_direction_x + self.velocity.y * goal_direction_y) / velocity_magnitude
        else:
            velocity_toward_goal = 0
        
        # 4. Progress rate (how quickly getting to goal, -1 to 1)
        if self.last_distance_to_goal != float('inf'):
            progress_rate = (self.last_distance_to_goal - distance_to_goal) / max(self.last_distance_to_goal, 1)
            progress_rate = np.clip(progress_rate, -1, 1)
        else:
            progress_rate = 0
        
        return [normalized_distance, normalized_angle, velocity_toward_goal, progress_rate]
    
    def _make_memory_enhanced_decision(self, inputs: np.ndarray) -> np.ndarray:
        """Make decision using memory-enhanced neural network"""
        # Get standard neural network output
        nn_output = self.brain.forward(inputs)
        
        # Check if we have a memory-enhanced network
        if isinstance(self.brain, MemoryNeuralNetwork):
            # Try to recall similar past situations
            recalled_action = self.brain.recall_similar_situation(inputs, threshold=0.75)
            
            if recalled_action is not None:
                # Blend current decision with recalled successful action
                # Weight: 70% current NN, 30% memory recall
                blended_output = 0.7 * nn_output + 0.3 * recalled_action
                return np.clip(blended_output, -1, 1)
        
        return nn_output

    def _get_pattern_steering(self) -> float:
        """Generate steering based on movement pattern"""
        if self.movement_pattern == 'explorer':
            # Random exploration with occasional direction changes
            if self.pattern_timer > 2.0:  # Change direction every 2 seconds
                self.turn_direction = random.choice([-1, 1])
                self.pattern_timer = 0
            return self.turn_direction * 0.3 + (random.random() - 0.5) * 0.4
            
        elif self.movement_pattern == 'zigzag':
            # Zigzag pattern
            self.zigzag_phase += 0.1
            return math.sin(self.zigzag_phase) * 0.8
            
        elif self.movement_pattern == 'spiral':
            # Spiral/circular movement with varying radius
            spiral_intensity = 0.5 + 0.3 * math.sin(self.pattern_timer * 0.5)
            return self.turn_direction * spiral_intensity
            
        elif self.movement_pattern == 'random_turn':
            # Completely random turning
            if self.pattern_timer > 0.5:  # Change every 0.5 seconds
                self.turn_direction = (random.random() - 0.5) * 2
                self.pattern_timer = 0
            return self.turn_direction
            
        return 0

    def _check_goal(self) -> bool:
        distance_to_goal = self.position.distance_to(self.track.goal)
        goal_reached_this_step = False
        
        # Reward for getting closer to goal
        if self.last_distance_to_goal != float('inf'):
            if distance_to_goal < self.last_distance_to_goal:
                self.progress_reward += (self.last_distance_to_goal - distance_to_goal) * 2
        self.last_distance_to_goal = distance_to_goal
        
        # Big reward for reaching goal
        if distance_to_goal < 40 and not self.reached_goal:  # Goal radius
            self.reached_goal = True
            goal_reached_this_step = True
            self.goal_reward = 10000 + (1000 / max(self.time_alive, 1))  # Bonus for speed
            self.state = CarState.FINISHED
        
        return goal_reached_this_step
    
    def _store_successful_pattern(self):
        """Store successful behavior pattern when goal is reached"""
        if isinstance(self.brain, MemoryNeuralNetwork):
            # Mark recent memory entries as successful
            for memory in self.brain.memory_buffer[-5:]:  # Last 5 steps before goal
                pattern_key = self._create_pattern_key(memory['input'])
                self.brain.learned_patterns[pattern_key] = memory['output']
                self.brain.pattern_usage_count[pattern_key] = (
                    self.brain.pattern_usage_count.get(pattern_key, 0) + 1
                )
    
    def _create_pattern_key(self, inputs: np.ndarray) -> str:
        """Create a hashable key for input patterns"""
        # Discretize inputs to create pattern key
        discretized = np.round(inputs * 4) / 4  # Round to quarters
        return str(discretized.tolist())

    def _calculate_fitness(self):
        """Advanced fitness calculation with multiple sophisticated components"""
        fitness_components = {}
        
        # 1. GOAL ACHIEVEMENT (highest priority)
        if self.reached_goal:
            # Exponential reward for reaching goal
            time_bonus = max(0, 1000 / max(self.time_alive, 1))  # Faster = better
            efficiency_bonus = self.path_efficiency * 500
            fitness_components['goal_achievement'] = self.goal_reward + time_bonus + efficiency_bonus
        else:
            # Exponential proximity reward (much stronger for being close)
            distance_to_goal = self.position.distance_to(self.track.goal)
            max_distance = math.sqrt((1350-100)**2 + (750-400)**2)
            proximity_ratio = 1.0 - (distance_to_goal / max_distance)
            
            # Exponential curve: being 90% close is much better than 50% close
            exponential_proximity = math.pow(proximity_ratio, 0.5) * 2000
            fitness_components['proximity'] = exponential_proximity
            
            # Progress reward (for consistent movement toward goal)
            fitness_components['progress'] = self.progress_reward
        
        # 2. PATH EFFICIENCY (crucial for intelligent behavior)
        self._update_path_efficiency()
        fitness_components['path_efficiency'] = self.path_efficiency * 300
        
        # 3. MOVEMENT QUALITY
        movement_quality = self._calculate_movement_quality()
        fitness_components['movement_quality'] = movement_quality * 200
        
        # 4. EXPLORATION REWARD (balanced with goal-seeking)
        exploration_score = self._calculate_exploration_score()
        fitness_components['exploration'] = exploration_score * 100
        
        # 5. SPEED AND EFFICIENCY
        speed_efficiency = self._calculate_speed_efficiency()
        fitness_components['speed_efficiency'] = speed_efficiency * 150
        
        # 6. SURVIVAL AND TIME
        survival_base = math.log(1 + self.time_alive) * 20  # Logarithmic to prevent over-emphasis
        fitness_components['survival'] = survival_base
        
        # 7. PENALTIES (exponential punishment for bad behavior)
        penalties = 0
        
        # Wall hit penalties (exponential)
        if self.wall_hits > 0:
            wall_penalty = math.pow(self.wall_hits, 1.5) * 100
            penalties += wall_penalty
        
        # Crash penalty
        if self.state == CarState.CRASHED:
            crash_penalty = 800 + (self.wall_hits * 100)  # Worse if multiple hits
            penalties += crash_penalty
        
        # Stagnation penalty
        if self.stagnation_time > 3.0:
            stagnation_penalty = math.pow(self.stagnation_time - 3.0, 2) * 50
            penalties += stagnation_penalty
        
        fitness_components['penalties'] = -penalties
        
        # 8. BONUS MODIFIERS
        bonuses = 0
        
        # Goal approach efficiency bonus
        if self.goal_approach_efficiency > 0.7:
            bonuses += (self.goal_approach_efficiency - 0.7) * 500
        
        # Smooth movement bonus
        if self.smooth_movement_score > 0.8:
            bonuses += (self.smooth_movement_score - 0.8) * 200
        
        fitness_components['bonuses'] = bonuses
        
        # COMBINE ALL COMPONENTS
        total_fitness = sum(fitness_components.values())
        
        # Apply multipliers for exceptional performance
        if self.reached_goal and self.path_efficiency > 0.85:
            total_fitness *= 1.5  # Exceptional path efficiency multiplier
        
        # Ensure minimum fitness
        self.fitness = max(1, total_fitness)
        
        # Store component breakdown for analysis
        self.fitness_breakdown = fitness_components
    
    def _update_movement_tracking(self, dt: float, old_pos: Point):
        """Track movement quality metrics"""
        # Track direction changes for smooth movement scoring
        angle_change = abs(self.angle - self.last_angle)
        if angle_change > math.pi:
            angle_change = 2 * math.pi - angle_change
        
        if angle_change > 0.2:  # Significant direction change
            self.direction_changes += 1
        
        self.last_angle = self.angle
        
        # Track velocity consistency
        current_speed = math.sqrt(self.velocity.x**2 + self.velocity.y**2)
        self.velocity_history.append(current_speed)
        
        # Keep only recent history
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)
        
        # Check for stagnation
        movement_distance = old_pos.distance_to(self.position)
        if movement_distance < 0.5:  # Very slow movement
            self.stagnation_time += dt
        else:
            self.stagnation_time = max(0, self.stagnation_time - dt * 0.5)
    
    def _update_exploration_tracking(self):
        """Track explored areas for exploration bonus"""
        # Divide world into grid cells (50x50 pixels each)
        grid_x = int(self.position.x // 50)
        grid_y = int(self.position.y // 50)
        self.visited_areas.add((grid_x, grid_y))
    
    def _update_path_efficiency(self):
        """Calculate path efficiency relative to straight-line distance"""
        if self.total_path_length > 0:
            start_pos = self.track.start_pos
            straight_line = start_pos.distance_to(self.position)
            
            # Efficiency = straight line distance / actual path length
            # Perfect efficiency = 1.0, lower = more wandering
            self.path_efficiency = min(1.0, straight_line / self.total_path_length)
        else:
            self.path_efficiency = 0.0
    
    def _calculate_movement_quality(self) -> float:
        """Calculate quality of movement (smoothness, consistency)"""
        quality_score = 0.0
        
        # Smooth movement (fewer direction changes)
        if self.time_alive > 0:
            direction_change_rate = self.direction_changes / self.time_alive
            smoothness = max(0, 1.0 - direction_change_rate / 5.0)  # Normalize
            quality_score += smoothness * 0.4
        
        # Speed consistency
        if len(self.velocity_history) > 3:
            speed_std = np.std(self.velocity_history)
            speed_consistency = max(0, 1.0 - speed_std / 5.0)  # Normalize
            quality_score += speed_consistency * 0.3
        
        # Goal approach efficiency
        distance_to_goal = self.position.distance_to(self.track.goal)
        if distance_to_goal > 0 and len(self.velocity_history) > 0:
            # Check if moving toward goal
            goal_dx = self.track.goal.x - self.position.x
            goal_dy = self.track.goal.y - self.position.y
            goal_distance = math.sqrt(goal_dx**2 + goal_dy**2)
            
            if goal_distance > 0:
                goal_direction_x = goal_dx / goal_distance
                goal_direction_y = goal_dy / goal_distance
                
                # Dot product with velocity direction
                if len(self.velocity_history) > 0:
                    velocity_magnitude = self.velocity_history[-1]
                    if velocity_magnitude > 0:
                        vel_direction_x = self.velocity.x / velocity_magnitude
                        vel_direction_y = self.velocity.y / velocity_magnitude
                        
                        goal_alignment = (goal_direction_x * vel_direction_x + 
                                        goal_direction_y * vel_direction_y)
                        self.goal_approach_efficiency = max(0, goal_alignment)
                        quality_score += self.goal_approach_efficiency * 0.3
        
        self.smooth_movement_score = quality_score
        return quality_score
    
    def _calculate_exploration_score(self) -> float:
        """Calculate exploration bonus (balanced with goal-seeking)"""
        # Reward for visiting new areas, but not too much
        exploration_coverage = len(self.visited_areas)
        
        # Diminishing returns on exploration
        exploration_score = math.log(1 + exploration_coverage) / math.log(100)  # Normalize
        
        # Reduce exploration reward if close to goal (focus on goal)
        distance_to_goal = self.position.distance_to(self.track.goal)
        max_distance = math.sqrt((1350-100)**2 + (750-400)**2)
        proximity_ratio = 1.0 - (distance_to_goal / max_distance)
        
        # When close to goal, reduce exploration importance
        exploration_modifier = 1.0 - (proximity_ratio * 0.7)
        
        return exploration_score * exploration_modifier
    
    def _calculate_speed_efficiency(self) -> float:
        """Calculate speed efficiency (not too fast, not too slow)"""
        if len(self.velocity_history) == 0:
            return 0.0
        
        avg_speed = np.mean(self.velocity_history)
        optimal_speed = self.max_speed * 0.7  # 70% of max speed is optimal
        
        # Efficiency peaks at optimal speed, decreases for too fast/slow
        speed_ratio = avg_speed / optimal_speed
        if speed_ratio <= 1.0:
            efficiency = speed_ratio  # Linear increase up to optimal
        else:
            efficiency = 1.0 / speed_ratio  # Decrease for speeds above optimal
        
        return min(1.0, efficiency)
    
    def reset_for_new_generation(self):
        """Reset memory state for new generation"""
        if isinstance(self.brain, MemoryNeuralNetwork):
            self.brain.reset_memory()

    def draw(self, screen, show_sensors: bool = False):
        if self.state == CarState.CRASHED:
            return

        # Draw car body
        car_length = 20
        car_width = 10

        # Calculate car corners
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)

        front_x = self.position.x + cos_angle * car_length / 2
        front_y = self.position.y + sin_angle * car_length / 2
        back_x = self.position.x - cos_angle * car_length / 2
        back_y = self.position.y - sin_angle * car_length / 2

        left_x = -sin_angle * car_width / 2
        left_y = cos_angle * car_width / 2
        right_x = sin_angle * car_width / 2
        right_y = -cos_angle * car_width / 2

        corners = [
            (front_x + left_x, front_y + left_y),
            (front_x + right_x, front_y + right_y),
            (back_x + right_x, back_y + right_y),
            (back_x + left_x, back_y + left_y)
        ]

        pygame.draw.polygon(screen, self.color, corners)

        # Draw direction indicator
        pygame.draw.circle(screen, WHITE, (int(front_x), int(front_y)), 3)

        # Draw sensors
        if show_sensors:
            for start, end in self.sensor_lines:
                pygame.draw.line(screen, YELLOW, start, end, 1)


class GeneticAlgorithm:
    def __init__(self, population_size: int, track: Track):
        self.population_size = population_size
        self.track = track
        self.generation = 1
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Adaptive mutation system
        self.base_mutation_rate = MUTATION_RATE
        self.current_mutation_rate = MUTATION_RATE
        self.stagnation_counter = 0
        self.fitness_improvement_threshold = 5.0
        self.max_stagnation = 5
        self.mutation_decay = 0.95
        self.mutation_boost = 1.5
        self.min_mutation_rate = 0.01
        self.max_mutation_rate = 0.4
        
        # Performance tracking for adaptation
        self.recent_best_fitnesses = []
        self.performance_window = 3
        self.elite_preservation_rate = 0.15  # 15% of population preserved as elites

        # Create initial population
        for _ in range(population_size):
            car = Car(track)
            self.population.append(car)

    def evolve(self):
        # Sort population by fitness
        self.population.sort(key=lambda car: car.fitness, reverse=True)

        # Record statistics
        best_fitness = self.population[0].fitness
        avg_fitness = sum(car.fitness for car in self.population) / len(self.population)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Update adaptive mutation rate
        self._update_adaptive_mutation(best_fitness)

        # Dynamic elite count based on performance
        elite_count = max(3, int(self.population_size * self.elite_preservation_rate))
        elites = self.population[:elite_count]

        # Selection for breeding (enhanced tournament selection)
        parents = self._enhanced_tournament_selection(self.population_size - elite_count)

        # Create new generation
        new_population = []

        # Add elites with slight variation to prevent exact copies
        for i, elite in enumerate(elites):
            new_car = Car(self.track, elite.brain.copy())
            # Reset memory for new generation
            new_car.reset_for_new_generation()
            # Add minimal mutation to elites (except the very best)
            if i > 0:  # Don't mutate the absolute best
                new_car.brain.mutate(self.current_mutation_rate * 0.1, 0.05)
            new_population.append(new_car)

        # Create offspring with adaptive mutation
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

            child1_brain, child2_brain = crossover(parent1.brain, parent2.brain)
            
            # Apply adaptive mutation with varying strengths
            mutation_strength1 = self._calculate_mutation_strength(parent1.fitness, best_fitness)
            mutation_strength2 = self._calculate_mutation_strength(parent2.fitness, best_fitness)
            
            child1_brain.mutate(self.current_mutation_rate, mutation_strength1)
            child2_brain.mutate(self.current_mutation_rate, mutation_strength2)

            new_car1 = Car(self.track, child1_brain)
            new_car1.reset_for_new_generation()
            new_population.append(new_car1)
            
            if len(new_population) < self.population_size:
                new_car2 = Car(self.track, child2_brain)
                new_car2.reset_for_new_generation()
                new_population.append(new_car2)

        self.population = new_population[:self.population_size]
        self.generation += 1

    def _update_adaptive_mutation(self, current_best_fitness: float):
        """Update mutation rate based on population performance"""
        self.recent_best_fitnesses.append(current_best_fitness)
        
        # Keep only recent history
        if len(self.recent_best_fitnesses) > self.performance_window:
            self.recent_best_fitnesses.pop(0)
        
        # Check for improvement
        if len(self.recent_best_fitnesses) >= 2:
            recent_improvement = self.recent_best_fitnesses[-1] - self.recent_best_fitnesses[-2]
            
            if recent_improvement < self.fitness_improvement_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        # Adjust mutation rate based on stagnation
        if self.stagnation_counter >= self.max_stagnation:
            # Increase mutation to encourage exploration
            self.current_mutation_rate = min(self.max_mutation_rate, 
                                           self.current_mutation_rate * self.mutation_boost)
            self.stagnation_counter = 0  # Reset counter
        else:
            # Gradually reduce mutation rate for exploitation
            self.current_mutation_rate = max(self.min_mutation_rate,
                                           self.current_mutation_rate * self.mutation_decay)
    
    def _calculate_mutation_strength(self, individual_fitness: float, best_fitness: float) -> float:
        """Calculate mutation strength based on individual performance"""
        if best_fitness == 0:
            return 0.2
        
        # Weaker individuals get stronger mutations
        fitness_ratio = individual_fitness / best_fitness
        # Inverse relationship: lower fitness = higher mutation strength
        mutation_strength = 0.1 + (1.0 - fitness_ratio) * 0.3
        return np.clip(mutation_strength, 0.05, 0.5)
    
    def _enhanced_tournament_selection(self, count: int) -> List[Car]:
        """Enhanced tournament selection with diversity consideration"""
        selected = []
        tournament_size = max(3, min(7, self.population_size // 10))
        
        for _ in range(count):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            
            # 80% of time: select best fitness
            # 20% of time: select for diversity (different behavior patterns)
            if random.random() < 0.8:
                winner = max(tournament, key=lambda car: car.fitness)
            else:
                # Diversity selection: prefer different movement patterns
                pattern_counts = {}
                for car in selected:
                    pattern = getattr(car, 'movement_pattern', 'unknown')
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                # Find least represented pattern in tournament
                best_diversity_score = float('inf')
                winner = tournament[0]
                for car in tournament:
                    pattern = getattr(car, 'movement_pattern', 'unknown')
                    diversity_score = pattern_counts.get(pattern, 0)
                    if diversity_score < best_diversity_score:
                        best_diversity_score = diversity_score
                        winner = car
            
            selected.append(winner)
        
        return selected
    
    def _tournament_selection(self, count: int) -> List[Car]:
        selected = []
        tournament_size = 5

        for _ in range(count):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda car: car.fitness)
            selected.append(winner)

        return selected

    def get_statistics(self) -> dict:
        alive_cars = [car for car in self.population if car.state == CarState.ALIVE]

        if not self.population:
            return {}

        best_car = max(self.population, key=lambda car: car.fitness)

        return {
            'generation': self.generation,
            'alive_count': len(alive_cars),
            'total_count': len(self.population),
            'best_fitness': best_car.fitness,
            'avg_fitness': sum(car.fitness for car in self.population) / len(self.population),
            'best_distance': best_car.distance_traveled,
            'goal_reached': best_car.reached_goal,
            'best_time': best_car.time_alive
        }


class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Deep Learning Cars - Neuroevolution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        self.track = Track()
        self.genetic_algorithm = GeneticAlgorithm(POPULATION_SIZE, self.track)

        self.generation_start_time = 0
        self.show_sensors = False
        self.simulation_speed = 1
        self.paused = False

        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.show_sensors = not self.show_sensors
                elif event.key == pygame.K_UP:
                    self.simulation_speed = min(20, self.simulation_speed + 1)
                elif event.key == pygame.K_DOWN:
                    self.simulation_speed = max(1, self.simulation_speed - 1)
                elif event.key == pygame.K_r:
                    self._reset_generation()
                elif event.key == pygame.K_n:
                    self._next_generation()

    def update(self, dt: float):
        if self.paused:
            return

        adjusted_dt = dt * self.simulation_speed

        # Update all cars
        for car in self.genetic_algorithm.population:
            car.update(adjusted_dt)

        # Check if generation should end
        alive_cars = [car for car in self.genetic_algorithm.population
                      if car.state == CarState.ALIVE]
        
        # Check if any car reached the goal
        goal_reached = any(car.reached_goal for car in self.genetic_algorithm.population)

        generation_time = pygame.time.get_ticks() / 1000.0 - self.generation_start_time

        if not alive_cars or generation_time > MAX_GENERATION_TIME or goal_reached:
            self._next_generation()

    def _reset_generation(self):
        for car in self.genetic_algorithm.population:
            car.__init__(self.track, car.brain)
        self.generation_start_time = pygame.time.get_ticks() / 1000.0

    def _next_generation(self):
        self.genetic_algorithm.evolve()
        self.generation_start_time = pygame.time.get_ticks() / 1000.0

    def draw(self):
        self.screen.fill(BLACK)

        # Draw track
        self.track.draw(self.screen)

        # Draw cars
        best_car = max(self.genetic_algorithm.population, key=lambda car: car.fitness)

        for car in self.genetic_algorithm.population:
            if car != best_car:
                car.draw(self.screen, False)

        # Draw best car last (on top) with sensors if enabled
        best_car.draw(self.screen, self.show_sensors)

        # Draw UI
        self._draw_ui()

        pygame.display.flip()

    def _draw_ui(self):
        stats = self.genetic_algorithm.get_statistics()
        if not stats:
            return

        # Background for stats
        ui_rect = pygame.Rect(10, 10, 300, 250)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), ui_rect)
        pygame.draw.rect(self.screen, WHITE, ui_rect, 2)

        y_offset = 20
        line_height = 22

        # Generation info
        texts = [
            f"Generation: {stats['generation']}",
            f"Cars Alive: {stats['alive_count']}/{stats['total_count']}",
            f"Best Fitness: {stats['best_fitness']:.0f}",
            f"Avg Fitness: {stats['avg_fitness']:.0f}",
            f"Best Distance: {stats['best_distance']:.0f}",
            f"Goal Reached: {'YES' if any(car.reached_goal for car in self.genetic_algorithm.population) else 'NO'}",
            f"Best Time: {stats['best_time']:.1f}s",
            "",
            f"Speed: {self.simulation_speed}x",
            f"Sensors: {'ON' if self.show_sensors else 'OFF'}",
            "" if not self.paused else "PAUSED"
        ]

        for text in texts:
            if text:
                surface = self.font.render(text, True, WHITE)
                self.screen.blit(surface, (20, y_offset))
            y_offset += line_height

        # Controls
        controls_y = SCREEN_HEIGHT - 120
        control_texts = [
            "Controls:",
            "SPACE - Pause/Resume",
            "S - Toggle Sensors",
            "↑/↓ - Speed Up/Down",
            "R - Reset Generation",
            "N - Next Generation"
        ]

        for text in control_texts:
            surface = self.small_font.render(text, True, WHITE)
            self.screen.blit(surface, (20, controls_y))
            controls_y += 18

    def run(self):
        self.generation_start_time = pygame.time.get_ticks() / 1000.0

        while self.running:
            dt = self.clock.tick(FPS) / 1000.0

            self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()


if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()