"""
Neural Network Visualization Tool for Genetic Algorithm Car Simulation
Provides real-time and static visualization of neural network structures and activations
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx
from typing import List, Optional, Tuple, Dict, Any
import math
import colorsys


class NeuralNetworkVisualizer:
    """Real-time neural network visualizer using Pygame"""
    
    def __init__(self, screen_width: int = 400, screen_height: int = 300):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = None
        self.font = None
        self.small_font = None
        
        # Visualization parameters
        self.layer_spacing = 80
        self.node_radius = 15
        self.max_connection_width = 5
        self.margin = 30
        
        # Colors
        self.bg_color = (20, 20, 30)
        self.node_color = (60, 60, 80)
        self.active_node_color = (100, 255, 100)
        self.positive_weight_color = (0, 255, 0)
        self.negative_weight_color = (255, 0, 0)
        self.text_color = (255, 255, 255)
        
        # Animation
        self.pulse_timer = 0
        
    def initialize_pygame(self):
        """Initialize Pygame components"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.font = pygame.font.Font(None, 16)
            self.small_font = pygame.font.Font(None, 12)
    
    def draw_network(self, neural_network, inputs: Optional[List[float]] = None, 
                    outputs: Optional[List[float]] = None) -> pygame.Surface:
        """Draw the neural network with current activations"""
        self.initialize_pygame()
        self.screen.fill(self.bg_color)
        
        if not hasattr(neural_network, 'layer_sizes'):
            return self.screen
        
        layer_sizes = neural_network.layer_sizes
        num_layers = len(layer_sizes)
        
        # Calculate layout
        total_width = (num_layers - 1) * self.layer_spacing
        start_x = (self.screen_width - total_width) // 2
        
        # Get network activations
        activations = self._get_activations(neural_network, inputs)
        
        # Draw connections first (so they appear behind nodes)
        self._draw_connections(neural_network, layer_sizes, start_x, activations)
        
        # Draw nodes
        self._draw_nodes(layer_sizes, start_x, activations, inputs, outputs)
        
        # Draw labels
        self._draw_labels(layer_sizes, start_x, inputs, outputs)
        
        # Update pulse timer for animations
        self.pulse_timer += 0.1
        
        return self.screen
    
    def _get_activations(self, neural_network, inputs: Optional[List[float]]) -> List[List[float]]:
        """Get current activations for all layers"""
        if inputs is None or not hasattr(neural_network, 'forward'):
            # Return zero activations
            return [[0.0] * size for size in neural_network.layer_sizes]
        
        try:
            # Feed forward to get activations
            neural_network.forward(np.array(inputs))
            if hasattr(neural_network, 'layerOutputs') and neural_network.layerOutputs:
                return neural_network.layerOutputs
            else:
                return [[0.0] * size for size in neural_network.layer_sizes]
        except:
            return [[0.0] * size for size in neural_network.layer_sizes]
    
    def _draw_connections(self, neural_network, layer_sizes: List[int], start_x: int, 
                         activations: List[List[float]]):
        """Draw connections between neurons"""
        if not hasattr(neural_network, 'weights'):
            return
        
        for layer_idx in range(len(layer_sizes) - 1):
            current_layer_size = layer_sizes[layer_idx]
            next_layer_size = layer_sizes[layer_idx + 1]
            
            current_x = start_x + layer_idx * self.layer_spacing
            next_x = start_x + (layer_idx + 1) * self.layer_spacing
            
            current_y_spacing = (self.screen_height - 2 * self.margin) / max(1, current_layer_size - 1)
            next_y_spacing = (self.screen_height - 2 * self.margin) / max(1, next_layer_size - 1)
            
            try:
                weights = neural_network.weights[layer_idx]
                
                for from_idx in range(current_layer_size):
                    for to_idx in range(next_layer_size):
                        weight = weights[to_idx][from_idx] if len(weights) > to_idx and len(weights[to_idx]) > from_idx else 0
                        
                        # Calculate positions
                        from_y = self.margin + from_idx * current_y_spacing if current_layer_size > 1 else self.screen_height // 2
                        to_y = self.margin + to_idx * next_y_spacing if next_layer_size > 1 else self.screen_height // 2
                        
                        # Calculate connection properties
                        abs_weight = abs(weight)
                        line_width = max(1, int(abs_weight * self.max_connection_width))
                        
                        # Connection color based on weight and activation
                        if weight > 0:
                            base_color = self.positive_weight_color
                        else:
                            base_color = self.negative_weight_color
                        
                        # Modulate opacity based on activation
                        activation_strength = activations[layer_idx][from_idx] if len(activations) > layer_idx and len(activations[layer_idx]) > from_idx else 0
                        opacity = min(255, int(50 + activation_strength * 200))
                        
                        color = (*base_color[:3], opacity)
                        
                        # Draw connection
                        pygame.draw.line(self.screen, base_color, 
                                       (current_x + self.node_radius, from_y), 
                                       (next_x - self.node_radius, to_y), line_width)
            except (IndexError, AttributeError):
                continue
    
    def _draw_nodes(self, layer_sizes: List[int], start_x: int, activations: List[List[float]], 
                   inputs: Optional[List[float]], outputs: Optional[List[float]]):
        """Draw neural network nodes"""
        for layer_idx, layer_size in enumerate(layer_sizes):
            x = start_x + layer_idx * self.layer_spacing
            y_spacing = (self.screen_height - 2 * self.margin) / max(1, layer_size - 1)
            
            for node_idx in range(layer_size):
                y = self.margin + node_idx * y_spacing if layer_size > 1 else self.screen_height // 2
                
                # Get activation value
                activation = 0.0
                if len(activations) > layer_idx and len(activations[layer_idx]) > node_idx:
                    activation = activations[layer_idx][node_idx]
                
                # Node color based on activation
                base_intensity = 60
                activation_intensity = int(activation * 195)
                node_color = (base_intensity + activation_intensity, base_intensity + activation_intensity, 80)
                
                # Add pulse effect for highly active nodes
                if activation > 0.7:
                    pulse = int(20 * math.sin(self.pulse_timer * 5))
                    node_color = tuple(min(255, c + pulse) for c in node_color)
                
                # Draw node
                pygame.draw.circle(self.screen, node_color, (int(x), int(y)), self.node_radius)
                pygame.draw.circle(self.screen, self.text_color, (int(x), int(y)), self.node_radius, 2)
                
                # Draw activation value
                if activation > 0.01:
                    text = self.small_font.render(f"{activation:.2f}", True, self.text_color)
                    text_rect = text.get_rect(center=(int(x), int(y)))
                    self.screen.blit(text, text_rect)
    
    def _draw_labels(self, layer_sizes: List[int], start_x: int, 
                    inputs: Optional[List[float]], outputs: Optional[List[float]]):
        """Draw layer labels and input/output information"""
        layer_labels = ["Input", "Hidden", "Hidden", "Output"]
        
        for layer_idx, layer_size in enumerate(layer_sizes):
            x = start_x + layer_idx * self.layer_spacing
            
            # Layer label
            if layer_idx < len(layer_labels):
                label = layer_labels[layer_idx] if layer_idx < 2 or layer_idx == len(layer_sizes) - 1 else f"Hidden {layer_idx - 1}"
            else:
                label = f"Hidden {layer_idx - 1}"
            
            text = self.font.render(label, True, self.text_color)
            text_rect = text.get_rect(center=(x, 15))
            self.screen.blit(text, text_rect)
            
            # Layer size
            size_text = self.small_font.render(f"({layer_size})", True, (150, 150, 150))
            size_rect = size_text.get_rect(center=(x, self.screen_height - 15))
            self.screen.blit(size_text, size_rect)


class MatplotlibNetworkVisualizer:
    """Static neural network visualizer using Matplotlib"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def create_network_diagram(self, neural_network, inputs: Optional[List[float]] = None, 
                             outputs: Optional[List[float]] = None, save_path: Optional[str] = None):
        """Create a detailed network diagram using matplotlib"""
        if not hasattr(neural_network, 'layer_sizes'):
            print("Neural network must have layer_sizes attribute")
            return
        
        layer_sizes = neural_network.layer_sizes
        
        # Create figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.ax.set_xlim(0, len(layer_sizes) + 1)
        self.ax.set_ylim(0, max(layer_sizes) + 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Calculate positions
        layer_positions = {}
        for layer_idx, layer_size in enumerate(layer_sizes):
            x = layer_idx + 1
            positions = []
            if layer_size == 1:
                positions = [(x, max(layer_sizes) / 2)]
            else:
                y_spacing = (max(layer_sizes) - 1) / (layer_size - 1)
                for node_idx in range(layer_size):
                    y = 1 + node_idx * y_spacing
                    positions.append((x, y))
            layer_positions[layer_idx] = positions
        
        # Draw connections
        self._draw_matplotlib_connections(neural_network, layer_positions, inputs)
        
        # Draw nodes
        self._draw_matplotlib_nodes(neural_network, layer_positions, inputs, outputs)
        
        # Add labels
        self._add_matplotlib_labels(layer_sizes, layer_positions)
        
        plt.title("Neural Network Architecture", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network diagram saved to {save_path}")
        
        return self.fig
    
    def _draw_matplotlib_connections(self, neural_network, layer_positions: Dict, 
                                   inputs: Optional[List[float]]):
        """Draw connections in matplotlib"""
        if not hasattr(neural_network, 'weights'):
            return
        
        for layer_idx in range(len(neural_network.layer_sizes) - 1):
            current_positions = layer_positions[layer_idx]
            next_positions = layer_positions[layer_idx + 1]
            
            try:
                weights = neural_network.weights[layer_idx]
                
                for from_idx, (from_x, from_y) in enumerate(current_positions):
                    for to_idx, (to_x, to_y) in enumerate(next_positions):
                        if len(weights) > to_idx and len(weights[to_idx]) > from_idx:
                            weight = weights[to_idx][from_idx]
                            
                            # Line properties based on weight
                            abs_weight = abs(weight)
                            line_width = abs_weight * 3
                            alpha = min(0.8, abs_weight)
                            color = 'green' if weight > 0 else 'red'
                            
                            self.ax.plot([from_x, to_x], [from_y, to_y], 
                                       color=color, alpha=alpha, linewidth=line_width)
            except (IndexError, AttributeError):
                continue
    
    def _draw_matplotlib_nodes(self, neural_network, layer_positions: Dict, 
                             inputs: Optional[List[float]], outputs: Optional[List[float]]):
        """Draw nodes in matplotlib"""
        # Get activations if possible
        activations = []
        if inputs is not None and hasattr(neural_network, 'forward'):
            try:
                neural_network.forward(np.array(inputs))
                if hasattr(neural_network, 'layerOutputs'):
                    activations = neural_network.layerOutputs
            except:
                pass
        
        for layer_idx, positions in layer_positions.items():
            for node_idx, (x, y) in enumerate(positions):
                # Determine node color based on activation
                activation = 0.0
                if len(activations) > layer_idx and len(activations[layer_idx]) > node_idx:
                    activation = activations[layer_idx][node_idx]
                
                # Color intensity based on activation
                intensity = 0.3 + activation * 0.7
                if layer_idx == 0:  # Input layer
                    color = plt.cm.Blues(intensity)
                elif layer_idx == len(neural_network.layer_sizes) - 1:  # Output layer
                    color = plt.cm.Reds(intensity)
                else:  # Hidden layers
                    color = plt.cm.Greens(intensity)
                
                # Draw node
                circle = patches.Circle((x, y), 0.2, facecolor=color, edgecolor='black', linewidth=2)
                self.ax.add_patch(circle)
                
                # Add activation value
                if activation > 0.01:
                    self.ax.text(x, y, f"{activation:.2f}", ha='center', va='center', 
                               fontsize=8, fontweight='bold')
    
    def _add_matplotlib_labels(self, layer_sizes: List[int], layer_positions: Dict):
        """Add labels to matplotlib diagram"""
        layer_names = ["Input", "Hidden", "Hidden", "Output"]
        
        for layer_idx, positions in layer_positions.items():
            x = positions[0][0]
            y_min = min(pos[1] for pos in positions)
            y_max = max(pos[1] for pos in positions)
            
            # Layer name
            if layer_idx < len(layer_names):
                name = layer_names[layer_idx] if layer_idx < 2 or layer_idx == len(layer_sizes) - 1 else f"Hidden {layer_idx - 1}"
            else:
                name = f"Hidden {layer_idx - 1}"
            
            self.ax.text(x, y_max + 0.5, name, ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
            
            # Layer size
            self.ax.text(x, y_min - 0.5, f"({layer_sizes[layer_idx]})", ha='center', va='top', 
                        fontsize=10, style='italic')


class NetworkEvolutionVisualizer:
    """Visualize how neural networks evolve over generations"""
    
    def __init__(self):
        self.generation_networks = []
        self.generation_fitnesses = []
        
    def record_generation(self, best_network, fitness: float):
        """Record the best network from a generation"""
        if hasattr(best_network, 'copy'):
            self.generation_networks.append(best_network.copy())
        else:
            self.generation_networks.append(best_network)
        self.generation_fitnesses.append(fitness)
    
    def create_evolution_animation(self, save_path: str = "network_evolution.gif"):
        """Create an animated visualization of network evolution"""
        if not self.generation_networks:
            print("No generation data to animate")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def animate(frame):
            if frame >= len(self.generation_networks):
                return
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Plot fitness evolution
            ax1.plot(range(len(self.generation_fitnesses[:frame+1])), 
                    self.generation_fitnesses[:frame+1], 'b-', linewidth=2)
            ax1.scatter(frame, self.generation_fitnesses[frame], color='red', s=100, zorder=5)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Best Fitness')
            ax1.set_title(f'Fitness Evolution (Generation {frame+1})')
            ax1.grid(True, alpha=0.3)
            
            # Visualize current network
            network = self.generation_networks[frame]
            visualizer = MatplotlibNetworkVisualizer()
            visualizer.ax = ax2
            visualizer._draw_simplified_network(network)
            ax2.set_title(f'Best Network - Generation {frame+1}')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.generation_networks), 
                           interval=1000, repeat=True)
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=1)
        print(f"Evolution animation saved to {save_path}")
        
        return anim
    
    def compare_network_generations(self, generation_indices: List[int], save_path: Optional[str] = None):
        """Compare networks from different generations"""
        if not generation_indices or max(generation_indices) >= len(self.generation_networks):
            print("Invalid generation indices")
            return
        
        fig, axes = plt.subplots(1, len(generation_indices), figsize=(5*len(generation_indices), 6))
        if len(generation_indices) == 1:
            axes = [axes]
        
        for i, gen_idx in enumerate(generation_indices):
            network = self.generation_networks[gen_idx]
            fitness = self.generation_fitnesses[gen_idx]
            
            visualizer = MatplotlibNetworkVisualizer()
            visualizer.ax = axes[i]
            visualizer._draw_simplified_network(network)
            axes[i].set_title(f'Generation {gen_idx+1}\nFitness: {fitness:.1f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Generation comparison saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the visualizers with a mock neural network
    class MockNeuralNetwork:
        def __init__(self):
            self.layer_sizes = [5, 8, 6, 2]
            self.weights = []
            self.biases = []
            self.layerOutputs = []
            
            # Initialize with random weights
            for i in range(len(self.layer_sizes) - 1):
                layer_weights = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * 0.5
                layer_biases = np.random.randn(self.layer_sizes[i+1]) * 0.5
                self.weights.append(layer_weights)
                self.biases.append(layer_biases)
        
        def forward(self, inputs):
            self.layerOutputs = [inputs]
            current = inputs
            
            for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                current = np.tanh(np.dot(weight, current) + bias)
                self.layerOutputs.append(current)
            
            return current
        
        def copy(self):
            new_net = MockNeuralNetwork()
            new_net.weights = [w.copy() for w in self.weights]
            new_net.biases = [b.copy() for b in self.biases]
            return new_net
    
    # Test pygame visualizer
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Neural Network Visualizer Test")
    clock = pygame.time.Clock()
    
    network = MockNeuralNetwork()
    visualizer = NeuralNetworkVisualizer(400, 300)
    
    # Test matplotlib visualizer
    plt_visualizer = MatplotlibNetworkVisualizer()
    inputs = [0.5, 0.3, 0.8, 0.1, 0.9]
    network.forward(np.array(inputs))
    
    fig = plt_visualizer.create_network_diagram(network, inputs, save_path="test_network.png")
    plt.show()
    
    # Test pygame visualization
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update inputs with some animation
        time = pygame.time.get_ticks() / 1000.0
        inputs = [0.5 + 0.3 * math.sin(time + i) for i in range(5)]
        
        # Draw network
        network_surface = visualizer.draw_network(network, inputs)
        screen.fill((40, 40, 50))
        screen.blit(network_surface, (200, 150))
        
        # Add title
        font = pygame.font.Font(None, 36)
        title = font.render("Neural Network Visualization", True, (255, 255, 255))
        screen.blit(title, (200, 50))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("Neural network visualizer test completed!")