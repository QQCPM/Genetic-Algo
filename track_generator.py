"""
Track Generator for Genetic Algorithm Car Simulation
Generates various track layouts and environments for testing cars
"""

import numpy as np
import pygame
import math
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import json


class TrackType(Enum):
    OVAL = "oval"
    FIGURE_EIGHT = "figure_eight"
    MAZE = "maze"
    SPIRAL = "spiral"
    RANDOM = "random"
    CIRCUIT = "circuit"
    OBSTACLE_COURSE = "obstacle_course"


@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: int = 3


@dataclass
class Checkpoint:
    position: Point
    radius: float
    checkpoint_id: int


class TrackGenerator:
    """Generates different types of racing tracks"""
    
    def __init__(self, width: int = 1400, height: int = 800, margin: int = 50):
        self.width = width
        self.height = height
        self.margin = margin
        self.walls = []
        self.checkpoints = []
        self.start_position = Point(100, height // 2)
        self.start_angle = 0
        
    def generate_track(self, track_type: TrackType, **kwargs) -> Dict:
        """Generate a track of the specified type"""
        self.walls = []
        self.checkpoints = []
        
        if track_type == TrackType.OVAL:
            return self._generate_oval(**kwargs)
        elif track_type == TrackType.FIGURE_EIGHT:
            return self._generate_figure_eight(**kwargs)
        elif track_type == TrackType.MAZE:
            return self._generate_maze(**kwargs)
        elif track_type == TrackType.SPIRAL:
            return self._generate_spiral(**kwargs)
        elif track_type == TrackType.RANDOM:
            return self._generate_random(**kwargs)
        elif track_type == TrackType.CIRCUIT:
            return self._generate_circuit(**kwargs)
        elif track_type == TrackType.OBSTACLE_COURSE:
            return self._generate_obstacle_course(**kwargs)
        else:
            return self._generate_oval()
    
    def _generate_oval(self, width_ratio: float = 0.7, height_ratio: float = 0.5) -> Dict:
        """Generate a simple oval track"""
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Outer oval
        outer_width = int((self.width - 2 * self.margin) * width_ratio)
        outer_height = int((self.height - 2 * self.margin) * height_ratio)
        
        # Inner oval (smaller)
        inner_width = outer_width - 200
        inner_height = outer_height - 120
        
        # Generate oval points
        num_points = 64
        outer_points = self._generate_oval_points(center_x, center_y, outer_width, outer_height, num_points)
        inner_points = self._generate_oval_points(center_x, center_y, inner_width, inner_height, num_points)
        
        # Create walls from points
        self._points_to_walls(outer_points)
        self._points_to_walls(inner_points)
        
        # Generate checkpoints along the track
        checkpoint_count = 8
        for i in range(checkpoint_count):
            angle = (i / checkpoint_count) * 2 * math.pi
            # Place checkpoints between inner and outer walls
            mid_width = (outer_width + inner_width) // 4
            mid_height = (outer_height + inner_height) // 4
            x = center_x + math.cos(angle) * mid_width
            y = center_y + math.sin(angle) * mid_height
            self.checkpoints.append(Checkpoint(Point(x, y), 25, i))
        
        # Set start position
        self.start_position = Point(center_x - outer_width//2 + 50, center_y)
        self.start_angle = 0
        
        return self._create_track_dict("Oval Track")
    
    def _generate_figure_eight(self) -> Dict:
        """Generate a figure-eight track"""
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Two circles for figure-eight
        circle_radius = min(self.width, self.height) // 6
        offset = circle_radius + 50
        
        # Left circle
        left_center = (center_x - offset, center_y)
        left_outer = self._generate_circle_points(*left_center, circle_radius + 60, 32)
        left_inner = self._generate_circle_points(*left_center, circle_radius, 32)
        
        # Right circle
        right_center = (center_x + offset, center_y)
        right_outer = self._generate_circle_points(*right_center, circle_radius + 60, 32)
        right_inner = self._generate_circle_points(*right_center, circle_radius, 32)
        
        # Create walls
        self._points_to_walls(left_outer)
        self._points_to_walls(left_inner)
        self._points_to_walls(right_outer)
        self._points_to_walls(right_inner)
        
        # Add connecting sections
        self._add_figure_eight_connections(left_center, right_center, circle_radius)
        
        # Generate checkpoints
        self._generate_figure_eight_checkpoints(left_center, right_center, circle_radius)
        
        self.start_position = Point(left_center[0] - circle_radius - 30, left_center[1])
        return self._create_track_dict("Figure-Eight Track")
    
    def _generate_maze(self, grid_size: int = 15) -> Dict:
        """Generate a maze-like track"""
        cell_width = (self.width - 2 * self.margin) // grid_size
        cell_height = (self.height - 2 * self.margin) // grid_size
        
        # Create maze grid
        maze = self._create_maze_grid(grid_size, grid_size)
        
        # Convert maze to walls
        for row in range(grid_size):
            for col in range(grid_size):
                x = self.margin + col * cell_width
                y = self.margin + row * cell_height
                
                if maze[row][col] == 1:  # Wall
                    # Create a wall block
                    self.walls.extend([
                        Wall(x, y, x + cell_width, y),
                        Wall(x + cell_width, y, x + cell_width, y + cell_height),
                        Wall(x + cell_width, y + cell_height, x, y + cell_height),
                        Wall(x, y + cell_height, x, y)
                    ])
        
        # Add boundary walls
        self._add_boundary_walls()
        
        # Generate sparse checkpoints in open areas
        self._generate_maze_checkpoints(maze, grid_size, cell_width, cell_height)
        
        self.start_position = Point(self.margin + cell_width//2, self.margin + cell_height//2)
        return self._create_track_dict("Maze Track")
    
    def _generate_spiral(self, turns: int = 4) -> Dict:
        """Generate a spiral track"""
        center_x = self.width // 2
        center_y = self.height // 2
        max_radius = min(self.width, self.height) // 3
        
        # Generate spiral points
        num_points = 100
        outer_spiral = []
        inner_spiral = []
        
        for i in range(num_points):
            progress = i / num_points
            angle = progress * turns * 2 * math.pi
            radius = max_radius * (0.2 + 0.8 * progress)
            
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            outer_spiral.append((x, y))
            
            # Inner spiral (offset)
            inner_radius = radius - 80
            if inner_radius > 0:
                inner_x = center_x + math.cos(angle) * inner_radius
                inner_y = center_y + math.sin(angle) * inner_radius
                inner_spiral.append((inner_x, inner_y))
        
        # Create walls from spiral
        self._points_to_walls(outer_spiral)
        if inner_spiral:
            self._points_to_walls(inner_spiral)
        
        # Generate checkpoints along spiral
        checkpoint_interval = len(outer_spiral) // 6
        for i in range(0, len(outer_spiral), checkpoint_interval):
            if i < len(outer_spiral) and i < len(inner_spiral):
                # Place checkpoint between spirals
                outer_point = outer_spiral[i]
                inner_point = inner_spiral[min(i, len(inner_spiral)-1)]
                mid_x = (outer_point[0] + inner_point[0]) / 2
                mid_y = (outer_point[1] + inner_point[1]) / 2
                self.checkpoints.append(Checkpoint(Point(mid_x, mid_y), 20, len(self.checkpoints)))
        
        self.start_position = Point(center_x + 50, center_y)
        return self._create_track_dict("Spiral Track")
    
    def _generate_random(self, obstacle_count: int = 15) -> Dict:
        """Generate a random obstacle course"""
        # Add boundary walls
        self._add_boundary_walls()
        
        # Generate random obstacles
        for _ in range(obstacle_count):
            # Random rectangular obstacles
            width = random.randint(20, 100)
            height = random.randint(20, 100)
            x = random.randint(self.margin + 100, self.width - self.margin - width - 100)
            y = random.randint(self.margin + 50, self.height - self.margin - height - 50)
            
            self.walls.extend([
                Wall(x, y, x + width, y),
                Wall(x + width, y, x + width, y + height),
                Wall(x + width, y + height, x, y + height),
                Wall(x, y + height, x, y)
            ])
        
        # Generate random checkpoints
        for i in range(8):
            while True:
                x = random.randint(self.margin + 50, self.width - self.margin - 50)
                y = random.randint(self.margin + 50, self.height - self.margin - 50)
                point = Point(x, y)
                
                # Check if point is clear of obstacles
                if self._is_point_clear(point, 30):
                    self.checkpoints.append(Checkpoint(point, 25, i))
                    break
        
        self.start_position = Point(self.margin + 50, self.height // 2)
        return self._create_track_dict("Random Obstacle Course")
    
    def _generate_circuit(self) -> Dict:
        """Generate a complex racing circuit"""
        # Main outer boundary
        self._add_boundary_walls()
        
        # Add chicanes and turns
        self._add_chicane(300, 200, 150, 100)
        self._add_chicane(800, 500, 200, 120)
        self._add_chicane(1100, 300, 100, 150)
        
        # Add hairpin turn
        self._add_hairpin(600, 600, 80)
        
        # Add S-curve
        self._add_s_curve(400, 100, 300, 200)
        
        # Generate circuit checkpoints
        checkpoint_positions = [
            (200, 400), (400, 200), (700, 150), (1000, 300),
            (1200, 500), (1000, 650), (600, 700), (300, 600)
        ]
        
        for i, (x, y) in enumerate(checkpoint_positions):
            self.checkpoints.append(Checkpoint(Point(x, y), 30, i))
        
        self.start_position = Point(150, 400)
        return self._create_track_dict("Racing Circuit")
    
    def _generate_obstacle_course(self, difficulty: str = "medium") -> Dict:
        """Generate an obstacle course with varying difficulty"""
        obstacle_count = {"easy": 8, "medium": 12, "hard": 18}[difficulty]
        
        # Add boundary
        self._add_boundary_walls()
        
        # Add specific obstacle patterns
        self._add_slalom_course(200, 300, 800, 5)
        self._add_tunnel_obstacles(300, 500, 600, 3)
        self._add_maze_section(900, 200, 400, 300)
        
        # Add random obstacles
        for _ in range(obstacle_count - 8):
            self._add_random_obstacle()
        
        # Strategic checkpoint placement
        self._generate_strategic_checkpoints()
        
        self.start_position = Point(self.margin + 30, self.height // 2)
        return self._create_track_dict(f"Obstacle Course ({difficulty.title()})")
    
    # Helper methods for track generation
    def _generate_oval_points(self, center_x: float, center_y: float, 
                            width: float, height: float, num_points: int) -> List[Tuple[float, float]]:
        """Generate points for an oval shape"""
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = center_x + (width / 2) * math.cos(angle)
            y = center_y + (height / 2) * math.sin(angle)
            points.append((x, y))
        return points
    
    def _generate_circle_points(self, center_x: float, center_y: float, 
                              radius: float, num_points: int) -> List[Tuple[float, float]]:
        """Generate points for a circle"""
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        return points
    
    def _points_to_walls(self, points: List[Tuple[float, float]]):
        """Convert a list of points to walls"""
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            self.walls.append(Wall(x1, y1, x2, y2))
    
    def _add_boundary_walls(self):
        """Add boundary walls around the track"""
        self.walls.extend([
            Wall(self.margin, self.margin, self.width - self.margin, self.margin),
            Wall(self.width - self.margin, self.margin, self.width - self.margin, self.height - self.margin),
            Wall(self.width - self.margin, self.height - self.margin, self.margin, self.height - self.margin),
            Wall(self.margin, self.height - self.margin, self.margin, self.margin)
        ])
    
    def _add_chicane(self, x: float, y: float, width: float, height: float):
        """Add a chicane obstacle"""
        self.walls.extend([
            Wall(x, y, x + width//3, y),
            Wall(x + width//3, y, x + width//3, y + height//2),
            Wall(x + 2*width//3, y + height//2, x + 2*width//3, y + height),
            Wall(x + 2*width//3, y + height, x + width, y + height)
        ])
    
    def _add_hairpin(self, center_x: float, center_y: float, radius: float):
        """Add a hairpin turn"""
        points = self._generate_circle_points(center_x, center_y, radius, 16)
        # Only use half the circle for hairpin
        half_points = points[:len(points)//2]
        self._points_to_walls(half_points)
    
    def _add_s_curve(self, start_x: float, start_y: float, width: float, height: float):
        """Add an S-shaped curve"""
        # Create S-curve using bezier-like points
        points = []
        num_points = 20
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start_x + width * t
            y = start_y + height * math.sin(t * math.pi) * 0.5
            points.append((x, y))
        self._points_to_walls(points)
    
    def _add_slalom_course(self, start_x: float, start_y: float, length: float, obstacle_count: int):
        """Add a slalom course with alternating obstacles"""
        spacing = length / obstacle_count
        for i in range(obstacle_count):
            x = start_x + i * spacing
            y = start_y + (50 if i % 2 == 0 else -50)
            # Small rectangular obstacle
            self.walls.extend([
                Wall(x, y, x + 30, y),
                Wall(x + 30, y, x + 30, y + 60),
                Wall(x + 30, y + 60, x, y + 60),
                Wall(x, y + 60, x, y)
            ])
    
    def _add_tunnel_obstacles(self, x: float, y: float, length: float, tunnel_count: int):
        """Add tunnel-like obstacles"""
        tunnel_spacing = length / tunnel_count
        for i in range(tunnel_count):
            tunnel_x = x + i * tunnel_spacing
            # Top and bottom walls creating a tunnel
            self.walls.extend([
                Wall(tunnel_x, y, tunnel_x + 60, y),
                Wall(tunnel_x, y + 120, tunnel_x + 60, y + 120)
            ])
    
    def _add_maze_section(self, x: float, y: float, width: float, height: float):
        """Add a small maze section"""
        # Simple maze pattern
        cell_size = 40
        rows = int(height // cell_size)
        cols = int(width // cell_size)
        
        for row in range(rows):
            for col in range(cols):
                if (row + col) % 3 == 0:  # Create some pattern
                    cell_x = x + col * cell_size
                    cell_y = y + row * cell_size
                    self.walls.append(Wall(cell_x, cell_y, cell_x + cell_size, cell_y + cell_size))
    
    def _add_random_obstacle(self):
        """Add a random obstacle"""
        width = random.randint(30, 80)
        height = random.randint(30, 80)
        x = random.randint(self.margin + 100, self.width - self.margin - width - 100)
        y = random.randint(self.margin + 100, self.height - self.margin - height - 100)
        
        self.walls.extend([
            Wall(x, y, x + width, y),
            Wall(x + width, y, x + width, y + height),
            Wall(x + width, y + height, x, y + height),
            Wall(x, y + height, x, y)
        ])
    
    def _create_maze_grid(self, rows: int, cols: int) -> List[List[int]]:
        """Create a simple maze using recursive backtracking"""
        maze = [[1 for _ in range(cols)] for _ in range(rows)]
        
        def carve_path(row: int, col: int):
            maze[row][col] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols and maze[new_row][new_col] == 1:
                    maze[row + dr//2][col + dc//2] = 0
                    carve_path(new_row, new_col)
        
        carve_path(1, 1)
        return maze
    
    def _is_point_clear(self, point: Point, radius: float) -> bool:
        """Check if a point is clear of walls"""
        for wall in self.walls:
            if self._point_to_line_distance(point, wall) < radius:
                return False
        return True
    
    def _point_to_line_distance(self, point: Point, wall: Wall) -> float:
        """Calculate distance from point to line segment"""
        A = point.x - wall.x1
        B = point.y - wall.y1
        C = wall.x2 - wall.x1
        D = wall.y2 - wall.y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return math.sqrt(A * A + B * B)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = wall.x1, wall.y1
        elif param > 1:
            xx, yy = wall.x2, wall.y2
        else:
            xx = wall.x1 + param * C
            yy = wall.y1 + param * D
        
        dx = point.x - xx
        dy = point.y - yy
        return math.sqrt(dx * dx + dy * dy)
    
    def _generate_figure_eight_checkpoints(self, left_center: Tuple[float, float], 
                                         right_center: Tuple[float, float], radius: float):
        """Generate checkpoints for figure-eight track"""
        # Left circle checkpoints
        for i in range(3):
            angle = i * 2 * math.pi / 3
            x = left_center[0] + math.cos(angle) * (radius + 30)
            y = left_center[1] + math.sin(angle) * (radius + 30)
            self.checkpoints.append(Checkpoint(Point(x, y), 25, len(self.checkpoints)))
        
        # Right circle checkpoints
        for i in range(3):
            angle = i * 2 * math.pi / 3
            x = right_center[0] + math.cos(angle) * (radius + 30)
            y = right_center[1] + math.sin(angle) * (radius + 30)
            self.checkpoints.append(Checkpoint(Point(x, y), 25, len(self.checkpoints)))
    
    def _add_figure_eight_connections(self, left_center: Tuple[float, float], 
                                    right_center: Tuple[float, float], radius: float):
        """Add connecting walls for figure-eight"""
        # Create intersection area
        mid_x = (left_center[0] + right_center[0]) / 2
        mid_y = (left_center[1] + right_center[1]) / 2
        
        # Add crossing barriers
        barrier_size = 20
        self.walls.extend([
            Wall(mid_x - barrier_size, mid_y - 5, mid_x - 5, mid_y - 5),
            Wall(mid_x + 5, mid_y - 5, mid_x + barrier_size, mid_y - 5),
            Wall(mid_x - barrier_size, mid_y + 5, mid_x - 5, mid_y + 5),
            Wall(mid_x + 5, mid_y + 5, mid_x + barrier_size, mid_y + 5)
        ])
    
    def _generate_maze_checkpoints(self, maze: List[List[int]], grid_size: int, 
                                 cell_width: int, cell_height: int):
        """Generate checkpoints for maze track"""
        checkpoint_count = 0
        for row in range(0, grid_size, grid_size//4):
            for col in range(0, grid_size, grid_size//4):
                if maze[row][col] == 0:  # Open cell
                    x = self.margin + col * cell_width + cell_width//2
                    y = self.margin + row * cell_height + cell_height//2
                    self.checkpoints.append(Checkpoint(Point(x, y), 20, checkpoint_count))
                    checkpoint_count += 1
    
    def _generate_strategic_checkpoints(self):
        """Generate strategically placed checkpoints"""
        # Place checkpoints in clear areas
        potential_positions = [
            (self.width * 0.2, self.height * 0.3),
            (self.width * 0.5, self.height * 0.2),
            (self.width * 0.8, self.height * 0.4),
            (self.width * 0.7, self.height * 0.7),
            (self.width * 0.3, self.height * 0.8),
            (self.width * 0.1, self.height * 0.6)
        ]
        
        for i, (x, y) in enumerate(potential_positions):
            point = Point(x, y)
            if self._is_point_clear(point, 40):
                self.checkpoints.append(Checkpoint(point, 30, i))
    
    def _create_track_dict(self, name: str) -> Dict:
        """Create a dictionary representation of the track"""
        return {
            "name": name,
            "width": self.width,
            "height": self.height,
            "walls": [(w.x1, w.y1, w.x2, w.y2, w.thickness) for w in self.walls],
            "checkpoints": [(c.position.x, c.position.y, c.radius, c.checkpoint_id) for c in self.checkpoints],
            "start_position": (self.start_position.x, self.start_position.y),
            "start_angle": self.start_angle
        }
    
    def save_track(self, filename: str, track_data: Dict):
        """Save track data to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(track_data, f, indent=2)
        print(f"Track saved to {filename}")
    
    def load_track(self, filename: str) -> Dict:
        """Load track data from a JSON file"""
        with open(filename, 'r') as f:
            track_data = json.load(f)
        
        # Reconstruct track objects
        self.walls = [Wall(w[0], w[1], w[2], w[3], w[4]) for w in track_data["walls"]]
        self.checkpoints = [Checkpoint(Point(c[0], c[1]), c[2], c[3]) for c in track_data["checkpoints"]]
        self.start_position = Point(*track_data["start_position"])
        self.start_angle = track_data["start_angle"]
        self.width = track_data["width"]
        self.height = track_data["height"]
        
        return track_data


# Example usage and testing
if __name__ == "__main__":
    generator = TrackGenerator(1400, 800)
    
    # Generate different track types
    track_types = [
        (TrackType.OVAL, {}),
        (TrackType.FIGURE_EIGHT, {}),
        (TrackType.SPIRAL, {"turns": 3}),
        (TrackType.CIRCUIT, {}),
        (TrackType.OBSTACLE_COURSE, {"difficulty": "medium"}),
        (TrackType.MAZE, {"grid_size": 20})
    ]
    
    for track_type, kwargs in track_types:
        print(f"Generating {track_type.value} track...")
        track_data = generator.generate_track(track_type, **kwargs)
        filename = f"track_{track_type.value}.json"
        generator.save_track(filename, track_data)
        print(f"Generated {len(generator.walls)} walls and {len(generator.checkpoints)} checkpoints")
        print(f"Track saved as {filename}\n")
    
    print("All tracks generated successfully!")