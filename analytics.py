"""
Data Analytics Module for Genetic Algorithm Car Simulation
Tracks performance metrics, generates visualizations, and provides insights
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import seaborn as sns
from pathlib import Path


@dataclass
class GenerationStats:
    """Statistics for a single generation"""
    generation: int
    timestamp: datetime
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    std_fitness: float
    best_distance: float
    avg_distance: float
    best_checkpoints: int
    avg_checkpoints: float
    alive_cars: int
    total_cars: int
    generation_time: float
    best_network_weights: Optional[List] = None


@dataclass
class ExperimentRun:
    """Complete experiment run data"""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime]
    config: Dict[str, Any]
    generations: List[GenerationStats]
    final_best_fitness: float
    total_generations: int
    experiment_notes: str = ""


class PerformanceTracker:
    """Tracks and analyzes performance metrics during simulation"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data_dir = Path("experiment_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.current_run = ExperimentRun(
            run_id=self.experiment_name,
            start_time=datetime.now(),
            end_time=None,
            config={},
            generations=[],
            final_best_fitness=0.0,
            total_generations=0
        )
        
        self.generation_metrics = []
        self.car_histories = []
        
    def set_config(self, config_dict: Dict[str, Any]):
        """Set the configuration for this experiment"""
        self.current_run.config = config_dict
    
    def record_generation(self, generation: int, population: List, generation_time: float, 
                         save_best_network: bool = True):
        """Record statistics for a completed generation"""
        if not population:
            return
        
        # Calculate fitness statistics
        fitnesses = [car.fitness for car in population]
        distances = [car.distance_traveled for car in population]
        checkpoints = [car.checkpoints_passed for car in population]
        alive_count = sum(1 for car in population if car.state.name == 'ALIVE')
        
        # Find best car
        best_car = max(population, key=lambda c: c.fitness)
        best_network = None
        
        if save_best_network and hasattr(best_car, 'brain'):
            try:
                # Save neural network weights
                best_network = self._extract_network_weights(best_car.brain)
            except Exception as e:
                print(f"Warning: Could not save network weights: {e}")
        
        # Create generation statistics
        stats = GenerationStats(
            generation=generation,
            timestamp=datetime.now(),
            best_fitness=max(fitnesses),
            avg_fitness=np.mean(fitnesses),
            worst_fitness=min(fitnesses),
            std_fitness=np.std(fitnesses),
            best_distance=max(distances),
            avg_distance=np.mean(distances),
            best_checkpoints=max(checkpoints),
            avg_checkpoints=np.mean(checkpoints),
            alive_cars=alive_count,
            total_cars=len(population),
            generation_time=generation_time,
            best_network_weights=best_network
        )
        
        self.current_run.generations.append(stats)
        self.generation_metrics.append(asdict(stats))
        
        # Update experiment totals
        self.current_run.final_best_fitness = stats.best_fitness
        self.current_run.total_generations = generation
        
        print(f"Generation {generation}: Best={stats.best_fitness:.1f}, "
              f"Avg={stats.avg_fitness:.1f}, Checkpoints={stats.best_checkpoints}")
    
    def _extract_network_weights(self, brain) -> List:
        """Extract weights from neural network for storage"""
        try:
            weights = []
            for layer_weights in brain.weights:
                weights.append(layer_weights.tolist() if hasattr(layer_weights, 'tolist') else layer_weights)
            return weights
        except:
            return None
    
    def finalize_experiment(self):
        """Finalize the current experiment"""
        self.current_run.end_time = datetime.now()
        self.save_experiment_data()
    
    def save_experiment_data(self):
        """Save experiment data to files"""
        # Save as JSON for easy reading
        json_file = self.data_dir / f"{self.experiment_name}.json"
        experiment_dict = asdict(self.current_run)
        
        # Convert datetime objects to strings for JSON serialization
        experiment_dict['start_time'] = self.current_run.start_time.isoformat()
        if self.current_run.end_time:
            experiment_dict['end_time'] = self.current_run.end_time.isoformat()
        
        for gen_data in experiment_dict['generations']:
            gen_data['timestamp'] = gen_data['timestamp'].isoformat() if gen_data['timestamp'] else None
        
        with open(json_file, 'w') as f:
            json.dump(experiment_dict, f, indent=2)
        
        # Save as pickle for Python objects
        pickle_file = self.data_dir / f"{self.experiment_name}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.current_run, f)
        
        print(f"Experiment data saved to {json_file} and {pickle_file}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert generation metrics to pandas DataFrame"""
        if not self.generation_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.generation_metrics)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def plot_fitness_evolution(self, save_plot: bool = True, show_plot: bool = True):
        """Plot fitness evolution over generations"""
        df = self.get_dataframe()
        if df.empty:
            print("No data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Fitness evolution
        plt.subplot(2, 2, 1)
        plt.plot(df['generation'], df['best_fitness'], 'g-', label='Best Fitness', linewidth=2)
        plt.plot(df['generation'], df['avg_fitness'], 'b-', label='Average Fitness', alpha=0.7)
        plt.fill_between(df['generation'], 
                        df['avg_fitness'] - df['std_fitness'],
                        df['avg_fitness'] + df['std_fitness'], 
                        alpha=0.2, color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Distance traveled
        plt.subplot(2, 2, 2)
        plt.plot(df['generation'], df['best_distance'], 'r-', label='Best Distance', linewidth=2)
        plt.plot(df['generation'], df['avg_distance'], 'orange', label='Average Distance', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.title('Distance Traveled')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Checkpoints passed
        plt.subplot(2, 2, 3)
        plt.plot(df['generation'], df['best_checkpoints'], 'purple', label='Best Checkpoints', linewidth=2)
        plt.plot(df['generation'], df['avg_checkpoints'], 'magenta', label='Average Checkpoints', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Checkpoints')
        plt.title('Checkpoints Passed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Survival rate
        plt.subplot(2, 2, 4)
        survival_rate = (df['alive_cars'] / df['total_cars']) * 100
        plt.plot(df['generation'], survival_rate, 'cyan', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Survival Rate (%)')
        plt.title('Car Survival Rate')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.data_dir / f"{self.experiment_name}_fitness_evolution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_file}")
        
        if show_plot:
            plt.show()
    
    def plot_performance_correlation(self, save_plot: bool = True, show_plot: bool = True):
        """Plot correlation between different performance metrics"""
        df = self.get_dataframe()
        if df.empty:
            print("No data to plot")
            return
        
        # Select numeric columns for correlation
        numeric_cols = ['best_fitness', 'avg_fitness', 'best_distance', 'avg_distance', 
                       'best_checkpoints', 'avg_checkpoints', 'generation_time']
        correlation_data = df[numeric_cols]
        
        plt.figure(figsize=(10, 8))
        
        # Correlation heatmap
        plt.subplot(2, 1, 1)
        correlation_matrix = correlation_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Performance Metrics Correlation')
        
        # Scatter plot: Fitness vs Distance
        plt.subplot(2, 1, 2)
        plt.scatter(df['best_distance'], df['best_fitness'], alpha=0.6, c=df['generation'], 
                   cmap='viridis', s=50)
        plt.colorbar(label='Generation')
        plt.xlabel('Best Distance')
        plt.ylabel('Best Fitness')
        plt.title('Fitness vs Distance Over Generations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.data_dir / f"{self.experiment_name}_correlation.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Correlation plot saved to {plot_file}")
        
        if show_plot:
            plt.show()
    
    def generate_report(self) -> str:
        """Generate a text summary report of the experiment"""
        df = self.get_dataframe()
        if df.empty:
            return "No data available for report"
        
        report = []
        report.append(f"GENETIC ALGORITHM EXPERIMENT REPORT")
        report.append(f"="*50)
        report.append(f"Experiment: {self.experiment_name}")
        report.append(f"Start Time: {self.current_run.start_time}")
        if self.current_run.end_time:
            duration = self.current_run.end_time - self.current_run.start_time
            report.append(f"Duration: {duration}")
        report.append(f"Total Generations: {self.current_run.total_generations}")
        report.append("")
        
        # Performance summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 20)
        report.append(f"Final Best Fitness: {self.current_run.final_best_fitness:.2f}")
        report.append(f"Best Distance: {df['best_distance'].max():.2f}")
        report.append(f"Best Checkpoints: {df['best_checkpoints'].max()}")
        report.append(f"Average Generation Time: {df['generation_time'].mean():.2f}s")
        report.append("")
        
        # Evolution analysis
        report.append("EVOLUTION ANALYSIS")
        report.append("-" * 20)
        initial_fitness = df['best_fitness'].iloc[0] if len(df) > 0 else 0
        final_fitness = df['best_fitness'].iloc[-1] if len(df) > 0 else 0
        improvement = final_fitness - initial_fitness
        report.append(f"Fitness Improvement: {improvement:.2f} ({improvement/initial_fitness*100:.1f}%)")
        
        # Find best performing generation
        best_gen = df.loc[df['best_fitness'].idxmax()]
        report.append(f"Best Generation: {best_gen['generation']}")
        report.append(f"Best Generation Fitness: {best_gen['best_fitness']:.2f}")
        report.append("")
        
        # Configuration summary
        if self.current_run.config:
            report.append("CONFIGURATION")
            report.append("-" * 20)
            for section, settings in self.current_run.config.items():
                if isinstance(settings, dict):
                    report.append(f"{section.upper()}:")
                    for key, value in settings.items():
                        report.append(f"  {key}: {value}")
                else:
                    report.append(f"{section}: {settings}")
        
        report_text = "\n".join(report)
        
        # Save report to file
        report_file = self.data_dir / f"{self.experiment_name}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to {report_file}")
        return report_text


class ExperimentComparison:
    """Compare multiple experiments"""
    
    def __init__(self):
        self.experiments = {}
        self.data_dir = Path("experiment_data")
    
    def load_experiment(self, experiment_name: str):
        """Load an experiment from saved data"""
        pickle_file = self.data_dir / f"{experiment_name}.pkl"
        try:
            with open(pickle_file, 'rb') as f:
                experiment = pickle.load(f)
                self.experiments[experiment_name] = experiment
                print(f"Loaded experiment: {experiment_name}")
        except FileNotFoundError:
            print(f"Experiment file not found: {pickle_file}")
    
    def compare_experiments(self, experiment_names: List[str], save_plot: bool = True):
        """Compare multiple experiments"""
        if not all(name in self.experiments for name in experiment_names):
            missing = [name for name in experiment_names if name not in self.experiments]
            print(f"Missing experiments: {missing}")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Compare fitness evolution
        plt.subplot(2, 2, 1)
        for name in experiment_names:
            exp = self.experiments[name]
            generations = [g.generation for g in exp.generations]
            best_fitness = [g.best_fitness for g in exp.generations]
            plt.plot(generations, best_fitness, label=f"{name} (Best)", linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Best Fitness Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Compare average fitness
        plt.subplot(2, 2, 2)
        for name in experiment_names:
            exp = self.experiments[name]
            generations = [g.generation for g in exp.generations]
            avg_fitness = [g.avg_fitness for g in exp.generations]
            plt.plot(generations, avg_fitness, label=f"{name} (Avg)", linewidth=2, alpha=0.7)
        
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.title('Average Fitness Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Compare final performance
        plt.subplot(2, 2, 3)
        final_fitness = [self.experiments[name].final_best_fitness for name in experiment_names]
        plt.bar(experiment_names, final_fitness, alpha=0.7)
        plt.ylabel('Final Best Fitness')
        plt.title('Final Performance Comparison')
        plt.xticks(rotation=45)
        
        # Compare total generations
        plt.subplot(2, 2, 4)
        total_gens = [self.experiments[name].total_generations for name in experiment_names]
        plt.bar(experiment_names, total_gens, alpha=0.7, color='orange')
        plt.ylabel('Total Generations')
        plt.title('Generations to Completion')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.data_dir / f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {plot_file}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example of how to use the analytics module
    tracker = PerformanceTracker("test_experiment")
    
    # Simulate some generation data
    for gen in range(1, 21):
        # Create mock population data
        mock_population = []
        for i in range(20):
            class MockCar:
                def __init__(self, fitness, distance, checkpoints):
                    self.fitness = fitness
                    self.distance_traveled = distance
                    self.checkpoints_passed = checkpoints
                    self.state = type('State', (), {'name': 'ALIVE' if fitness > 50 else 'CRASHED'})()
            
            fitness = np.random.normal(100 + gen * 10, 20)
            distance = np.random.normal(50 + gen * 5, 10)
            checkpoints = np.random.poisson(gen * 0.5)
            
            mock_population.append(MockCar(fitness, distance, checkpoints))
        
        tracker.record_generation(gen, mock_population, np.random.uniform(1.0, 3.0))
    
    # Generate visualizations and report
    tracker.plot_fitness_evolution()
    tracker.plot_performance_correlation()
    report = tracker.generate_report()
    print(report)
    
    tracker.finalize_experiment()