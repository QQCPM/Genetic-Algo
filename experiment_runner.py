"""
Experiment Runner and Comparison Tool for Genetic Algorithm Car Simulation
Runs multiple experiments with different configurations and compares results
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, asdict

# Import our custom modules
from config import Config, TrackConfig, GeneticConfig, NeuralNetworkConfig
from analytics import PerformanceTracker, ExperimentComparison
from track_generator import TrackGenerator, TrackType


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    description: str
    config: Config
    track_type: TrackType
    max_generations: int = 100
    timeout_hours: float = 2.0
    save_interval: int = 10


class ExperimentRunner:
    """Runs and manages multiple genetic algorithm experiments"""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.experiments = []
        self.running_experiments = {}
        self.completed_experiments = {}
        self.comparison_tool = ExperimentComparison()
        
        # Track overall statistics
        self.total_experiments_run = 0
        self.successful_experiments = 0
        self.failed_experiments = 0
        
    def add_experiment(self, experiment: ExperimentConfig):
        """Add an experiment to the queue"""
        self.experiments.append(experiment)
        print(f"Added experiment: {experiment.name}")
    
    def create_preset_experiments(self) -> List[ExperimentConfig]:
        """Create a set of predefined experiments for comparison"""
        experiments = []
        
        # Experiment 1: Small population, fast evolution
        config1 = Config()
        config1.genetic.POPULATION_SIZE = 20
        config1.genetic.MUTATION_RATE = 0.15
        config1.genetic.MAX_GENERATION_TIME = 15.0
        config1.neural_network.LAYER_SIZES = [5, 6, 2]
        
        experiments.append(ExperimentConfig(
            name="small_fast",
            description="Small population with fast evolution",
            config=config1,
            track_type=TrackType.OVAL,
            max_generations=50
        ))
        
        # Experiment 2: Large population, slow evolution
        config2 = Config()
        config2.genetic.POPULATION_SIZE = 100
        config2.genetic.MUTATION_RATE = 0.05
        config2.genetic.MAX_GENERATION_TIME = 45.0
        config2.neural_network.LAYER_SIZES = [5, 12, 8, 2]
        
        experiments.append(ExperimentConfig(
            name="large_slow",
            description="Large population with slow evolution",
            config=config2,
            track_type=TrackType.OVAL,
            max_generations=30
        ))
        
        # Experiment 3: Complex neural network
        config3 = Config()
        config3.genetic.POPULATION_SIZE = 50
        config3.genetic.MUTATION_RATE = 0.1
        config3.neural_network.LAYER_SIZES = [5, 16, 12, 8, 2]
        
        experiments.append(ExperimentConfig(
            name="complex_network",
            description="Complex neural network architecture",
            config=config3,
            track_type=TrackType.FIGURE_EIGHT,
            max_generations=40
        ))
        
        # Experiment 4: High mutation rate
        config4 = Config()
        config4.genetic.POPULATION_SIZE = 50
        config4.genetic.MUTATION_RATE = 0.25
        config4.genetic.CROSSOVER_RATE = 0.8
        
        experiments.append(ExperimentConfig(
            name="high_mutation",
            description="High mutation rate for exploration",
            config=config4,
            track_type=TrackType.MAZE,
            max_generations=60
        ))
        
        # Experiment 5: Different track complexity
        config5 = Config()
        config5.genetic.POPULATION_SIZE = 50
        config5.car.NUM_SENSORS = 7
        config5.neural_network.LAYER_SIZES = [7, 10, 6, 2]
        
        experiments.append(ExperimentConfig(
            name="complex_track",
            description="Complex track with more sensors",
            config=config5,
            track_type=TrackType.CIRCUIT,
            max_generations=50
        ))
        
        return experiments
    
    def run_single_experiment(self, experiment: ExperimentConfig) -> Dict[str, Any]:
        """Run a single experiment (mock implementation)"""
        print(f"Starting experiment: {experiment.name}")
        
        # Initialize tracker
        tracker = PerformanceTracker(experiment.name)
        tracker.set_config(asdict(experiment.config))
        
        # Generate track
        track_generator = TrackGenerator(
            experiment.config.display.SCREEN_WIDTH,
            experiment.config.display.SCREEN_HEIGHT
        )
        track_data = track_generator.generate_track(experiment.track_type)
        
        # Save experiment configuration
        config_file = self.results_dir / f"{experiment.name}_config.json"
        experiment.config.save_to_file(str(config_file))
        
        # Mock simulation loop
        start_time = time.time()
        best_fitness_ever = 0
        generation = 1
        
        print(f"Running {experiment.name} with {experiment.config.genetic.POPULATION_SIZE} cars...")
        
        while generation <= experiment.max_generations:
            # Check timeout
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > experiment.timeout_hours:
                print(f"Experiment {experiment.name} timed out after {elapsed_hours:.2f} hours")
                break
            
            # Simulate generation
            generation_start = time.time()
            
            # Mock population performance (improving over time)
            base_fitness = 100 + generation * 10
            noise = np.random.normal(0, 20, experiment.config.genetic.POPULATION_SIZE)
            fitnesses = base_fitness + noise
            distances = fitnesses * 0.5 + np.random.normal(0, 5, len(fitnesses))
            checkpoints = np.random.poisson(min(generation * 0.2, 8), len(fitnesses))
            
            # Create mock population
            mock_population = []
            for i in range(len(fitnesses)):
                class MockCar:
                    def __init__(self, fitness, distance, checkpoints):
                        self.fitness = max(0, fitness)
                        self.distance_traveled = max(0, distance)
                        self.checkpoints_passed = max(0, checkpoints)
                        self.state = type('State', (), {'name': 'ALIVE' if fitness > 50 else 'CRASHED'})()
                
                mock_population.append(MockCar(fitnesses[i], distances[i], checkpoints[i]))
            
            generation_time = time.time() - generation_start
            
            # Record generation
            tracker.record_generation(generation, mock_population, generation_time)
            
            # Update best fitness
            current_best = max(car.fitness for car in mock_population)
            if current_best > best_fitness_ever:
                best_fitness_ever = current_best
            
            # Progress update
            if generation % 10 == 0:
                print(f"  Generation {generation}/{experiment.max_generations}, "
                      f"Best: {current_best:.1f}, Time: {generation_time:.2f}s")
            
            generation += 1
            
            # Simulate processing time
            time.sleep(0.1)  # Small delay to simulate real processing
        
        # Finalize experiment
        tracker.finalize_experiment()
        
        # Generate reports and visualizations
        tracker.plot_fitness_evolution(save_plot=True, show_plot=False)
        tracker.plot_performance_correlation(save_plot=True, show_plot=False)
        report = tracker.generate_report()
        
        # Move generated files to experiment directory
        experiment_dir = self.results_dir / experiment.name
        experiment_dir.mkdir(exist_ok=True)
        
        # Move analytics files
        for file_pattern in [f"{experiment.name}*"]:
            for file_path in Path("experiment_data").glob(file_pattern):
                if file_path.is_file():
                    file_path.rename(experiment_dir / file_path.name)
        
        total_time = time.time() - start_time
        
        result = {
            "name": experiment.name,
            "description": experiment.description,
            "status": "completed",
            "total_time": total_time,
            "generations_completed": generation - 1,
            "best_fitness": best_fitness_ever,
            "config": asdict(experiment.config),
            "track_type": experiment.track_type.value,
            "output_directory": str(experiment_dir)
        }
        
        print(f"Completed experiment {experiment.name}: "
              f"Best fitness {best_fitness_ever:.1f} in {generation-1} generations "
              f"({total_time:.1f}s)")
        
        return result
    
    def run_all_experiments(self, parallel: bool = False):
        """Run all queued experiments"""
        if not self.experiments:
            print("No experiments queued")
            return
        
        print(f"Starting {len(self.experiments)} experiments...")
        
        if parallel:
            self._run_experiments_parallel()
        else:
            self._run_experiments_sequential()
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def _run_experiments_sequential(self):
        """Run experiments one after another"""
        for experiment in self.experiments:
            try:
                result = self.run_single_experiment(experiment)
                self.completed_experiments[experiment.name] = result
                self.successful_experiments += 1
            except Exception as e:
                print(f"Experiment {experiment.name} failed: {e}")
                self.failed_experiments += 1
                self.completed_experiments[experiment.name] = {
                    "name": experiment.name,
                    "status": "failed",
                    "error": str(e)
                }
            
            self.total_experiments_run += 1
    
    def _run_experiments_parallel(self):
        """Run experiments in parallel (simplified version)"""
        print("Note: Parallel execution is simplified for this demo")
        # For real implementation, you would use threading or multiprocessing
        self._run_experiments_sequential()
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report"""
        if not self.completed_experiments:
            print("No completed experiments to compare")
            return
        
        # Create comparison data
        comparison_data = []
        for name, result in self.completed_experiments.items():
            if result["status"] == "completed":
                comparison_data.append({
                    "experiment": name,
                    "description": result["description"],
                    "best_fitness": result["best_fitness"],
                    "generations": result["generations_completed"],
                    "total_time": result["total_time"],
                    "track_type": result["track_type"],
                    "population_size": result["config"]["genetic"]["POPULATION_SIZE"],
                    "mutation_rate": result["config"]["genetic"]["MUTATION_RATE"],
                    "network_complexity": len(result["config"]["neural_network"]["LAYER_SIZES"])
                })
        
        if not comparison_data:
            print("No successful experiments to compare")
            return
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison data
        comparison_file = self.results_dir / "experiment_comparison.csv"
        df.to_csv(comparison_file, index=False)
        
        # Generate comparison plots
        self._create_comparison_plots(df)
        
        # Generate text report
        self._create_text_report(df)
        
        print(f"Comparison report generated in {self.results_dir}")
    
    def _create_comparison_plots(self, df: pd.DataFrame):
        """Create comparison visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Best fitness comparison
        axes[0, 0].bar(df['experiment'], df['best_fitness'], alpha=0.7)
        axes[0, 0].set_title('Best Fitness by Experiment')
        axes[0, 0].set_ylabel('Best Fitness')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Generations to completion
        axes[0, 1].bar(df['experiment'], df['generations'], alpha=0.7, color='orange')
        axes[0, 1].set_title('Generations to Completion')
        axes[0, 1].set_ylabel('Generations')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Time vs Performance
        axes[1, 0].scatter(df['total_time'], df['best_fitness'], 
                          s=df['population_size']*2, alpha=0.6, c=df['mutation_rate'], 
                          cmap='viridis')
        axes[1, 0].set_xlabel('Total Time (seconds)')
        axes[1, 0].set_ylabel('Best Fitness')
        axes[1, 0].set_title('Performance vs Time (size = population, color = mutation rate)')
        
        # Population size vs Performance
        axes[1, 1].scatter(df['population_size'], df['best_fitness'], 
                          s=100, alpha=0.6, c=df['mutation_rate'], cmap='plasma')
        axes[1, 1].set_xlabel('Population Size')
        axes[1, 1].set_ylabel('Best Fitness')
        axes[1, 1].set_title('Population Size vs Performance')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "experiment_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to {plot_file}")
    
    def _create_text_report(self, df: pd.DataFrame):
        """Create a detailed text report"""
        report = []
        report.append("GENETIC ALGORITHM EXPERIMENT COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Experiments: {self.total_experiments_run}")
        report.append(f"Successful: {self.successful_experiments}")
        report.append(f"Failed: {self.failed_experiments}")
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"Best Overall Fitness: {df['best_fitness'].max():.2f}")
        report.append(f"Average Best Fitness: {df['best_fitness'].mean():.2f}")
        report.append(f"Fitness Standard Deviation: {df['best_fitness'].std():.2f}")
        report.append(f"Fastest Completion: {df['total_time'].min():.1f} seconds")
        report.append(f"Average Time: {df['total_time'].mean():.1f} seconds")
        report.append("")
        
        # Best performing experiment
        best_idx = df['best_fitness'].idxmax()
        best_experiment = df.iloc[best_idx]
        report.append("BEST PERFORMING EXPERIMENT")
        report.append("-" * 30)
        report.append(f"Name: {best_experiment['experiment']}")
        report.append(f"Description: {best_experiment['description']}")
        report.append(f"Best Fitness: {best_experiment['best_fitness']:.2f}")
        report.append(f"Generations: {best_experiment['generations']}")
        report.append(f"Population Size: {best_experiment['population_size']}")
        report.append(f"Mutation Rate: {best_experiment['mutation_rate']}")
        report.append(f"Track Type: {best_experiment['track_type']}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 30)
        for _, row in df.iterrows():
            report.append(f"Experiment: {row['experiment']}")
            report.append(f"  Description: {row['description']}")
            report.append(f"  Best Fitness: {row['best_fitness']:.2f}")
            report.append(f"  Generations: {row['generations']}")
            report.append(f"  Time: {row['total_time']:.1f}s")
            report.append(f"  Population: {row['population_size']}")
            report.append(f"  Mutation Rate: {row['mutation_rate']}")
            report.append(f"  Track: {row['track_type']}")
            report.append("")
        
        # Analysis and recommendations
        report.append("ANALYSIS AND RECOMMENDATIONS")
        report.append("-" * 30)
        
        # Population size analysis
        pop_correlation = df['population_size'].corr(df['best_fitness'])
        report.append(f"Population Size vs Performance Correlation: {pop_correlation:.3f}")
        
        # Mutation rate analysis
        mutation_correlation = df['mutation_rate'].corr(df['best_fitness'])
        report.append(f"Mutation Rate vs Performance Correlation: {mutation_correlation:.3f}")
        
        # Time efficiency
        time_efficiency = df['best_fitness'] / df['total_time']
        most_efficient_idx = time_efficiency.idxmax()
        most_efficient = df.iloc[most_efficient_idx]
        report.append(f"Most Time-Efficient: {most_efficient['experiment']} "
                     f"({time_efficiency.iloc[most_efficient_idx]:.2f} fitness/second)")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        if pop_correlation > 0.3:
            report.append("- Larger populations tend to perform better")
        elif pop_correlation < -0.3:
            report.append("- Smaller populations may be more efficient")
        else:
            report.append("- Population size shows no clear correlation with performance")
        
        if mutation_correlation > 0.3:
            report.append("- Higher mutation rates appear beneficial")
        elif mutation_correlation < -0.3:
            report.append("- Lower mutation rates appear beneficial")
        else:
            report.append("- Mutation rate shows no clear correlation with performance")
        
        # Save report
        report_text = "\n".join(report)
        report_file = self.results_dir / "experiment_comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Detailed report saved to {report_file}")
        return report_text
    
    def load_experiment_results(self, experiment_names: List[str]):
        """Load previously run experiment results"""
        for name in experiment_names:
            experiment_dir = self.results_dir / name
            if experiment_dir.exists():
                self.comparison_tool.load_experiment(name)
            else:
                print(f"Experiment directory not found: {experiment_dir}")
    
    def quick_comparison(self, experiment_names: List[str]):
        """Quick comparison of specific experiments"""
        self.load_experiment_results(experiment_names)
        self.comparison_tool.compare_experiments(experiment_names, save_plot=True)


def main():
    """Main function to demonstrate the experiment runner"""
    runner = ExperimentRunner()
    
    # Add preset experiments
    preset_experiments = runner.create_preset_experiments()
    
    for exp in preset_experiments:
        runner.add_experiment(exp)
    
    print("Starting genetic algorithm experiment comparison...")
    print(f"Queued {len(runner.experiments)} experiments")
    
    # Run all experiments
    runner.run_all_experiments()
    
    print("\nExperiment suite completed!")
    print(f"Results saved in: {runner.results_dir}")
    
    # Show summary
    if runner.completed_experiments:
        successful = [r for r in runner.completed_experiments.values() if r["status"] == "completed"]
        if successful:
            best_experiment = max(successful, key=lambda x: x["best_fitness"])
            print(f"\nBest performing experiment: {best_experiment['name']}")
            print(f"Best fitness achieved: {best_experiment['best_fitness']:.2f}")


if __name__ == "__main__":
    main()