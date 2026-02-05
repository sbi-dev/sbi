import torch
import time
from typing import List, Callable, Union, Dict, Any
from tqdm import tqdm
from sbi.benchmark.tasks import BenchmarkTask
from sbi.benchmark.results import BenchmarkResult
from sbi.utils.metrics import c2st
from sbi.inference import NPE, NPSE, FMPE # Import standard solvers if available

class BenchmarkRunner:
    """Orchestrates the running of benchmarks."""
    
    def __init__(self, tasks: List[BenchmarkTask], solvers: List[Union[str, Callable]], num_simulations: int = 1000, num_observation_samples: int = 10000):
        self.tasks = tasks
        self.solvers = solvers
        self.num_simulations = num_simulations
        self.num_observation_samples = num_observation_samples
        self.results_data = []

    def _get_solver_instance(self, solver_name: str, prior, device='cpu'):
        if solver_name.lower() == 'npe':
            return NPE(prior=prior, device=device)
        elif solver_name.lower() == 'fmpe':
             # Check if FMPE is importable
            try:
                from sbi.inference import FMPE
                return FMPE(prior=prior, device=device)
            except ImportError:
                 raise ValueError("FMPE not available in this version.")
        else:
            raise ValueError(f"Unknown solver string: {solver_name}")

    def run(self, num_epochs: int = 10, batch_size: int = 100, device: str = 'cpu') -> BenchmarkResult:
        """Runs the benchmark."""
        
        for task in self.tasks:
            print(f"Running Task: {task.name}")
            
            # Generate common training data for fairness (optional, but good for benchmarks)
            # Actually, standard is to let the trainer simulate.
            # But for fair comparison, we might want fixed data?
            # sbi standard trainers simulate themselves. Let's stick to that for now.
            
            # Get Ground Truth
            theta_true = task.get_ground_truth_posterior_samples(self.num_observation_samples)
            x_o = task.get_observation()
            
            for solver_item in self.solvers:
                solver_name = solver_item if isinstance(solver_item, str) else solver_item.__name__
                print(f"  Solver: {solver_name}")
                
                # Setup
                if isinstance(solver_item, str):
                    inference = self._get_solver_instance(solver_item, task.prior, device)
                else:
                    # Assume callable is a class or builder
                    inference = solver_item(prior=task.prior, device=device)
                    
                # Train
                try:
                    start_time = time.time()
                    
                    # Simulate
                    theta = task.prior.sample((self.num_simulations,))
                    x = task.get_simulator()(theta)
                    
                    inference.append_simulations(theta, x)
                    density_estimator = inference.train(
                        training_batch_size=batch_size,
                        max_num_epochs=num_epochs,
                        show_train_summary=False
                    )
                    
                    posterior = inference.build_posterior(density_estimator)
                    
                    # Sample
                    theta_est = posterior.sample((self.num_observation_samples,), x=x_o, show_progress_bars=False)
                    
                    runtime = time.time() - start_time
                    
                    # Metric: C2ST
                    c2st_score = c2st(theta_true, theta_est).item()
                except Exception as e:
                    print(f"    Failed: {e}")
                    continue
                
                self.results_data.append({
                    'task': task.name,
                    'solver': solver_name,
                    'c2st': c2st_score,
                    'runtime': runtime
                })
                
        return BenchmarkResult(self.results_data)
