import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

class BenchmarkResult:
    """Stores and visualizes benchmark results."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Args:
            results: List of dicts, each containing:
                     'task': str
                     'solver': str
                     'c2st': float
                     'mmd': float (optional)
                     'runtime': float (optional)
        """
        self.df = pd.DataFrame(results)
        
    def plot_c2st(self, save_path: Optional[str] = None):
        """Plots C2ST scores for each task and solver."""
        if 'c2st' not in self.df.columns:
            print("No C2ST data to plot.")
            return
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x='task', y='c2st', hue='solver')
        plt.title("C2ST Benchmark Results")
        plt.ylabel("C2ST (Lower is better, 0.5 is ideal)") # Wait, C2ST: 0.5 is indistinguishable (good)
        # Actually usually C2ST is accuracy: 0.5 is best, 1.0 is worst.
        # But commonly we plot 'accuracy' so 0.5 is target.
        plt.axhline(0.5, color='gray', linestyle='--', label='Ideal (0.5)')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def get_summary(self) -> pd.DataFrame:
        """Returns mean/std of metrics grouped by task and solver."""
        return self.df.groupby(['task', 'solver']).agg(['mean', 'std'])
