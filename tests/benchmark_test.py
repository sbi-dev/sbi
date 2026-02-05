import torch
from sbi.benchmark import BenchmarkRunner, GaussianMixtureTask
from sbi.inference import NPE

def test_benchmark_runner_simple():
    # Use a small configuration for testing speed
    task = GaussianMixtureTask(seed=0)
    
    # Define a simple solver list with just NPE
    solvers = ["NPE"]
    
    runner = BenchmarkRunner(
        tasks=[task],
        solvers=solvers,
        num_simulations=100, # Small number for speed
        num_observation_samples=100 # Small number
    )
    
    # Run with limited epochs
    results = runner.run(num_epochs=1, batch_size=50)
    
    # Check results
    assert len(results.df) == 1
    assert results.df.iloc[0]['task'] == "GaussianMixture"
    assert results.df.iloc[0]['solver'] == "NPE"
    assert 'c2st' in results.df.columns
    assert 0.0 <= results.df.iloc[0]['c2st'] <= 1.0
    
    print("Benchmark Task Test Passed!")
    print(results.df)

if __name__ == "__main__":
    test_benchmark_runner_simple()
