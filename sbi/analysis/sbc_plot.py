#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats
from scipy.stats import binom

# ============================================================================
# Data Classes for Configuration
# ============================================================================

@dataclass
class SBCPlotConfig:
    """Configuration for SBC rank plots."""
    
    num_bins: Optional[int] = None
    bin_method: Literal["fd", "sturges", "scott", "sqrt"] = "fd"
    plot_type: Literal["hist", "cdf", "ecdf_diff", "combo"] = "ecdf_diff"
    confidence_level: float = 0.99
    show_ci_band: bool = True
    ci_band_alpha: float = 0.3
    line_alpha: float = 0.8
    num_cdf_points: int = 100
    figsize: Optional[Tuple[float, float]] = None
    colors: Optional[List[str]] = None
    

@dataclass
class SBCStatistics:
    """Container for SBC statistical results."""
    
    num_sbc_runs: int
    num_parameters: int
    num_posterior_samples: int
    num_bins: int
    uniformity_pvalue: Optional[float] = None
    max_deviation: Optional[float] = None


# ============================================================================
# Core Data Processing
# ============================================================================

class SBCRankData:
    """Handles SBC rank data validation and processing."""
    
    def __init__(
        self,
        ranks: Union[np.ndarray, List[np.ndarray]],
        num_posterior_samples: int
    ):
        """
        Initialize SBC rank data.
        
        Args:
            ranks: Array(s) of shape (num_sbc_runs, num_parameters)
            num_posterior_samples: Number of posterior samples used for ranking
        """
        self.ranks_list = self._validate_and_convert_ranks(ranks)
        self.num_posterior_samples = num_posterior_samples
        self.num_sbc_runs = self.ranks_list[0].shape[0]
        self.num_parameters = self.ranks_list[0].shape[1]
        
    def _validate_and_convert_ranks(
        self,
        ranks: Union[np.ndarray, List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Validate and convert ranks to list of numpy arrays."""
        if isinstance(ranks, np.ndarray):
            ranks_list = [ranks]
        else:
            ranks_list = list(ranks)
            
        # Validate all ranks have the same shape
        base_shape = ranks_list[0].shape
        for i, rank in enumerate(ranks_list):
            if not isinstance(rank, np.ndarray):
                ranks_list[i] = np.array(rank)
            if ranks_list[i].shape != base_shape:
                raise ValueError(
                    f"All ranks must have the same shape. "
                    f"Expected {base_shape}, got {ranks_list[i].shape} at index {i}"
                )
                
        return ranks_list
    
    def calculate_optimal_bins(
        self,
        method: Literal["fd", "sturges", "scott", "sqrt"] = "fd"
    ) -> int:
        """
        Calculate optimal number of bins using various heuristics.
        
        Args:
            method: Binning method
                - 'fd': Freedman-Diaconis rule (recommended for SBC)
                - 'sturges': Sturges' formula
                - 'scott': Scott's rule
                - 'sqrt': Square root rule
                
        Returns:
            Optimal number of bins
        """
        n = self.num_sbc_runs
        
        if method == "fd":
            # Freedman-Diaconis: bin_width = 2 * IQR * n^(-1/3)
            # For uniform distribution on [0, L], IQR ≈ L/2
            # This gives: num_bins ≈ L / (2 * (L/2) * n^(-1/3)) = n^(1/3)
            return max(int(np.ceil(n ** (1/3))), 10)
        
        elif method == "sturges":
            # Sturges: k = ceil(log2(n)) + 1
            return int(np.ceil(np.log2(n))) + 1
        
        elif method == "scott":
            # Scott's rule adapted for uniform distribution
            # Similar to FD but uses std instead of IQR
            return max(int(np.ceil(n ** (1/3))), 10)
        
        elif method == "sqrt":
            # Square root rule: k = ceil(sqrt(n))
            return int(np.ceil(np.sqrt(n)))
        
        else:
            raise ValueError(f"Unknown binning method: {method}")


# ============================================================================
# Plotting Strategies
# ============================================================================

class SBCPlotter:
    """Base class for SBC plotting strategies."""
    
    def __init__(self, data: SBCRankData, config: SBCPlotConfig):
        self.data = data
        self.config = config
        
        # Calculate num_bins if not provided
        if config.num_bins is None:
            self.num_bins = data.calculate_optimal_bins(config.bin_method)
        else:
            self.num_bins = config.num_bins
            
    def _get_uniform_ci_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence interval bounds under uniformity assumption."""
        n = self.data.num_sbc_runs
        
        # For histogram: binomial confidence intervals
        p_uniform = 1 / (self.num_bins + 1)
        lower = binom(n, p_uniform).ppf((1 - self.config.confidence_level) / 2)
        upper = binom(n, p_uniform).ppf(1 - (1 - self.config.confidence_level) / 2)
        
        return lower, upper


class HistogramPlotter(SBCPlotter):
    """Histogram-based SBC visualization."""
    
    def plot(
        self,
        ax: Axes,
        ranks: np.ndarray,
        param_idx: int,
        label: Optional[str] = None,
        color: Optional[str] = None
    ) -> None:
        """Plot histogram of ranks for a single parameter."""
        ax.hist(
            ranks[:, param_idx],
            bins=self.num_bins,
            label=label,
            color=color or 'firebrick',
            alpha=self.config.line_alpha,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add uniform expectation band
        if self.config.show_ci_band:
            lower, upper = self._get_uniform_ci_bounds()
            ax.axhline(float(lower), color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(float(upper), color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.fill_between(
                [0, self.data.num_posterior_samples],
                lower, upper,
                color='gray',
                alpha=self.config.ci_band_alpha,
                label=f'{int(self.config.confidence_level * 100)}% CI under uniformity'
            )


class ECDFPlotter(SBCPlotter):
    """Empirical CDF visualization (Step Plot)."""
    
    def plot(self, ax: Axes, ranks: np.ndarray, param_idx: int, 
             label: Optional[str] = None, color: Optional[str] = None) -> None:
        """Plot empirical CDF of ranks."""
        
        # Get raw data (NO BINNING)
        raw_ranks = ranks[:, param_idx]
        N = len(raw_ranks)
        
        # Normalize and sort
        data = np.sort(raw_ranks) / self.data.num_posterior_samples
        
        # Generate Y axis (0 to 1)
        y = np.arange(1, N + 1) / N
        
        # Step plot is standard for CDFs
        ax.step(data, y, where='post', label=label, color=color,
                alpha=self.config.line_alpha, linewidth=1.5)
        
        # Uniform diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Uniform')
        
        # Add smooth confidence band (DKW-like)
        if self.config.show_ci_band:
            p = np.linspace(0, 1, 1000)
            
            # Approx CI around the diagonal
            std_dev = np.sqrt(p * (1 - p) / N)
            alpha_val = (1 - self.config.confidence_level) / 2
            z_score = stats.norm.ppf(1 - alpha_val)
            
            upper = p + z_score * std_dev
            lower = p - z_score * std_dev
            
            ax.fill_between(p, lower, upper, color='gray', 
                          alpha=self.config.ci_band_alpha,
                          label=f'{int(self.config.confidence_level*100)}% CI')


class ECDFDifferencePlotter(SBCPlotter):
    """
    ECDF difference from uniform (Deviation Plot).
    Improved: Uses raw sorted data (infinite resolution) and Ellipse CI.
    """
    
    def plot(self, ax: Axes, ranks: np.ndarray, param_idx: int, 
             label: Optional[str] = None, color: Optional[str] = None) -> None:
        """Plot ECDF difference from uniform expectation."""
        
        # 1. Get raw data (NO BINNING)
        raw_ranks = ranks[:, param_idx]
        N = self.data.num_sbc_runs
        
        # 2. Normalize and Sort
        # We divide by num_posterior_samples to get the rank probability (0 to 1)
        y_observed = np.sort(raw_ranks) / self.data.num_posterior_samples
        
        # 3. Theoretical Uniform Quantiles
        # Ideally, the i-th point should be at i/N
        x_theoretical = np.linspace(0, 1, N)
        
        # 4. Calculate Deviation
        diff = y_observed - x_theoretical
        
        # 5. Plot the smooth line
        ax.plot(
            x_theoretical, 
            diff, 
            label=label, 
            color=color, 
            alpha=self.config.line_alpha, 
            linewidth=1.5
        )
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        # 6. Add the "Ellipse" Confidence Band
        if self.config.show_ci_band:
            # High-resolution x-axis for the smooth ellipse shape
            p = np.linspace(0, 1, 1000)
            
            # Standard deviation of the ECDF diff is sqrt(p(1-p)/N)
            std_dev = np.sqrt(p * (1 - p) / N)
            
            # Z-score for confidence level
            alpha_val = (1 - self.config.confidence_level) / 2
            z_score = stats.norm.ppf(1 - alpha_val)
            
            upper = z_score * std_dev
            lower = -z_score * std_dev
            
            ax.fill_between(
                p, lower, upper,
                color='gray',
                alpha=self.config.ci_band_alpha,
                label=f'{int(self.config.confidence_level * 100)}% CI',
                zorder=0
            )


# ============================================================================
# Main Interface
# ============================================================================

def sbc_rank_plot(
    ranks: Union[np.ndarray, List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    bin_method: Literal["fd", "sturges", "scott", "sqrt"] = "fd",
    plot_type: Literal["hist", "cdf", "ecdf_diff", "combo"] = "ecdf_diff",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    confidence_level: float = 0.99,
    show_ci_band: bool = True,
    fig: Optional[Figure] = None,
    ax: Optional[Union[Axes, np.ndarray]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[Figure, np.ndarray]:
    """
    Plot simulation-based calibration ranks with improved defaults and features.
    
    This function provides multiple visualization options for SBC diagnostics:
    - Histograms: Traditional rank histogram
    - CDF: Empirical cumulative distribution function
    - ECDF difference: Deviation from uniform (recommended, most sensitive)
    - Combo: Side-by-side comparison
    
    Args:
        ranks: SBC ranks array(s) of shape (num_sbc_runs, num_parameters)
        num_posterior_samples: Number of posterior samples used for ranking
        num_bins: Number of bins (auto-calculated if None using bin_method)
        bin_method: Method for calculating bins if num_bins is None
            - 'fd': Freedman-Diaconis (recommended, ~n^(1/3) bins)
            - 'sturges': Sturges' formula (~log2(n) bins)
            - 'scott': Scott's rule
            - 'sqrt': Square root choice (~sqrt(n) bins)
        plot_type: Type of visualization
            - 'hist': Traditional histogram
            - 'cdf': Empirical CDF
            - 'ecdf_diff': ECDF difference from uniform (recommended)
            - 'combo': Show both histogram and ECDF difference
        parameter_labels: Labels for each parameter dimension
        ranks_labels: Labels for different rank sets (if comparing methods)
        colors: Colors for each rank set
        confidence_level: Confidence level for uniform bands (default 0.99)
        show_ci_band: Whether to show confidence interval band
        fig: Existing figure to plot on
        ax: Existing axes to plot on
        figsize: Figure size (width, height)
        **kwargs: Additional plotting arguments
        
    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axes (array if multiple parameters)
    """
    # Create configuration
    config = SBCPlotConfig(
        num_bins=num_bins,
        bin_method=bin_method,
        plot_type=plot_type,
        confidence_level=confidence_level,
        show_ci_band=show_ci_band,
        colors=colors,
        figsize=figsize,
        **kwargs
    )
    
    # Initialize data handler
    data = SBCRankData(ranks, num_posterior_samples)
    
    # Set up labels
    if parameter_labels is None:
        parameter_labels = [f"θ{i+1}" for i in range(data.num_parameters)]
    if ranks_labels is None:
        ranks_labels = [f"Ranks {i+1}" for i in range(len(data.ranks_list))]
    
    # Normalize any single Axes input into a 2D ndarray for consistent indexing
    if ax is not None and isinstance(ax, Axes):
        ax = np.array([[ax]])

    # Determine subplot layout
    params_in_subplots = len(data.ranks_list) > 1 or plot_type == "hist"
    
    if plot_type == "combo":
        num_cols, num_rows = 2, data.num_parameters
    elif params_in_subplots:
        num_cols = min(data.num_parameters, 4)
        num_rows = int(np.ceil(data.num_parameters / num_cols))
    else:
        num_cols = 1
        num_rows = 1
    
    # Create figure if needed
    if fig is None or ax is None:
        if figsize is None:
            if plot_type == "combo":
                figsize = (12, 4 * num_rows)
            else:
                figsize = (4 * num_cols, 4 * num_rows)
        
        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    
    # Create appropriate plotter
    if plot_type == "combo":
        hist_plotter = HistogramPlotter(data, config)
        diff_plotter = ECDFDifferencePlotter(data, config)
        
        for param_idx in range(data.num_parameters):
            for rank_idx, ranks_array in enumerate(data.ranks_list):
                color = colors[rank_idx] if colors else f"C{rank_idx}"
                label = ranks_labels[rank_idx]
                
                # Histogram on left
                hist_plotter.plot(
                    ax[param_idx, 0], ranks_array, param_idx, label, color
                )
                # ECDF diff on right
                diff_plotter.plot(
                    ax[param_idx, 1], ranks_array, param_idx, label, color
                )
            
            ax[param_idx, 0].set_title(f"{parameter_labels[param_idx]} - Histogram")
            ax[param_idx, 1].set_title(f"{parameter_labels[param_idx]} - ECDF Diff")
            ax[param_idx, 0].set_ylabel("Count")
            ax[param_idx, 1].set_ylabel("ECDF - Uniform")
            
            if param_idx == data.num_parameters - 1:
                ax[param_idx, 0].set_xlabel("Rank")
                ax[param_idx, 1].set_xlabel("Rank")
                
            if param_idx == 0:
                ax[param_idx, 1].legend(loc='best')
    
    else:
        # Single plot type
        if plot_type == "hist":
            plotter = HistogramPlotter(data, config)
        elif plot_type == "cdf":
            plotter = ECDFPlotter(data, config)
        else:
            plotter = ECDFDifferencePlotter(data, config)
        
        if params_in_subplots:
            for param_idx in range(data.num_parameters):
                row_idx = param_idx // num_cols
                col_idx = param_idx % num_cols
                current_ax = ax[row_idx, col_idx]
                
                for rank_idx, ranks_array in enumerate(data.ranks_list):
                    color = colors[rank_idx] if colors else f"C{rank_idx}"
                    label = ranks_labels[rank_idx]
                    plotter.plot(current_ax, ranks_array, param_idx, label, color)
                
                current_ax.set_title(parameter_labels[param_idx])
                current_ax.set_xlabel("Rank")
                
                if plot_type == "hist":
                    current_ax.set_ylabel("Count")
                elif plot_type == "cdf":
                    current_ax.set_ylabel("ECDF")
                else:
                    current_ax.set_ylabel("ECDF - Uniform")
                
                if param_idx == 0:
                    current_ax.legend(loc='best')
        else:
            # All parameters in one plot
            current_ax = ax[0, 0] if isinstance(ax, np.ndarray) else ax
            for param_idx in range(data.num_parameters):
                color = colors[param_idx] if colors else f"C{param_idx}"
                label = parameter_labels[param_idx]
                plotter.plot(current_ax, data.ranks_list[0], param_idx, label, color)
            
            current_ax.set_xlabel("Rank")
            if plot_type == "hist":
                current_ax.set_ylabel("Count")
            elif plot_type == "cdf":
                current_ax.set_ylabel("ECDF")
            else:
                current_ax.set_ylabel("ECDF - Uniform")
            current_ax.legend(loc='best')
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# YOUR DATA LOADING AND ANALYSIS (CUSTOMIZE THIS PART)
# ============================================================================

def main():
    """
    Main analysis - CUSTOMIZE THIS!
    
    Replace the example data generation with your actual SBC ranks.
    """
    
    print("=" * 60)
    print("SBC Rank Plot - Improved Version")
    print("=" * 60)
    print()
    
    # -----------------------------------------------------------------------
    # OPTION A: Load your actual ranks from file
    # -----------------------------------------------------------------------
    # print("Loading your SBC ranks from file...")
    # ranks = np.load('your_sbc_ranks.npy')  # Shape: (num_sbc_runs, num_parameters)
    # num_posterior_samples = 1000  # Set this to your actual number
    
    # -----------------------------------------------------------------------
    # OPTION B: Example data (for testing)
    # -----------------------------------------------------------------------
    
    

    print("Generating example data...")
    num_sbc_runs = 500
    num_parameters = 3
    num_posterior_samples = 1000
    
    # Perfect calibration example
    ranks = np.random.randint(
        0, num_posterior_samples + 1,
        size=(num_sbc_runs, num_parameters)
    )
    print(f"✓ Generated {num_sbc_runs} SBC runs with {num_parameters} parameters")
    print()
    
    # -----------------------------------------------------------------------
    # Create the plot
    # -----------------------------------------------------------------------
    print("Creating SBC diagnostic plot...")
    
    fig, ax = sbc_rank_plot(
        ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type='ecdf_diff',  # Try: 'hist', 'cdf', 'ecdf_diff', 'combo'
        parameter_labels=['μ', 'σ', 'ρ'],  # Customize your parameter names
        confidence_level=0.99,
        show_ci_band=True
    )
    
    fig.suptitle('SBC Diagnostic: My Model', fontsize=14, y=1.02)
    
    # -----------------------------------------------------------------------
    # Save and show
    # -----------------------------------------------------------------------
    output_file = 'sbc_diagnostic.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")
    
    plt.show()
    
    # -----------------------------------------------------------------------
    print()
    print("Creating comparison of all plot types...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_types = ['hist', 'cdf', 'ecdf_diff']
    
    for ax_single, ptype in zip(axes, plot_types):
        sbc_rank_plot(
            ranks[:, [0]],  # Just first parameter for comparison
            num_posterior_samples=num_posterior_samples,
            plot_type=ptype,  # type: ignore
            parameter_labels=['μ'],
            fig=fig,
            ax=np.array([[ax_single]])
        )
        ax_single.set_title(f'{ptype.upper()}', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sbc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison to: sbc_comparison.png")
    plt.show()
    
    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main() 



