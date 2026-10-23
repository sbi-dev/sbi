# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from sbi.utils.sbiutils import nan_tolerant_input_net
from sbi.utils.torchutils import assert_all_finite
from sbi.utils.user_input_checks import process_x


@dataclass
class ObservationDiagnostic:
    r"""Diagnostic result evaluating whether an observation $x_o$ is in-distribution.

    Attributes:
        is_in_distribution: Whether the observation is considered in-distribution
            relative to reference simulations. `True` if empirical percentile is
            greater than or equal to threshold, `False` if below threshold, or
            `None` if the diagnostic is indeterminate (e.g. no reference data
            or insufficient sample size).
        empirical_percentile: Empirical tail probability (survival percentile) of the
            observation's distance relative to reference simulations. Represents the
            proportion of reference simulations that have a distance greater than or
            equal to that of $x_o$. A value near 1.0 indicates a central observation,
            around 0.50 typical, and near 0.0 an extreme tail observation. `None` if
            indeterminate.
        mahalanobis_distance: Mahalanobis distance of the observation from the
            reference distribution mean. `None` if indeterminate.
        warning: Human-readable explanation of why the observation is flagged,
            or `None` if no warning is warranted.
        method: Scoring method used (default: `"mahalanobis"`).
        threshold: Percentile threshold used for decision (default: `0.01`).
        space: Space in which the check was performed (`"observation"` or `"embedded"`).
        details: Additional metrics such as reference sample size, reference
            distance summary statistics, subspace residuals, or constant features.
    """

    is_in_distribution: Optional[bool]
    empirical_percentile: Optional[float]
    mahalanobis_distance: Optional[float]
    warning: Optional[str]
    method: str = "mahalanobis"
    threshold: float = 0.01
    space: str = "observation"
    details: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like indexing for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for key if key is in the diagnostic, else default."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def keys(self) -> Iterator[str]:
        """Return an iterator over field names."""
        return iter(asdict(self).keys())

    def values(self) -> Iterator[Any]:
        """Return an iterator over field values."""
        return iter(asdict(self).values())

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Return an iterator over field (name, value) pairs."""
        return iter(asdict(self).items())

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic result to a dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        fields = [
            f"is_in_distribution={self.is_in_distribution}",
            (
                f"empirical_percentile={self.empirical_percentile:.4f}"
                if self.empirical_percentile is not None
                else "empirical_percentile=None"
            ),
            (
                f"mahalanobis_distance={self.mahalanobis_distance:.4f}"
                if self.mahalanobis_distance is not None
                else "mahalanobis_distance=None"
            ),
            f"warning={repr(self.warning)}",
        ]
        return f"ObservationDiagnostic({', '.join(fields)})"


def _get_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    """Safely obtain the device of a module, falling back if parameterless."""
    param = next(module.parameters(), None)
    if param is not None:
        return param.device
    buf = next(module.buffers(), None)
    if buf is not None:
        return buf.device
    return fallback


class ReferenceObservationStats:
    r"""Stores compact summary statistics from reference simulations for OOD checks.

    Computes and retains the mean, precision decomposition, and reference distance
    distribution from simulation outputs $X_{ref}$ without requiring the full raw
    dataset to be kept in memory indefinitely.
    """

    def __init__(
        self,
        mean: Tensor,
        active_indices: Tensor,
        constant_indices: Tensor,
        constant_values: Tensor,
        reference_distances: Tensor,
        n_simulations: int,
        event_shape: torch.Size,
        cholesky_factor: Optional[Tensor] = None,
        eigenvectors: Optional[Tensor] = None,
        eigenvalues: Optional[Tensor] = None,
        embedding_net: Optional[nn.Module] = None,
        is_embedded: bool = False,
        feature_scale: Optional[Tensor] = None,
    ):
        self.mean = mean
        self.active_indices = active_indices
        self.constant_indices = constant_indices
        self.constant_values = constant_values
        self.reference_distances = reference_distances  # Sorted 1D tensor
        self.n_simulations = n_simulations
        self.event_shape = event_shape
        self.cholesky_factor = cholesky_factor
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.embedding_net = embedding_net
        self.is_embedded = is_embedded
        self.feature_scale = feature_scale

    @classmethod
    def from_simulations(
        cls,
        x: Tensor,
        embedding_net: Optional[nn.Module] = None,
        max_reference_samples: int = 10_000,
        subspace_tol: float = 1e-6,
        is_embedded: bool = False,
    ) -> "ReferenceObservationStats":
        r"""Compute compact reference statistics from training simulations.

        Args:
            x: Tensor of simulation outputs with shape `(N, *event_shape)`.
            embedding_net: Optional neural network mapping $x$ to embedding space.
            max_reference_samples: Maximum number of reference samples retained
                for the reference distance score distribution. If $N$ exceeds this,
                a deterministic linspace subset or subsample is used.
            subspace_tol: Relative eigenvalue cutoff below which dimensions are
                treated as null space (for rank-deficient covariance).
            is_embedded: Whether simulations are already in embedded space.

        Returns:
            A `ReferenceObservationStats` instance.
        """
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        if x_tensor.ndim < 2:
            x_tensor = x_tensor.unsqueeze(0)

        n_sims = x_tensor.shape[0]
        event_shape = x_tensor.shape[1:]

        nan_net = nan_tolerant_input_net(embedding_net)

        if torch.isnan(x_tensor).any():
            if nan_net is not None:
                assert_all_finite(x_tensor, "Reference simulations x", allow_nan=True)
                validate_nan_input = getattr(nan_net, "validate_nan_input", None)
                if validate_nan_input is not None:
                    validate_nan_input(x_tensor)

                assert embedding_net is not None
                target_device = _get_module_device(embedding_net, x_tensor.device)
                orig_mode = embedding_net.training
                try:
                    embedding_net.eval()
                    with torch.no_grad():
                        x_tensor = embedding_net(x_tensor.to(target_device)).to(
                            x_tensor.device
                        )
                finally:
                    embedding_net.train(orig_mode)

                is_embedded = True
                event_shape = x_tensor.shape[1:]
            else:
                assert_all_finite(x_tensor, "Reference simulations x", allow_nan=False)
        elif torch.isinf(x_tensor).any():
            assert_all_finite(x_tensor, "Reference simulations x", allow_nan=False)

        # Flatten event dimensions: (N, D)
        x_flat = x_tensor.flatten(start_dim=1)
        d_dim = x_flat.shape[1]

        if n_sims < 2:
            # Insufficient samples to compute variance or covariance
            dummy_dist = torch.empty(0, dtype=torch.float32, device=x_tensor.device)
            return cls(
                mean=x_flat.mean(dim=0),
                active_indices=torch.arange(d_dim, device=x_tensor.device),
                constant_indices=torch.empty(
                    0, dtype=torch.long, device=x_tensor.device
                ),
                constant_values=torch.empty(
                    0, dtype=torch.float32, device=x_tensor.device
                ),
                reference_distances=dummy_dist,
                n_simulations=n_sims,
                event_shape=event_shape,
                embedding_net=embedding_net,
                is_embedded=is_embedded,
                feature_scale=None,
            )

        mean = x_flat.mean(dim=0)
        x_centered = x_flat - mean

        # Identify constant features (variance near zero)
        var = (x_centered**2).sum(dim=0) / (n_sims - 1)
        const_mask = var <= 1e-12
        constant_indices = torch.where(const_mask)[0]
        active_indices = torch.where(~const_mask)[0]
        constant_values = mean[constant_indices]

        if len(active_indices) == 0:
            # All features are constant
            zero_dist = torch.zeros(
                min(n_sims, max_reference_samples),
                dtype=torch.float32,
                device=x_tensor.device,
            )
            return cls(
                mean=mean,
                active_indices=active_indices,
                constant_indices=constant_indices,
                constant_values=constant_values,
                reference_distances=zero_dist,
                n_simulations=n_sims,
                event_shape=event_shape,
                embedding_net=embedding_net,
                is_embedded=is_embedded,
                feature_scale=None,
            )

        x_active = x_centered[:, active_indices]
        n_active = len(active_indices)

        # Standardize active features by sample standard deviations so decomposition
        # operates on the correlation matrix R. This preserves valid low-variance
        # physical dimensions and avoids conflating scales with rank deficiency.
        feature_scale = torch.sqrt(torch.clamp(var[active_indices], min=1e-12))
        x_scaled = x_active / feature_scale

        # Sample correlation matrix for active dimensions
        R = (x_scaled.t() @ x_scaled) / (n_sims - 1)

        # Attempt Cholesky factorization on correlation matrix for stable, fast solves
        cholesky_factor = None
        eigenvectors = None
        eigenvalues = None

        is_full_rank = n_sims > n_active
        if is_full_rank:
            try:
                L = torch.linalg.cholesky(R)
                # In correlation matrix, diagonal elements are <= 1.0. A pivot
                # < 1e-5 indicates true multicollinearity / rank deficiency.
                min_pivot = L.diagonal().min()
                if min_pivot > 1e-5:
                    cholesky_factor = L
            except torch.linalg.LinAlgError:
                cholesky_factor = None

        if cholesky_factor is None:
            # Fall back to spectral decomposition on correlation matrix
            try:
                eigs, vecs = torch.linalg.eigh(R)
                # eigh returns ascending order; flip to descending
                eigs = torch.flip(eigs, dims=(0,))
                vecs = torch.flip(vecs, dims=(1,))
            except torch.linalg.LinAlgError:
                # If eigh fails to converge due to ill-conditioning or repeated
                # eigenvalues, compute the spectral decomposition via SVD of x_scaled
                _, S, Vh = torch.linalg.svd(x_scaled, full_matrices=False)
                eigs = (S**2) / (n_sims - 1)
                vecs = Vh.t()

            max_eig = torch.clamp(eigs[0], min=1e-12)
            valid_eig_mask = eigs > (max_eig * subspace_tol)

            eigenvalues = torch.clamp(eigs[valid_eig_mask], min=1e-12)
            eigenvectors = vecs[:, valid_eig_mask]

        # Compute reference distances for training points
        subsample_idx = torch.arange(n_sims, device=x_tensor.device)
        if n_sims > max_reference_samples:
            subsample_idx = torch.linspace(
                0, n_sims - 1, max_reference_samples, device=x_tensor.device
            ).long()

        x_sub_scaled = x_scaled[subsample_idx]
        ref_dists = cls._compute_distances_internal(
            x_sub_scaled, cholesky_factor, eigenvectors, eigenvalues
        )

        # Apply exact leave-one-out scaling for symmetric out-of-sample comparison
        # d_{-i}^2 = [N^2 (N - 2) / ((N - 1) * ((N - 1)^2 - N * d_i^2))] * d_i^2
        if n_sims > n_active + 2 and cholesky_factor is not None:
            denom = (n_sims - 1) ** 2 - n_sims * (ref_dists**2)
            valid_loo = denom > 0
            scaled_dists = ref_dists.clone()
            scaled_dists[valid_loo] = torch.sqrt(
                torch.clamp(
                    ((n_sims**2 * (n_sims - 2)) / ((n_sims - 1) * denom[valid_loo]))
                    * (ref_dists[valid_loo] ** 2),
                    min=0.0,
                )
            )
            ref_dists = scaled_dists

        sorted_ref_dists, _ = torch.sort(ref_dists)

        return cls(
            mean=mean,
            active_indices=active_indices,
            constant_indices=constant_indices,
            constant_values=constant_values,
            reference_distances=sorted_ref_dists,
            n_simulations=n_sims,
            event_shape=event_shape,
            cholesky_factor=cholesky_factor,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            embedding_net=embedding_net,
            is_embedded=is_embedded,
            feature_scale=feature_scale,
        )

    @staticmethod
    def _compute_distances_internal(
        x_centered_active: Tensor,
        cholesky_factor: Optional[Tensor],
        eigenvectors: Optional[Tensor],
        eigenvalues: Optional[Tensor],
    ) -> Tensor:
        """Compute Mahalanobis distances given Cholesky factor or SVD components."""
        if cholesky_factor is not None:
            # Solve L @ w = x_centered_active.t() -> w shape (n_active, B)
            w = torch.linalg.solve_triangular(
                cholesky_factor, x_centered_active.t(), upper=False
            )
            dists = torch.sqrt(torch.clamp((w**2).sum(dim=0), min=0.0))
            return dists
        elif eigenvectors is not None and eigenvalues is not None:
            # y = eigenvectors.t() @ x_centered_active.t() -> (k, B)
            y = eigenvectors.t() @ x_centered_active.t()
            scaled_y = y / torch.sqrt(eigenvalues).unsqueeze(1)
            in_subspace_dist_sq = (scaled_y**2).sum(dim=0)
            return torch.sqrt(torch.clamp(in_subspace_dist_sq, min=0.0))
        else:
            return torch.zeros(
                x_centered_active.shape[0], device=x_centered_active.device
            )

    def compute_distance(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        r"""Compute Mahalanobis distance for observation(s) $x$.

        Args:
            x: Observed tensor of shape `(B, *event_shape)` or `(*event_shape)`.

        Returns:
            Tuple of (distances, details_dict).
        """
        is_embedded = getattr(self, "is_embedded", False)
        if is_embedded and self.embedding_net is not None:
            nan_net = nan_tolerant_input_net(self.embedding_net)
            if nan_net is not None:
                assert_all_finite(x, "Observation x", allow_nan=True)
                validate_nan_input = getattr(nan_net, "validate_nan_input", None)
                if validate_nan_input is not None and torch.isnan(x).any():
                    validate_nan_input(x)
            else:
                if torch.isnan(x).any():
                    raise ValueError("Observation x contains NaN values.")
            if torch.isinf(x).any():
                raise ValueError("Observation x contains Inf values.")

            needs_embedding = torch.isnan(x).any() or (
                tuple(x.shape[-len(self.event_shape) :]) != tuple(self.event_shape)
            )
            if needs_embedding:
                target_device = _get_module_device(self.embedding_net, x.device)
                orig_mode = self.embedding_net.training
                try:
                    self.embedding_net.eval()
                    with torch.no_grad():
                        x = self.embedding_net(x.to(target_device)).to(x.device)
                finally:
                    self.embedding_net.train(orig_mode)
        else:
            if torch.isnan(x).any():
                raise ValueError("Observation x contains NaN values.")
            if torch.isinf(x).any():
                raise ValueError("Observation x contains Inf values.")

        # Ensure batch dimension
        if x.dim() == len(self.event_shape):
            if tuple(x.shape) != tuple(self.event_shape):
                raise ValueError(
                    f"Dimension mismatch: expected observation with event shape "
                    f"{tuple(self.event_shape)}, but got shape {tuple(x.shape)}."
                )
            x_batched = x.unsqueeze(0)
        elif x.dim() == len(self.event_shape) + 1:
            if tuple(x.shape[1:]) != tuple(self.event_shape):
                raise ValueError(
                    f"Dimension mismatch: expected observation with event shape "
                    f"{tuple(self.event_shape)}, but got batch with event shape "
                    f"{tuple(x.shape[1:])}."
                )
            x_batched = x
        elif len(self.event_shape) == 1 and x.numel() == self.event_shape[0]:
            x_batched = x.reshape(1, -1)
        elif (
            len(self.event_shape) == 1
            and x.dim() == 2
            and x.shape[-1] == self.event_shape[0]
        ):
            x_batched = x
        else:
            raise ValueError(
                f"Dimension mismatch: expected observation with event shape "
                f"{tuple(self.event_shape)}, but got tensor with shape "
                f"{tuple(x.shape)}."
            )

        x_flat = x_batched.flatten(start_dim=1)
        mean = self.mean.to(x_flat.device)
        x_centered = x_flat - mean

        details: Dict[str, Any] = {}
        if is_embedded:
            details["space"] = "embedded"
            details["embedding_dim"] = self.event_shape[-1]

        # 1. Check constant features
        if len(self.constant_indices) > 0:
            const_idx = self.constant_indices.to(x_flat.device)
            const_val = self.constant_values.to(x_flat.device)
            const_diff = torch.abs(x_flat[:, const_idx] - const_val)
            const_violation = (const_diff > 1e-5).any(dim=1)
            if const_violation.any():
                details["constant_feature_violation"] = True
                # Return infinite distance for points violating constant features
                distances = torch.full(
                    (x_flat.shape[0],),
                    float("inf"),
                    device=x_flat.device,
                    dtype=torch.float32,
                )
                distances[~const_violation] = 0.0
                return distances, details

        if len(self.active_indices) == 0:
            # All features were constant and none were violated
            return torch.zeros(
                x_flat.shape[0], device=x_flat.device, dtype=torch.float32
            ), details

        active_idx = self.active_indices.to(x_flat.device)
        x_active = x_centered[:, active_idx]
        if self.feature_scale is not None:
            feature_scale = self.feature_scale.to(x_flat.device)
            x_scaled = x_active / feature_scale
        else:
            x_scaled = x_active

        cholesky = (
            self.cholesky_factor.to(x_flat.device)
            if self.cholesky_factor is not None
            else None
        )
        evecs = (
            self.eigenvectors.to(x_flat.device)
            if self.eigenvectors is not None
            else None
        )
        evals = (
            self.eigenvalues.to(x_flat.device) if self.eigenvalues is not None else None
        )

        dists = self._compute_distances_internal(x_scaled, cholesky, evecs, evals)

        # If spectral decomposition was used because of collinear features
        # (N > D, but rank < D), check for out-of-subspace residual. When N <= D,
        # rank deficiency is an artifact of sample size rather than a constraint.
        if evecs is not None and self.n_simulations > len(self.active_indices):
            # Project onto subspace in standardized coordinates:
            # y = evecs.t() @ x_scaled.t()
            y = evecs.t() @ x_scaled.t()
            x_proj = (evecs @ y).t()  # (B, n_active)
            residual = x_scaled - x_proj
            res_norm = torch.norm(residual, dim=1)
            scale = torch.norm(x_scaled, dim=1) + 1e-8
            rel_res = res_norm / scale
            details["subspace_residual"] = rel_res.cpu().tolist()
            if (rel_res > 0.05).any():
                details["out_of_subspace"] = True

        return dists, details

    def diagnose(
        self,
        x_o: Tensor,
        threshold: float = 0.01,
        space: str = "observation",
    ) -> ObservationDiagnostic:
        r"""Diagnose observation $x_o$ against reference simulations.

        Args:
            x_o: Observation tensor.
            threshold: Percentile threshold below which observation is flagged.
            space: Space in which the check is evaluated.

        Returns:
            `ObservationDiagnostic` result.
        """
        if self.n_simulations < 2 or len(self.reference_distances) == 0:
            return ObservationDiagnostic(
                is_in_distribution=None,
                empirical_percentile=None,
                mahalanobis_distance=None,
                warning=(
                    f"Insufficient reference simulations (N={self.n_simulations}) "
                    "to compute observation diagnostic. At least 2 reference "
                    "simulations are required."
                ),
                threshold=threshold,
                space=space,
                details={"n_reference": self.n_simulations},
            )

        dists, details = self.compute_distance(x_o)
        ref_dists = self.reference_distances.to(x_o.device)

        # Single observation vs batch
        is_batched = dists.shape[0] > 1

        # Empirical tail percentile / survival probability:
        # proportion of reference simulations with distance >= d_o
        # P(D_ref >= d_o)
        percentiles = []
        for d in dists:
            if math.isinf(d.item()):
                percentiles.append(0.0)
            else:
                p = (
                    (ref_dists > d).float().sum() + 0.5 * (ref_dists == d).float().sum()
                ) / len(ref_dists)
                percentiles.append(p.item())

        details["n_reference"] = self.n_simulations
        details["ref_distance_min"] = ref_dists[0].item()
        details["ref_distance_median"] = ref_dists[len(ref_dists) // 2].item()
        details["ref_distance_max"] = ref_dists[-1].item()

        if self.n_simulations < int(1.0 / threshold):
            details["sample_size_note"] = (
                f"Reference sample size N={self.n_simulations} is smaller than "
                f"1/threshold={int(1.0 / threshold)}, limiting empirical percentile "
                "resolution."
            )

        space_to_report = "embedded" if getattr(self, "is_embedded", False) else space

        if not is_batched:
            dist_val = float(dists[0].item())
            perc_val = float(percentiles[0])
            is_in_dist = (
                (perc_val >= threshold)
                and not details.get("constant_feature_violation", False)
                and not details.get("out_of_subspace", False)
            )

            warning = None
            if not is_in_dist:
                if details.get("constant_feature_violation", False):
                    warning = (
                        "Observation x_o violates one or more constant features "
                        "present in training simulations. The observation is strictly "
                        "out-of-distribution."
                    )
                elif details.get("out_of_subspace", False):
                    dist_val = float("inf")
                    perc_val = 0.0
                    warning = (
                        "Observation x_o lies outside the subspace spanned by the "
                        "training simulations (subspace residual > 0.05). The "
                        "observation is strictly out-of-distribution."
                    )
                else:
                    warning = (
                        "Observation x_o falls in the extreme tails of the "
                        f"training simulation distribution (empirical percentile: "
                        f"{perc_val:.4f} < {threshold:.4f}, Mahalanobis distance: "
                        f"{dist_val:.2f}). Posterior samples may be misleading due to "
                        "misspecification or out-of-distribution conditioning."
                    )

            return ObservationDiagnostic(
                is_in_distribution=is_in_dist,
                empirical_percentile=perc_val,
                mahalanobis_distance=dist_val,
                warning=warning,
                threshold=threshold,
                space=space_to_report,
                details=details,
            )
        else:
            # Batch of observations:
            all_in_dist = (
                all((p >= threshold) for p in percentiles)
                and not details.get("constant_feature_violation", False)
                and not details.get("out_of_subspace", False)
            )
            max_dist = float(torch.max(dists).item())
            min_perc = float(min(percentiles))

            warning = None
            if not all_in_dist:
                n_ood = sum(1 for p in percentiles if p < threshold)
                warning = (
                    f"{n_ood}/{len(percentiles)} observations in batch fall below the "
                    f"in-distribution percentile threshold (min empirical percentile: "
                    f"{min_perc:.4f} < {threshold:.4f}, max Mahalanobis distance: "
                    f"{max_dist:.2f}). Posterior samples may be misleading."
                )

            details["batch_percentiles"] = percentiles
            details["batch_distances"] = dists.cpu().tolist()

            return ObservationDiagnostic(
                is_in_distribution=all_in_dist,
                empirical_percentile=min_perc,
                mahalanobis_distance=max_dist,
                warning=warning,
                threshold=threshold,
                space=space_to_report,
                details=details,
            )

    def to(self, device: Union[str, torch.device]) -> "ReferenceObservationStats":
        """Move all stored statistics tensors to the target device."""
        self.mean = self.mean.to(device)
        self.active_indices = self.active_indices.to(device)
        self.constant_indices = self.constant_indices.to(device)
        self.constant_values = self.constant_values.to(device)
        self.reference_distances = self.reference_distances.to(device)
        if self.cholesky_factor is not None:
            self.cholesky_factor = self.cholesky_factor.to(device)
        if self.eigenvectors is not None:
            self.eigenvectors = self.eigenvectors.to(device)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.to(device)
        if self.embedding_net is not None:
            self.embedding_net.to(device)
        if self.feature_scale is not None:
            self.feature_scale = self.feature_scale.to(device)
        return self


def diagnose_observation(
    x_o: Tensor,
    x_ref: Optional[Tensor] = None,
    *,
    reference_stats: Optional[ReferenceObservationStats] = None,
    threshold: float = 0.01,
    space: Literal["observation", "embedded"] = "observation",
    embedding_net: Optional[nn.Module] = None,
) -> ObservationDiagnostic:
    r"""Diagnose whether an observation $x_o$ is supported by simulation distribution.

    Evaluates the Mahalanobis distance of $x_o$ relative to the reference simulation
    distribution and ranks it against the reference score distribution to determine
    the empirical tail percentile.

    Args:
        x_o: The observed data to diagnose.
        x_ref: Optional reference simulations. If provided, reference statistics are
            computed from `x_ref`. Either `x_ref` or `reference_stats` must be provided.
        reference_stats: Optional precomputed `ReferenceObservationStats`.
        threshold: Empirical percentile threshold below which the observation is
            flagged as out-of-distribution (default: `0.01`, corresponding to the
            bottom 1% tail probability).
        space: Space in which to perform the diagnostic (`"observation"` or
            `"embedded"`). When `"embedded"`, observations are projected through
            `embedding_net`.
        embedding_net: Optional neural embedding network used when `space="embedded"`.

    Returns:
        An `ObservationDiagnostic` object containing `is_in_distribution`,
        `empirical_percentile`, `mahalanobis_distance`, and `warning`.
    """
    x_proc = process_x(x_o, x_event_shape=None)

    effective_net = embedding_net or (
        reference_stats.embedding_net if reference_stats is not None else None
    )
    nan_net = nan_tolerant_input_net(effective_net)

    if nan_net is not None:
        assert_all_finite(x_proc, "Observation x_o", allow_nan=True)
        validate_nan_input = getattr(nan_net, "validate_nan_input", None)
        if validate_nan_input is not None and torch.isnan(x_proc).any():
            validate_nan_input(x_proc)
    else:
        assert_all_finite(x_proc, "Observation x_o", allow_nan=False)

    if reference_stats is None and x_ref is None:
        return ObservationDiagnostic(
            is_in_distribution=None,
            empirical_percentile=None,
            mahalanobis_distance=None,
            warning=(
                "No reference simulations available to diagnose observation. "
                "Provide `x_ref` or precomputed `reference_stats`."
            ),
            threshold=threshold,
            space=space,
        )

    if reference_stats is None and x_ref is not None:
        x_ref_proc = process_x(x_ref, x_event_shape=None)
        if nan_net is not None:
            assert_all_finite(x_ref_proc, "Reference simulations x_ref", allow_nan=True)
            validate_nan_input = getattr(nan_net, "validate_nan_input", None)
            if validate_nan_input is not None and torch.isnan(x_ref_proc).any():
                validate_nan_input(x_ref_proc)
        else:
            assert_all_finite(
                x_ref_proc, "Reference simulations x_ref", allow_nan=False
            )

        if space == "embedded":
            if effective_net is None:
                raise ValueError(
                    "Cannot evaluate diagnostic in 'embedded' space without an "
                    "embedding_net."
                )
            net_device = _get_module_device(effective_net, x_ref_proc.device)
            orig_mode = effective_net.training
            try:
                effective_net.eval()
                with torch.no_grad():
                    x_ref_eval = effective_net(x_ref_proc.to(net_device)).to(
                        x_ref_proc.device
                    )
                    x_o_eval = effective_net(x_proc.to(net_device)).to(x_proc.device)
            finally:
                effective_net.train(orig_mode)
            reference_stats = ReferenceObservationStats.from_simulations(
                x_ref_eval, embedding_net=effective_net, is_embedded=True
            )
            diag = reference_stats.diagnose(x_o_eval, threshold=threshold, space=space)
            diag.details["embedding_dim"] = x_ref_eval.shape[-1]
            return diag
        else:
            reference_stats = ReferenceObservationStats.from_simulations(
                x_ref_proc, embedding_net=effective_net, is_embedded=False
            )
            return reference_stats.diagnose(x_proc, threshold=threshold, space=space)

    assert reference_stats is not None

    if getattr(reference_stats, "is_embedded", False):
        # When reference statistics are embedded (e.g. from NaN-tolerant net or
        # embedded simulations), diagnosis is evaluated in embedded space.
        if (
            effective_net is not None
            and x_proc.shape[1:] != reference_stats.event_shape
        ):
            net_device = _get_module_device(effective_net, x_proc.device)
            orig_mode = effective_net.training
            try:
                effective_net.eval()
                with torch.no_grad():
                    x_proc_eval = effective_net(x_proc.to(net_device)).to(x_proc.device)
            finally:
                effective_net.train(orig_mode)
        else:
            x_proc_eval = x_proc

        diag = reference_stats.diagnose(
            x_proc_eval, threshold=threshold, space="embedded"
        )
        diag.details["embedding_dim"] = reference_stats.event_shape[-1]
        return diag

    if space == "embedded":
        raise ValueError(
            "Cannot evaluate diagnostic in 'embedded' space using reference_stats "
            "computed in observation space. Pass reference_stats computed from "
            "embedded simulations, provide x_ref, or call "
            "posterior.diagnose_observation(..., space='embedded')."
        )

    return reference_stats.diagnose(x_proc, threshold=threshold, space=space)
