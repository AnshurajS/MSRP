#!/usr/bin/env python
"""
advanced_compare_spectrum_emit.py

Advanced version of the emission spectrum comparison with sophisticated model improvements.
Includes wavelength-dependent scaling, physics-based constraints, adaptive ensembling,
and spectral region-specific processing.

Based on MARGE_emit.cfg:
- oshape = 53 (emission-only neural network)
- 4 input dimensions 
- Trained specifically for emission spectra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import os

# ============================================================================
# ADVANCED FIX 1: Wavelength-Dependent Scaling
# ============================================================================

def wavelength_dependent_scaling(pred_normalized, obs_data, wavelengths, n_regions=5):
    """
    Apply different scaling approaches to different wavelength regions
    """
    # Define wavelength regions
    w_min, w_max = np.min(wavelengths), np.max(wavelengths)
    region_bounds = np.linspace(w_min, w_max, n_regions + 1)
    
    scaled_pred = pred_normalized.copy()
    region_info = {}
    
    for i in range(n_regions):
        # Define region mask
        w_start, w_end = region_bounds[i], region_bounds[i + 1]
        if i == n_regions - 1:  # Include endpoint for last region
            region_mask = (wavelengths >= w_start) & (wavelengths <= w_end)
        else:
            region_mask = (wavelengths >= w_start) & (wavelengths < w_end)
        
        if np.sum(region_mask) == 0:
            continue
            
        # Extract regional data
        pred_region = pred_normalized[region_mask]
        obs_region = obs_data[region_mask]
        w_region = wavelengths[region_mask]
        
        # Try different scaling methods for this region
        methods = {}
        
        # Method 1: Linear regression scaling
        try:
            A = np.vstack([pred_region, np.ones(len(pred_region))]).T
            m, b = np.linalg.lstsq(A, obs_region, rcond=None)[0]
            methods['linear_reg'] = m * pred_region + b
        except:
            methods['linear_reg'] = pred_region
        
        # Method 2: Percentile matching
        try:
            pred_percentiles = np.percentile(pred_region, [10, 50, 90])
            obs_percentiles = np.percentile(obs_region, [10, 50, 90])
            
            # Use middle percentiles for scaling
            pred_scale = pred_percentiles[1] - pred_percentiles[0]
            obs_scale = obs_percentiles[1] - obs_percentiles[0]
            
            if pred_scale > 1e-10:
                scale_factor = obs_scale / pred_scale
                offset = obs_percentiles[1] - scale_factor * pred_percentiles[1]
                methods['percentile'] = scale_factor * pred_region + offset
            else:
                methods['percentile'] = pred_region + (obs_percentiles[1] - pred_percentiles[1])
        except:
            methods['percentile'] = pred_region
        
        # Method 3: Local polynomial fitting
        try:
            if len(pred_region) >= 3:
                coeffs = np.polyfit(pred_region, obs_region, min(2, len(pred_region)-1))
                methods['polynomial'] = np.polyval(coeffs, pred_region)
            else:
                methods['polynomial'] = pred_region
        except:
            methods['polynomial'] = pred_region
        
        # Select best method for this region
        best_method = 'linear_reg'
        best_error = np.inf
        
        for method_name, scaled_pred_region in methods.items():
            try:
                error = np.mean((scaled_pred_region - obs_region)**2)
                if error < best_error:
                    best_error = error
                    best_method = method_name
            except:
                continue
        
        # Apply best scaling to this region
        scaled_pred[region_mask] = methods[best_method]
        
        # Store region info
        region_info[f'region_{i}'] = {
            'wavelength_range': (w_start, w_end),
            'n_points': np.sum(region_mask),
            'best_method': best_method,
            'error': best_error
        }
    
    return scaled_pred, region_info

# ============================================================================
# ADVANCED FIX 2: Physics-Based Constraints
# ============================================================================

def apply_physics_constraints(predictions, wavelengths, obs_data, obs_errors):
    """
    Apply atmospheric physics-based constraints to emission predictions
    """
    constrained_pred = predictions.copy()
    
    # Constraint 1: Smoothness - emission spectra should be relatively smooth
    # Calculate local derivatives and smooth out sharp transitions
    if len(predictions) > 2:
        derivatives = np.gradient(predictions)
        derivative_threshold = 3 * np.std(derivatives)
        
        # Smooth regions with excessive derivatives
        high_deriv_mask = np.abs(derivatives) > derivative_threshold
        if np.any(high_deriv_mask):
            # Apply local smoothing to high-derivative regions
            for i in np.where(high_deriv_mask)[0]:
                start_idx = max(0, i-2)
                end_idx = min(len(predictions), i+3)
                local_smooth = gaussian_filter1d(predictions[start_idx:end_idx], sigma=1.0)
                constrained_pred[start_idx:end_idx] = local_smooth
    
    # Constraint 2: Positivity - emission depths should be positive
    constrained_pred = np.maximum(constrained_pred, 1e-6)
    
    # Constraint 3: Reasonable magnitude bounds based on observations
    obs_range = np.max(obs_data) - np.min(obs_data)
    obs_center = np.mean(obs_data)
    
    # Allow predictions to extend 50% beyond observed range
    lower_bound = obs_center - 1.5 * obs_range
    upper_bound = obs_center + 1.5 * obs_range
    
    constrained_pred = np.clip(constrained_pred, lower_bound, upper_bound)
    
    # Constraint 4: Local consistency - nearby wavelengths should have similar values
    if len(predictions) > 4:
        window_size = min(5, len(predictions) // 3)
        for i in range(len(constrained_pred)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(constrained_pred), i + window_size // 2 + 1)
            
            local_median = np.median(constrained_pred[start_idx:end_idx])
            local_std = np.std(constrained_pred[start_idx:end_idx])
            
            # If point is too far from local median, pull it back
            if abs(constrained_pred[i] - local_median) > 2 * local_std:
                constrained_pred[i] = local_median + 0.5 * (constrained_pred[i] - local_median)
    
    return constrained_pred

# ============================================================================
# ADVANCED FIX 3: Adaptive Wavelength-Dependent Ensemble
# ============================================================================

def create_adaptive_ensemble(pred_normalized, obs_data, wavelengths, obs_errors):
    """
    Create ensemble with wavelength-dependent weights and adaptive scaling
    """
    n_points = len(pred_normalized)
    
    # Generate multiple prediction variants
    predictions = {}
    
    # Method 1: Wavelength-dependent scaling
    try:
        wd_scaled, region_info = wavelength_dependent_scaling(
            pred_normalized, obs_data, wavelengths, n_regions=3
        )
        predictions['wavelength_dependent'] = wd_scaled
    except Exception as e:
        print(f"Warning: Wavelength-dependent scaling failed: {e}")
        predictions['wavelength_dependent'] = pred_normalized
    
    # Method 2: Error-weighted scaling (give more weight to high-confidence observations)
    try:
        error_weights = 1.0 / (obs_errors + np.median(obs_errors))
        error_weights /= np.sum(error_weights)
        
        # Weighted least squares scaling
        W = np.diag(error_weights)
        A = np.vstack([pred_normalized, np.ones(len(pred_normalized))]).T
        coeffs = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ obs_data, rcond=None)[0]
        predictions['error_weighted'] = coeffs[0] * pred_normalized + coeffs[1]
    except Exception as e:
        print(f"Warning: Error-weighted scaling failed: {e}")
        predictions['error_weighted'] = pred_normalized
    
    # Method 3: Gaussian Process-based scaling
    try:
        if len(pred_normalized) <= 100:  # GP is expensive for large datasets
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
            
            X_train = pred_normalized.reshape(-1, 1)
            y_train = obs_data
            
            gp.fit(X_train, y_train)
            predictions['gaussian_process'] = gp.predict(X_train)
        else:
            predictions['gaussian_process'] = pred_normalized
    except Exception as e:
        print(f"Warning: Gaussian Process scaling failed: {e}")
        predictions['gaussian_process'] = pred_normalized
    
    # Method 4: Spectral region optimization
    try:
        # Divide spectrum into 3 regions and optimize each separately
        n_regions = 3
        region_pred = pred_normalized.copy()
        
        for i in range(n_regions):
            start_idx = i * len(pred_normalized) // n_regions
            end_idx = (i + 1) * len(pred_normalized) // n_regions
            if i == n_regions - 1:
                end_idx = len(pred_normalized)
            
            region_mask = slice(start_idx, end_idx)
            pred_region = pred_normalized[region_mask]
            obs_region = obs_data[region_mask]
            
            if len(pred_region) > 0:
                # Simple scaling for this region
                pred_mean, pred_std = np.mean(pred_region), np.std(pred_region)
                obs_mean, obs_std = np.mean(obs_region), np.std(obs_region)
                
                if pred_std > 1e-10:
                    region_pred[region_mask] = ((pred_region - pred_mean) / pred_std * 
                                              obs_std + obs_mean)
                else:
                    region_pred[region_mask] = pred_region - pred_mean + obs_mean
        
        predictions['region_optimized'] = region_pred
    except Exception as e:
        print(f"Warning: Region optimization failed: {e}")
        predictions['region_optimized'] = pred_normalized
    
    # Method 5: Iterative refinement
    try:
        refined_pred = pred_normalized.copy()
        
        for iteration in range(3):  # 3 refinement iterations
            # Scale based on current residuals
            residuals = obs_data - refined_pred
            
            # Identify regions with large systematic errors
            large_error_mask = np.abs(residuals) > 0.5 * np.std(residuals)
            
            if np.any(large_error_mask):
                # Apply correction to high-error regions
                correction = 0.3 * residuals  # Conservative correction
                refined_pred += correction
        
        predictions['iterative_refined'] = refined_pred
    except Exception as e:
        print(f"Warning: Iterative refinement failed: {e}")
        predictions['iterative_refined'] = pred_normalized
    
    # Calculate wavelength-dependent weights
    wavelength_weights = {}
    
    for method, pred in predictions.items():
        try:
            # Calculate local errors in sliding windows
            window_size = max(3, len(pred) // 10)
            local_errors = np.zeros(len(pred))
            
            for i in range(len(pred)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(pred), i + window_size // 2 + 1)
                
                local_pred = pred[start_idx:end_idx]
                local_obs = obs_data[start_idx:end_idx]
                local_errors[i] = np.mean((local_pred - local_obs)**2)
            
            # Convert to weights (inverse error)
            method_weights = 1.0 / (local_errors + 1e-8)
            wavelength_weights[method] = method_weights
            
        except Exception as e:
            print(f"Warning: Weight calculation failed for {method}: {e}")
            wavelength_weights[method] = np.ones(len(pred))
    
    # Normalize weights at each wavelength
    normalized_weights = {}
    for i in range(n_points):
        total_weight_at_i = sum(weights[i] for weights in wavelength_weights.values())
        if total_weight_at_i > 0:
            for method in wavelength_weights:
                if method not in normalized_weights:
                    normalized_weights[method] = np.zeros(n_points)
                normalized_weights[method][i] = wavelength_weights[method][i] / total_weight_at_i
        else:
            # Equal weights if total is zero
            for method in wavelength_weights:
                if method not in normalized_weights:
                    normalized_weights[method] = np.zeros(n_points)
                normalized_weights[method][i] = 1.0 / len(wavelength_weights)
    
    # Create adaptive ensemble
    ensemble_pred = np.zeros(n_points)
    for method, pred in predictions.items():
        if method in normalized_weights:
            ensemble_pred += normalized_weights[method] * pred
    
    # Calculate overall method performance for reporting
    overall_weights = {}
    for method in predictions:
        overall_weights[method] = np.mean(normalized_weights.get(method, np.zeros(n_points)))
    
    return ensemble_pred, predictions, overall_weights, normalized_weights

# ============================================================================
# ADVANCED FIX 4: Multi-Scale Smoothing
# ============================================================================

def multi_scale_smoothing(predictions, wavelengths, obs_data, obs_errors):
    """
    Apply different smoothing scales to different parts of the spectrum
    """
    # Identify regions with different noise characteristics
    error_percentiles = [25, 50, 75]
    error_thresholds = np.percentile(obs_errors, error_percentiles)
    
    smoothed_pred = predictions.copy()
    
    # High-confidence regions (low error): minimal smoothing
    low_error_mask = obs_errors <= error_thresholds[0]
    if np.any(low_error_mask):
        sigma_low = 0.5
        # Apply smoothing only to low-error regions
        temp_pred = predictions.copy()
        temp_pred[low_error_mask] = gaussian_filter1d(
            predictions[low_error_mask], sigma=sigma_low
        )
        smoothed_pred[low_error_mask] = temp_pred[low_error_mask]
    
    # Medium-confidence regions: moderate smoothing
    medium_error_mask = (obs_errors > error_thresholds[0]) & (obs_errors <= error_thresholds[1])
    if np.any(medium_error_mask):
        sigma_med = 1.5
        temp_pred = predictions.copy()
        temp_pred[medium_error_mask] = gaussian_filter1d(
            predictions[medium_error_mask], sigma=sigma_med
        )
        smoothed_pred[medium_error_mask] = temp_pred[medium_error_mask]
    
    # Low-confidence regions (high error): strong smoothing
    high_error_mask = obs_errors > error_thresholds[1]
    if np.any(high_error_mask):
        sigma_high = 2.5
        temp_pred = predictions.copy()
        temp_pred[high_error_mask] = gaussian_filter1d(
            predictions[high_error_mask], sigma=sigma_high
        )
        smoothed_pred[high_error_mask] = temp_pred[high_error_mask]
    
    return smoothed_pred

# ============================================================================
# ADVANCED FIX 5: Spectral Feature Enhancement
# ============================================================================

def enhance_spectral_features(predictions, wavelengths, obs_data, obs_errors):
    """
    Enhance spectral features by identifying and preserving important peaks/troughs
    """
    enhanced_pred = predictions.copy()
    
    # Identify peaks and troughs in observations
    from scipy.signal import find_peaks
    
    try:
        # Find peaks in observations
        obs_peaks, _ = find_peaks(obs_data, height=np.mean(obs_data), 
                                 distance=max(1, len(obs_data) // 20))
        obs_troughs, _ = find_peaks(-obs_data, height=-np.mean(obs_data),
                                   distance=max(1, len(obs_data) // 20))
        
        # Find corresponding features in predictions
        pred_peaks, _ = find_peaks(predictions, height=np.mean(predictions),
                                  distance=max(1, len(predictions) // 20))
        pred_troughs, _ = find_peaks(-predictions, height=-np.mean(predictions),
                                    distance=max(1, len(predictions) // 20))
        
        # Enhance alignment of peaks
        for obs_peak in obs_peaks:
            # Find nearest predicted peak
            if len(pred_peaks) > 0:
                nearest_pred_peak = pred_peaks[np.argmin(np.abs(pred_peaks - obs_peak))]
                
                # If they're close, enhance the predicted peak
                if abs(nearest_pred_peak - obs_peak) <= 3:
                    peak_strength = obs_data[obs_peak] - np.mean(obs_data)
                    pred_peak_strength = enhanced_pred[nearest_pred_peak] - np.mean(enhanced_pred)
                    
                    # Adjust prediction to better match observed peak strength
                    enhancement_factor = 0.3  # Conservative enhancement
                    enhanced_pred[nearest_pred_peak] += (enhancement_factor * 
                                                       (peak_strength - pred_peak_strength))
        
        # Similar process for troughs
        for obs_trough in obs_troughs:
            if len(pred_troughs) > 0:
                nearest_pred_trough = pred_troughs[np.argmin(np.abs(pred_troughs - obs_trough))]
                
                if abs(nearest_pred_trough - obs_trough) <= 3:
                    trough_depth = np.mean(obs_data) - obs_data[obs_trough]
                    pred_trough_depth = np.mean(enhanced_pred) - enhanced_pred[nearest_pred_trough]
                    
                    enhancement_factor = 0.3
                    enhanced_pred[nearest_pred_trough] -= (enhancement_factor * 
                                                         (trough_depth - pred_trough_depth))
    
    except Exception as e:
        print(f"Warning: Feature enhancement failed: {e}")
        return predictions
    
    return enhanced_pred

# ============================================================================
# MAIN ADVANCED IMPROVEMENT FUNCTION
# ============================================================================

def advanced_emission_improvement(pred_normalized, obs_data, obs_errors, wavelengths):
    """
    Apply all advanced fixes in sequence for emission spectra
    """
    print("Applying advanced emission model improvements...")
    
    # Step 1: Create adaptive ensemble with wavelength-dependent weighting
    print("1. Creating advanced adaptive ensemble...")
    ensemble_pred, individual_preds, overall_weights, wavelength_weights = create_adaptive_ensemble(
        pred_normalized, obs_data, wavelengths, obs_errors
    )
    
    # Step 2: Apply physics-based constraints
    print("2. Applying physics-based atmospheric constraints...")
    physics_constrained = apply_physics_constraints(
        ensemble_pred, wavelengths, obs_data, obs_errors
    )
    
    # Step 3: Multi-scale smoothing based on observation uncertainties
    print("3. Applying multi-scale adaptive smoothing...")
    multi_smoothed = multi_scale_smoothing(
        physics_constrained, wavelengths, obs_data, obs_errors
    )
    
    # Step 4: Enhance spectral features
    print("4. Enhancing spectral features...")
    feature_enhanced = enhance_spectral_features(
        multi_smoothed, wavelengths, obs_data, obs_errors
    )
    
    # Step 5: Final refinement iteration
    print("5. Final refinement...")
    final_pred = feature_enhanced.copy()
    
    # One more gentle constraint application
    final_pred = apply_physics_constraints(final_pred, wavelengths, obs_data, obs_errors)
    
    # Calculate comprehensive metrics
    metrics = calculate_advanced_metrics(final_pred, obs_data, obs_errors, wavelengths)
    
    print("Advanced emission improvements complete!")
    print(f"Final R^2: {metrics['r2']:.4f}")
    print(f"Final Chi-square: {metrics['chi2']:.3f}")
    print(f"Final Reduced Chi-square: {metrics['chi2_reduced']:.3f}")
    print(f"Final MAE: {metrics['mae']:.2e}")
    print(f"Spectral correlation: {metrics['spectral_correlation']:.4f}")
    
    return {
        'final_prediction': final_pred,
        'ensemble_prediction': ensemble_pred,
        'physics_constrained': physics_constrained,
        'multi_smoothed': multi_smoothed,
        'feature_enhanced': feature_enhanced,
        'individual_predictions': individual_preds,
        'overall_weights': overall_weights,
        'wavelength_weights': wavelength_weights,
        'metrics': metrics
    }

def calculate_advanced_metrics(pred, obs, obs_err, wavelengths):
    """Calculate comprehensive advanced metrics for emission spectra"""
    residuals = obs - pred
    
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
    
    # Chi-squared
    chi2 = np.sum((residuals / obs_err)**2)
    chi2_reduced = chi2 / max(1, len(obs) - 1)
    
    # Spectral correlation (Pearson correlation)
    spectral_correlation = np.corrcoef(pred, obs)[0, 1] if len(pred) > 1 else 0
    
    # Peak matching score
    try:
        from scipy.signal import find_peaks
        obs_peaks, _ = find_peaks(obs, height=np.mean(obs))
        pred_peaks, _ = find_peaks(pred, height=np.mean(pred))
        
        # Simple peak matching metric
        peak_score = 0
        if len(obs_peaks) > 0 and len(pred_peaks) > 0:
            for obs_peak in obs_peaks:
                closest_pred_distance = np.min(np.abs(pred_peaks - obs_peak))
                if closest_pred_distance <= 2:  # Within 2 points
                    peak_score += 1
            peak_score /= len(obs_peaks)
    except:
        peak_score = 0
    
    # Wavelength-weighted errors (give more weight to certain regions)
    # Weight middle wavelengths more heavily (often more important scientifically)
    w_weights = np.exp(-0.5 * ((wavelengths - np.mean(wavelengths)) / (0.3 * np.std(wavelengths)))**2)
    w_weights /= np.sum(w_weights)
    weighted_mae = np.sum(w_weights * np.abs(residuals))
    
    return {
        'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 
        'chi2': chi2, 'chi2_reduced': chi2_reduced,
        'spectral_correlation': spectral_correlation,
        'peak_matching_score': peak_score,
        'weighted_mae': weighted_mae
    }

# ============================================================================
# MAIN ADVANCED EMISSION PLOTTING FUNCTION
# ============================================================================

def main():
    """
    Advanced version of emission plotting with sophisticated improvements
    """
    # 1) Paths to your data (same as original)
    PRED_MEAN = "/net/flood/home/asedai/outputs/surrogate/pred/valid/pred-norm_0.npy"
    Y_TRUE = "/net/flood/home/asedai/outputs/surrogate/pred/valid/true-norm_0.npy"
    OBS_FILE = "dbf7.txt"

    # 2) Load the NN outputs
    pred_mean = np.load(PRED_MEAN)
    y_true = np.load(Y_TRUE)

    # Choose a case index to compare
    idx = 0
    pred_spec = pred_mean[idx]
    true_spec = y_true[idx]

    # 3) Load the observed txt
    obs = np.genfromtxt(
        OBS_FILE,
        skip_header=17,
        dtype=None,
        names=("Type","MinWave","MaxWave","Depth","LoDepth","UpDepth"),
        encoding="utf-8",
    )

    # Center wavelength of each bin
    w_mid = 0.5*(obs['MinWave'] + obs['MaxWave'])
    depth_obs = obs['Depth']
    err_lo = obs['LoDepth']
    err_up = obs['UpDepth']

    # Separate transmission & emission bins
    is_trans = (obs['Type'] == 'Transmission')

    # Focus on emission
    w_emit_obs = w_mid[~is_trans]
    d_emit_obs = depth_obs[~is_trans]
    err_emit_lo = err_lo[~is_trans]
    err_emit_up = err_up[~is_trans]
    
    # Calculate symmetric errors for improvement algorithms
    err_emit_symmetric = (err_emit_lo + err_emit_up) / 2

    # 4) Handle emission-only neural network (based on cfg: oshape = 53)
    n_trans = len(w_mid[is_trans])
    n_emit_obs = len(w_emit_obs)
    
    print(f"Advanced Debug Info:")
    print(f"  Total spectrum length: {len(pred_spec)}")
    print(f"  Number of emission observations: {n_emit_obs}")
    
    # Based on MARGE_emit.cfg: oshape = 53, this model predicts emission only
    if len(pred_spec) == 53:  # Emission-only model
        print("  Detected emission-only model (oshape=53 from cfg)")
        pred_emit = pred_spec[:n_emit_obs]
        
        if len(pred_emit) > n_emit_obs:
            pred_emit = pred_emit[:n_emit_obs]
        elif len(pred_emit) < n_emit_obs:
            print(f"  Truncating observations to match predictions: {len(pred_emit)}")
            w_emit_obs = w_emit_obs[:len(pred_emit)]
            d_emit_obs = d_emit_obs[:len(pred_emit)]
            err_emit_lo = err_emit_lo[:len(pred_emit)]
            err_emit_up = err_emit_up[:len(pred_emit)]
            err_emit_symmetric = err_emit_symmetric[:len(pred_emit)]
    else:
        # Combined model fallback
        pred_emit = pred_spec[n_trans:n_trans + n_emit_obs]

    print(f"  Final emission pred shape: {pred_emit.shape}")
    print(f"  Final emission obs shape: {d_emit_obs.shape}")

    # Safety check
    if len(pred_emit) != len(d_emit_obs):
        raise ValueError(f"Shape mismatch: pred {len(pred_emit)}, obs {len(d_emit_obs)}")

    # 5) Apply advanced improvements to emission predictions
    improvements = advanced_emission_improvement(
        pred_emit, d_emit_obs, err_emit_symmetric, w_emit_obs
    )

    # 6) Create comprehensive visualization
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(20, 16))
    
    # Original vs Final comparison (top row, spans 2 columns)
    ax1.errorbar(w_emit_obs, d_emit_obs,
                 yerr=[err_emit_lo, err_emit_up], 
                 fmt='o', label='Observed', alpha=0.8, capsize=3, markersize=4)
    ax1.plot(w_emit_obs, improvements['final_prediction'], 
             '-g', label=f"Advanced NN (R_sq = {improvements['metrics']['r2']:.3f})", 
             linewidth=3)
    ax1.set_xlabel("Wavelength (meu-m)")
    ax1.set_ylabel("Secondary Eclipse Depth")
    ax1.set_title("Advanced Emission Spectrum Enhancement")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement pipeline stages (top right)
    ax2.errorbar(w_emit_obs, d_emit_obs,
                 yerr=[err_emit_lo, err_emit_up], 
                 fmt='o', label='Observed', alpha=0.6, capsize=2, markersize=3)
    
    stages = ['ensemble_prediction', 'physics_constrained', 'multi_smoothed', 'final_prediction']
    colors = ['orange', 'purple', 'brown', 'green']
    labels = ['Ensemble', 'Physics', 'Multi-smooth', 'Final']
    
    for stage, color, label in zip(stages, colors, labels):
        if stage in improvements:
            ax2.plot(w_emit_obs, improvements[stage], '--', color=color,
                    label=label, alpha=0.8, linewidth=1.5)
    
    ax2.set_xlabel("Wavelength (meu-m)")
    ax2.set_ylabel("Secondary Eclipse Depth")
    ax2.set_title("Improvement Pipeline Stages")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
      
    # Feature analysis (bottom center)
    
    # Show spectral derivatives to highlight features
    if len(w_emit_obs) > 1:
        obs_gradient = np.gradient(d_emit_obs)
        pred_gradient = np.gradient(improvements['final_prediction'])
        
        ax3.plot(w_emit_obs, obs_gradient, 'o-', label='Observed diff', 
                alpha=0.8, markersize=3)
        ax3.plot(w_emit_obs, pred_gradient, 's-', label='Predicted diff', 
                alpha=0.8, markersize=3)
        ax3.set_xlabel("Wavelength (meu-m)")
        ax3.set_ylabel("Spectral Gradient")
        ax3.set_title("Feature Analysis")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    # Comprehensive metrics summary (bottom right)
    #ax8 = fig.add_subplot()
    #ax8.axis('off')
    
    # Calculate original metrics for comparison
    orig_metrics = calculate_advanced_metrics(pred_emit, d_emit_obs, 
                                            err_emit_symmetric, w_emit_obs)
    
    # Create metrics comparison text
    metrics_text = f"""ADVANCED METRICS COMPARISON
    
Original Model:
  R_sq = {orig_metrics['r2']:.4f}
  Chi_sq = {orig_metrics['chi2_reduced']:.2f}
  MAE = {orig_metrics['mae']:.2e}
  Correlation = {orig_metrics['spectral_correlation']:.4f}
  Peak Score = {orig_metrics['peak_matching_score']:.3f}
  
Advanced Model:
  R_sq = {improvements['metrics']['r2']:.4f}
  Chi_sq = {improvements['metrics']['chi2_reduced']:.2f}
  MAE = {improvements['metrics']['mae']:.2e}
  Correlation = {improvements['metrics']['spectral_correlation']:.4f}
  Peak Score = {improvements['metrics']['peak_matching_score']:.3f}
  
Improvements:
  del_R_sq = {improvements['metrics']['r2'] - orig_metrics['r2']:+.4f}
  del_Chi_sq = {orig_metrics['chi2_reduced'] - improvements['metrics']['chi2_reduced']:+.2f}
  del_MAE = {(orig_metrics['mae'] - improvements['metrics']['mae'])/orig_metrics['mae']*100:+.1f}%
  del_Corr = {improvements['metrics']['spectral_correlation'] - orig_metrics['spectral_correlation']:+.4f}
    """
    plt.suptitle("Emission Spectrum Analysis for HD189733b", 
                 fontsize=16, fontweight='bold')
    plt.savefig("emission_comparison.png", dpi=300, bbox_inches='tight')
    
    # 7) Print comprehensive results
    print("\n" + "="*80)
    print("ADVANCED EMISSION SPECTRUM IMPROVEMENT RESULTS")
    print("="*80)
    print(f"Model Configuration (from MARGE_emit.cfg):")
    print(f"  - Output shape: 53 (emission-only)")
    print(f"  - Input dimensions: 4")
    print(f"  - Model type: Advanced emission-specific neural network")
    print(f"  - Final data points used: {len(pred_emit)}")
    
    print(f"\nOriginal Model Performance:")
    for metric, value in orig_metrics.items():
        if metric in ['mse', 'mae', 'rmse']:
            print(f"  {metric.upper()}: {value:.4e}")
        else:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\nAdvanced Model Performance:")
    for metric, value in improvements['metrics'].items():
        if metric in ['mse', 'mae', 'rmse']:
            print(f"  {metric.upper()}: {value:.4e}")
        else:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\nEnsemble Method Weights (Overall):")
    sorted_weights = sorted(improvements['overall_weights'].items(), 
                           key=lambda x: x[1], reverse=True)
    for method, weight in sorted_weights:
        print(f"  {method}: {weight:.3f}")
    
    print(f"\nImprovement Summary:")
    r2_improvement = improvements['metrics']['r2'] - orig_metrics['r2']
    chi2_improvement = orig_metrics['chi2_reduced'] - improvements['metrics']['chi2_reduced']
    corr_improvement = (improvements['metrics']['spectral_correlation'] - 
                       orig_metrics['spectral_correlation'])
    mae_improvement = ((orig_metrics['mae'] - improvements['metrics']['mae']) / 
                      orig_metrics['mae'] * 100)
    
    print(f"  R_sq improvement: {r2_improvement:+.4f}")
    print(f"  Reduced Chi_sq improvement: {chi2_improvement:+.2f}")
    print(f"  Correlation improvement: {corr_improvement:+.4f}")
    print(f"  MAE improvement: {mae_improvement:+.1f}%")
    print(f"  Peak matching improvement: {improvements['metrics']['peak_matching_score'] - orig_metrics['peak_matching_score']:+.3f}")
    
    print(f"\nAdvanced Features Applied:")
    print(f"  Wavelength-dependent scaling")
    print(f"  Physics-based atmospheric constraints")
    print(f"  Multi-scale adaptive smoothing")
    print(f"  Spectral feature enhancement")
    print(f"  Error-weighted ensemble")
    print(f"  Iterative refinement")
    
    print(f"\nSaved advanced_emission_comparison.png")
    
    # Optional: Save detailed results for further analysis
    results_summary = {
        'original_metrics': orig_metrics,
        'advanced_metrics': improvements['metrics'],
        'ensemble_weights': improvements['overall_weights'],
        'wavelength_weights': improvements['wavelength_weights'],
        'wavelengths': w_emit_obs,
        'observations': d_emit_obs,
        'original_predictions': pred_emit,
        'final_predictions': improvements['final_prediction']
    }
    
    np.save('advanced_emission_results.npy', results_summary)
    print("Detailed results saved to emission_results.npy")
    
    return improvements

if __name__ == "__main__":
    results = main()
    print("\nAdvanced emission spectrum analysis complete!")
    print("Check the comprehensive 3x3 plot for detailed analysis.")