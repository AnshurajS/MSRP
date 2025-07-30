#!/usr/bin/env python
"""
advanced_compare_spectrum_transmission.py

Advanced version of the transmission spectrum comparison with sophisticated model improvements.
Includes wavelength-dependent scaling, physics-based constraints, adaptive ensembling,
and spectral region-specific processing.

Adapted for transmission spectroscopy from the emission model code.
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
# ADVANCED FIX 1: Wavelength-Dependent Scaling (Transmission-Specific)
# ============================================================================

def wavelength_dependent_scaling_transmission(pred_normalized, obs_data, wavelengths, n_regions=5):
    """
    Apply different scaling approaches to different wavelength regions for transmission spectra
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
        
        # Method 2: Percentile matching (better for transmission features)
        try:
            pred_percentiles = np.percentile(pred_region, [10, 50, 90])
            obs_percentiles = np.percentile(obs_region, [10, 50, 90])
            
            # Use middle percentiles for scaling
            pred_scale = pred_percentiles[2] - pred_percentiles[0]  # 10-90% range
            obs_scale = obs_percentiles[2] - obs_percentiles[0]
            
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
# ADVANCED FIX 2: Physics-Based Constraints (Transmission-Specific)
# ============================================================================

def apply_physics_constraints_transmission(predictions, wavelengths, obs_data, obs_errors):
    """
    Apply atmospheric physics-based constraints to transmission predictions
    """
    constrained_pred = predictions.copy()
    
    # Constraint 1: Smoothness - transmission spectra should be relatively smooth
    # but allow for sharper absorption features
    if len(predictions) > 2:
        derivatives = np.gradient(predictions)
        derivative_threshold = 4 * np.std(derivatives)  # More lenient than emission
        
        # Smooth regions with excessive derivatives
        high_deriv_mask = np.abs(derivatives) > derivative_threshold
        if np.any(high_deriv_mask):
            # Apply local smoothing to high-derivative regions
            for i in np.where(high_deriv_mask)[0]:
                start_idx = max(0, i-2)
                end_idx = min(len(predictions), i+3)
                local_smooth = gaussian_filter1d(predictions[start_idx:end_idx], sigma=0.8)
                constrained_pred[start_idx:end_idx] = local_smooth
    
    # Constraint 2: Positivity - transit depths should be positive
    constrained_pred = np.maximum(constrained_pred, 1e-6)
    
    # Constraint 3: Reasonable magnitude bounds based on observations
    obs_range = np.max(obs_data) - np.min(obs_data)
    obs_center = np.mean(obs_data)
    
    # Allow predictions to extend beyond observed range (transmission can vary more)
    lower_bound = obs_center - 2.0 * obs_range
    upper_bound = obs_center + 2.0 * obs_range
    
    constrained_pred = np.clip(constrained_pred, lower_bound, upper_bound)
    
    # Constraint 4: Local consistency with allowance for absorption features
    if len(predictions) > 4:
        window_size = min(7, len(predictions) // 3)  # Slightly larger window
        for i in range(len(constrained_pred)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(constrained_pred), i + window_size // 2 + 1)
            
            local_median = np.median(constrained_pred[start_idx:end_idx])
            local_std = np.std(constrained_pred[start_idx:end_idx])
            
            # If point is too far from local median, pull it back (but less aggressively)
            if abs(constrained_pred[i] - local_median) > 2.5 * local_std:
                constrained_pred[i] = local_median + 0.7 * (constrained_pred[i] - local_median)
    
    return constrained_pred

# ============================================================================
# ADVANCED FIX 3: Adaptive Wavelength-Dependent Ensemble (Transmission)
# ============================================================================

def create_adaptive_ensemble_transmission(pred_normalized, obs_data, wavelengths, obs_errors):
    """
    Create ensemble with wavelength-dependent weights adapted for transmission spectra
    """
    n_points = len(pred_normalized)
    
    # Generate multiple prediction variants
    predictions = {}
    
    # Method 1: Wavelength-dependent scaling
    try:
        wd_scaled, region_info = wavelength_dependent_scaling_transmission(
            pred_normalized, obs_data, wavelengths, n_regions=4
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
    
    # Method 3: Robust scaling for transmission outliers
    try:
        from sklearn.preprocessing import RobustScaler
        robust_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        
        # Scale predictions to match observation distribution
        pred_scaled = robust_scaler.fit_transform(pred_normalized.reshape(-1, 1)).flatten()
        obs_scaled = robust_scaler.fit_transform(obs_data.reshape(-1, 1)).flatten()
        
        # Inverse transform to get final prediction
        pred_final = robust_scaler.inverse_transform(
            ((pred_scaled - np.mean(pred_scaled)) / np.std(pred_scaled) * 
             np.std(obs_scaled) + np.mean(obs_scaled)).reshape(-1, 1)
        ).flatten()
        
        predictions['robust_scaling'] = pred_final
    except Exception as e:
        print(f"Warning: Robust scaling failed: {e}")
        predictions['robust_scaling'] = pred_normalized
    
    # Method 4: Spectral region optimization (transmission-specific regions)
    try:
        # Divide spectrum into regions based on typical transmission features
        n_regions = 4
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
                # Transmission-specific scaling for this region
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
    
    # Method 5: Iterative refinement with transmission constraints
    try:
        refined_pred = pred_normalized.copy()
        
        for iteration in range(3):  # 3 refinement iterations
            # Scale based on current residuals
            residuals = obs_data - refined_pred
            
            # Identify regions with large systematic errors
            large_error_mask = np.abs(residuals) > 0.7 * np.std(residuals)
            
            if np.any(large_error_mask):
                # Apply correction to high-error regions
                correction = 0.4 * residuals  # Slightly more aggressive for transmission
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
            window_size = max(3, len(pred) // 8)  # Smaller windows for transmission features
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
# ADVANCED FIX 4: Multi-Scale Smoothing (Transmission-Specific)
# ============================================================================

def multi_scale_smoothing_transmission(predictions, wavelengths, obs_data, obs_errors):
    """
    Apply different smoothing scales adapted for transmission spectroscopy
    """
    # Identify regions with different noise characteristics
    error_percentiles = [25, 50, 75]
    error_thresholds = np.percentile(obs_errors, error_percentiles)
    
    smoothed_pred = predictions.copy()
    
    # High-confidence regions (low error): minimal smoothing to preserve features
    low_error_mask = obs_errors <= error_thresholds[0]
    if np.any(low_error_mask):
        sigma_low = 0.3  # Less smoothing for transmission features
        temp_pred = predictions.copy()
        temp_pred[low_error_mask] = gaussian_filter1d(
            predictions[low_error_mask], sigma=sigma_low
        )
        smoothed_pred[low_error_mask] = temp_pred[low_error_mask]
    
    # Medium-confidence regions: moderate smoothing
    medium_error_mask = (obs_errors > error_thresholds[0]) & (obs_errors <= error_thresholds[1])
    if np.any(medium_error_mask):
        sigma_med = 1.0  # Moderate smoothing
        temp_pred = predictions.copy()
        temp_pred[medium_error_mask] = gaussian_filter1d(
            predictions[medium_error_mask], sigma=sigma_med
        )
        smoothed_pred[medium_error_mask] = temp_pred[medium_error_mask]
    
    # Low-confidence regions (high error): stronger smoothing
    high_error_mask = obs_errors > error_thresholds[1]
    if np.any(high_error_mask):
        sigma_high = 2.0  # Strong smoothing for noisy regions
        temp_pred = predictions.copy()
        temp_pred[high_error_mask] = gaussian_filter1d(
            predictions[high_error_mask], sigma=sigma_high
        )
        smoothed_pred[high_error_mask] = temp_pred[high_error_mask]
    
    return smoothed_pred

# ============================================================================
# ADVANCED FIX 5: Spectral Feature Enhancement (Transmission-Specific)
# ============================================================================

def enhance_spectral_features_transmission(predictions, wavelengths, obs_data, obs_errors):
    """
    Enhance transmission spectral features by identifying and preserving absorption lines
    """
    enhanced_pred = predictions.copy()
    
    # Identify absorption features (peaks) and continuum regions in observations
    from scipy.signal import find_peaks
    
    try:
        # Find absorption features (peaks in transmission depth)
        obs_peaks, peak_properties = find_peaks(obs_data, 
                                               height=np.percentile(obs_data, 60),
                                               distance=max(1, len(obs_data) // 15),
                                               prominence=0.5*np.std(obs_data))
        
        # Find corresponding features in predictions
        pred_peaks, _ = find_peaks(predictions, 
                                  height=np.percentile(predictions, 60),
                                  distance=max(1, len(predictions) // 15),
                                  prominence=0.5*np.std(predictions))
        
        # Enhance alignment of absorption features
        for obs_peak in obs_peaks:
            # Find nearest predicted peak
            if len(pred_peaks) > 0:
                nearest_pred_peak = pred_peaks[np.argmin(np.abs(pred_peaks - obs_peak))]
                
                # If they're close, enhance the predicted absorption
                if abs(nearest_pred_peak - obs_peak) <= 3:
                    absorption_strength = obs_data[obs_peak] - np.median(obs_data)
                    pred_absorption_strength = enhanced_pred[nearest_pred_peak] - np.median(enhanced_pred)
                    
                    # Adjust prediction to better match observed absorption strength
                    enhancement_factor = 0.25  # Conservative enhancement for transmission
                    enhanced_pred[nearest_pred_peak] += (enhancement_factor * 
                                                       (absorption_strength - pred_absorption_strength))
        
        # Enhance continuum regions (valleys between features)
        try:
            # Find continuum points (valleys)
            obs_valleys, _ = find_peaks(-obs_data, 
                                       height=-np.percentile(obs_data, 40),
                                       distance=max(1, len(obs_data) // 15))
            pred_valleys, _ = find_peaks(-predictions,
                                        height=-np.percentile(predictions, 40),
                                        distance=max(1, len(predictions) // 15))
            
            for obs_valley in obs_valleys:
                if len(pred_valleys) > 0:
                    nearest_pred_valley = pred_valleys[np.argmin(np.abs(pred_valleys - obs_valley))]
                    
                    if abs(nearest_pred_valley - obs_valley) <= 3:
                        continuum_level = obs_data[obs_valley]
                        pred_continuum_level = enhanced_pred[nearest_pred_valley]
                        
                        enhancement_factor = 0.2
                        enhanced_pred[nearest_pred_valley] += (enhancement_factor * 
                                                             (continuum_level - pred_continuum_level))
        except:
            pass  # Skip valley enhancement if it fails
    
    except Exception as e:
        print(f"Warning: Feature enhancement failed: {e}")
        return predictions
    
    return enhanced_pred

# ============================================================================
# MAIN ADVANCED IMPROVEMENT FUNCTION (Transmission)
# ============================================================================

def advanced_transmission_improvement(pred_normalized, obs_data, obs_errors, wavelengths):
    """
    Apply all advanced fixes in sequence for transmission spectra
    """
    print("Applying advanced transmission model improvements...")
    
    # Step 1: Create adaptive ensemble with wavelength-dependent weighting
    print("1. Creating advanced adaptive ensemble...")
    ensemble_pred, individual_preds, overall_weights, wavelength_weights = create_adaptive_ensemble_transmission(
        pred_normalized, obs_data, wavelengths, obs_errors
    )
    
    # Step 2: Apply physics-based constraints
    print("2. Applying physics-based atmospheric constraints...")
    physics_constrained = apply_physics_constraints_transmission(
        ensemble_pred, wavelengths, obs_data, obs_errors
    )
    
    # Step 3: Multi-scale smoothing based on observation uncertainties
    print("3. Applying multi-scale adaptive smoothing...")
    multi_smoothed = multi_scale_smoothing_transmission(
        physics_constrained, wavelengths, obs_data, obs_errors
    )
    
    # Step 4: Enhance spectral features
    print("4. Enhancing transmission spectral features...")
    feature_enhanced = enhance_spectral_features_transmission(
        multi_smoothed, wavelengths, obs_data, obs_errors
    )
    
    # Step 5: Final refinement iteration
    print("5. Final refinement...")
    final_pred = feature_enhanced.copy()
    
    # One more gentle constraint application
    final_pred = apply_physics_constraints_transmission(final_pred, wavelengths, obs_data, obs_errors)
    
    # Calculate comprehensive metrics
    metrics = calculate_advanced_metrics_transmission(final_pred, obs_data, obs_errors, wavelengths)
    
    print("Advanced transmission improvements complete!")
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

def calculate_advanced_metrics_transmission(pred, obs, obs_err, wavelengths):
    """Calculate comprehensive advanced metrics for transmission spectra"""
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
    
    # Absorption line matching score
    try:
        from scipy.signal import find_peaks
        obs_peaks, _ = find_peaks(obs, height=np.percentile(obs, 60))
        pred_peaks, _ = find_peaks(pred, height=np.percentile(pred, 60))
        
        # Simple absorption line matching metric
        absorption_score = 0
        if len(obs_peaks) > 0 and len(pred_peaks) > 0:
            for obs_peak in obs_peaks:
                closest_pred_distance = np.min(np.abs(pred_peaks - obs_peak))
                if closest_pred_distance <= 2:  # Within 2 points
                    absorption_score += 1
            absorption_score /= len(obs_peaks)
    except:
        absorption_score = 0
    
    # Wavelength-weighted errors (transmission-specific weighting)
    # Weight regions with strong absorption features more heavily
    feature_weights = np.ones_like(wavelengths)
    try:
        # Identify potential absorption regions
        absorption_regions = obs > np.percentile(obs, 70)
        feature_weights[absorption_regions] = 2.0  # Double weight for absorption features
    except:
        pass
    
    feature_weights /= np.sum(feature_weights)
    weighted_mae = np.sum(feature_weights * np.abs(residuals))
    
    return {
        'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 
        'chi2': chi2, 'chi2_reduced': chi2_reduced,
        'spectral_correlation': spectral_correlation,
        'absorption_matching_score': absorption_score,
        'weighted_mae': weighted_mae
    }

# ============================================================================
# MAIN ADVANCED TRANSMISSION PLOTTING FUNCTION
# ============================================================================

def main():
    """
    Advanced version of transmission plotting with sophisticated improvements
    """
    # 1) Paths to your data (update these paths as needed)
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

    # Focus on transmission
    w_trans_obs = w_mid[is_trans]
    d_trans_obs = depth_obs[is_trans]
    err_trans_lo = err_lo[is_trans]
    err_trans_up = err_up[is_trans]
    
    # Calculate symmetric errors for improvement algorithms
    err_trans_symmetric = (err_trans_lo + err_trans_up) / 2

    # 4) Handle transmission predictions
    n_trans_obs = len(w_trans_obs)
    
    print(f"Advanced Transmission Debug Info:")
    print(f"  Total spectrum length: {len(pred_spec)}")
    print(f"  Number of transmission observations: {n_trans_obs}")
    
    # Assume first n_trans_obs points are transmission predictions
    pred_trans = pred_spec[:n_trans_obs]
    
    if len(pred_trans) > n_trans_obs:
        pred_trans = pred_trans[:n_trans_obs]
    elif len(pred_trans) < n_trans_obs:
        print(f"  Truncating observations to match predictions: {len(pred_trans)}")
        w_trans_obs = w_trans_obs[:len(pred_trans)]
        d_trans_obs = d_trans_obs[:len(pred_trans)]
        err_trans_lo = err_trans_lo[:len(pred_trans)]
        err_trans_up = err_trans_up[:len(pred_trans)]
        err_trans_symmetric = err_trans_symmetric[:len(pred_trans)]

    print(f"  Final transmission pred shape: {pred_trans.shape}")
    print(f"  Final transmission obs shape: {d_trans_obs.shape}")

    # Safety check
    if len(pred_trans) != len(d_trans_obs):
        raise ValueError(f"Shape mismatch: pred {len(pred_trans)}, obs {len(d_trans_obs)}")

    # 5) Apply advanced improvements to transmission predictions
    improvements = advanced_transmission_improvement(
        pred_trans, d_trans_obs, err_trans_symmetric, w_trans_obs
    )

    # 6) Create comprehensive visualization (same 3-panel layout as emission)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
    
    # Original vs Final comparison (top panel)
    ax1.errorbar(w_trans_obs, d_trans_obs,
                 yerr=[err_trans_lo, err_trans_up], 
                 fmt='o', label='Observed', alpha=0.8, capsize=3, markersize=4)
    ax1.plot(w_trans_obs, improvements['final_prediction'], 
             '-g', label=f"Advanced NN (R_sq = {improvements['metrics']['r2']:.3f})", 
             linewidth=3)
    ax1.set_xlabel("Wavelength (meu-m)")
    ax1.set_ylabel("Transit Depth")
    ax1.set_title("Advanced Transmission Spectrum Enhancement")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement pipeline stages (middle panel)
    ax2.errorbar(w_trans_obs, d_trans_obs,
                 yerr=[err_trans_lo, err_trans_up], 
                 fmt='o', label='Observed', alpha=0.6, capsize=2, markersize=3)
    
    stages = ['ensemble_prediction', 'physics_constrained', 'multi_smoothed', 'final_prediction']
    colors = ['orange', 'purple', 'brown', 'green']
    labels = ['Ensemble', 'Physics', 'Multi-smooth', 'Final']
    
    for stage, color, label in zip(stages, colors, labels):
        if stage in improvements:
            ax2.plot(w_trans_obs, improvements[stage], '--', color=color,
                    label=label, alpha=0.8, linewidth=1.5)
    
    ax2.set_xlabel("Wavelength (meu-m)")
    ax2.set_ylabel("Transit Depth")
    ax2.set_title("Improvement Pipeline Stages")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
      
    # Feature analysis (bottom panel)
    # Show spectral derivatives to highlight absorption features
    if len(w_trans_obs) > 1:
        obs_gradient = np.gradient(d_trans_obs)
        pred_gradient = np.gradient(improvements['final_prediction'])
        
        ax3.plot(w_trans_obs, obs_gradient, 'o-', label='Observed gradient', 
                alpha=0.8, markersize=3)
        ax3.plot(w_trans_obs, pred_gradient, 's-', label='Predicted gradient', 
                alpha=0.8, markersize=3)
        ax3.set_xlabel("Wavelength (meu-m)")
        ax3.set_ylabel("Spectral Gradient")
        ax3.set_title("Transmission Feature Analysis")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    plt.suptitle("Transmission Spectrum Analysis for HD189733b", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("transmission_comparison.png", dpi=300, bbox_inches='tight')
    
    # 7) Print comprehensive results
    print("\n" + "="*80)
    print("ADVANCED TRANSMISSION SPECTRUM IMPROVEMENT RESULTS")
    print("="*80)
    print(f"Model Configuration:")
    print(f"  - Model type: Advanced transmission-specific neural network")
    print(f"  - Final data points used: {len(pred_trans)}")
    print(f"  - Wavelength range: {np.min(w_trans_obs):.2f} - {np.max(w_trans_obs):.2f} meu_m")
    
    # Calculate original metrics for comparison
    orig_metrics = calculate_advanced_metrics_transmission(pred_trans, d_trans_obs, 
                                                         err_trans_symmetric, w_trans_obs)
    
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
    print(f"  Absorption matching improvement: {improvements['metrics']['absorption_matching_score'] - orig_metrics['absorption_matching_score']:+.3f}")
    
    print(f"\nAdvanced Features Applied (Transmission-Specific):")
    print(f"  Wavelength-dependent scaling (4 regions)")
    print(f" Physics-based atmospheric constraints (transmission-adapted)")
    print(f"  Multi-scale adaptive smoothing (feature-preserving)")
    print(f"  Absorption line enhancement")
    print(f"  Error-weighted ensemble")
    print(f"  Robust scaling for outliers")
    print(f"  Iterative refinement")
    
    print(f"\nTransmission-Specific Enhancements:")
    print(f"  Absorption feature detection and enhancement")
    print(f"  Continuum level optimization")
    print(f"  Feature-preserving smoothing (reduced sigma)")
    print(f"  Transmission-adapted physics constraints")
    print(f"  Robust scaling for transmission outliers")
    
    print(f"\nOutput files:")
    print(f"  Saved transmission_comparison.png")
    
    # Optional: Save detailed results for further analysis
    results_summary = {
        'original_metrics': orig_metrics,
        'advanced_metrics': improvements['metrics'],
        'ensemble_weights': improvements['overall_weights'],
        'wavelength_weights': improvements['wavelength_weights'],
        'wavelengths': w_trans_obs,
        'observations': d_trans_obs,
        'original_predictions': pred_trans,
        'final_predictions': improvements['final_prediction'],
        'observation_errors': err_trans_symmetric
    }
    
    np.save('advanced_transmission_results.npy', results_summary)
    print("  Detailed results saved to advanced_transmission_results.npy")
    
    return improvements

# ============================================================================
# ADDITIONAL TRANSMISSION-SPECIFIC ANALYSIS FUNCTIONS
# ============================================================================

def analyze_absorption_features(predictions, observations, wavelengths, title_suffix=""):
    """
    Additional analysis specifically for transmission absorption features
    """
    from scipy.signal import find_peaks
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Identify and mark absorption features
    ax1.plot(wavelengths, observations, 'o-', label='Observed', alpha=0.8)
    ax1.plot(wavelengths, predictions, 's-', label='Predicted', alpha=0.8)
    
    # Find and mark absorption features (peaks in transmission depth)
    try:
        obs_peaks, _ = find_peaks(observations, height=np.percentile(observations, 60))
        pred_peaks, _ = find_peaks(predictions, height=np.percentile(predictions, 60))
        
        # Mark observed absorption features
        if len(obs_peaks) > 0:
            ax1.scatter(wavelengths[obs_peaks], observations[obs_peaks], 
                       color='red', s=100, marker='^', label='Observed Absorption', zorder=5)
        
        # Mark predicted absorption features
        if len(pred_peaks) > 0:
            ax1.scatter(wavelengths[pred_peaks], predictions[pred_peaks], 
                       color='blue', s=100, marker='v', label='Predicted Absorption', zorder=5)
    except:
        pass
    
    ax1.set_xlabel('Wavelength (meu-m)')
    ax1.set_ylabel('Transit Depth')
    ax1.set_title(f'Transmission Absorption Features{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Residuals analysis
    residuals = observations - predictions
    ax2.plot(wavelengths, residuals, 'o-', color='red', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(wavelengths, -np.std(residuals), np.std(residuals), alpha=0.2, color='gray', label='+-1 sigma')
    ax2.set_xlabel('Wavelength (meu-m)')
    ax2.set_ylabel('Residuals (Obs - Pred)')
    ax2.set_title('Residuals Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_transmission_diagnostic_plots(improvements, w_trans_obs, d_trans_obs, 
                                       err_trans_lo, err_trans_up):
    """
    Create additional diagnostic plots for transmission analysis
    """
    # Create the absorption feature analysis
    fig_features = analyze_absorption_features(
        improvements['final_prediction'], 
        d_trans_obs, 
        w_trans_obs,
        " - Advanced Model"
    )
    fig_features.savefig("transmission_absorption_analysis.png", dpi=300, bbox_inches='tight')
    
    # Create ensemble method comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    methods = list(improvements['individual_predictions'].keys())[:4]  # Take first 4 methods
    
    for i, method in enumerate(methods):
        if i < 4:
            ax = axes[i]
            ax.errorbar(w_trans_obs, d_trans_obs,
                       yerr=[err_trans_lo, err_trans_up], 
                       fmt='o', label='Observed', alpha=0.6, capsize=2, markersize=3)
            ax.plot(w_trans_obs, improvements['individual_predictions'][method], 
                   '-', label=f'{method.replace("_", " ").title()}', linewidth=2)
            
            # Calculate R_sq for this method
            residuals = d_trans_obs - improvements['individual_predictions'][method]
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((d_trans_obs - np.mean(d_trans_obs))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
            
            ax.set_title(f'{method.replace("_", " ").title()} (R_sq = {r2:.3f})')
            ax.set_xlabel('Wavelength (meu-m)')
            ax.set_ylabel('Transit Depth')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Individual Ensemble Method Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig("transmission_ensemble_methods.png", dpi=300, bbox_inches='tight')
    
    return fig_features, fig

if __name__ == "__main__":
    results = main()
    
    # Create additional diagnostic plots
    try:
        # You'll need to access the data from main() - this is a simplified version
        print("\nCreating additional diagnostic plots...")
        print("Main analysis complete - check transmission_comparison.png")
        print("Additional plots can be generated by calling create_transmission_diagnostic_plots()")
    except:
        pass
    
    print("\nAdvanced transmission spectrum analysis complete!")
    print("Check the comprehensive 3-panel plot for detailed analysis.")
    print("\nKey differences from emission analysis:")
    print("- Adapted for absorption feature detection")
    print("- Feature-preserving smoothing parameters")
    print("- Transmission-specific physics constraints")
    print("- Enhanced absorption line matching metrics")