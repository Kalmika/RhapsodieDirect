# =============================================================================
# 3. CORE FUNCTIONS FOR CORRELATED NOISE
# =============================================================================

"""
    compute_power_spectrum(A, σ², N) -> Matrix{Float64}

Compute the power spectral density P(k) = A * exp(-σ²|k|²)
"""
function compute_power_spectrum(A::Float64, σ²::Float64, N::Int)
    freq = fftfreq(N)
    k_grid = freq' .^ 2 .+ freq .^ 2  # Broadcasting
    println("Power spectrum computed for N=$N, A=$A, σ²=$σ²")
    return A .* exp.(-σ² .* (2π)^2 .* k_grid)
end

"""
    generate_correlated_noise(model::CorrelatedNoise) -> Matrix{Float64}

Generate one realization of correlated noise using FFT method.
"""
function generate_correlated_noise(model::CorrelatedNoise)
    white_noise = randn(model.N, model.N)
    fourier_noise = fft(white_noise)
    filtered_noise = fourier_noise .* model.sqrt_P
    return real.(ifft(filtered_noise))
end

# =============================================================================
# 4. UNIFIED INTERFACE - Multiple Dispatch
# =============================================================================

"""
    generate_noise(model::NoiseModel, ...) -> Array

Generate noise according to the specified model.
Multiple dispatch handles different noise types automatically.
"""

# Diagonal noise generation (existing approach)
function generate_noise(model::DiagonalNoise, signal::AbstractArray, weights::AbstractArray)
    noise = similar(signal)
    
    @inbounds for i in eachindex(noise, weights)
        w = weights[i]
        if isfinite(w) && w > 0
            noise[i] = randn() / sqrt(w)
        else
            noise[i] = 0.0
        end
    end
    
    return noise
end

"""
    apply_precision_matrix(model::NoiseModel, residual, ...) -> Array

Apply precision matrix W = C^(-1) to residual.
"""

# Diagonal precision (existing approach)
function apply_precision_matrix(model::DiagonalNoise, residual::AbstractArray, weights::AbstractArray)
    return weights .* residual
end

# Correlated precision (new approach - using FFT for efficiency)
function apply_precision_matrix(model::CorrelatedNoise, residual::AbstractArray)
    if ndims(residual) == 2
        return apply_precision_2d(model, residual)
    elseif ndims(residual) == 3  # For polarimetric data (H, W, frames)
        H, W, F = size(residual)
        
        # W must be even to split into left/right analyzers
        W % 2 == 0 || error("Width dimension must be even for analyzer splitting: got W=$W")
        
        result = similar(residual)
        W_half = W ÷ 2
        
        # Apply precision for each frame and each analyzer (left/right)
        for f in 1:F
            # Left analyzer: columns 1:W_half
            result[:, 1:W_half, f] = apply_precision_2d(model, residual[:, 1:W_half, f])
            # Right analyzer: columns (W_half+1):W  
            result[:, (W_half+1):W, f] = apply_precision_2d(model, residual[:, (W_half+1):W, f])
        end
        
        return result
    else
        error("Unsupported residual dimensions: $(size(residual)). Expected 2D or 3D (H, W, F)")
    end
end

"""
    apply_precision_2d(model::CorrelatedNoise, residual_2d) -> Matrix

Apply 2D precision matrix via FFT: W * residual = F^(-1) * diag(1/P(k)) * F * residual
"""
function apply_precision_2d(model::CorrelatedNoise, residual_2d::AbstractMatrix)
    # Fourier transform of residual
    fourier_residual = fft(residual_2d)
    
    # Apply precision in Fourier domain: multiply by 1/P(k)
    # Add small regularization to avoid division by zero
    epsilon = 1e-12
    precision_fourier = fourier_residual ./ (model.P .+ epsilon)
    
    # Transform back to spatial domain
    return real.(ifft(precision_fourier))
end

# =============================================================================
# 5. FACTORY FUNCTIONS FOR EASY CREATION
# =============================================================================

"""
    create_noise_model(type::Symbol, params...) -> NoiseModel

Factory function to create noise models based on type.

# Examples
```julia
# Diagonal noise
diagonal_model = create_noise_model(:diagonal)

# Correlated noise
corr_model = create_noise_model(:correlated, A=2.0, σ²=5.0, N=128)
```
"""
function create_noise_model(type::Symbol, args...; kwargs...)
    if type == :diagonal
        return DiagonalNoise()
    elseif type == :correlated
        # Extract parameters from kwargs
        A = Base.get(kwargs, :A, 1.0)
        σ² = Base.get(kwargs, :σ², 1.0) 
        N = Base.get(kwargs, :N, 128)
        return CorrelatedNoise(A, σ², N)
    else
        error("Unknown noise model type: $type")
    end
end

# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================
function theoretical_variance(model::CorrelatedNoise)
    return model.A / (4π * model.σ²)
end

"""
    get_noise_type(model::NoiseModel) -> Symbol

Get the type of noise model as a symbol.
"""
get_noise_type(::DiagonalNoise) = :diagonal
get_noise_type(::CorrelatedNoise) = :correlated

# =============================================================================
# 7. VALIDATION AND TESTING
# =============================================================================

# Validation functions moved to noise_validation.jl (optional plotting)
# Include them only if Plots is available

function validate_noise_model(model::NoiseModel; kwargs...)
    @warn "Use include(\"noise_validation.jl\") to enable plotting validation functions"
    return nothing
end