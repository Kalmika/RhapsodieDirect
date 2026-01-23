# =============================================================================
# 3. CORE FUNCTIONS FOR CORRELATED NOISE
# =============================================================================

"""
    compute_power_spectrum(A, σ, N) -> Matrix{Float64}

Compute the power spectral density P(k) = A * exp(-σ²|k|²)
"""
function compute_power_spectrum(A::Float64, σ::Float64, N::Int)
    freq = fftfreq(N)
    k_grid = freq' .^ 2 .+ freq .^ 2  # Broadcasting
    # println("Power spectrum computed for N=$N, A=$A, σ=$σ")
    return A .* exp.(-σ^2 .* (2π)^2 .* k_grid)
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

function generate_correlated_toeplitz_noise(model::CorrelatedNoise)
    white_noise = randn(model.N*2, model.N*2)
    fourier_noise = fft(white_noise)
    filtered_noise = fourier_noise .* model.sqrt_P_double
    return real.(ifft(filtered_noise))[1:model.N, 1:model.N] # Crop to original size
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
function generate_noise(::DiagonalNoise{T}, signal::AbstractArray{T}, weights::AbstractArray{T}) where {T<:AbstractFloat}
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
function apply_precision_matrix(::DiagonalNoise{T}, residual::AbstractArray{T}, weights::AbstractArray{T}) where {T<:AbstractFloat}
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
    create_noise_model(type::Symbol, params...; T=Float64, kwargs...) -> NoiseModel

Factory function to create noise models based on type.

# Examples
```julia
# Diagonal noise (Float64 by default)
diagonal_model = create_noise_model(:diagonal)

# Diagonal noise with explicit type
diagonal_model = create_noise_model(:diagonal, T=Float32)

# Correlated noise
corr_model = create_noise_model(:correlated, A=2.0, σ=5.0, N=128)
```
"""
function create_noise_model(type::Symbol, args...; T::Type{<:AbstractFloat}=Float64, kwargs...)
    if type == :diagonal
        return DiagonalNoise{T}()
    elseif type == :correlated
        # Extract parameters from kwargs
        A = Base.get(kwargs, :A, one(T))
        σ = Base.get(kwargs, :σ, one(T))
        N = Base.get(kwargs, :N, 128)
        return CorrelatedNoise(T(A), T(σ), N)
    else
        error("Unknown noise model type: $type")
    end
end

# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================

"""
    with_weights(model::DiagonalNoise{T}, weights::AbstractArray{T}) -> DiagonalNoise{T}

Create a new DiagonalNoise instance with the specified weights.
Useful for adding weights to a model that was initially created without them.
"""
function with_weights(_::DiagonalNoise{T}, weights::AbstractArray{T}) where {T<:AbstractFloat}
    return DiagonalNoise(weights)
end

"""
    with_weights(model::DiagonalAndCorrelatedNoise{T}, weights::AbstractArray{T}) -> DiagonalAndCorrelatedNoise{T}

Create a new DiagonalAndCorrelatedNoise instance with updated weights in the diagonal component.
The correlated component remains unchanged.
"""
function with_weights(model::DiagonalAndCorrelatedNoise{T}, weights::AbstractArray{T}) where {T<:AbstractFloat}
    updated_diag = DiagonalNoise(weights)
    return DiagonalAndCorrelatedNoise(updated_diag, model.corr_noise)
end

function theoretical_variance(model::CorrelatedNoise)
    return model.A / (4π * model.σ^2)
end

"""
    get_noise_type(model::NoiseModel) -> Symbol

Get the type of noise model as a symbol.
"""
get_noise_type(::DiagonalNoise{T}) where {T} = :diagonal
get_noise_type(::CorrelatedNoise{T}) where {T} = :correlated

# =============================================================================
# 7. VALIDATION AND TESTING
# =============================================================================

# Validation functions moved to noise_validation.jl (optional plotting)
# Include them only if Plots is available

function validate_noise_model(model::NoiseModel; kwargs...)
    @warn "Use include(\"noise_validation.jl\") to enable plotting validation functions"
    return nothing
end

# =============================================================================
# 8. COVARIANCE APPLICATION - Multiply noise models with arrays
# =============================================================================

"""
    toeplitz_convolve(img::AbstractMatrix, padded_kernel) -> Matrix

Apply Toeplitz convolution using pre-computed FFT of kernel.
Efficient convolution via FFT domain multiplication.
For Toeplitz convolution, padded_kernel must have size (2H, 2W) for image size (H, W).
"""
function toeplitz_convolve(img::AbstractMatrix{T}, padded_kernel::AbstractMatrix{K}) where {T<:AbstractFloat, K}
    H, W = size(img)
    Ph, Pw = size(padded_kernel)
    if Ph != 2*H || Pw != 2*W
        throw(ArgumentError("padded_kernel must have size (2H,2W) for image size (H,W). Got $(Ph)x$(Pw) vs expected $(2*H)x$(2*W)."))
    end

    # Pad image to (2H, 2W)
    P = similar(padded_kernel, Complex{promote_type(T, Float64)})
    P .= 0
    # Place real image in top-left corner
    P[1:H, 1:W] .= complex.(img)

    # FFT, multiply, IFFT
    result_padded = ifft(fft(P) .* padded_kernel)
    # Keep real part and crop
    return real(result_padded[1:H, 1:W])
end

"""
    apply_covariance(diag_noise::DiagonalNoise{T}, input_array, direct_model)

Apply diagonal covariance matrix C = W^(-1) to input array.
For diagonal noise: result = input / weights (element-wise).
"""
function apply_covariance(
    diag_noise::DiagonalNoise{T},
    input_array::AbstractArray{T,3},
    _::DirectModel{T},  # Unused - kept for uniform interface
) where {T<:AbstractFloat}
    # For diagonal noise: C = diag(σ_i²) = W^(-1)
    # Apply element-wise: result = x / W (since C = W^(-1))

    if diag_noise.weights === nothing
        throw(ArgumentError("DiagonalNoise does not contain weights. Cannot apply covariance. Use with_weights() to add weights first."))
    end

    return input_array ./ diag_noise.weights
end

"""
    apply_covariance(noise_model::CorrelatedNoise, input_array, direct_model, [weights])

Apply correlated covariance matrix C = B C_(δs) B^T to input array.
Where C_(δs) = F^(-1) Λ F is the spatial correlation operator.

# Process:
1. Apply B^T (adjoint of direct model) to transform data to object space
2. Apply spatial filter C_(δs) in Fourier domain using power spectrum
3. Apply B (direct model) to transform back to data space
"""
function apply_covariance(
    noise_model::CorrelatedNoise{T},
    input_array::AbstractArray{T,3},
    direct_model::DirectModel{T}
) where {T<:AbstractFloat}

    H, W2, n_frames = size(input_array)
    if W2 % 2 != 0
        throw(ArgumentError("Second dimension must be even (2W). Got $W2"))
    end
    W = div(W2, 2)

    # Output array
    out = zeros(T, H, W2, n_frames)

    # Process each frame independently
    # IMPORTANT: Must use full DirectModel (not individual TR[i]) to handle global operations
    for frame_idx in 1:n_frames
        # Step 1: Isolate current frame with zeros elsewhere (like Python)
        # This allows DirectModel to apply correct per-frame B while preserving global operations
        single_frame_image = zeros(T, H, W2, direct_model.rows[3])
        single_frame_image[:, :, frame_idx] = input_array[:, :, frame_idx]

        # Step 2: Apply B^T (adjoint of full DirectModel) to the isolated frame
        # Transform from data space to object space
        transposed_result = direct_model' * single_frame_image

        # Step 3: Extract intensity component (assuming speckles are non-polarized)
        # transposed_result is a PolarimetricMap
        total_intensity = transposed_result.I

        # Step 4: Apply spatial filter C_(δs) = F^(-1) Λ F in Fourier domain
        filtered_intensity = toeplitz_convolve(total_intensity, noise_model.P_double)

        # Step 5: Reconstruct Stokes parameters matching apply_direct_model format
        # Build input_array format: [filtered_intensity, zeros, zeros]
        # Apply same transformation as apply_direct_model:
        # S = PolarimetricMap("intensities", input_array[1] - input_array[2], input_array[2], input_array[3])
        zero_pol = zeros(T, size(filtered_intensity))
        S_filtered = PolarimetricMap("intensities",
                                     filtered_intensity .- zero_pol,  # input_array[1] - input_array[2]
                                     zero_pol,                          # input_array[2]
                                     zero_pol)                          # input_array[3]

        # Step 6: Apply B (full DirectModel) to transform back to data space
        reprojected = direct_model * S_filtered

        # Step 7: Extract only the frame_idx channel from the result
        out[:, :, frame_idx] = reprojected[:, :, frame_idx]
    end

    return out
end

"""
    apply_covariance(noise_model::DiagonalAndCorrelatedNoise, input_array, direct_model, weights)

Apply combined covariance matrix C = (B C_(δs) B^T + Σ_n) to input array.
Sum of diagonal and correlated components.
"""
function apply_covariance(
    noise_model::DiagonalAndCorrelatedNoise{T},
    input_array::AbstractArray{T,3},
    direct_model::DirectModel{T},
) where {T<:AbstractFloat}

    # Apply diagonal component: Σ_n * x
    diag_result = apply_covariance( noise_model.diag_noise, input_array, direct_model )

    # Apply correlated component: B C_(δs) B^T * x
    corr_result = apply_covariance( noise_model.corr_noise, input_array, direct_model )

    # Sum both components
    return diag_result .+ corr_result
end

"""
    apply_covariance(x::AbstractArray{T,3}, dataset::Dataset{T}) -> AbstractArray{T,3}

Apply covariance matrix to input array using noise model from dataset.
Dispatches to appropriate method based on noise model type.

# Arguments
- `x`: Input array (H × W × frames)
- `dataset`: Dataset containing noise_model and direct_model
"""
function apply_covariance(
    x::AbstractArray{T,3},
    dataset::Dataset{T}
) where {T<:AbstractFloat}
    # Dispatch based on noise model type
    return apply_covariance(dataset.noise_model, x, dataset.direct_model)
end