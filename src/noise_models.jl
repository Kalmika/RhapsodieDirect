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

# =============================================================================
# PRE-COMPUTED FFT OPERATOR FOR ZERO-ALLOCATION TOEPLITZ CONVOLUTION
# =============================================================================

"""
    PrecomputedToeplitzFFT{T}

Pre-computed FFT plans and buffers for efficient Toeplitz convolution.
Avoids allocations by reusing buffers and pre-computed FFT plans.

# Fields
- `H, W`: Original image dimensions
- `plan_fft`: Pre-computed forward FFT plan
- `plan_ifft`: Pre-computed inverse FFT plan
- `kernel_fft`: Pre-computed FFT of kernel (stored in Fourier domain)
- `buffer_padded`: Pre-allocated buffer for padded image (2H × 2W)
- `buffer_fft`: Pre-allocated buffer for FFT result (2H × 2W)

# Example
```julia
# Create once (expensive)
fft_op = PrecomputedToeplitzFFT(128, 128, kernel_2N)

# Use many times (cheap, zero-allocation)
result = similar(img)
toeplitz_convolve!(result, fft_op, img)
```
"""
struct PrecomputedToeplitzFFT{T<:AbstractFloat, P<:FFTW.cFFTWPlan, IP<:AbstractFFTs.ScaledPlan}
    H::Int
    W::Int
    plan_fft::P                       # In-place FFT plan
    plan_ifft::IP                     # In-place IFFT plan
    kernel_fft::Matrix{ComplexF64}    # Pre-computed kernel FFT
    buffer_padded::Matrix{ComplexF64} # (2H, 2W) padded buffer
    buffer_fft::Matrix{ComplexF64}    # (2H, 2W) FFT buffer
end

"""
    PrecomputedToeplitzFFT(H, W, kernel)

Create a PrecomputedToeplitzFFT operator for images of size (H, W).
The kernel should have size (2H, 2W) for proper Toeplitz embedding.

# Arguments
- `H, W`: Image dimensions
- `kernel`: Kernel matrix of size (2H, 2W)
"""
function PrecomputedToeplitzFFT(H::Int, W::Int, kernel::AbstractMatrix{T}) where {T}
    Ph, Pw = size(kernel)
    if Ph != 2*H || Pw != 2*W
        throw(ArgumentError("kernel must have size (2H,2W)=$(2*H)×$(2*W). Got $(Ph)×$(Pw)."))
    end

    # Pre-allocate buffers
    buffer_padded = zeros(ComplexF64, 2*H, 2*W)
    buffer_fft = zeros(ComplexF64, 2*H, 2*W)

    # Create in-place FFT plans (MEASURE flag for optimal performance)
    plan_fft = plan_fft!(buffer_padded; flags=FFTW.MEASURE)
    plan_ifft = plan_ifft!(buffer_fft; flags=FFTW.MEASURE)

    # Store kernel directly (already in Fourier domain for P_double/Delta_kernel)
    # DO NOT FFT - the kernel is already a power spectrum in frequency domain
    kernel_fft = ComplexF64.(kernel)

    return PrecomputedToeplitzFFT{Float64, typeof(plan_fft), typeof(plan_ifft)}(H, W, plan_fft, plan_ifft, kernel_fft, buffer_padded, buffer_fft)
end

"""
    toeplitz_convolve!(result, op::PrecomputedToeplitzFFT, img)

In-place Toeplitz convolution using pre-computed FFT plans.
Zero allocations after initial setup.

# Arguments
- `result`: Output matrix (H × W), modified in-place
- `op`: PrecomputedToeplitzFFT operator
- `img`: Input image (H × W)
"""
function toeplitz_convolve!(
    result::AbstractMatrix{T},
    op::PrecomputedToeplitzFFT,
    img::AbstractMatrix{T}
) where {T<:AbstractFloat}
    H, W = op.H, op.W

    # Clear and copy to padded buffer
    op.buffer_padded .= 0
    @views op.buffer_padded[1:H, 1:W] .= img

    # In-place FFT
    op.plan_fft * op.buffer_padded  # Modifies buffer_padded in-place

    # Multiply with pre-computed kernel FFT
    op.buffer_fft .= op.buffer_padded .* op.kernel_fft

    # In-place IFFT
    op.plan_ifft * op.buffer_fft  # Modifies buffer_fft in-place

    # Extract real part of result
    @views result .= real.(op.buffer_fft[1:H, 1:W])

    return result
end


function apply_special_inverse_covariance( input_array::AbstractArray{T,3}, kernel, σ2::T ) where {T<:AbstractFloat}
    H, W2, C = size(input_array)
    if W2 % 2 != 0
        throw(ArgumentError("second dimension must be even (2W). Got $W2"))
    end
    W = div(W2, 2)

    # output: same layout (H, 2W, C)
    out = Array{Float64}(undef, H, W2, C)

    for c in 1:C
        # views avoid copies; we materialize only when needed
        x_view = @view input_array[:, 1:W, c]        # (H, W)
        y_view = @view input_array[:, W+1:2W, c]     # (H, W)

        # convert to Float64 matrices for FFT stability
        x = Array{Float64}(x_view)
        y = Array{Float64}(y_view)

        sum_xy = x .+ y

        # F† Δ F (x + y)
        conv = toeplitz_convolve(sum_xy, kernel)   # (H, W)

        # apply formula
        left  = ((4.0 * σ2) .* x .- conv) ./ (4.0 * σ2^2)
        right = ((4.0 * σ2) .* y .- conv) ./ (4.0 * σ2^2)

        # write back
        out[:, 1:W, c]     .= left
        out[:, W+1:2W, c]  .= right
    end

    return out
end


"""
    apply_special_transform_inverse_covariance(input_array, kernel, σ2, R)

DirectModel `R` containing the list of geometric transformations (Hᵥ and H_w for each frame).

Implements the formula:
C⁻¹[x; y] = (1 / 4σ⁴) * [ 4σ²x - Hᵥ F†ΔF(Hᵥ†x + H_w†y) ; 4σ²y - H_w F†ΔF(Hᵥ†x + H_w†y) ]
"""
function apply_special_transform_inverse_covariance( input_array::AbstractArray{T,3}, kernel::AbstractMatrix, σ2::T, R::DirectModel{T} ) where {T<:AbstractFloat}
    H, W2, C = size(input_array)
    if W2 % 2 != 0
        throw(ArgumentError("The second dimension must be even (2W). Received: $W2"))
    end
    if C != length(R.TR)
        throw(ArgumentError(
            "The number of image channels ($C) must match the number of transformations in R ($(length(R.TR)))."
        ))
    end
    W = div(W2, 2)

    out = Array{Float64}(undef, H, W2, C)

    for c in 1:C
        # Extract the x and y halves for the current channel
        x = Array{Float64}(@view input_array[:, 1:W, c])
        y = Array{Float64}(@view input_array[:, W+1:2W, c])

        # Extract the H_v and H_w operators for THIS SPECIFIC CHANNEL
        # H_v is R.TR[c].H_l and H_w is R.TR[c].H_r
        H_v = R.TR[c].H_l
        H_w = R.TR[c].H_r

        # 1. Compute the inner term: (Hᵥ†x + H_w†y)
        sum_term = (H_v' * x) .+ (H_w' * y)

        # 2. Apply the convolution: F†ΔF(...)
        conv = toeplitz_convolve(sum_term, kernel)

        # 3. Apply the forward operators: Hᵥ(...) and H_w(...)
        left_term  = H_v * conv
        right_term = H_w * conv

        # 4. Apply the final formula
        left  = (4.0 * σ2 .* x .- left_term)  ./ (4.0 * σ2^2)
        right = (4.0 * σ2 .* y .- right_term) ./ (4.0 * σ2^2)

        # 5. Store the results
        out[:, 1:W, c]     .= left
        out[:, W+1:2W, c]  .= right
    end

    return out
end

# =============================================================================
# PRE-ALLOCATED WORKSPACE FOR PRECISION OPERATIONS
# =============================================================================

"""
    PrecisionWorkspace{T}

Pre-allocated workspace for zero-allocation precision matrix operations.
Holds buffers for PCG iterations and intermediate computations.

# Fields
- `pcg_ws`: PCGWorkspace for the PCG solver
- `fft_op`: PrecomputedToeplitzFFT for fast convolutions
- `x_buffer, y_buffer`: Buffers for left/right analyzer data (H × W)
- `sum_buffer`: Buffer for sum terms (H × W)
- `conv_buffer`: Buffer for convolution results (H × W)
- `left_buffer, right_buffer`: Buffers for left/right results (H × W)
"""
struct PrecisionWorkspace{T<:AbstractFloat, FFT<:PrecomputedToeplitzFFT}
    pcg_ws::PCGWorkspace{T,3}
    fft_op::FFT
    P_double::Matrix{Float64}           # Power spectrum for dynamic kernel computation
    mean_white_noise::Float64           # Mean diagonal noise variance
    weighted_residual::Array{Float64,3} # Buffer for PCG solution (H × 2W × n_frames)
    x_buffer::Matrix{Float64}
    y_buffer::Matrix{Float64}
    sum_buffer::Matrix{Float64}
    conv_buffer::Matrix{Float64}
    left_buffer::Matrix{Float64}
    right_buffer::Matrix{Float64}
end

"""
    PrecisionWorkspace(H, W, n_frames, kernel)

Create a PrecisionWorkspace for data of size (H, 2W, n_frames).

# Arguments
- `H`: Height of each half-image
- `W`: Width of each half-image (total width is 2W)
- `n_frames`: Number of frames
- `kernel`: Kernel matrix of size (2H, 2W) for Toeplitz convolution
"""
function PrecisionWorkspace(H::Int, W::Int, n_frames::Int, kernel::AbstractMatrix{T}, P_double::AbstractMatrix{Float64}, mean_white_noise::Float64) where {T}
    # PCG workspace for arrays of size (H, 2W, n_frames)
    pcg_ws = PCGWorkspace((H, 2*W, n_frames), Float64)

    # Pre-computed FFT operator
    fft_op = PrecomputedToeplitzFFT(H, W, kernel)

    # Buffer for PCG solution (weighted_residual)
    weighted_residual = zeros(Float64, H, 2*W, n_frames)

    # Buffers for intermediate computations (H × W each)
    x_buffer = zeros(Float64, H, W)
    y_buffer = zeros(Float64, H, W)
    sum_buffer = zeros(Float64, H, W)
    conv_buffer = zeros(Float64, H, W)
    left_buffer = zeros(Float64, H, W)
    right_buffer = zeros(Float64, H, W)

    return PrecisionWorkspace{Float64, typeof(fft_op)}(
        pcg_ws, fft_op, Matrix{Float64}(P_double), mean_white_noise, weighted_residual,
        x_buffer, y_buffer, sum_buffer, conv_buffer, left_buffer, right_buffer
    )
end

"""
    apply_special_transform_inverse_covariance!(out, input_array, ws, σ2, R)

In-place version of apply_special_transform_inverse_covariance using pre-allocated workspace.
Zero allocations after initial setup.

# Arguments
- `out`: Output array (H × 2W × C), modified in-place
- `input_array`: Input array (H × 2W × C)
- `ws`: PrecisionWorkspace with pre-allocated buffers
- `σ2`: Variance parameter
- `R`: DirectModel containing geometric transformations
"""
function apply_special_transform_inverse_covariance!(
    out::AbstractArray{T,3},
    input_array::AbstractArray{T,3},
    ws::PrecisionWorkspace,
    σ2::T,
    R::DirectModel{T}
) where {T<:AbstractFloat}
    H, W2, C = size(input_array)
    W = div(W2, 2)

    # Pre-compute scaling factors
    scale_4σ2 = 4.0 * σ2
    inv_4σ4 = 1.0 / (4.0 * σ2^2)

    for c in 1:C
        # Copy to buffers (avoiding allocation)
        @views ws.x_buffer .= input_array[:, 1:W, c]
        @views ws.y_buffer .= input_array[:, W+1:2W, c]

        # Get operators for this channel
        H_v = R.TR[c].H_l
        H_w = R.TR[c].H_r

        # 1. Compute sum term: (Hᵥ†x + H_w†y)
        # Note: H_v' * x allocates, but this is unavoidable without modifying the operator
        ws.sum_buffer .= (H_v' * ws.x_buffer) .+ (H_w' * ws.y_buffer)

        # 2. Apply convolution in-place: F†ΔF(...)
        toeplitz_convolve!(ws.conv_buffer, ws.fft_op, ws.sum_buffer)

        # 3. Apply forward operators and store in left/right buffers
        ws.left_buffer .= H_v * ws.conv_buffer
        ws.right_buffer .= H_w * ws.conv_buffer

        # 4. Apply final formula in-place
        @views @. out[:, 1:W, c] = (scale_4σ2 * ws.x_buffer - ws.left_buffer) * inv_4σ4
        @views @. out[:, W+1:2W, c] = (scale_4σ2 * ws.y_buffer - ws.right_buffer) * inv_4σ4
    end

    return out
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

# =============================================================================
# 9. PRECISION APPLICATION - Apply inverse covariance (precision matrix)
# =============================================================================

"""
    apply_precision(diag_noise::DiagonalNoise{T}, input_array, direct_model; kwargs...)

Apply diagonal precision matrix W = C^(-1) to input array.
For diagonal noise: result = input * weights (element-wise).

Note: kwargs are accepted for interface consistency but ignored for diagonal noise.
"""
function apply_precision(
    diag_noise::DiagonalNoise{T},
    input_array::AbstractArray{T,3},
    direct_model::DirectModel{T};
    kwargs...  # Accept but ignore kwargs for interface consistency
) where {T<:AbstractFloat}
    println("[apply_precision Diagonal] Called, input size=$(size(input_array))")
    flush(stdout)
    if diag_noise.weights === nothing
        throw(ArgumentError("DiagonalNoise does not contain weights. Cannot apply covariance. Use with_weights() to add weights first."))
    end
    result = input_array .* diag_noise.weights
    println("[apply_precision Diagonal] Done")
    flush(stdout)
    return result
end

"""
    apply_precision(noise_model::CorrelatedNoise{T}, input_array, direct_model; kwargs...)

Apply correlated precision matrix W = (B C_(δs) B^T + λI)^(-1) to input array via PCG.
Solves the system (C_corr + λI) * x = input_array where:
- C_corr = B C_(δs) B^T (spatially correlated covariance)
- λI is a small regularization term for numerical stability

# Keyword Arguments
- `additive_variance::Union{Nothing, T}`: Regularization parameter λ (default: small value for stability)
- `rtol::Real`: Relative tolerance for PCG convergence (default: 1e-5)
- `maxiter::Union{Nothing, Int}`: Maximum PCG iterations (default: nothing = auto)
- `verbose::Bool`: Print convergence information (default: false)

# Algorithm
Uses apply_special_transform_inverse_covariance as preconditioner.
"""
function apply_precision(
    noise_model::CorrelatedNoise{T},
    input_array::AbstractArray{T,3},
    direct_model::DirectModel{T};
    additive_variance::Union{Nothing, T} = nothing,
    rtol::Real = 1e-5,
    maxiter::Union{Nothing, Int} = nothing,
    verbose::Bool = false
) where {T<:AbstractFloat}
    println("[apply_precision Correlated] Starting, input size=$(size(input_array))")
    println("  additive_variance=$additive_variance, rtol=$rtol")
    flush(stdout)

    # Use small regularization if none provided (for numerical stability)
    reg_variance = additive_variance !== nothing ? additive_variance : T(1e-10)

    # Create covariance operator: A = C_corr + λI
    apply_A = function(x::AbstractArray{T,3})
        cov_result = apply_covariance(noise_model, x, direct_model)
        return cov_result .+ reg_variance .* x
    end

    # Preconditioner using special inverse formula
    # Wiener filter: Δ = (2σ² × P) / (P + 2σ²)
    P_double = noise_model.P_double
    Delta_kernel = (2.0 * reg_variance .* P_double) ./ (P_double .+ 2.0 * reg_variance)

    apply_M_inv = function(x::AbstractArray{T,3})
        return apply_special_transform_inverse_covariance(
            x,
            Delta_kernel,
            reg_variance,
            direct_model
        )
    end

    # Solve via PCG
    println("[apply_precision Correlated] Starting PCG with reg_variance=$reg_variance")
    flush(stdout)
    x_solution, info = pcg( apply_A, input_array; apply_M_inv = apply_M_inv, rtol = rtol, maxiter = maxiter, verbose = verbose )
    # x_solution, info = pcg( apply_A, input_array; rtol = rtol, maxiter = maxiter, verbose = verbose )
    println("[apply_precision Correlated] PCG done: converged=$(info.converged), iterations=$(info.iterations)")
    flush(stdout)

    if verbose && !info.converged
        @warn "PCG did not converge in apply_precision for CorrelatedNoise" iterations=info.iterations final_residual=info.residual_norm
    end

    return x_solution
end

"""
    apply_precision(noise_model::DiagonalAndCorrelatedNoise{T}, input_array, direct_model; kwargs...)

Apply combined precision matrix W = (C + λI)^(-1) to input array via PCG.
Solves the regularized system (C + λI) * x = input_array where:
- C = B C_(δs) B^T + Σ_n (correlated + diagonal covariance)
- λI is an optional Tikhonov regularization term

# Keyword Arguments
- `additive_variance::Union{Nothing, T}`: Tikhonov regularization parameter λ (default: nothing)
- `rtol::Real`: Relative tolerance for PCG convergence (default: 1e-5)
- `maxiter::Union{Nothing, Int}`: Maximum PCG iterations (default: nothing = auto)
- `verbose::Bool`: Print convergence information (default: false)

# Algorithm
1. Compute mean diagonal variance: σ²_mean = mean(1/weights)
2. Create preconditioner: M ≈ C_corr + (σ²_mean + λ)I
   - Uses apply_special_transform_inverse_covariance for M^{-1}
3. Solve via PCG: (C + λI) * x = b with preconditioner M
"""
function apply_precision(
    noise_model::DiagonalAndCorrelatedNoise{T},
    input_array::AbstractArray{T,3},
    direct_model::DirectModel{T};
    additive_variance::Union{Nothing, T} = nothing,
    rtol::Real = 1e-5,
    maxiter::Union{Nothing, Int} = 200,
    verbose::Bool = false
) where {T<:AbstractFloat}
    if verbose
        println("[apply_precision DiagonalAndCorrelated] Starting...")
        println("  input size=$(size(input_array)), additive_variance=$additive_variance, rtol=$rtol")
        flush(stdout)
    end

    # 1. Calculate mean variance of diagonal noise
    # This represents the average white noise level
    mean_white_noise = if noise_model.diag_noise.weights !== nothing
        # Mean of inverse weights = mean variance
        sum(1.0 ./ noise_model.diag_noise.weights) / length(noise_model.diag_noise.weights)
    else
        zero(T)
    end

    # 2. Add optional additive variance (e.g., gamma_t^2 for diffusion)
    total_variance = mean_white_noise + (additive_variance !== nothing ? additive_variance : zero(T))
    if verbose
        println("  mean_white_noise=$mean_white_noise, total_variance=$total_variance")
        flush(stdout)
    end

    # 3. Create covariance operator: A = C + λI
    #    where C = apply_covariance(noise_model, ...)
    apply_A = function(x::AbstractArray{T,3})
        cov_result = apply_covariance(noise_model, x, direct_model)
        if additive_variance !== nothing
            return cov_result .+ additive_variance .* x
        else
            return cov_result
        end
    end

    # 4. Create preconditioner: M^{-1} ≈ (C_corr + total_variance*I)^{-1}
    #    Uses the special inverse formula in Fourier domain
    #    Wiener filter: Δ = (2σ² × P) / (P + 2σ²)
    P_double = noise_model.corr_noise.P_double
    Delta_kernel = (2.0 * total_variance .* P_double) ./ (P_double .+ 2.0 * total_variance)

    apply_M_inv = function(x::AbstractArray{T,3})
        return apply_special_transform_inverse_covariance( x, Delta_kernel, total_variance, direct_model )
    end

    # 5. Solve (C + λI) * x = input_array using PCG
    if verbose
        println("[apply_precision DiagonalAndCorrelated] Starting PCG with total_variance=$total_variance")
        flush(stdout)
    end
    x_solution, info = pcg( apply_A, input_array; apply_M_inv = apply_M_inv, rtol = rtol, maxiter = maxiter, verbose = verbose )

    if verbose
        println("[apply_precision DiagonalAndCorrelated] PCG done: converged=$(info.converged), iterations=$(info.iterations)")
        flush(stdout)
    end

    # 6. Warn if not converged
    if !info.converged
        @warn "PCG did not converge in apply_precision" iterations=info.iterations final_residual=info.residual_norm
    end

    return x_solution, info
end

"""
    apply_precision(x::AbstractArray{T,3}, dataset::Dataset{T}; kwargs...) -> AbstractArray{T,3}

Apply precision matrix (inverse covariance) to input array using noise model from dataset.
Dispatches to appropriate method based on noise model type.

# Arguments
- `x`: Input array (H × W × frames)
- `dataset`: Dataset containing noise_model and direct_model

# Keyword Arguments (for CorrelatedNoise and DiagonalAndCorrelatedNoise)
- `additive_variance::Union{Nothing, T}`: Tikhonov regularization parameter (default: nothing)
- `rtol::Real`: PCG relative tolerance (default: 1e-5)
- `maxiter::Union{Nothing, Int}`: PCG max iterations (default: nothing = auto)
- `verbose::Bool`: Print convergence info (default: false)
"""
function apply_precision(
    x::AbstractArray{T,3},
    dataset::Dataset{T};
    kwargs...
) where {T<:AbstractFloat}
    verbose = get(kwargs, :verbose, false)
    if verbose
        println("[apply_precision] Called with x size=$(size(x)), type=$(typeof(dataset.noise_model))")
        println("[apply_precision] kwargs=$kwargs")
        flush(stdout)
    end
    # Dispatch based on noise model type, forwarding all kwargs
    result, info = apply_precision(dataset.noise_model, x, dataset.direct_model; kwargs...)
    if verbose
        println("[apply_precision] Done, result size=$(size(result))")
        flush(stdout)
    end
    return result, info
end

# =============================================================================
# OPTIMIZED PRECISION APPLICATION WITH PRE-ALLOCATED WORKSPACE
# =============================================================================

"""
    apply_precision_fast!(noise_model, input_array, direct_model, workspace; kwargs...)

Optimized version of apply_precision using pre-allocated workspace.
Significantly reduces allocations during PCG iterations.
The solution is stored in `workspace.weighted_residual`.

# Arguments
- `noise_model`: DiagonalAndCorrelatedNoise model
- `input_array`: Input array (H × 2W × n_frames) - the right-hand side
- `direct_model`: DirectModel for geometric transformations
- `workspace`: PrecisionWorkspace with pre-allocated buffers (includes output buffer)

# Keyword Arguments
- `additive_variance::Union{Nothing, T}`: Tikhonov regularization parameter λ
- `rtol::Real`: Relative tolerance for PCG convergence (default: 1e-5)
- `maxiter::Int`: Maximum PCG iterations (default: 200)
- `verbose::Bool`: Print convergence information (default: false)

# Returns
NamedTuple with PCG convergence info (converged, iterations, residual_norm)

# Example
```julia
# Create workspace once (expensive, do at setup)
H, W, n_frames = 128, 128, 4  # W is half-width
ws = create_precision_workspace(noise_model, (H, 2W, n_frames), additive_variance)

# Use many times (cheap, minimal allocations)
info = apply_precision_fast!(noise_model, b, direct_model, ws; rtol=1e-5)
# Result is in ws.weighted_residual
```
"""
function apply_precision_fast!(
    noise_model::DiagonalAndCorrelatedNoise{T},
    input_array::AbstractArray{T,3},
    direct_model::DirectModel{T},
    workspace::PrecisionWorkspace;
    additive_variance::Union{Nothing, T} = nothing,
    rtol::Real = 1e-5,
    maxiter::Int = 200,
    verbose::Bool = false
) where {T<:AbstractFloat}

    # Use workspace's internal weighted_residual buffer
    x_solution = workspace.weighted_residual

    # Initialize solution to zero (or warm start if provided)
    x_solution .= 0

    # 1. Use pre-computed mean_white_noise from workspace (avoids recomputing)
    mean_white_noise = workspace.mean_white_noise

    # 2. Total variance for preconditioner
    total_variance = mean_white_noise + (additive_variance !== nothing ? additive_variance : zero(T))

    if verbose
        println("[apply_precision_fast!] mean_white_noise=$mean_white_noise, total_variance=$total_variance")
        flush(stdout)
    end

    # 3. Update kernel for current total_variance (dynamic Wiener filter)
    # Δ = (2σ² × P) / (P + 2σ²) in Fourier domain
    # Note: kernel_fft is ComplexF64, P_double is Float64
    P_double = workspace.P_double
    @. workspace.fft_op.kernel_fft = ComplexF64((2.0 * total_variance * P_double) / (P_double + 2.0 * total_variance))

    # 4. Create in-place covariance operator: A!(out, x) computes out = (C + λI) * x
    function apply_A!(out::AbstractArray{T,3}, x::AbstractArray{T,3})
        # Apply full covariance: C * x
        # Note: apply_covariance still allocates, but it's unavoidable without major refactoring
        cov_result = apply_covariance(noise_model, x, direct_model)
        if additive_variance !== nothing
            @. out = cov_result + additive_variance * x
        else
            out .= cov_result
        end
        return out
    end

    # 5. Create in-place preconditioner: M_inv!(out, r) computes out = M^{-1} * r
    function apply_M_inv!(out::AbstractArray{T,3}, r::AbstractArray{T,3})
        apply_special_transform_inverse_covariance!(out, r, workspace, T(total_variance), direct_model)
        return out
    end

    # 5. Solve using pre-allocated PCG
    if verbose
        println("[apply_precision_fast!] Starting PCG with maxiter=$maxiter, rtol=$rtol")
        flush(stdout)
    end

    info = pcg_preallocated!(
        x_solution,
        apply_A!,
        apply_M_inv!,
        input_array,
        workspace.pcg_ws;
        rtol = rtol,
        maxiter = maxiter,
        verbose = verbose
    )

    if verbose
        println("[apply_precision_fast!] PCG done: converged=$(info.converged), iterations=$(info.iterations)")
        flush(stdout)
    end

    if !info.converged
        @warn "PCG did not converge in apply_precision_fast!" iterations=info.iterations final_residual=info.residual_norm
    end

    return info
end

"""
    apply_precision_fast!(input_array, dataset, workspace; kwargs...)

Convenience wrapper for apply_precision_fast! using Dataset.
The solution is stored in `workspace.weighted_residual`.
"""
function apply_precision_fast!(
    input_array::AbstractArray{T,3},
    dataset::Dataset{T},
    workspace::PrecisionWorkspace;
    kwargs...
) where {T<:AbstractFloat}
    return apply_precision_fast!(
        dataset.noise_model,
        input_array,
        dataset.direct_model,
        workspace;
        kwargs...
    )
end

"""
    create_precision_workspace(noise_model::DiagonalAndCorrelatedNoise, data_shape, additive_variance)

Helper function to create a PrecisionWorkspace for the given noise model and data shape.

# Arguments
- `noise_model`: DiagonalAndCorrelatedNoise model
- `data_shape`: Tuple (H, W2, n_frames) where W2 = 2W (full width including both analyzers)
- `additive_variance`: Regularization parameter λ (used to compute Wiener kernel)

# Returns
PrecisionWorkspace ready for use with apply_precision_fast!
"""
function create_precision_workspace(
    noise_model::DiagonalAndCorrelatedNoise{T},
    data_shape::Tuple{Int, Int, Int},
    additive_variance::T
) where {T<:AbstractFloat}
    H, W2, n_frames = data_shape
    W = div(W2, 2)

    # Compute mean white noise variance
    mean_white_noise = if noise_model.diag_noise.weights !== nothing
        sum(1.0 ./ noise_model.diag_noise.weights) / length(noise_model.diag_noise.weights)
    else
        zero(T)
    end
    total_variance = mean_white_noise + additive_variance

    # Compute Wiener filter kernel: Δ = (2σ² × P) / (P + 2σ²)
    P_double = noise_model.corr_noise.P_double
    Delta_kernel = (2.0 * total_variance .* P_double) ./ (P_double .+ 2.0 * total_variance)

    # Store P_double and mean_white_noise for dynamic kernel recomputation
    return PrecisionWorkspace(H, W, n_frames, Delta_kernel, P_double, Float64(mean_white_noise))
end