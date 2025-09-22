# Méthodes d'application des opérateurs

"""
    *(W::DiagonalWeights, x::AbstractArray)

Apply diagonal weights element-wise: W * x = weights .* x
"""
function Base.:*(W::DiagonalWeights{T,N}, x::AbstractArray{T,N}) where {T,N}
    return W.weights .* x  # Changer weights_op en weights
end

"""
    *(W::FourierPrecisionOperator, x::AbstractArray{T,2})

Apply precision operator with bad pixel masking in Fourier domain.
Implements W * x = M .* IFFT(inv_psd .* FFT(x))
where M is the bad pixel mask.
"""
function Base.:*(W::FourierPrecisionOperator, x::AbstractArray{T,2}) where T
    # Transform to Fourier domain using pre-computed plan
    Fx = W.fft_plan * x
    
    # Apply inverse filter
    W_Fx = W.inv_psd .* Fx
    
    # Transform back and take real part
    result = real.(W.ifft_plan * W_Fx)
    
    # Apply bad pixel mask: M ∘ (F^(-1) ∘ diag(1/P(k)) ∘ F)
    return W.good_pix .* result
end

"""
    *(W::FourierPrecisionOperator, x::AbstractArray{T,3})

Apply precision operator to 3D arrays (e.g., I, Ip, θ stacks) with bad pixel masking.
Applies operator slice by slice with consistent masking.
"""
function Base.:*(W::FourierPrecisionOperator, x::AbstractArray{T,3}) where T
    result = similar(x)
    for i in 1:size(x, 3)
        result[:, :, i] = W * x[:, :, i]
    end
    return result
end