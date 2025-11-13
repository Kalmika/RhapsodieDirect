# Méthodes d'application des opérateurs
import Base: broadcastable

"""
    *(W::DiagonalWeights, x::AbstractArray)

Apply diagonal weights element-wise: W * x = weights .* x
"""
function Base.:*(W::DiagonalWeights{T,N}, x::AbstractArray{T,N}) where {T,N}
    return W.weights .* x  
end

"""
    /(x::AbstractArray, W::DiagonalWeights)

Apply diagonal weights element-wise via division: x / W = x ./ weights
"""
function Base.:/(x::AbstractArray{T,N}, W::DiagonalWeights{T,N}) where {T,N}
    return x ./ W.weights
end

"""
    broadcastable(W::DiagonalWeights)

Allow DiagonalWeights to be used in broadcasted operations (like .+, .*, etc.).
The broadcasting will be applied to the underlying `weights` array.
"""
Base.broadcastable(W::DiagonalWeights) = W.weights

"""
    *(W::FourierPrecisionOperator, x::AbstractArray{T,3})

Apply precision operator with bad pixel masking in Fourier domain.
Implements W * x = IFFT(inv_psd .* FFT(x))
where M is the bad pixel mask.
"""
function Base.:*(W::FourierPrecisionOperator, x::AbstractArray{T,3}) where T
    # x shape : (128, 256, 4), so 8 images of 128x128 side by side
    largeur_sous_image = size(x, 2) ÷ 2
    result = similar(x)
    # Boucle externe : sur les tranches
    for k in 1:size(x, 3)
        # Boucle interne : sur les 2 parties (gauche/droite)
        for j in 1:2
            start_col = (j - 1) * largeur_sous_image + 1
            end_col = j * largeur_sous_image
            # Transform to Fourier domain using pre-computed plan
            Fx = W.fft_plan * x[:, start_col:end_col, k]
            # Apply inverse filter
            W_Fx = W.inv_psd .* Fx
            # Transform back and take real part
            result[:, start_col:end_col, k] = real.(W.ifft_plan * W_Fx)
        end
    end
    return result
end