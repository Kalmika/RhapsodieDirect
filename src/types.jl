const ImageInterpolator{T<:AbstractFloat, K<:Kernel{T}} = TwoDimensionalTransformInterpolator{T,K,K}

"""
    ObjectParameters(size, center)
    
where:

* `size` is the size of one map of the object
* `center` is the center of the object. 

""" ObjectParameters
struct ObjectParameters{T<:AbstractFloat, U<:Int}
    size::NTuple{2,U}
    center::NTuple{2,T}
end

"""
    DatasetParameters(size, frames_total, frames_per_hwp_pos, hwp_cycles, center)
    
* `size` is the total size of one frame on ne camera (with both side)
* `frames_total` is the total number of frames in the dataset
* `frames_per_hwp_pos`is the number of frames for a given half-wave-plate (hwp) position
* `hwp_cycles` is the number of hwp cycles
* `center` is the center of the object in this dataset

""" DatasetParameters
struct DatasetParameters{T<:AbstractFloat, U<:Int}
    size::NTuple{2,U}
    frames_total::U
    frames_per_hwp_pos::U
    hwp_cycles::U
    center::NTuple{2,T}
end


"""
    FieldTransformParameters(ker,field_angle, translation_left, translation_right, polarization_left, polarization_right)

* `ker` is an interpolation Kernel from the package `InterpolationKernels`
* `field_angle` is the field rotation for the given frame
* `translation_left` is the vector of translation composed with the difference of the object_center on the left side of the camera and the dataset center.
* `translation_right`is the vector of translation composed with the difference of the object_center on the right side of the camera and the dataset center.
* `polarization_left` are the polarization coefficient (the three first mueller matrix coefficients) of the left side of the camera
* `polarization_right`are the polarization coefficient (the three first mueller matrix coefficients) of the right side of the camera

""" FieldTransformParameters
struct FieldTransformParameters{T<:AbstractFloat,K<:Kernel}
    ker::K
    field_angle::T
    translation_left::NTuple{2,T}
    translation_right::NTuple{2,T}
    polarization_left::NTuple{3,T}
    polarization_right::NTuple{3,T}
end


"""
    FieldTransformOperator
    
provides the linear combination of the geometrical transform and polarization coefficient, with the blurred Stokes parameters.

""" FieldTransformOperator
struct FieldTransformOperator{T<:AbstractFloat, 
                              ColType<:NTuple{3,Int},
                              RowType<:NTuple{2,Int},
                              P<:NTuple{3,T},
                              L<:Mapping, 
                              R<:Mapping} <: LinearMapping
    cols::ColType
    rows::RowType
    v_l::P
    v_r::P
    H_l::L              
    H_r::R      
end

"""
    NoiseModel

Abstract type for noise modeling approaches.
"""
abstract type NoiseModel end

"""
    DiagonalNoise{T} <: NoiseModel

Classical diagonal noise model using independent Gaussian noise.
Can be initialized with or without explicit weights.

# Fields
- `weights::Union{AbstractArray{T}, Nothing}`: Optional weight matrix W. If `nothing`, acts as identity.

# Constructors
- `DiagonalNoise{T}()`: Create without weights (identity covariance)
- `DiagonalNoise(weights)`: Create with specific weights (type inferred)
"""
struct DiagonalNoise{T<:AbstractFloat} <: NoiseModel
    weights::Union{AbstractArray{T}, Nothing}

    # Constructor without weights - requires explicit type parameter
    DiagonalNoise{T}() where {T<:AbstractFloat} = new{T}(nothing)

    # Constructor with weights - infers type from weights
    DiagonalNoise(weights::AbstractArray{T}) where {T<:AbstractFloat} = new{T}(weights)
end

# Default constructor - uses Float64 when no type is specified
DiagonalNoise() = DiagonalNoise{Float64}()



"""
    CorrelatedNoise <: NoiseModel

Spatially correlated noise model using power spectral density P(k).
Based on filtered Gaussian noise in Fourier domain.

* `A` - Amplitude parameter
* `σ` - Spectral width parameter
* `N` - Image size (assuming square images)
* `P` - Pre-computed P(k) matrix (N x N)
* `P_double` - Pre-computed P(k) matrix (2N x 2N) for Toeplitz convolution
* `sqrt_P` - Pre-computed sqrt(P(k)) for efficiency (N x N)
* `sqrt_P_double` - Pre-computed sqrt(P(k)) of size 2N x 2N
"""
struct CorrelatedNoise{T<:AbstractFloat} <: NoiseModel
    A::T
    σ::T
    N::Int
    P::Matrix{T}
    P_double::Matrix{T}
    sqrt_P::Matrix{T}
    sqrt_P_double::Matrix{T}

    function CorrelatedNoise(A::T, σ::T, N::Int) where {T<:AbstractFloat}
        P = compute_power_spectrum(A, σ, N)
        P_double = compute_power_spectrum(A, σ, N*2)
        sqrt_P = sqrt.(P)
        sqrt_P_double = sqrt.(P_double)
        new{T}(A, σ, N, P, P_double, sqrt_P, sqrt_P_double)
    end
end


"""
    DiagonalAndCorrelatedNoise{T} <: NoiseModel

A noise model that combines a diagonal and a correlated component.
It is constructed via composition, holding instances of the two sub-models.

# Fields
- `diag_noise::DiagonalNoise{T}`: The diagonal noise component.
- `corr_noise::CorrelatedNoise{T}`: The correlated noise component.
"""
struct DiagonalAndCorrelatedNoise{T<:AbstractFloat} <: NoiseModel
    diag_noise::DiagonalNoise{T}
    corr_noise::CorrelatedNoise{T}

    """
        DiagonalAndCorrelatedNoise(diag_noise::DiagonalNoise{T}, corr_noise::CorrelatedNoise{T})

    Direct constructor from component noise models.
    """
    function DiagonalAndCorrelatedNoise(diag_noise::DiagonalNoise{T}, corr_noise::CorrelatedNoise{T}) where {T<:AbstractFloat}
        new{T}(diag_noise, corr_noise)
    end

    """
        DiagonalAndCorrelatedNoise(A::T, σ::T, N::Int) where {T}

    Convenience constructor to create a `DiagonalAndCorrelatedNoise` model
    directly from the physical parameters of its correlated part.
    """
    function DiagonalAndCorrelatedNoise(A::T, σ::T, N::Int) where {T<:AbstractFloat}
        new{T}(DiagonalNoise{T}(), CorrelatedNoise(A, σ, N))
    end
end

"""
    AbstractWeightOperator

Abstract type for precision/weight operators in optimization.
"""
abstract type AbstractWeightOperator end

"""
    DiagonalWeights <: AbstractWeightOperator

Traditional diagonal weight operator W = diag(w).
Compatible with existing weight arrays.
"""
struct DiagonalWeights{T<:AbstractFloat, N} <: AbstractWeightOperator
    weights::AbstractArray{T,N}
    
    DiagonalWeights(weights::AbstractArray{T,N}) where {T,N} = new{T,N}(weights)
end

"""
    FourierPrecisionOperator <: AbstractWeightOperator

Fourier-domain precision operator W = M ∘ F^(-1) ∘ diag(1/P(k)) ∘ F
where M is the bad pixel mask and F is the Fourier transform.
Applies inverse filter 1/(P(k) + ε) in frequency domain, then masks bad pixels.
"""
struct FourierPrecisionOperator{T<:AbstractFloat} <: AbstractWeightOperator
    inv_psd::Matrix{ComplexF64}    # 1/(P(k) + ε) 
    good_pix::AbstractArray{T,2}   # Bad pixel mask M (0 = dead, ≠0 = good)
    fft_plan::Any                  # Pre-computed FFT plan (can be ScaledPlan or cFFTWPlan)
    ifft_plan::Any                 # Pre-computed IFFT plan (can be ScaledPlan or cFFTWPlan)
    function FourierPrecisionOperator(noise::CorrelatedNoise{T}, 
                                    good_pix::AbstractArray{T,2}; 
                                    reg_param_relative::Float64=1e-3) where T
        reg_param = reg_param_relative * maximum(noise.P)
        inv_psd = complex.(1.0 ./ (noise.P .+ reg_param))
        # Pre-compute FFT plans for efficiency - use same size as good_pix
        dummy = zeros(ComplexF64, size(noise.P))
        fft_plan = plan_fft(dummy)
        ifft_plan = plan_ifft(dummy)
        new{T}(inv_psd, good_pix, fft_plan, ifft_plan)
    end
end

"""
    DirectModel
    
TODO: documentation
""" DirectModel
struct DirectModel{T<:AbstractFloat, 
                   S<:AbstractString,
                   ColType<:NTuple{2,Int},
                   RowType<:NTuple{3,Int},
                   PerFrameTransformsType<:Vector{FieldTransformOperator{T}},
                   GlobalTransformsType<:Mapping} <: LinearMapping
    cols::ColType
    rows::RowType
    parameter_type::S
    TR::PerFrameTransformsType               
    A::GlobalTransformsType
    A_pseudo_inv::Matrix{T}  # Matrice pseudo-inverse des coefficients de polarisation
end  

# Constructeur sans A (A = Identity)
DirectModel(cols::ColType, 
            rows::RowType, 
            parameter_type::S,
            TR::PerFrameTransformsType) where {T<:AbstractFloat, 
                        S<:AbstractString,
                        ColType<:NTuple{2,Int},
                        RowType<:NTuple{3,Int},
                        PerFrameTransformsType<:Vector{FieldTransformOperator{T}}} =
                   DirectModel(cols, rows, parameter_type, TR, LazyAlgebra.Id)

# Constructeur complet avec calcul automatique de A_pseudo_inv
function DirectModel(cols::ColType, 
                    rows::RowType, 
                    parameter_type::S,
                    TR::PerFrameTransformsType,
                    A::GlobalTransformsType) where {T<:AbstractFloat, 
                                S<:AbstractString,
                                ColType<:NTuple{2,Int},
                                RowType<:NTuple{3,Int},
                                PerFrameTransformsType<:Vector{FieldTransformOperator{T}},
                                GlobalTransformsType<:Mapping}
    
    # Calcul de la matrice pseudo-inverse des coefficients de polarisation
    A_matrix = zeros(T, length(TR)*2, length(TR[1].v_l))
    @inbounds for k=1:length(TR)	 
        A_matrix[2k-1, :] .= TR[k].v_l
        A_matrix[2k, :]   .= TR[k].v_r
    end
    A_pseudo_inv = inv(A_matrix' * A_matrix)
    
    return DirectModel{T, S, ColType, RowType, PerFrameTransformsType, GlobalTransformsType}(
        cols, rows, parameter_type, TR, A, A_pseudo_inv)
end

"""
    Dataset

Container for observational data with associated noise model and direct model.
"""
struct Dataset{T<:AbstractFloat, N<:NoiseModel, H<:DirectModel{T}, D<:AbstractArray{T,3}}
    data::D
    noise_model::N
    direct_model::H

    function Dataset(
        data::AbstractArray{T,3},
        noise_model::NoiseModel,
        direct_model::H
    ) where {T, H<:DirectModel{T}}
        new{T, typeof(noise_model), H, typeof(data)}(data, noise_model, direct_model)
    end
end

