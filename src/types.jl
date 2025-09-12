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
end  

DirectModel(cols::ColType, 
            rows::RowType, 
            parameter_type::S,
            TR::PerFrameTransformsType) where {T<:AbstractFloat, 
                        S<:AbstractString,
                        ColType<:NTuple{2,Int},
                        RowType<:NTuple{3,Int},
                        PerFrameTransformsType<:Vector{FieldTransformOperator{T}}} =
                   DirectModel(cols, rows, parameter_type, TR, LazyAlgebra.Id)


"""
    Dataset
    
TODO: documentation
""" Dataset
struct Dataset{T<:AbstractFloat,
               M<:AbstractArray{T,3},
               H<:DirectModel{T}}
               
        data::M
        weights::M
        direct_model::H
end

"""
    NoiseModel

Abstract type for noise modeling approaches.
"""
abstract type NoiseModel end

"""
    DiagonalNoise <: NoiseModel

Classical diagonal noise model using independent Gaussian noise.
Compatible with existing weight-based approach.
"""
struct DiagonalNoise <: NoiseModel end

"""
    CorrelatedNoise <: NoiseModel

Spatially correlated noise model using power spectral density P(k).
Based on filtered Gaussian noise in Fourier domain.

* `A` - Amplitude parameter  
* `σ²` - Spectral width parameter
* `N` - Image size (assuming square images)
* `P` - Pre-computed P(k) matrix
* `sqrt_P` - Pre-computed sqrt(P(k)) for efficiency
"""
struct CorrelatedNoise{T<:AbstractFloat} <: NoiseModel
    A::T
    σ²::T  
    N::Int
    P::Matrix{T}
    sqrt_P::Matrix{T}
    
    function CorrelatedNoise(A::T, σ²::T, N::Int) where {T<:AbstractFloat}
        P = compute_power_spectrum(A, σ², N)
        sqrt_P = sqrt.(P)
        new{T}(A, σ², N, P, sqrt_P)
    end
end

