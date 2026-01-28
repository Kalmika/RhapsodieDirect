#
# Rhapsodie.jl
#
# Package for the Reconstruction of High-contrAst Polarized
# SOurces and Deconvolution for cIrcumstellar Environments (Rhapsodie)
#
#----------------------------------------------------------
#
# 
# Copiyright (c) 2017-2021 Laurence Denneulin (see LICENCE.md)
#

module RhapsodieDirect

    export
        apply!,
        chi_square!,
        convert,
        data_generator,
        data_simulator_dual_component,
        data_simulator_dual_component_bis,
        data_simulator,
        Dataset,
        DatasetParameters,
        DirectModel,
        direct_model!,
        field_transform,
        FieldTransformOperator,
        FieldTransformParameters,
        generate_parameters,
        get_indices_table,
        load_field_transforms,
        load_identity_field_transforms,
        ObjectParameters,
        PolarimetricMap,
        PolarimetricPixel,
        read,
        set_default_polarisation_coefficients,
        set_fft_operator,
        vcreate,
        write,
        #  Noise models
        NoiseModel, DiagonalNoise, CorrelatedNoise, DiagonalAndCorrelatedNoise, create_noise_model, generate_noise, validate_noise_model,
        generate_correlated_noise,
        apply_covariance,
        apply_precision,
        toeplitz_convolve,
        apply_special_inverse_covariance,
        apply_special_transform_inverse_covariance,
        with_weights,
        AbstractWeightOperator,
        DiagonalWeights,
        FourierPrecisionOperator,
        compute_polarization_inverse_matrix,
        apply_direct_model_inverse_no_blur,
        #  PCG Solver
        pcg,
        pcg_solve_covariance

    import Base: +, -, *, /, ==, getindex, setindex!, read, write, convert, copy, fill!

    using TwoDimensional
    using FFTW
    using InterpolationKernels
    using LinearInterpolators
    using LazyAlgebra
    import LazyAlgebra: Mapping, vcreate, vcopy, apply!
    using FFTW
    using EasyFITS
    
    include("types.jl")
    include("polarimetric_parameters.jl")
    include("mappings.jl")
    include("methods.jl")
    include("utils.jl")
    include("loaders.jl")
    include("datasimul_tools.jl")
    include("noise_models.jl")
    include("weight_operators.jl")
    include("pcg_solver.jl")
end

