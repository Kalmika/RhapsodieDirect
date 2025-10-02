#
# datasimul_tools.jl
#
# Provide tools to simulate synthetic parameters and dataset.
#
# ------------------------------------------------
#
# This file is part of Rhapsodie
#
#
# Copyright (c) 2017-2021 Laurence Denneulin (see LICENCE.md)
#

#------------------------------------------------

function data_simulator_dual_component(Good_Pix::AbstractArray{T,2},
                        F::Vector{FieldTransformOperator{T}}, 
                        S_disk::PolarimetricMap, S_star::PolarimetricMap; A_disk::Mapping = LazyAlgebra.Id, ro_noise=8.5) where {T <:AbstractFloat}
   
    M=zeros(size(Good_Pix)[1],size(Good_Pix)[2],length(F))
    H_disk = DirectModel(size(S_disk), size(M), S_disk.parameter_type, F, A_disk)
	H_star = DirectModel(size(S_star), size(M), S_star.parameter_type, F)
    M = H_disk*S_disk + H_star*S_star
    
    println("hellox!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    VAR=max.(M,zero(eltype(M))) .+ro_noise^2
	W=Good_Pix ./ VAR
	D=data_generator(M, W)
	
	return D,W
end

function data_generator(model::AbstractArray{T,N}, weights::AbstractArray{T,N};bad=zero(T)) where {T<:AbstractFloat,N}   
    #seed === nothing ||  Random.seed!(seed);
    
    data = Array{T}(undef, size(model));
    @inbounds for i in eachindex(data, weights)
        w=weights[i] 
        (isfinite(w) && w >= 0 ) || error("invalid weights")
        if w >0            
            data[i] = model[i]  +randn()/sqrt(w)    
        elseif w ==0 
            data[i]=bad;
        end
    end
    return data
end


"""
    data_simulator_dual_component_flexible(BadPixMap, field_transforms, S_disk, S_star, noise_model; A_disk=nothing)

Flexible dual component simulator for disk + star systems.
"""
function data_simulator_dual_component_bis(
    Good_Pix::AbstractArray{T,2},
    F::Vector{FieldTransformOperator{T}}, 
    S_disk::PolarimetricMap, 
    S_star::PolarimetricMap;
    noise_model::NoiseModel = DiagonalNoise(),
    A_disk::Mapping = LazyAlgebra.Id,
    verbose::Bool = false,
    ro_noise::Float64 = 8.5,
    reg_param_relative::Float64 = 1e-3) where {T <:AbstractFloat}

    if verbose
        println("Good_Pix size: ", size(Good_Pix))
        println("Number of field transforms: ", length(F))
        println("S_disk size: ", size(S_disk))
        println("S_star size: ", size(S_star))
        println("Noise model type: ", typeof(noise_model))
        println("A_disk type: ", typeof(A_disk))
        println("ro_noise value: ", ro_noise)
    end
    
    if isa(noise_model, DiagonalNoise)
        return dsdc_diagonal_noise(Good_Pix, F, S_disk, S_star; A_disk=A_disk, ro_noise=ro_noise)        
    elseif isa(noise_model, CorrelatedNoise)
        return dsdc_correlated_noise(Good_Pix, F, S_disk, S_star, noise_model; A_disk=A_disk, ro_noise=ro_noise, reg_param_relative=reg_param_relative)
    else
        error("Unsupported noise model type: $(typeof(noise_model))")
    end

end

"""  
Data simulator dual component (disk + star) with diagonal noise.
"""
function dsdc_diagonal_noise(
    Good_Pix::AbstractArray{T,2},
    F::Vector{FieldTransformOperator{T}}, 
    S_disk::PolarimetricMap, 
    S_star::PolarimetricMap; 
    A_disk::Mapping = LazyAlgebra.Id, 
    ro_noise=8.5) where {T <:AbstractFloat}
   
    M=zeros(size(Good_Pix)[1],size(Good_Pix)[2],length(F))
    H_disk = DirectModel(size(S_disk), size(M), S_disk.parameter_type, F, A_disk)
	H_star = DirectModel(size(S_star), size(M), S_star.parameter_type, F)
    data = H_disk*S_disk + H_star*S_star

    weights_operator = compute_weights_and_add_noise!(data, Good_Pix, ro_noise)
	return data, weights_operator
end

"""  
Data simulator dual component (disk + star) with correlated noise.
"""
function dsdc_correlated_noise(
    Good_Pix::AbstractArray{T,2},
    F::Vector{FieldTransformOperator{T}}, 
    S_disk::PolarimetricMap, 
    S_star::PolarimetricMap,
    noise_model::CorrelatedNoise; 
    A_disk::Mapping = LazyAlgebra.Id, 
    reg_param_relative::Float64 = 1e-3,  # Regularization parameter
    ro_noise::Float64 = 8.5) where {T <:AbstractFloat}

    M=zeros(size(Good_Pix)[1],size(Good_Pix)[2],length(F))
    H_disk = DirectModel(size(S_disk), size(M), S_disk.parameter_type, F, A_disk)
	H_star = DirectModel(size(S_star), size(M), S_star.parameter_type, F)

    correlated_noise = generate_correlated_noise(noise_model)
    S_star_correlated_noise = PolarimetricMap("intensities",  S_star.Iu .+ correlated_noise, zero(correlated_noise), zero(correlated_noise)) 
    data = H_disk*S_disk + H_star*S_star_correlated_noise

    weights_operator = FourierPrecisionOperator(noise_model, Good_Pix, reg_param_relative=reg_param_relative)
	return data, weights_operator
end

function data_simulator(Good_Pix::AbstractArray{T,2},
                        F::Vector{FieldTransformOperator{T}}, 
                        S::PolarimetricMap; A::Mapping = LazyAlgebra.Id, ro_noise=8.5) where {T <:AbstractFloat}
   
    data=zeros(size(Good_Pix)[1],size(Good_Pix)[2],length(F))
    H = DirectModel(size(S), size(data),S.parameter_type,F,A)
    data = H*S
    weights = compute_weights_and_add_noise!(data, Good_Pix, ro_noise)
	return data, weights
end

function compute_weights_and_add_noise!(data::AbstractArray{T,N1}, 
                                      good_pixels::AbstractArray{T,N2}, 
                                      ro_noise::Real) where {T<:AbstractFloat,N1,N2}
    # Compute variance: max(signal, 0) + readout_noise²
    VAR = max.(data, zero(eltype(data))) .+ ro_noise^2    
    # Compute weights: good_pixels / variance
    weights = good_pixels ./ VAR    
    add_noise_with_masking!(data, weights)

    return DiagonalWeights(weights)
end

function add_noise_with_masking!(data::AbstractArray{T,N}, weights::AbstractArray{T,N};bad=zero(T)) where {T<:AbstractFloat,N}   
    @inbounds for i in eachindex(data, weights)
        (isfinite(weights[i]) && weights[i] >= 0 ) || error("invalid weights")
        if weights[i] >0            
            data[i] += randn()/sqrt(weights[i])    
        elseif weights[i] ==0 
            data[i]=bad;
        end
    end
    return data
end

function generate_parameters(parameters::ObjectParameters, tau::Float64)
	Ip=zeros(parameters.size);
	Iu=zeros(parameters.size);
	θ=zeros(parameters.size);
	STAR1=zeros(parameters.size);
	STAR2=zeros(parameters.size);
	
	for i=1:parameters.size[1]
    	for j=1:parameters.size[2]
    		r1=parameters.center[1]-i;
    		r2=parameters.center[2]-j;
    		if (r1^2+r2^2<=20^2)
        		Iu[i,j]=1000;
        		Ip[i,j]=tau*Iu[i,j]/(1-tau);
    		end 
    		if ((r1^2+r2^2>=25^2)&&(r1^2+r2^2<=27^2))
        		Iu[i,j]=1000;
        		Ip[i,j]=tau*Iu[i,j]/(1-tau);
    		end
    		if ((r1^2+r2^2>=32^2)&&(r1^2+r2^2<=40^2))
        		Iu[i,j]=1000;
        		Ip[i,j]=tau*Iu[i,j]/(1-tau);
    		end
    		θ[i,parameters.size[2]+1-j]=atan(j-parameters.center[2],i-parameters.center[1]);
			STAR1[i,j]=200*exp(-((i-parameters.center[1])^2+(j-parameters.center[2])^2)/(2*75^2))
			STAR2[i,j]=100000*exp(-((i-parameters.center[1])^2+(j-parameters.center[2])^2)/(2*7^2))
			if (((parameters.center[1]-i)^2+(parameters.center[2]-j)^2)<=10^2)
        		STAR2[i,j]=800;
        		Iu[i,j]=0;
        		Ip[i,j]=0;	
    		end
			if (((parameters.center[1]-i)^2+(parameters.center[2]-j)^2)<=70^2)
        		STAR1[i,j]=50;		
    		end
		end
	end    
	θ=θ.*(Ip.!=0);
	STAR=STAR1+STAR2
	STAR[round(Int64,10*parameters.size[1]/16)-3,round(Int64,10*parameters.size[2]/16)]=20000.0;
	STAR[round(Int64,10*parameters.size[1]/16),round(Int64,10*parameters.size[2]/16)-3]=100000.0;

    return PolarimetricMap("intensities", Iu+STAR, Ip, θ);
end
