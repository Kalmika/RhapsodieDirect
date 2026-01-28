#
# Polarimetric_Parameters.jl
#
# Provide the type polarimetric parameter.
#
# ------------------------------------------------
#
# This file is part of Rhapsodie
#
#
# Copyright (c) 2017-2021 Laurence Denneulin (see LICENCE.md)

# Dictionaries for parameters order depending of the type
const MAPDICT = Dict("stokes" => Dict(1 => :I, 2 => :Q,  3 => :U),
                      "intensities" => Dict(1 => :Iu, 2 => :Ip,  3 => :θ),
                      "mixed" => Dict(1 => :Iu, 2 => :Q,  3 => :U))

#------------------------------------------------
# Structure definition
 
struct PolarimetricMap{T<: AbstractFloat,
                       M<:AbstractArray{T,2}} 
    parameter_type::AbstractString   #either "stokes", "intensities" or "mixed"
    I::M
    Q::M
    U::M
    Iu::M
    Ip::M
    θ::M
end

function PolarimetricMap{T}(parameter_type::AbstractString,        
                            I::AbstractArray{<:Any,2},
                            Q::AbstractArray{<:Any,2},
                            U::AbstractArray{<:Any,2},
                            Iu::AbstractArray{<:Any,2},
                            Ip::AbstractArray{<:Any,2},
                            θ::AbstractArray{<:Any,2}) where {T<:AbstractFloat}        
    @assert size(I) == size(Q) == size(U) == size(Iu) == size(Ip) == size(θ)
    # (minimum(I) < 0) && @warn "Negative intensities in I"  
    # (minimum(Iu) < 0) && @warn "Negative intensities in Iu"  
    # (minimum(Ip) < 0) && @warn "Negative intensities in Ip"  

    PolarimetricMap(parameter_type,        
                    convert(Array{T},I),
                    convert(Array{T},Q),
                    convert(Array{T},U),
                    convert(Array{T},Iu),
                    convert(Array{T},Ip),
                    convert(Array{T},θ))
end
                               
#------------------------------------------------
# Constructors 
"""
PolarimetricMap(parameter_type, x) -> PolarimetricMap

create an object of type PolarimetricParameter from either:
- Parameters I, Q, U (i.e. parameter_type = 'stokes')
- Parameters Iu, Ip and θ (i.e. parameter_type = 'intensities')
- Parameters Iu, Q, U (i.e. parameter_type = 'mixed')

Each parameter can be called from the structur. For exemple with a 
construction from Stokes parameters S=(I,Q,U):

using Rhapsodie
X = PolarimetricParameter(S, 'stokes');
X.I #yields the Stokes parameter I
X.Ip #yields the polarized intensity Ip
X.I[1,1] #yields the Stokes parameter I at the CartesianIndex (1,1); 

PolarimetricMap(parameter_type, n1, n2) -> PolarimetricMap

yields an empty    
""" PolarimetricMap    
function PolarimetricMap(parameter_type::AbstractString, 
                         x1::A, 
                         x2::A, 
                         x3::A) where {T<:AbstractFloat, A<:AbstractArray{T,2}}                   
    n1, n2 = size(x1)
    @assert ((n1,n2) == size(x2)) && ((n1,n2) == size(x3)) 
        if parameter_type == "stokes"
        I = x1            # Stokes parameter I (total light intensity)
        Q = x2            # Stokes parameter Q
        U = x3            # Stokes parameter U
        Iu = Array{T}(undef, n1, n2)    # intensity of unpolarized light
        Ip = Array{T}(undef, n1, n2)    # intensity of linearly polarized light
        θ = Array{T}(undef, n1, n2) # angle of linearly polarized light
        @inbounds for i2 in 1:n2
            @simd for i1 in 1:n1
                Ip[i1,i2] = sqrt(Q[i1,i2]^2 + U[i1,i2]^2);
                Iu[i1,i2] = I[i1, i2]-Ip[i1,i2]; 
                θ[i1,i2] = atan(U[i1,i2], Q[i1,i2])/2;
            end
        end
    elseif parameter_type == "intensities"
        I = Array{T}(undef, n1, n2)    # Stokes parameter I (total light intensity)
        Q = Array{T}(undef, n1, n2)    # Stokes parameter Q
        U = Array{T}(undef, n1, n2)    # Stokes parameter U
        Iu = x1          # intensity of unpolarized light
        Ip = x2          # intensity of linearly polarized light
        θ = x3       # angle of linearly polarized light
         @inbounds for i2 in 1:n2
            @simd for i1 in 1:n1
                I[i1,i2] = Iu[i1,i2] + Ip[i1,i2];
                Q[i1,i2] = Ip[i1,i2] .*cos.(2*θ[i1,i2]);
                U[i1,i2] = Ip[i1,i2] .*sin.(2*θ[i1,i2]);
            end
        end
    elseif parameter_type == "mixed" 
        I = Array{T}(undef, n1, n2)     # Stokes parameter I (total light intensity
        Q = x2            # Stokes parameter Q
        U = x3            # Stokes parameter U
        Iu = x1          # intensity of unpolarized light     
        Ip = Array{T}(undef, n1, n2)    # intensity of linearly polarized light
        θ = Array{T}(undef, n1, n2) # angle of linearly polarized light
        @inbounds for i2 in 1:n2
            @simd for i1 in 1:n1
                Ip[i1,i2] = sqrt(Q[i1,i2]^2 + U[i1,i2]^2);
                I[i1,i2] = Iu[i1,i2] + Ip[i1,i2];
                θ[i1,i2] = atan(U[i1,i2], Q[i1,i2])/2;
            end
        end

    else
        error("unkown type, only known types : stokes, intensities and mixed")
    end
    PolarimetricMap{T}(parameter_type,I,Q,U,Iu,Ip,θ)
end

function PolarimetricMap(parameter_type::AbstractString, 
                         x::Array{T,3}) where {T<:AbstractFloat}
    n1, n2, n3 = size(x)
    @assert n3 == 3 
    PolarimetricMap(parameter_type, 
                    copy(x[:,:,1]),
                    copy(x[:,:,2]),
                    copy(x[:,:,3]));
end

function PolarimetricMap(parameter_type::AbstractString, n1::Int, n2::Int)
    return PolarimetricMap(parameter_type,
                           Array{Float64,2}(undef, n1, n2),
                           Array{Float64,2}(undef, n1, n2),
                           Array{Float64,2}(undef, n1, n2),
                           Array{Float64,2}(undef, n1, n2),
                           Array{Float64,2}(undef, n1, n2),
                           Array{Float64,2}(undef, n1, n2))
end

function (P::PolarimetricMap)(X::PolarimetricMap)
    P.I .= copy(X.I)
    P.Q .= copy(X.Q)
    P.U .= copy(X.U)
    P.Iu .= copy(X.Iu)
    P.Ip .= copy(X.Ip)
    P.θ .= copy(X.θ)
end


"""
    
""" rebuild
function rebuild(parameter_type::AbstractString,P::PolarimetricMap)
    if parameter_type == "stokes"
        P.Ip .= sqrt.(P.Q.^2 +P.U.^2)
        P.Iu .= P.I - P.Ip
        P.θ .= atan.(P.U,P.Q)/2          
    elseif parameter_type == "intensities"
        P.I .= P.Iu + P.Ip
        P.Q .= P.Ip .* cos.(2*P.θ)
        P.U .= P.Ip .* sin.(2*P.θ)
    elseif parameter_type == "mixed" 
        P.Ip .= sqrt.(P.Q.^2 +P.U.^2)
        P.I .= P.Iu + P.Ip
        P.θ .= atan.(P.U,P.Q)/2 
    else
        error("unkown type, only known types : stokes, intensities and mixed")
    end
    # (minimum(P.I) < 0) && @warn "Negative intensities in I"  
    # (minimum(P.Iu) < 0) && @warn "Negative intensities in Iu"  
    # (minimum(P.Ip) < 0) && @warn "Negative intensities in Ip"  
    return nothing
end


function (P::PolarimetricMap)(X::AbstractArray{T,3}) where {T<: AbstractFloat}
    @inbounds for (i,map) in enumerate(P)
        map .= copy(X[:,:,i])
    end
    rebuild(P.parameter_type,P)
end

#------------------------------------------------
# Base fonction redefinitions

Base.size(A::PolarimetricMap) = size(A.I)
Base.length(A::PolarimetricMap) = 3
Base.copy(X::PolarimetricMap{Float64}) = PolarimetricMap(X.parameter_type, 
                                                         copy(X.I), 
                                                         copy(X.Q), 
                                                         copy(X.U), 
                                                         copy(X.Iu), 
                                                         copy(X.Ip), 
                                                         copy(X.θ))

function Base.get(x::PolarimetricMap{Float64}, i::Int64)
    return eval(:($(x).$(MAPDICT[x.parameter_type][i])))
end

function Base.iterate(x::PolarimetricMap{Float64}, state=1)
    n = length(x)
    if state > length(x)
        return nothing
    else
        return (get(x,state),state+1)
    end
end

function get_stokes(x::PolarimetricMap{Float64})
    return [x.I,x.Q,x.U]
end

function get_mixed(x::PolarimetricMap{Float64})
    return [x.Iu,x.Q,x.U]
end

function get_intensities(x::PolarimetricMap{Float64})
    return [x.Iu,x.Ip,x.θ]
end

function Base.fill!(P::PolarimetricMap, r::Real)
    @inbounds for (i,map) in enumerate(P)
        map .= r
    end
    rebuild(P.parameter_type,P)
end

 function +(x::PolarimetricMap, y::Array{T,3}) where {T<:AbstractFloat} 
    @assert size(y)[1:2] == size(x)       
    if x.parameter_type == "stokes"
       I=x.I + view(y,:,:,1);
       Q=x.Q + view(y,:,:,2);
       U=x.U + view(y,:,:,3);
       return PolarimetricMap("stokes", I, Q, U)
    elseif x.parameter_type == "intensities"
       Iu=x.Iu + view(y,:,:,1);
       Ip=x.Ip + view(y,:,:,2);
       θ=x.θ + view(y,:,:,3);
       return PolarimetricMap("intensities", Iu, Ip, θ)
    elseif x.parameter_type == "mixed"
       Iu=x.Iu + view(y,:,:,1);
       Q=x.Q + view(y,:,:,2);
       U=x.U + view(y,:,:,3);
       return PolarimetricMap("mixed", Iu, Q, U)
    else
        error("unknown parameter type")
    end
 end
 
 +(y::Array{T,3}, x::PolarimetricMap) where {T<:AbstractFloat} = x + y
 -(x::PolarimetricMap, y::Array{T,3}) where {T<:AbstractFloat} = x + (-y)
 

function +(x::PolarimetricMap, y::PolarimetricMap) 
    if x.parameter_type != y.parameter_type
        @warn "x.parameter_type : "*x.parameter_type*" is different of y.parameter_type : "*y.parameter_type*". The result of the sum will be of parameter_type : "*x.parameter_type*"."
    end
    return x + convert(Array{Float64,3}, y, x.parameter_type)
end                  


 function -(x::PolarimetricMap, y::PolarimetricMap) 
    if x.parameter_type != y.parameter_type
        @warn "x.parameter_type : "*x.parameter_type*" is different of y.parameter_type : "*y.parameter_type*". The result of the sum will be of parameter_type : "*x.parameter_type*"."
    end
    return x - convert(Array{Float64,3}, y, x.parameter_type)
end
  
 vcopy(x::PolarimetricMap) = PolarimetricMap(x.parameter_type,
                                             x.I, 
                                             x.Q, 
                                             x.U,
                                             x.Iu,
                                             x.Ip, 
                                             x.θ)
 function vcreate(x::PolarimetricMap)
 
    @assert (x.parameter_type == "stokes") | 
            (x.parameter_type == "intensities") | 
            (x.parameter_type == "mixed")
     n1,n2=size(x);
     return PolarimetricMap(x.parameter_type, n1, n2)
 end     
                                    
 function convert(::Type{Array{T,3}}, x::PolarimetricMap{T}, parameter_type::AbstractString) where {T <:AbstractFloat}
     if parameter_type == "stokes"
       return cat(x.I, x.Q, x.U, dims=3)
    elseif parameter_type == "intensities"
       return cat(x.Iu, x.Ip, x.θ, dims=3)
    elseif parameter_type == "mixed"
       return cat(x.Iu, x.Q, x.U, dims=3)
    else
        error("unknown parameter type")
    end
 end
 
 function convert(::Type{Array{T,3}}, x::PolarimetricMap{T}) where {T <:AbstractFloat}
     convert(Array{T,3},x, x.parameter_type)
 end

#------------------------------------------------
# Writting function to save PolarimetricMap in fits file
"""
    write(X,'filename.fits') 
    
where X is a PolarimetricMap, write a fitsfile

"""

function write(X::PolarimetricMap, filename::AbstractString)
    data=cat(X.Iu', X.Ip', X.θ', X.I', X.Q', X.U',dims=3)
 
    writefits!(filename,
              ["MAPORDER" => "Iu, Ip, Theta, I, Q, U"],
              data)
end

"""
    read('parameter_type','filename.fits') -> PolarimetricMap
    
create an object of type PolarimetricMap from a fits file with:
    - Parameters I, Q, U (i.e. parameter_type = 'stokes')
    - Parameters Iu, Ip and θ (i.e. parameter_type = 'intensities')
    - Parameters Iu, Q, U (i.e. parameter_type = 'mixed')
   
"""


function read(parameter_type::AbstractString, filename::AbstractString)
    X=readfits(filename);
    return PolarimetricMap(parameter_type, 
                           view(X,:,:,4)', 
                           view(X,:,:,5)', 
                           view(X,:,:,6)', 
                           view(X,:,:,1)', 
                           view(X,:,:,2)', 
                           view(X,:,:,3)')
end

