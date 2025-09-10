# FieldTransformOperator mappings

function vcreate(::Type{LazyAlgebra.Direct}, A::FieldTransformOperator{T},
                 x::AbstractArray{T,3}, scratch::Bool = false) where {T <: AbstractFloat}
    @assert !Base.has_offset_axes(x)
    @assert size(x) == A.cols
    Array{T,2}(undef, A.rows)
end

function vcreate(::Type{LazyAlgebra.Adjoint}, A::FieldTransformOperator{T},
                 x::AbstractArray{T,2}, scratch::Bool = false) where {T <: AbstractFloat}
    @assert !Base.has_offset_axes(x)
    @assert size(x) == A.rows
    Array{T,3}(undef, A.cols)
end


#FIXME : refactor apply! to make α, β usefull (technically dst = α.R*src + β*dst)

function apply!(α::Real,
                ::Type{LazyAlgebra.Direct},
                R::FieldTransformOperator{T},
                src::AbstractArray{T,3},
                scratch::Bool,
                β::Real,
                dst::AbstractArray{T,2}) where {T<:AbstractFloat}
    @assert β==0 && α==1
    @assert size(src) == R.cols
    @assert size(dst) == R.rows
    n = R.rows[2]
    @assert iseven(n)
    fill!(dst,zero(T));
    # Allocating memory FIXME: find a way to calculate fully in place 
    z = zeros(R.cols[1:2]);
    
    #Compute left direct model
    @simd for i=1:length(R.v_l)
        vupdate!(z, R.v_l[i],view(src,:,:,i)) 
    end
    apply!(view(dst,:, 1:(n÷2)),R.H_l,z);
    
    # Reset the array values to 0. (faster than allocating two different arrays)
    vfill!(z,0.)
    
    # Compute right direct model
    @simd for i=1:length(R.v_r)
        vupdate!(z, R.v_r[i],view(src,:,:,i)) 
    end
    apply!(view(dst,:, (n÷2)+1:n),R.H_r,z);
    return dst
end

function apply!(α::Real,
                ::Type{LazyAlgebra.Adjoint},
                R::FieldTransformOperator{T},
                src::AbstractArray{T,2},
                scratch::Bool,
                β::Real,
                dst::AbstractArray{T,3}) where {T<:AbstractFloat}
    @assert β==0 && α==1
    @assert size(src) == R.rows
    @assert size(dst) == R.cols
    n = R.rows[2]
    @assert iseven(n)
    fill!(dst,zero(T));
  
    y = zeros(R.cols[1:2])
    vmul!(y, R.H_l', view(src, :, 1:(n÷2)))
    @simd for i=1:length(R.v_l)
         vupdate!(view(dst,:,:,i), R.v_l[i], y)
    end
    vmul!(y, R.H_r', view(src, :, (n÷2)+1:n))
    @simd for i=1:length(R.v_r)
         vupdate!(view(dst,:,:,i), R.v_r[i], y)
    end

    return dst;
end

# DirectModel mapping

function vcreate(::Type{LazyAlgebra.Direct}, A::DirectModel{T},
                 x::PolarimetricMap{T}, scratch::Bool = false) where {T <: AbstractFloat}
    @assert !Base.has_offset_axes(x)
    @assert size(x) == A.cols
    Array{T,3}(undef, A.rows)
end

function vcreate(::Type{LazyAlgebra.Adjoint}, A::DirectModel{T},
                 x::AbstractArray{T,3}, scratch::Bool = false) where {T <: AbstractFloat}
    @assert !Base.has_offset_axes(x)
    @assert size(x) == A.rows
    PolarimetricMap{T}(A.parameter_type,Array{T,2}(undef, A.cols),
                                        Array{T,2}(undef, A.cols),
                                        Array{T,2}(undef, A.cols),
                                        Array{T,2}(undef, A.cols),
                                        Array{T,2}(undef, A.cols),
                                        Array{T,2}(undef, A.cols))
end


#FIXME : refactor apply! to make α, β usefull (technically dst = α.R*src + β*dst)

function apply!(α::Real,
                ::Type{LazyAlgebra.Direct},
                R::DirectModel{T},
                src::PolarimetricMap{T},
                scratch::Bool,
                β::Real,
                dst::AbstractArray{T,3}) where {T<:AbstractFloat}
    @assert β==0 && α==1
    @assert size(src) == R.cols          # Vérification taille objet
    @assert size(dst) == R.rows          # Vérification taille données
    
    # Étape 1: Application du flou (PSF/convolution) à chaque paramètre de Stokes
    x = zeros(R.cols[1],R.cols[2], length(src))  # Buffer temporaire
    @inbounds for (i,map) in enumerate(get_stokes(src))
        setindex!(x,R.A*map,:,:,i)       # x[:,:,i] = R.A * map (convolution par la PSF)
    end
    
    # Étape 2: Application des transformations géométriques par frame
    @inbounds for k=1:length(R.TR)       # Pour chaque observation/frame
        apply!(view(dst,:,:,k),R.TR[k],x)  # Transformation géométrique + polarisation
    end
    return dst
end

# ADJOINT DU MODÈLE DIRECT : Données observées → PolarimetricMap (rétroprojection)
# Processus inverse : Données → Rétroprojection géométrique → Déconvolution → Objet reconstruit
function apply!(α::Real,
                ::Type{LazyAlgebra.Adjoint},
                R::DirectModel{T},
                src::AbstractArray{T,3},        # Données observées (entrée)
                scratch::Bool,
                β::Real,
                dst::PolarimetricMap{T}) where {T<:AbstractFloat}  # Carte polarimétrique reconstruite (sortie)
    @assert β==0 && α==1
    @assert size(src) == R.rows
    @assert size(dst) == R.cols
    x = zeros(R.cols[1],R.cols[2], length(dst))  # Accumulation finale
    y = zeros(R.cols[1],R.cols[2], length(dst))  # Buffer temporaire
    
    # Étape 1: Rétroprojection géométrique de toutes les observations
    @inbounds for k=1:length(R.TR)	 
        vmul!(y, R.TR[k]', view(src,:,:,k)) # Adjoint des transformations géométriques
        vupdate!(x,1.,y)  # Accumulation des contributions
	end
    
    # Étape 2: Déconvolution (adjoint de l'opérateur de flou) pour chaque Stokes
    @inbounds for (i,map) in enumerate(get_stokes(dst))
        apply!(map,R.A', x[:,:,i]) # Déconvolution par l'adjoint de la PSF
    end
    rebuild("stokes",dst)
    return dst;
end

