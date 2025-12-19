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

function vcreate(::Type{LazyAlgebra.Inverse}, A::DirectModel{T},
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

# INVERSE DU MODÈLE DIRECT (uniquement si A = Id, sans convolution)
# Stratégie: Inverse = A_pseudo_inv × Adjoint(données)
# Processus: Données → Adjoint → Correction par A_pseudo_inv → (I, Q, U) exacts
function apply!(α::Real,
                ::Type{LazyAlgebra.Inverse},
                R::DirectModel{T},
                src::AbstractArray{T,3},        # Données observées (entrée)
                scratch::Bool,
                β::Real,
                dst::PolarimetricMap{T}) where {T<:AbstractFloat}  # Carte polarimétrique reconstruite (sortie)
    @assert β==0 && α==1
    @assert size(src) == R.rows
    @assert size(dst) == R.cols
    
    # Vérification: l'inverse n'existe que si A = Id (pas de convolution)
    @assert R.A == LazyAlgebra.Id "Inverse operation requires A = Id (no blur/convolution)"
    
    # Étape 1: Calculer l'adjoint dans un buffer temporaire
    # Créer une copie temporaire de dst pour stocker le résultat de l'adjoint
    adjoint_result = PolarimetricMap{T}(R.parameter_type,
                                        zeros(T, R.cols),
                                        zeros(T, R.cols),
                                        zeros(T, R.cols),
                                        zeros(T, R.cols),
                                        zeros(T, R.cols),
                                        zeros(T, R.cols))
    
    # Appliquer l'adjoint (réutilisation du code existant)
    apply!(α, LazyAlgebra.Adjoint, R, src, scratch, β, adjoint_result)
    
    # Étape 2: Correction par A_pseudo_inv (3x3) sur le vecteur Stokes adjoint (3x1)
    # Pour chaque pixel: [I, Q, U]_inverse = (A' A)^{-1} × [I, Q, U]_adjoint
    n_stokes = length(dst)  # attendu 3
    @inbounds for i=1:R.cols[1]
        for j=1:R.cols[2]
            stokes_adjoint = get_stokes(adjoint_result)
            # Construire le vecteur 3x1 à partir de l'adjoint (sans StaticArrays)
            adjoint_vec3 = Vector{T}(undef, 3)
            adjoint_vec3[1] = stokes_adjoint[1][i, j]
            adjoint_vec3[2] = stokes_adjoint[2][i, j]
            adjoint_vec3[3] = stokes_adjoint[3][i, j]
            # Inversion via la matrice 3x3
            stokes_inverse = R.A_pseudo_inv * adjoint_vec3
            # Stocker le résultat
            stokes_dst = get_stokes(dst)
            stokes_dst[1][i, j] = stokes_inverse[1]
            stokes_dst[2][i, j] = stokes_inverse[2]
            stokes_dst[3][i, j] = stokes_inverse[3]
        end
    end
    
    rebuild("stokes", dst)
    return dst
end

# @inbounds for k=1:length(R.TR)	 
#     R.TR[k] \ view(src,:,:,k) # inverse
#     vupdate!(x,1.,y)  # Accumulation des contributions
# end



# # INVERSE DU MODÈLE DIRECT (SANS CONVOLUTION)
# # Utilisé pour retrouver les paramètres de Stokes (I, Q, U) à partir des mesures

# """
#     compute_polarization_inverse_matrix(v_l::NTuple{3,T}, v_r::NTuple{3,T}) where {T<:AbstractFloat}

# Calcule la matrice pseudo-inverse pour retrouver (I, Q, U) à partir de deux mesures.

# # Arguments
# - `v_l`: coefficients (I, Q, U) pour l'analyseur gauche (ψ=0)
# - `v_r`: coefficients (I, Q, U) pour l'analyseur droit (ψ=π/2)

# # Retourne
# - `pol_pinv`: matrice 3×2 qui satisfait [I; Q; U] = pol_pinv * [mesure_gauche; mesure_droite]

# # Description
# Pour chaque configuration HWP, on a deux mesures (une par analyseur) liées aux Stokes par:
# ```
# mesure_gauche = v_l[1]*I + v_l[2]*Q + v_l[3]*U
# mesure_droite = v_r[1]*I + v_r[2]*Q + v_r[3]*U
# ```
# Cette fonction calcule la pseudo-inverse pour résoudre ce système 2×3 surdéterminé.
# Elle utilise la résolution du système des moindres carrés via l'opérateur backslash.
# """
# function compute_polarization_inverse_matrix(v_l::NTuple{3,T}, v_r::NTuple{3,T}) where {T<:AbstractFloat}
#     # Matrice 2×3: chaque ligne contient les coefficients (I, Q, U) d'un analyseur
#     pol_matrix = [v_l[1] v_l[2] v_l[3];
#                   v_r[1] v_r[2] v_r[3]]
    
#     # Calculer la pseudo-inverse en résolvant (A' * A) * X = A' pour chaque colonne
#     # C'est équivalent à pinv(A) = (A' * A) \ A'
#     # L'opérateur \ résout le système de moindres carrés de manière robuste
#     AtA = pol_matrix' * pol_matrix  # 3×3
#     At = pol_matrix'                 # 3×2
    
#     # Résoudre le système colonne par colonne
#     pol_pinv = zeros(T, 3, 2)
#     for j in 1:2
#         pol_pinv[:, j] = AtA \ At[:, j]  # Résolution robuste
#     end
    
#     return pol_pinv
# end

# """
#     apply_direct_model_inverse_no_blur(data::AbstractArray{T,3}, direct_model::DirectModel{T}) where {T<:AbstractFloat}

# Applique l'inverse du modèle direct SANS convolution (A = Id).

# # Arguments
# - `data`: données observées de taille (hauteur, largeur, n_frames)
# - `direct_model`: modèle direct contenant les transformations géométriques et coefficients de polarisation

# # Retourne
# - `stokes_buffer`: array (hauteur, largeur, 3) contenant (I, Q, U) reconstruits

# # Description
# Pour chaque frame:
# 1. Inverse les transformations géométriques (rotation, translation, interpolation)
# 2. Inverse les coefficients de polarisation via pseudo-inverse
# 3. Reconstruit les paramètres de Stokes (I, Q, U)

# Cette fonction suppose que A = Id (pas de convolution/flou).
# """
# function apply_direct_model_inverse_no_blur(data::AbstractArray{T,3}, direct_model::DirectModel{T}) where {T<:AbstractFloat}
#     @assert size(data) == direct_model.rows "Data size mismatch: expected $(direct_model.rows), got $(size(data))"
    
#     n_stokes = 3
#     stokes_buffer = zeros(T, (direct_model.cols[1], 
#                               direct_model.cols[2], 
#                               n_stokes))
#     n = direct_model.rows[2]
#     @assert iseven(n) "Image width must be even (left/right split)"
    
#     # Pré-calculer les matrices inverses pour chaque frame (optimisation)
#     pol_inverses = [compute_polarization_inverse_matrix(TR.v_l, TR.v_r) 
#                     for TR in direct_model.TR]
    
#     @inbounds for k = 1:length(direct_model.TR)
#         TR = direct_model.TR[k]
#         pol_pinv = pol_inverses[k]
        
#         # Étape 1: Inverser les transformations géométriques
#         y_l = zeros(T, TR.cols[1:2])
#         vmul!(y_l, TR.H_l', view(data, :, 1:(n÷2), k))  # Côté gauche (analyseur 1)
        
#         y_r = zeros(T, TR.cols[1:2])
#         vmul!(y_r, TR.H_r', view(data, :, (n÷2)+1:n, k))  # Côté droit (analyseur 2)
        
#         # Étape 2: Inverser les coefficients de polarisation pixel par pixel
#         for i = 1:size(stokes_buffer, 1)
#             for j = 1:size(stokes_buffer, 2)
#                 # Vecteur des 2 mesures (une par analyseur)
#                 measurements = [y_l[i, j], y_r[i, j]]
                
#                 # Retrouve (I, Q, U) via la pseudo-inverse
#                 stokes_vec = pol_pinv * measurements
                
#                 # Stocke dans le buffer (accumulation si plusieurs frames)
#                 stokes_buffer[i, j, 1] += stokes_vec[1]  # I
#                 stokes_buffer[i, j, 2] += stokes_vec[2]  # Q
#                 stokes_buffer[i, j, 3] += stokes_vec[3]  # U
#             end
#         end
#     end
    
#     # Moyenne sur les frames
#     stokes_buffer ./= length(direct_model.TR)
    
#     return stokes_buffer
# end