"""
    chi_square!(x,g,D)
    
    Compute the chi square value (D.d - D.H*x)'.*D.w.*(D.d - D.H*x) where D(d,w,H) is a `Dataset` structure and (x,g) can either be PolarimetricMaps or Arrays.
""" chi_square!
function chi_square!(x::PolarimetricMap{T},
                     g::PolarimetricMap{T},
                     D::Dataset) where {T<:AbstractFloat}
    
    fill!(g,zero(T))
    res = D.direct_model*x - D.data;
    wres = D.weights_op .* res
    apply!(g, D.direct_model', wres)
    
    return vdot(res,wres)
end
    

function chi_square!(x::AbstractArray{T,3},
                     g::AbstractArray{T,3},
                     D::Dataset) where {T<:AbstractFloat}  

    fill!(g,zero(T))
    X=PolarimetricMap("stokes",x)
    res = D.direct_model*X - D.data;
    wres = D.weights_op .* res
    rwr = D.direct_model' * wres
    g .+= cat(rwr.I, rwr.Q, rwr.U, dims=3)
    return vdot(res,wres)
end    
