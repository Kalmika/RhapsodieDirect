#
# utils.jl
#
# Provide tools to load the instrumental parameters and the 
# calibrated data, and for the calculus of the RHAPSODIE direct model. 
#
# ------------------------------------------------
#
# This file is part of RhapsodieDirect
#
#
# Copyright (c) 2025 Laurence Denneulin (see LICENCE.md)
#

#------------------------------------------------
"""
Return the a table of indices of size 4 by N where 4 
is the number of position of half wave plate (HWP) 
and N is the product of the number of half wave plate cycles 
and the number of frame per position of HWP in one cycle. 
For example, for a dataset with two cycles and 2 frames 
per position of HWP, returns:

1 2  9 10
3 4 11 12
5 6 13 14
7 8 15 16

It is usefull to generate default polarisation values 
and to apply the Double Difference and Double Ratio.
""" get_indices_table

function get_indices_table(d::DatasetParameters;
                           hwp_pos=4)	
	indices=zeros(hwp_pos,d.frames_per_hwp_pos*d.hwp_cycles)
	for i=1:hwp_pos
		ind=repeat(range(0,
		                 stop=hwp_pos*d.frames_per_hwp_pos*(d.hwp_cycles-1),
		                 length=d.hwp_cycles), inner=d.frames_per_hwp_pos) +         
		    (d.frames_per_hwp_pos*i .-mod.(range(1,
		                                         stop=d.frames_per_hwp_pos*d.hwp_cycles,
		                                         length=d.frames_per_hwp_pos*d.hwp_cycles),
		                                    d.frames_per_hwp_pos))
		indices[i,:]=ind
	end
	indices=Int64.(indices)
	return indices
end

"""
Set default polarisation coefficient for an ideal 
optical scheme composed with a half-wave-plate, 
oriented with an angle α, and an analyzer, 
oriented with an angle ψ. Default is ESO/VLT-SPHERE IRDIS configuration.
Indices can be computed using `get_indices_table`.
""" set_default_polarisation_coefficients
function set_default_polarisation_coefficients(indices::AbstractArray{Int64,2}; 
                                               α=[0, pi/4, pi/8, 3*pi/8], 
                                               ψ=[0,pi/2])
	J(a)=[cos(2*a) sin(2*a); sin(2*a) -cos(2*a)]
	P(a)=[cos(a), -sin(a)]
    v=Vector{NTuple{2, NTuple{3, Float64}}}(undef, length(indices));
	for i=1:4
		v1=J(α[i])*P(ψ[1])
		vn1=v1[1]*v1[1]+v1[2]*v1[2]
		v2=J(α[i])*P(ψ[2])
		vn2=v2[1]*v2[1]+v2[2]*v2[2]
		for k=1:length(indices[i,:])		
		v[indices[i,k]] =((round(vn1/2, digits=4), 
		                   round((abs(v1[1])^2-abs(v1[2])^2)/2, digits=4), 
		                   round(real(v1[1]*v1[2]), digits=4)),
		                  (round(vn2/2, digits=4), 
		                  round((abs(v2[1])^2-abs(v2[2])^2)/2, digits=4), 
		                  round(real(v2[1]*v2[2]), digits=4)));
        end
	end
	return v
end

"""
Returns an AffineTransform2D of either a rotation 
followed by a center shift or only the center shift 
if the rotation angle is 0.
""" field_transform
function field_transform(A::AffineTransform2D{T}, 
                         epsilon::NTuple{2,T}, 
                         angle::T, 
                         center::NTuple{2,T}, 
                         newcenter::NTuple{2,T}) where {T<:AbstractFloat}
    if angle !=0
        center_to_origin=translate(A,-center[1],-center[2])
        rotation=rotate(angle,center_to_origin)
        field_transformation =translate(newcenter[1]+epsilon[1], newcenter[2]+epsilon[2],rotation)
    else
        field_transformation =translate(A,epsilon[1]+newcenter[1]-center[1], epsilon[2]+newcenter[2]-center[2])
	end
	return field_transformation
end


function set_fft_operator(object_parameters::ObjectParameters,
                          psf_map::AbstractArray{T,2}, 
                          psf_center::AbstractArray{T,1};
                          ker = CatmullRomSpline(Float64, Flat)) where {T <: AbstractFloat}
	#FIXME: PSF map is transformed to the same size as the object maps : the center of the psf is translated to fit the new map center. It thus involves interpolations, which doesn't seems to be a good idea. 
	
	resized_psf_map=zeros(object_parameters.size);
    new_psf_center=floor.(object_parameters.size./2).+1
	Id = AffineTransform2D{Float64}()
	centering=translate(-(new_psf_center[1]-psf_center[1]), -(new_psf_center[2]-psf_center[2]), Id)

	LazyAlgebra.apply!(resized_psf_map, ker, centering, psf_map);
	resized_psf_map./=sum(resized_psf_map);
	F=FFTOperator(resized_psf_map)
	FFT=F\Diag(F*ifftshift(resized_psf_map)) .*F;
	
	return FFT, resized_psf_map
end


function compute_mean(x) 
    return sum(x) / length(x)
end

function compute_variance(x)
    μ = compute_mean(x)
    return sum((xi - μ)^2 for xi in x) / (length(x) - 1)
end

#TODO: Refactore Cropping and Padding operators using LazyAlgebra Cropping mapping.      

#=      

"""
Compute the bounding box size for a given AffineTransform2D.
""" bbox_size
function bbox_size(inp_dims::NTuple{2,Integer},
                  A::AffineTransform2D{Float64})
                    
	xmin=typemax(Float64);
	ymin=typemax(Float64);
	xmax=typemin(Float64);
	ymax=typemin(Float64);
	
	xmin_int=typemin(Int64);
	ymin_int=typemin(Int64);
	xmax_int=typemin(Int64);
	ymax_int=typemin(Int64);
	
	width=typemin(Float64);
	height=typemin(Float64);
	
	Ind=[repeat(1:inp_dims[1], inner=inp_dims[2]) repeat(1:inp_dims[2], outer=inp_dims[1])]
	for it=1:(inp_dims[1]*inp_dims[2])
		(x,y)=A(Ind[it,1], Ind[it,2]);
		xmin=min(x, xmin);
		ymin=min(y, ymin);
		xmax=max(x, xmax);
		ymax=max(y, ymax);
	end
	xmin_int=floor(Int64,xmin);
	ymin_int=floor(Int64,ymin);
	xmax_int=ceil(Int64,xmax);
	ymax_int=ceil(Int64,ymax);

	width=xmax_int-xmin_int;
	height=ymax_int-ymin_int;
    out_dims=(width, height);

	return(out_dims, (xmin_int, xmax_int, ymin_int, ymax_int))
end

function SetCropOperator()
    DSIZE=get_par().rows[1]
    MASK=ones(get_par().cols[1:2]);
    for k=1:length(Trans_Table)
        X11=Trans_Table[k][1](1,1);
        X12=Trans_Table[k][1](1,DSIZE);
        X13=Trans_Table[k][1](DSIZE, DSIZE);
        X14=Trans_Table[k][1](DSIZE, 1);
        
        X21=Trans_Table[k][2](1,1);
        X22=Trans_Table[k][2](1,DSIZE);
        X23=Trans_Table[k][2](DSIZE, DSIZE);
        X24=Trans_Table[k][2](DSIZE, 1);
        Mask1=zeros(get_par().cols[1:2]);
        Mask2=zeros(get_par().cols[1:2]);

        for i=1:get_par().cols[1]
	        for j=1:get_par().cols[2]	
	                H11 = ((i-X11[1])*(X12[1]-X11[1]) +(j-X11[2])*(X12[2]-X11[2]) >0)
	                H12 = ((i-X12[1])*(X13[1]-X12[1]) +(j-X12[2])*(X13[2]-X12[2]) >0)
	                H13 = ((i-X13[1])*(X14[1]-X13[1]) +(j-X13[2])*(X14[2]-X13[2]) >0)
	                H14 = ((i-X14[1])*(X11[1]-X14[1]) +(j-X14[2])*(X11[2]-X14[2]) >0)	
	                H21 = ((i-X21[1])*(X22[1]-X21[1]) +(j-X21[2])*(X22[2]-X21[2]) >0)
	                H22 = ((i-X22[1])*(X23[1]-X22[1]) +(j-X22[2])*(X23[2]-X22[2]) >0)
	                H23 = ((i-X23[1])*(X24[1]-X23[1]) +(j-X23[2])*(X24[2]-X23[2]) >0)
	                H24 = ((i-X24[1])*(X21[1]-X24[1]) +(j-X24[2])*(X21[2]-X24[2]) >0)	
		        if H11 && H12 && H13 && H14
		            Mask1[i,j,:] .=1;
		            #MASK[i,j,:] .=1;
		        end
		        if H21 && H22 && H23 && H24
		            Mask2[i,j,:] .=1;
		            #MASK[i,j,:] .=1;
		        end
	        end
        end
    MASK .*= Mask1.*Mask2
    end
    #write(FitsFile, "MASK.fits", MASK, overwrite=true)
    push!(MASK_save, MASK)
end     

        
function crop(X::M)  where {T<:AbstractFloat, M<:AbstractArray{T,2}}
    #@assert size(X) .==   get_par().cols
    xymin_1=findfirst(get_MASK() .!= 0.)
    xymin_2=findfirst(get_MASK()' .!= 0.)
    xymax_1=findlast(get_MASK() .!= 0.)
    xymax_2=findlast(get_MASK()' .!= 0.)
    
    xmin=min(xymin_1[1], xymin_2[2]);
    ymin=min(xymin_1[2], xymin_2[1]);
    xmax=max(xymax_1[1], xymax_2[2]);
    ymax=max(xymax_1[2], xymax_2[1]);
    
    Y=X[xmin:xmax, ymin:ymax];
    Y[.!isfinite.(Y)].=0    
    return Y     
end        

function crop!(X::M)  where {T<:AbstractFloat, M<:AbstractArray{T,2}}
    #@assert size(X) .==   get_par().cols 
    X[.!isfinite.(X)].=0   
    X .*= get_MASK()        
end        
        
function crop(X::M)  where {T<:AbstractFloat, M<:AbstractArray{T,3}}
    #@assert size(X) .==   get_par().cols
    Y=copy(X);
    return crop!(Y)    
end        

function crop!(X::M)  where {T<:AbstractFloat, M<:AbstractArray{T,3}}
    #@assert size(X) .==   get_par().cols
    for k in size(X)[3] 
        crop!(view(X,:,:,k))
    end     
end        
   
                
function crop(X::PolarimetricMap{T}) where {T<:AbstractFloat}  
    #@assert size(X) .==   get_par().cols
    return PolarimetricMap(X.parameter_type,
                           crop(view(X.I,:,:)),
                           crop(view(X.Q,:,:)),
                           crop(view(X.U,:,:)),
                           crop(view(X.Iu,:,:)),
                           crop(view(X.Ip,:,:)),        
                           crop(view(X.θ,:,:)))   
end        

function crop!(X::PolarimetricMap{T})  where {T<:AbstractFloat}
        crop!(view(X.I,:,:));
        crop!(view(X.Q,:,:));
        crop!(view(X.U,:,:));
        crop!(view(X.Iu,:,:));
        crop!(view(X.Ip,:,:));        
        crop!(view(X.θ,:,:));
end        
        
function pad(X::M)  where {T<:AbstractFloat, M<:AbstractArray{T,2}}
#    X_size = size(X);
#    center_diff = X_size./2 .- get_par().cols[1:2]./2;
#    ker = LinearInterpolators.CatmullRomSpline(Float64, LinearInterpolators.Flat)    
#    Id = AffineTransform2D{Float64}()
#    center_change = translate(center_diff[1],center_diff[2], Id)   
#    PAD=TwoDimensionalTransformInterpolator(get_par().cols[1:2], X_size, ker, ker, center_change)
    PAD = ZeroPaddingOperator(get_par().cols[1:2], size(X)) 
    return PAD*X;
end         
        
function pad(X::PolarimetricMap{T}) where {T<:AbstractFloat}  
    #@assert size(X) .==   get_par().cols
    return PolarimetricMap(X.parameter_type,
                           pad(view(X.I,:,:)),
                           pad(view(X.Q,:,:)),
                           pad(view(X.U,:,:)),
                           pad(view(X.Iu,:,:)),
                           pad(view(X.Ip,:,:)),        
                           pad(view(X.θ,:,:)))   
end    


function check_MSE(model, data, weights)
	MSE = vdot(data-model, weights.*(data-model)) ;
	N=count(weights .> 0);
	println("MSE=$MSE, N=$N, MSE/N=$(MSE/N)");
end
=#


