function load_field_transforms(object::ObjectParameters,
                           data::DatasetParameters,
                           parameters::Vector{FieldTransformParameters})
    @assert data.frames_total == length(parameters)
                           
    Id = AffineTransform2D{Float64}()
    field_transforms=Vector{FieldTransformOperator{Float64}}()
    
    for k=1:data.frames_total
        T_left=field_transform(Id, 
                               parameters[k].translation_left, 
                               parameters[k].field_angle, 
                               object.center,data.center) 
        T_right=field_transform(Id, 
                               parameters[k].translation_right, 
                               parameters[k].field_angle, 
                               object.center,data.center)   

                                   
        output_size=(data.size[1], data.size[2]÷2)
        input_size= object.size
    	T1=TwoDimensionalTransformInterpolator(output_size, 
    	                                       input_size, 
    	                                       parameters[k].ker, 
    	                                       parameters[k].ker, 
    	                                       inv(T_left))
    	T2=TwoDimensionalTransformInterpolator(output_size, 
    	                                       input_size, 
    	                                       parameters[k].ker, 
    	                                       parameters[k].ker, 
    	                                       inv(T_right))
    	
	    push!(field_transforms, FieldTransformOperator((object.size[1],object.size[2],3), 
	                                                     data.size, 
	                                                     parameters[k].polarization_left,
	                                                     parameters[k].polarization_right,
	                                                     T1,
	                                                     T2));    

    end
    #TODO: Add bounding_box and mask calculus (cf. bbox_size and SetCropOperator) and export it somehow
    return field_transforms
end 

function load_identity_field_transforms(object::ObjectParameters,
                                      data::DatasetParameters)
    
    field_transforms = Vector{FieldTransformOperator{Float64}}()
    
    for k = 1:data.frames_total
        # Coefficients de polarisation identité (matrice identité 3x3)
        polarization_identity = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        
        # Créer un FieldTransformOperator identité sans interpolation
        # En passant `nothing` pour T1 et T2, cela devrait utiliser l'identité
        push!(field_transforms, FieldTransformOperator((object.size[1], object.size[2], 3),
                                                        data.size,
                                                        polarization_identity,
                                                        polarization_identity,
                                                        LazyAlgebra.Id,  # Identity mapping
                                                        LazyAlgebra.Id)) 
    end
    
    return field_transforms
end

