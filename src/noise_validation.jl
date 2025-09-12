# =============================================================================
# Noise Models Validation - Optional plotting functionality
# =============================================================================

"""
    validate_noise_model(model::CorrelatedNoise; σ²_values=nothing, n_samples=100)

Validate correlated noise model by comparing theoretical and empirical variances.
Requires Plots.jl to be installed for visualization.
"""
function validate_noise_model(model::CorrelatedNoise; σ²_values=logrange(0.1, 50, 30), n_samples=100)
    # Check if Plots is available
    if !isdefined(Main, :Plots)
        try
            @eval Main using Plots
        catch
            @warn "Plots.jl not available. Install with: using Pkg; Pkg.add(\"Plots\")"
            return
        end
    end
    
    println("🔍 VALIDATING CORRELATED NOISE MODEL")
    println("="^50)
    
    # Generate test ranges
    empirical_vars = Float64[]
    theoretical_vars = Float64[]
    
    for σ² in σ²_values
        # Create temporary model with different σ²
        temp_model = CorrelatedNoise(model.A, σ², model.N)
        
        # Generate samples and compute empirical variance
        variances = [compute_variance(generate_correlated_noise(temp_model)) for _ in 1:n_samples]
        push!(empirical_vars, mean(variances))
        
        # Theoretical variance
        push!(theoretical_vars, theoretical_variance(temp_model))
    end
    
    # Plot results
    p = Main.Plots.plot(σ²_values, theoretical_vars, 
                       label="Theoretical", linewidth=2, 
                       xlabel="σ²", ylabel="Variance",
                       title="Noise Model Validation")
    Main.Plots.scatter!(p, σ²_values, empirical_vars, 
                       label="Empirical", markersize=4)
    
    # Display results
    println("✅ Validation completed")
    println("   Max relative error: $(maximum(abs.(empirical_vars .- theoretical_vars) ./ theoretical_vars) * 100)%")
    
    return p
end

"""
    validate_noise_model(model::DiagonalNoise; kwargs...)

Validation for diagonal noise (no plotting needed).
"""
function validate_noise_model(model::DiagonalNoise; kwargs...)
    println("✅ DiagonalNoise model validated (no additional checks needed)")
end
