# examples/test_noise_models.jl
using RhapsodieDirect
using Plots 

using Pkg
Pkg.activate(".") 

# Include optional validation functions that need Plots
include("../src/noise_validation.jl")

# Quick noise comparison demo
function demo_noise_models()
    # Create the models
    diag_model = create_noise_model(:diagonal)
    corr_model = create_noise_model(:correlated, A=2.0, σ²=5.0, N=128)
    # Generate noise
    signal = zeros(128, 128)
    weights = ones(128, 128)
    noise_diag = generate_noise(diag_model, signal, weights)
    noise_corr = RhapsodieDirect.generate_correlated_noise(corr_model)
    # Plot
    p1 = heatmap(noise_diag, title="Diagonal Noise")
    p2 = heatmap(noise_corr, title="Correlated Noise") 
    plot(p1, p2, layout=(1,2), size=(800,300))
end

# Convergence test for noise model validation
function demo_validate_noise_model()
    # Create the correlated model
    model = create_noise_model(:correlated, A=2.0, σ²=5.0, N=128)
    # Use validation with plotting capability
    validate_noise_model(model, n_samples=200, σ²_values=logrange(0.1, 50, 30))
end

demo_noise_models()
demo_validate_noise_model()