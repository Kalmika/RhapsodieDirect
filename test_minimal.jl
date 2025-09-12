#!/usr/bin/env julia

# Test minimal pour diagnostiquer le problème
println("🔍 Diagnostic RhapsodieDirect")

try
    println("1. Activating environment...")
    import Pkg
    Pkg.activate(".")
    
    println("2. Loading RhapsodieDirect...")
    using RhapsodieDirect
    
    println("3. Testing noise model creation...")
    diag_model = create_noise_model(:diagonal)
    corr_model = create_noise_model(:correlated, A=1.0, σ²=2.0, N=32)
    
    println("4. Testing noise generation...")
    signal = randn(32, 32)
    weights = ones(32, 32)
    
    noise_diag = generate_noise(diag_model, signal, weights)
    noise_corr = generate_noise(corr_model, signal)
    
    println("✅ All basic functions work!")
    println("   Diagonal noise size: $(size(noise_diag))")
    println("   Correlated noise size: $(size(noise_corr))")
    
catch e
    println("❌ Error: $e")
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("🎯 Diagnostic complete")
