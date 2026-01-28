# =============================================================================
# DEBUG SCRIPT: Compare apply_precision components between Python and Julia
# =============================================================================
# Run this in Julia to get intermediate values, then compare with Python
#
# Usage:
#   include("test_precision_debug.jl")
#   debug_precision_components(dataset, residual)
# =============================================================================

using RhapsodieDirect
using LinearAlgebra
using Statistics

"""
    debug_precision_components(dataset, residual; additive_variance=1.0)

Test each component of apply_precision to find discrepancies with Python.
Returns a dictionary with intermediate values for comparison.
"""
function debug_precision_components(dataset, residual; additive_variance=1.0)
    println("=" ^ 60)
    println("DEBUG: Testing apply_precision components")
    println("=" ^ 60)
    
    noise_model = dataset.noise_model
    direct_model = dataset.direct_model
    
    H, W2, C = size(residual)
    W = div(W2, 2)
    
    results = Dict{String, Any}()
    
    # =========================================================================
    # 1. TEST: Input statistics
    # =========================================================================
    println("\n[1] INPUT STATISTICS:")
    println("  residual shape: $(size(residual))")
    println("  residual sum: $(sum(residual))")
    println("  residual mean: $(mean(residual))")
    println("  residual min/max: $(minimum(residual)) / $(maximum(residual))")
    println("  residual[1,1,1]: $(residual[1,1,1])")
    println("  residual[64,128,2]: $(residual[min(64,H), min(128,W2), min(2,C)])")
    results["input_sum"] = sum(residual)
    results["input_mean"] = mean(residual)
    flush(stdout)
    
    # =========================================================================
    # 2. TEST: Weights statistics
    # =========================================================================
    println("\n[2] WEIGHTS STATISTICS:")
    weights = noise_model.diag_noise.weights
    if weights !== nothing
        println("  weights shape: $(size(weights))")
        println("  weights sum: $(sum(weights))")
        println("  weights mean: $(mean(weights))")
        println("  1/weights mean (variance): $(mean(1.0 ./ weights))")
        results["weights_sum"] = sum(weights)
        results["mean_variance"] = mean(1.0 ./ weights)
    else
        println("  weights: nothing")
    end
    flush(stdout)
    
    # =========================================================================
    # 3. TEST: Correlated noise kernel
    # =========================================================================
    println("\n[3] CORRELATED NOISE KERNEL:")
    corr = noise_model.corr_noise
    println("  P_double shape: $(size(corr.P_double))")
    println("  P_double sum: $(sum(corr.P_double))")
    println("  P_double[1,1]: $(corr.P_double[1,1])")
    println("  P_double max: $(maximum(abs.(corr.P_double)))")
    results["P_double_sum"] = sum(corr.P_double)
    flush(stdout)
    
    # =========================================================================
    # 4. TEST: apply_covariance (diagonal part)
    # =========================================================================
    println("\n[4] TEST apply_covariance (diagonal):")
    diag_cov = RhapsodieDirect.apply_covariance(noise_model.diag_noise, residual, direct_model)
    println("  diag_cov sum: $(sum(diag_cov))")
    println("  diag_cov mean: $(mean(diag_cov))")
    results["diag_cov_sum"] = sum(diag_cov)
    flush(stdout)
    
    # =========================================================================
    # 5. TEST: apply_covariance (correlated part)
    # =========================================================================
    println("\n[5] TEST apply_covariance (correlated):")
    corr_cov = RhapsodieDirect.apply_covariance(noise_model.corr_noise, residual, direct_model)
    println("  corr_cov sum: $(sum(corr_cov))")
    println("  corr_cov mean: $(mean(corr_cov))")
    println("  corr_cov min/max: $(minimum(corr_cov)) / $(maximum(corr_cov))")
    results["corr_cov_sum"] = sum(corr_cov)
    flush(stdout)
    
    # =========================================================================
    # 6. TEST: apply_covariance (combined)
    # =========================================================================
    println("\n[6] TEST apply_covariance (combined = diag + corr):")
    combined_cov = RhapsodieDirect.apply_covariance(noise_model, residual, direct_model)
    println("  combined_cov sum: $(sum(combined_cov))")
    println("  combined_cov mean: $(mean(combined_cov))")
    results["combined_cov_sum"] = sum(combined_cov)
    flush(stdout)
    
    # =========================================================================
    # 7. TEST: Full operator A = C + λI
    # =========================================================================
    println("\n[7] TEST operator A = C + λI (with additive_variance=$additive_variance):")
    A_result = combined_cov .+ additive_variance .* residual
    println("  A(residual) sum: $(sum(A_result))")
    println("  A(residual) mean: $(mean(A_result))")
    results["A_result_sum"] = sum(A_result)
    flush(stdout)
    
    # =========================================================================
    # 8. TEST: Preconditioner M^{-1}
    # =========================================================================
    println("\n[8] TEST preconditioner M^{-1}:")
    total_variance = mean(1.0 ./ weights) + additive_variance
    println("  total_variance for preconditioner: $total_variance")
    
    M_inv_result = RhapsodieDirect.apply_special_transform_inverse_covariance(
        residual, 
        corr.P_double, 
        total_variance, 
        direct_model
    )
    println("  M^{-1}(residual) sum: $(sum(M_inv_result))")
    println("  M^{-1}(residual) mean: $(mean(M_inv_result))")
    println("  M^{-1}(residual) min/max: $(minimum(M_inv_result)) / $(maximum(M_inv_result))")
    results["M_inv_sum"] = sum(M_inv_result)
    flush(stdout)
    
    # =========================================================================
    # 9. TEST: Symmetry check for A
    # =========================================================================
    println("\n[9] SYMMETRY CHECK:")
    # Create random test vectors
    v1 = randn(size(residual))
    v2 = randn(size(residual))
    
    apply_A = function(x)
        cov = RhapsodieDirect.apply_covariance(noise_model, x, direct_model)
        return cov .+ additive_variance .* x
    end
    
    Av1 = apply_A(v1)
    Av2 = apply_A(v2)
    
    dot1 = sum(v2 .* Av1)  # <v2, A*v1>
    dot2 = sum(v1 .* Av2)  # <v1, A*v2>
    
    println("  <v2, A*v1> = $dot1")
    println("  <v1, A*v2> = $dot2")
    println("  Difference: $(abs(dot1 - dot2))")
    println("  Relative diff: $(abs(dot1 - dot2) / max(abs(dot1), abs(dot2)))")
    
    if abs(dot1 - dot2) / max(abs(dot1), abs(dot2)) > 1e-6
        println("  ⚠️  WARNING: Operator A may not be symmetric!")
    else
        println("  ✓ Operator A appears symmetric")
    end
    results["symmetry_diff"] = abs(dot1 - dot2)
    flush(stdout)
    
    # =========================================================================
    # 10. TEST: Positive definiteness check
    # =========================================================================
    println("\n[10] POSITIVE DEFINITENESS CHECK:")
    Av1 = apply_A(v1)
    inner_prod = sum(v1 .* Av1)  # <v1, A*v1>
    println("  <v1, A*v1> = $inner_prod")
    if inner_prod > 0
        println("  ✓ Positive for this test vector")
    else
        println("  ⚠️  WARNING: <v, Av> <= 0, operator may not be positive definite!")
    end
    flush(stdout)
    
    # =========================================================================
    # 11. TEST: Preconditioner quality M^{-1} * A ≈ I
    # =========================================================================
    println("\n[11] PRECONDITIONER QUALITY TEST:")
    apply_M_inv = function(x)
        return RhapsodieDirect.apply_special_transform_inverse_covariance(
            x, corr.P_double, total_variance, direct_model
        )
    end
    
    # M^{-1} * A * v should be close to v if M ≈ A
    MAv = apply_M_inv(Av1)
    relative_error = norm(MAv .- v1) / norm(v1)
    println("  ||M^{-1}*A*v - v|| / ||v|| = $relative_error")
    if relative_error < 0.5
        println("  ✓ Preconditioner is reasonable")
    else
        println("  ⚠️  WARNING: Preconditioner may be poor (error > 0.5)")
    end
    results["precond_error"] = relative_error
    flush(stdout)
    
    println("\n" * "=" ^ 60)
    println("DEBUG COMPLETE - Compare these values with Python")
    println("=" ^ 60)
    
    return results
end

"""
    test_single_frame(dataset, frame_idx=1)

Test apply_covariance on a single frame to isolate issues.
"""
function test_single_frame(dataset, frame_idx=1)
    noise_model = dataset.noise_model
    direct_model = dataset.direct_model
    
    H, W2, C = size(dataset.data)
    
    # Create test input with only one non-zero frame
    test_input = zeros(H, W2, C)
    test_input[:, :, frame_idx] = randn(H, W2)
    
    println("Testing single frame $frame_idx...")
    
    # Test correlated covariance
    corr_result = RhapsodieDirect.apply_covariance(noise_model.corr_noise, test_input, direct_model)
    
    println("  Input frame $frame_idx sum: $(sum(test_input[:,:,frame_idx]))")
    println("  Output frame $frame_idx sum: $(sum(corr_result[:,:,frame_idx]))")
    
    # Check if other frames are contaminated
    for f in 1:C
        if f != frame_idx
            other_sum = sum(abs.(corr_result[:,:,f]))
            if other_sum > 1e-10
                println("  ⚠️  Frame $f has non-zero output: $other_sum")
            end
        end
    end
end

println("Debug script loaded. Usage:")
println("  results = debug_precision_components(dataset, residual)")
println("  test_single_frame(dataset)")
