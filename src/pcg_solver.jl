# pcg_solver.jl
# Simple Preconditioned Conjugate Gradient solver for RhapsodieDirect
# Algorithm 5.3 from Nocedal & Wright

using LinearAlgebra

"""
    pcg(apply_A, b; x0=nothing, rtol=1e-5, atol=0.0, maxiter=nothing, 
        apply_M_inv=nothing, callback=nothing, verbose=false)

Preconditioned Conjugate Gradient solver for symmetric positive definite systems.

Solves: A x = b

# Arguments
- `apply_A`: Function that computes A*x (e.g., `x -> apply_covariance(x, dataset)`)
- `b`: Right-hand side (AbstractArray)

# Keyword Arguments
- `x0`: Initial guess. If `nothing`, uses zeros.
- `rtol`: Relative tolerance (default: 1e-5)
- `atol`: Absolute tolerance (default: 0.0)
- `maxiter`: Maximum iterations. If `nothing`, uses `length(b)`.
- `apply_M_inv`: Preconditioner function that solves M*y = r for y.
                 If `nothing`, uses identity (no preconditioning).
- `callback`: Optional function called at each iteration as `callback(k, x, r_norm)`
- `verbose`: Print convergence info (default: false)

# Returns
- `x`: Solution
- `info`: NamedTuple with fields:
    - `converged::Bool`: Whether the solver converged
    - `iterations::Int`: Number of iterations performed
    - `residual_norm::Float64`: Final residual norm
    - `residual_history::Vector{Float64}`: Residual norm at each iteration

# Example
```julia
# Define operators
apply_A = x -> apply_covariance(x, dataset)
apply_M_inv = x -> apply_preconditioner_inverse(x, dataset)

# Solve
x, info = pcg(apply_A, b; apply_M_inv=apply_M_inv, rtol=1e-6, verbose=true)

if info.converged
    println("Converged in \$(info.iterations) iterations")
end
```
"""
function pcg(
    apply_A,
    b::AbstractArray{T};
    x0::Union{Nothing, AbstractArray{T}} = nothing,
    rtol::Real = 1e-5,
    atol::Real = 0.0,
    maxiter::Union{Nothing, Int} = nothing,
    apply_M_inv = nothing,
    callback = nothing,
    verbose::Bool = false
) where {T <: AbstractFloat}

    # Initialize
    n = length(b)
    maxiter = isnothing(maxiter) ? n : maxiter
    
    # Initial guess
    x = isnothing(x0) ? zero(b) : copy(x0)
    
    # Identity preconditioner if none provided
    if isnothing(apply_M_inv)
        apply_M_inv = identity
    end
    
    # r_0 = A*x_0 - b
    Ax = apply_A(x)
    r = Ax .- b
    
    # Initial residual norm for convergence check
    r_norm = norm(r)
    b_norm = norm(b)
    r0_norm = r_norm
    
    # Tolerance: stop when ||r|| <= max(rtol * ||r_0||, atol)
    tol = max(rtol * r0_norm, atol)
    
    # History
    residual_history = Float64[r_norm]
    
    if verbose
        println("PCG: Initial residual = $(r_norm)")
    end
    
    # Check if already converged
    if r_norm <= tol
        return x, (
            converged = true,
            iterations = 0,
            residual_norm = r_norm,
            residual_history = residual_history
        )
    end
    
    # Solve M*y_0 = r_0
    y = apply_M_inv(r)
    
    # p_0 = -y_0
    p = -y
    
    # rho_k = r_k^T * y_k
    rho = dot(r, y)
    
    # Main loop
    k = 0
    converged = false
    
    while k < maxiter
        # A * p_k
        Ap = apply_A(p)
        
        # alpha_k = (r_k^T * y_k) / (p_k^T * A * p_k)
        pAp = dot(p, Ap)
        
        # Check for breakdown
        if abs(pAp) < eps(T) * n
            if verbose
                println("PCG: Breakdown at iteration $k (p^T A p ≈ 0)")
            end
            break
        end
        
        alpha = rho / pAp
        
        # x_{k+1} = x_k + alpha_k * p_k
        x .+= alpha .* p
        
        # r_{k+1} = r_k + alpha_k * A * p_k
        r .+= alpha .* Ap
        
        # Check convergence
        r_norm = norm(r)
        push!(residual_history, r_norm)
        
        k += 1
        
        if verbose && (k % 10 == 0 || k == 1)
            println("PCG: Iteration $k, residual = $(r_norm)")
        end
        
        # Callback
        if !isnothing(callback)
            callback(k, x, r_norm)
        end
        
        if r_norm <= tol
            converged = true
            break
        end
        
        # Solve M * y_{k+1} = r_{k+1}
        y = apply_M_inv(r)
        
        # rho_{k+1} = r_{k+1}^T * y_{k+1}
        rho_new = dot(r, y)
        
        # beta_{k+1} = rho_{k+1} / rho_k
        beta = rho_new / rho
        
        # p_{k+1} = -y_{k+1} + beta_{k+1} * p_k
        p .= -y .+ beta .* p
        
        # Update rho for next iteration
        rho = rho_new
    end
    
    if verbose
        status = converged ? "CONVERGED" : "NOT CONVERGED"
        println("PCG: $status after $k iterations, final residual = $(r_norm)")
    end
    
    return x, (
        converged = converged,
        iterations = k,
        residual_norm = r_norm,
        residual_history = residual_history
    )
end


# ============================================================================
# Convenience wrapper for Dataset-based usage
# ============================================================================

"""
    pcg_solve_covariance(dataset, b; kwargs...)

Convenience wrapper for solving covariance systems with a Dataset.

Solves: C * x = b where C is the covariance operator.

# Arguments
- `dataset`: Dataset containing covariance operator info
- `b`: Right-hand side

# Keyword Arguments
Same as `pcg`, plus:
- `apply_covariance_fn`: Function (x, dataset) -> C*x. Default uses `apply_covariance`.
- `apply_precond_fn`: Function (x, dataset) -> M^{-1}*x. Default is `nothing` (no precond).

# Example
```julia
x, info = pcg_solve_covariance(
    dataset, residual;
    rtol=1e-6,
    maxiter=100,
    verbose=true
)
```
"""
function pcg_solve_covariance(
    dataset,
    b::AbstractArray{T};
    apply_covariance_fn = apply_covariance,
    apply_precond_fn = nothing,
    kwargs...
) where {T <: AbstractFloat}

    # Create operator closures
    apply_A = x -> apply_covariance_fn(x, dataset)
    
    apply_M_inv = if isnothing(apply_precond_fn)
        nothing
    else
        x -> apply_precond_fn(x, dataset)
    end
    
    return pcg(apply_A, b; apply_M_inv=apply_M_inv, kwargs...)
end


# ============================================================================
# Simple test
# ============================================================================

function test_pcg()
    println("\n" * "="^50)
    println("Testing PCG solver")
    println("="^50)
    
    # Create a simple SPD matrix
    n = 100
    A_dense = randn(n, n)
    A_dense = A_dense' * A_dense + 5.0 * I  # Make SPD
    
    # True solution and RHS
    x_true = randn(n)
    b = A_dense * x_true
    
    # Operator
    apply_A = x -> A_dense * x
    
    # Test 1: No preconditioner
    println("\nTest 1: PCG without preconditioner")
    x1, info1 = pcg(apply_A, b; rtol=1e-8, verbose=true)
    err1 = norm(x1 - x_true) / norm(x_true)
    println("Relative error: $(err1)")
    
    # Test 2: Jacobi preconditioner
    println("\nTest 2: PCG with Jacobi preconditioner")
    diag_A = diag(A_dense)
    apply_M_inv = x -> x ./ diag_A
    
    x2, info2 = pcg(apply_A, b; apply_M_inv=apply_M_inv, rtol=1e-8, verbose=true)
    err2 = norm(x2 - x_true) / norm(x_true)
    println("Relative error: $(err2)")
    
    # Compare iterations
    println("\n" * "-"^50)
    println("Summary:")
    println("  Without precond: $(info1.iterations) iterations")
    println("  With Jacobi:     $(info2.iterations) iterations")
    println("="^50 * "\n")
    
    return info1.converged && info2.converged && err1 < 1e-6 && err2 < 1e-6
end

# Run test if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_pcg()
end

