# =============================================================================
# test_noise_models.jl - Validation des modèles de bruit
# =============================================================================

using Pkg
Pkg.activate(".")
# Ensure dependencies are installed (no-op if already instantiated)
try
    Pkg.instantiate()
catch e
    @warn "Pkg.instantiate() failed; run `Pkg.instantiate()` manually in the project folder: $e"
end

using FFTW
using Plots

# Import du module principal
using RhapsodieDirect




