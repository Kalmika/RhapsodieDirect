#
# test_interpolation_simple.jl
#
# Test simple pour l'interpolation/rotation avec CatmullRomSpline
# Sort juste la partie FieldTransformOperator du modèle complet
#

using Pkg
Pkg.activate(".")

using RhapsodieDirect
using InterpolationKernels
using Plots

println("Test simple des transformations d'interpolation...")

# --- 1. Paramètres ---
object_params = ObjectParameters((64, 64), (32.0, 32.0))
data_params = DatasetParameters((64, 128), 4, 1, 1, (32.0, 32.0))  # 4 frames seulement
ker = CatmullRomSpline(Float64, Flat)

# --- 2. Création d'un objet test simple ---
println("Création d'un carré test...")

# Carré simple de 16x16 pixels
I_map = zeros(Float64, object_params.size)
Q_map = zeros(Float64, object_params.size)  
U_map = zeros(Float64, object_params.size)

# Carré centré
center_x, center_y = round.(Int, object_params.center)
half_size = 8
y_range = (center_y - half_size):(center_y + half_size)
x_range = (center_x - half_size):(center_x + half_size)

I_map[y_range, x_range] .= 100.0
Q_map[y_range, x_range] .= 50.0   
U_map[y_range, x_range] .= -30.0  

test_object = PolarimetricMap("stokes", I_map, Q_map, U_map)
println("Objet créé - max I: $(maximum(I_map))")

# --- 3. Définition de quelques transformations test ---
println("Définition des transformations...")

indices = get_indices_table(data_params)
polar_params = set_default_polarisation_coefficients(indices)

# Créer 4 transformations avec des rotations/translations différentes
field_params = FieldTransformParameters[]

# Frame 1: pas de transformation  
push!(field_params, FieldTransformParameters(ker, 0.0, (0.0, 0.0), (0.0, 0.0), 
                                            polar_params[1][1], polar_params[1][2]))

# Frame 2: rotation de 15°
push!(field_params, FieldTransformParameters(ker, π/12, (0.0, 0.0), (0.0, 0.0), 
                                            polar_params[2][1], polar_params[2][2]))

# Frame 3: translation de 20 pixels
push!(field_params, FieldTransformParameters(ker, 0.0, (20.0, 10.0), (-20.0, -10.0), 
                                            polar_params[3][1], polar_params[3][2]))

# Frame 4: rotation + translation
push!(field_params, FieldTransformParameters(ker, -π/18, (1.5, -1.0), (-1.5, 1.0), 
                                            polar_params[4][1], polar_params[4][2]))

# --- 4. Construction des FieldTransformOperators ---
field_transforms = load_field_transforms(object_params, data_params, field_params)
println("$(length(field_transforms)) transformations créées")

# --- 5. Application manuelle (extrait de mappings.jl) ---
println("Application des transformations...")

# Préparer le tenseur d'entrée (comme dans DirectModel.apply!)
x = zeros(Float64, object_params.size[1], object_params.size[2], length(test_object));

# Accès direct aux composantes (équivalent à get_stokes)
stokes_components = [test_object.I, test_object.Q, test_object.U]
@inbounds for (i, map) in enumerate(stokes_components)
    setindex!(x, map, :, :, i)  # Pas de PSF dans ce test
end

# Appliquer chaque transformation (la partie qu'on veut tester !)
dst = zeros(Float64, data_params.size[1], data_params.size[2], data_params.frames_total)

@inbounds for k = 1:length(field_transforms)
    println("Application transformation $k...")
    apply!(view(dst, :, :, k), field_transforms[k], x)
end

println("Transformations appliquées - taille résultat: $(size(dst))")

# --- 6. Plots avant/après ---
println("Génération des plots...")

# Plot l'objet original
p1 = heatmap(I_map, title="Objet original (I)", aspect_ratio=:equal, c=:viridis)

# Plot les 4 résultats (prendre seulement la partie gauche du détecteur)
left_detector = dst[:, 1:(data_params.size[2]÷2), :]

p2 = heatmap(left_detector[:, :, 1], title="Frame 1 (identité)", aspect_ratio=:equal, c=:viridis)
p3 = heatmap(left_detector[:, :, 2], title="Frame 2 (rotation 15°)", aspect_ratio=:equal, c=:viridis)  
p4 = heatmap(left_detector[:, :, 3], title="Frame 3 (translation)", aspect_ratio=:equal, c=:viridis)
p5 = heatmap(left_detector[:, :, 4], title="Frame 4 (rotation + trans.)", aspect_ratio=:equal, c=:viridis)

# Assemblage final
plot_final = plot(p1, p2, p3, p4, p5, layout=(2, 3), size=(900, 600))
savefig(plot_final, "test_interpolation_results.png")

println("Plot sauvegardé: test_interpolation_results.png")

# --- 7. Vérifications numériques ---
println("\n=== VÉRIFICATIONS ===")
println("Objet original - min/max I: $(minimum(I_map)) / $(maximum(I_map))")
println("Frame 1 - min/max: $(minimum(left_detector[:,:,1])) / $(maximum(left_detector[:,:,1]))")
println("Frame 2 - min/max: $(minimum(left_detector[:,:,2])) / $(maximum(left_detector[:,:,2]))")
println("Frame 3 - min/max: $(minimum(left_detector[:,:,3])) / $(maximum(left_detector[:,:,3]))")
println("Frame 4 - min/max: $(minimum(left_detector[:,:,4])) / $(maximum(left_detector[:,:,4]))")

# Conservation d'énergie (approximative avec interpolation)
energy_orig = sum(I_map)
energy_1 = sum(left_detector[:,:,1])
energy_2 = sum(left_detector[:,:,2])
println("\nConservation d'énergie:")
println("Original: $energy_orig")
println("Frame 1: $energy_1 (ratio: $(energy_1/energy_orig))")
println("Frame 2: $energy_2 (ratio: $(energy_2/energy_orig))")

# Sauvegarde des données pour comparaison Python
using DelimitedFiles
writedlm("original_object.txt", I_map)
writedlm("transformed_frame1.txt", left_detector[:,:,1])
writedlm("transformed_frame2.txt", left_detector[:,:,2])
println("\nDonnées sauvegardées pour comparaison Python:")
println("- original_object.txt")
println("- transformed_frame1.txt (identité)")  
println("- transformed_frame2.txt (rotation 15°)")

println("\n✅ Test terminé avec succès !")