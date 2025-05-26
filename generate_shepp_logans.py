import numpy as np
from scipy.spatial.transform import Rotation
# import matplotlib.pyplot as plt # Optionnel, pour vérification
# import vedo # Optionnel, pour vérification
import os
import json
import random
from pathlib import Path # Pour une gestion propre des chemins

# --- (La fonction shepp_logan_3d reste la même que précédemment) ---
def shepp_logan_3d(size=128, ellipsoid_params_list=None):
    if ellipsoid_params_list is None:
        raise ValueError("ellipsoid_params_list ne peut pas être None.")
    grid_1d = np.linspace(-0.5, 0.5, size, endpoint=True)
    X, Y, Z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    phantom = np.zeros((size, size, size), dtype=np.float32)
    for params in ellipsoid_params_list:
        intensity = params['intensity']
        cx, cy, cz = params['center']
        ax, ay, az = params['semi_axes']
        rot_xyz_deg = params.get('rotation_euler_xyz_deg', (0, 0, 0))
        if ax <= 0 or ay <= 0 or az <= 0: continue
        x_c = X - cx; y_c = Y - cy; z_c = Z - cz
        x_r, y_r, z_r = x_c, y_c, z_c
        if not (rot_xyz_deg[0] == 0 and rot_xyz_deg[1] == 0 and rot_xyz_deg[2] == 0):
            rotation = Rotation.from_euler('xyz', rot_xyz_deg, degrees=True)
            points_translated = np.stack([x_c.ravel(), y_c.ravel(), z_c.ravel()], axis=-1)
            points_rotated = rotation.inv().apply(points_translated)
            x_r = points_rotated[:, 0].reshape(X.shape)
            y_r = points_rotated[:, 1].reshape(Y.shape)
            z_r = points_rotated[:, 2].reshape(Z.shape)
        mask = (x_r / ax)**2 + (y_r / ay)**2 + (z_r / az)**2 <= 1.0
        phantom[mask] += intensity
    return phantom

def generate_randomized_ellipsoid_parameters(num_total_ellipsoids=3, verbose=False):
    params_list = []

    e1_intensity = random.uniform(0.8, 1.2)
    e1_center_val = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
    e1_ax_val = random.uniform(0.40, 0.48)
    e1_ay_val = random.uniform(0.35, 0.45)
    e1_az_val = random.uniform(0.30, 0.42)
    e1_rot_xyz_deg_val = (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-5, 5))
    ellipsoid1_params = {
        'id': 1, 'type': 'outer_shell', 'intensity': e1_intensity, 'center': e1_center_val,
        'semi_axes': (e1_ax_val, e1_ay_val, e1_az_val), 'rotation_euler_xyz_deg': e1_rot_xyz_deg_val
    }
    params_list.append(ellipsoid1_params)
    
    R_e1 = Rotation.from_euler('xyz', e1_rot_xyz_deg_val, degrees=True)
    c_e1 = np.array(e1_center_val)
    e1_ax, e1_ay, e1_az = e1_ax_val, e1_ay_val, e1_az_val

    e2_intensity = random.uniform(-0.8 * e1_intensity, -0.5 * e1_intensity)
    e2_center_offset_scale = 0.01
    e2_center_val = (e1_center_val[0] + random.uniform(-e1_ax * e2_center_offset_scale, e1_ax * e2_center_offset_scale),
                     e1_center_val[1] + random.uniform(-e1_ay * e2_center_offset_scale, e1_ay * e2_center_offset_scale),
                     e1_center_val[2] + random.uniform(-e1_az * e2_center_offset_scale, e1_az * e2_center_offset_scale))
    reduction_factor = random.uniform(0.90, 0.98)
    e2_ax_val = e1_ax * reduction_factor
    e2_ay_val = e1_ay * reduction_factor
    e2_az_val = e1_az * reduction_factor
    e2_rot_xyz_deg_val = (e1_rot_xyz_deg_val[0] + random.uniform(-2, 2),
                          e1_rot_xyz_deg_val[1] + random.uniform(-2, 2),
                          e1_rot_xyz_deg_val[2] + random.uniform(-2, 2))
    ellipsoid2_params = {
        'id': 2, 'type': 'hollowing_shell', 'intensity': e2_intensity, 'center': e2_center_val,
        'semi_axes': (e2_ax_val, e2_ay_val, e2_az_val), 'rotation_euler_xyz_deg': e2_rot_xyz_deg_val
    }
    params_list.append(ellipsoid2_params)

    min_semi_axis_val = 0.005 
    clearance_for_sampling_range = 0.001 
    ABSOLUTE_SAFETY_GAP = 0.01 # Marge de sécurité absolue pour l'espace entre les surfaces
    CENTER_PLACEMENT_SCALE_FACTOR = 0.80 # Utiliser 80% de la zone de placement du centre

    for i in range(3, num_total_ellipsoids + 1):
        ei_intensity = random.uniform(0.1, 0.5)
        ei_ax_orig = random.uniform(0.03, 0.15) # Taille initiale un peu plus grande
        ei_ay_orig = random.uniform(0.03, 0.15)
        ei_az_orig = random.uniform(0.03, 0.15)
        ei_rot_xyz_deg = (random.uniform(-180, 180), random.uniform(-180, 180), random.uniform(-180, 180))
        
        ei_ax, ei_ay, ei_az = ei_ax_orig, ei_ay_orig, ei_az_orig # Copie pour modification

        max_resize_attempts = 7 # Augmenter un peu le nombre d'essais de redimensionnement
        for attempt in range(max_resize_attempts):
            R_ei_obj = Rotation.from_euler('xyz', ei_rot_xyz_deg, degrees=True)
            R_eff = R_e1.inv() * R_ei_obj 
            R_m = R_eff.as_matrix()

            ext_x = np.sqrt( (R_m[0,0]*ei_ax)**2 + (R_m[0,1]*ei_ay)**2 + (R_m[0,2]*ei_az)**2 )
            ext_y = np.sqrt( (R_m[1,0]*ei_ax)**2 + (R_m[1,1]*ei_ay)**2 + (R_m[1,2]*ei_az)**2 )
            ext_z = np.sqrt( (R_m[2,0]*ei_ax)**2 + (R_m[2,1]*ei_ay)**2 + (R_m[2,2]*ei_az)**2 )

            # Limites pour le centre de ei, incluant la marge de sécurité absolue
            limit_center_x = e1_ax - ext_x - ABSOLUTE_SAFETY_GAP
            limit_center_y = e1_ay - ext_y - ABSOLUTE_SAFETY_GAP
            limit_center_z = e1_az - ext_z - ABSOLUTE_SAFETY_GAP
            
            if (limit_center_x >= clearance_for_sampling_range and
                limit_center_y >= clearance_for_sampling_range and
                limit_center_z >= clearance_for_sampling_range):
                break 

            if verbose and attempt == 0:
                 print(f"    Ajustement de taille pour l'ellipsoïde interne {i} (essai {attempt+1})...")
            if attempt == max_resize_attempts - 1 and verbose:
                 print(f"    Avertissement: L'ellipsoïde interne {i} n'a pas pu être ajusté après {max_resize_attempts} essais.")

            scale_factors_needed = []
            if limit_center_x < clearance_for_sampling_range:
                target_ext_x = e1_ax - clearance_for_sampling_range - ABSOLUTE_SAFETY_GAP
                if target_ext_x < min_semi_axis_val : target_ext_x = min_semi_axis_val 
                scale_factors_needed.append(target_ext_x / (ext_x + 1e-9))
            if limit_center_y < clearance_for_sampling_range:
                target_ext_y = e1_ay - clearance_for_sampling_range - ABSOLUTE_SAFETY_GAP
                if target_ext_y < min_semi_axis_val : target_ext_y = min_semi_axis_val
                scale_factors_needed.append(target_ext_y / (ext_y + 1e-9))
            if limit_center_z < clearance_for_sampling_range:
                target_ext_z = e1_az - clearance_for_sampling_range - ABSOLUTE_SAFETY_GAP
                if target_ext_z < min_semi_axis_val : target_ext_z = min_semi_axis_val
                scale_factors_needed.append(target_ext_z / (ext_z + 1e-9))

            reduction_scale = min(scale_factors_needed) if scale_factors_needed else 0.90 # Réduction par défaut si pas de besoin spécifique
            reduction_scale = max(0.1, min(reduction_scale, 0.90)) # Plus agressif sur la réduction max

            ei_ax = max(min_semi_axis_val, ei_ax * reduction_scale)
            ei_ay = max(min_semi_axis_val, ei_ay * reduction_scale)
            ei_az = max(min_semi_axis_val, ei_az * reduction_scale)
        # Fin de la boucle de redimensionnement

        # Recalculer ext_x,y,z finaux et les limites pour le placement du centre
        R_ei_obj = Rotation.from_euler('xyz', ei_rot_xyz_deg, degrees=True)
        R_eff = R_e1.inv() * R_ei_obj; R_m = R_eff.as_matrix()
        ext_x = np.sqrt( (R_m[0,0]*ei_ax)**2 + (R_m[0,1]*ei_ay)**2 + (R_m[0,2]*ei_az)**2 )
        ext_y = np.sqrt( (R_m[1,0]*ei_ax)**2 + (R_m[1,1]*ei_ay)**2 + (R_m[1,2]*ei_az)**2 )
        ext_z = np.sqrt( (R_m[2,0]*ei_ax)**2 + (R_m[2,1]*ei_ay)**2 + (R_m[2,2]*ei_az)**2 )

        # Limites finales pour la zone de placement du centre
        final_limit_for_center_x = max(clearance_for_sampling_range, e1_ax - ext_x - ABSOLUTE_SAFETY_GAP)
        final_limit_for_center_y = max(clearance_for_sampling_range, e1_ay - ext_y - ABSOLUTE_SAFETY_GAP)
        final_limit_for_center_z = max(clearance_for_sampling_range, e1_az - ext_z - ABSOLUTE_SAFETY_GAP)

        # Réduire la plage de placement effective du centre
        actual_sampling_range_x = final_limit_for_center_x * CENTER_PLACEMENT_SCALE_FACTOR
        actual_sampling_range_y = final_limit_for_center_y * CENTER_PLACEMENT_SCALE_FACTOR
        actual_sampling_range_z = final_limit_for_center_z * CENTER_PLACEMENT_SCALE_FACTOR
        
        # S'assurer que la plage d'échantillonnage est valide pour random.uniform
        actual_sampling_range_x = max(clearance_for_sampling_range / 2.0, actual_sampling_range_x)
        actual_sampling_range_y = max(clearance_for_sampling_range / 2.0, actual_sampling_range_y)
        actual_sampling_range_z = max(clearance_for_sampling_range / 2.0, actual_sampling_range_z)


        c_ei_local_x = random.uniform(-actual_sampling_range_x, actual_sampling_range_x)
        c_ei_local_y = random.uniform(-actual_sampling_range_y, actual_sampling_range_y)
        c_ei_local_z = random.uniform(-actual_sampling_range_z, actual_sampling_range_z)
        c_ei_local = np.array([c_ei_local_x, c_ei_local_y, c_ei_local_z])

        c_ei_world = R_e1.apply(c_ei_local) + c_e1
        
        ellipsoidi_params = {
            'id': i, 'type': 'inner_structure', 'intensity': ei_intensity, 'center': tuple(c_ei_world),
            'semi_axes': (ei_ax, ei_ay, ei_az), 'rotation_euler_xyz_deg': ei_rot_xyz_deg
        }
        params_list.append(ellipsoidi_params)
    return params_list

# --- (Les fonctions generate_dataset, display_slices_mpl, visualize_with_vedo restent les mêmes) ---
def generate_dataset(num_samples, output_base_dir, phantom_size=128, 
                     num_ellipsoids_options=(3, 4, 5), verbose_generation=False, verbose_params=False):
    output_base_dir = Path(output_base_dir)
    phantoms_dir = output_base_dir / "phantoms"
    params_dir = output_base_dir / "parameters"
    
    phantoms_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose_generation: print(f"Début de la génération de {num_samples} fantômes dans '{output_base_dir}'...")
    
    generated_count = 0
    for i in range(num_samples):
        if verbose_generation and (i % (num_samples // 20 if num_samples >= 20 else 1) == 0) : # Moins de messages pour les grands datasets
            print(f"  Génération du fantôme {i+1}/{num_samples}...")
            
        num_ellips = random.choice(num_ellipsoids_options)
        
        retry_count = 0
        max_retries = 3 
        ellipsoid_params = None
        while ellipsoid_params is None and retry_count < max_retries:
            try:
                ellipsoid_params = generate_randomized_ellipsoid_parameters(num_ellips, verbose=verbose_params)
                for p_idx, p in enumerate(ellipsoid_params):
                    if not all(sa > 0 for sa in p['semi_axes']):
                        raise ValueError(f"Un demi-axe généré pour l'ellipsoide {p_idx+1} est non-positif: {p['semi_axes']}.")
            except Exception as e:
                if verbose_params:
                    print(f"    Erreur lors de la génération des paramètres (essai {retry_count+1}): {e}. Nouvel essai...")
                ellipsoid_params = None
                retry_count += 1
        
        if ellipsoid_params is None:
            if verbose_generation: print(f"  Échec de la génération des paramètres pour le fantôme {i+1} après {max_retries} essais. Ignoré.")
            continue

        phantom = shepp_logan_3d(size=phantom_size, ellipsoid_params_list=ellipsoid_params)
        
        phantom_filename = phantoms_dir / f"phantom_{generated_count:04d}.npy"
        params_filename = params_dir / f"params_{generated_count:04d}.json"
        
        np.save(phantom_filename, phantom)
        with open(params_filename, 'w') as f:
            json.dump(ellipsoid_params, f, indent=4)
        generated_count +=1
            
    print(f"{generated_count} fantômes générés avec succès dans '{output_base_dir}'.")


def display_slices_mpl(phantom_3d, slice_index=None, title="Shepp-Logan 3D - Matplotlib"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib n'est pas installé.")
        return
    size = phantom_3d.shape[0]
    if slice_index is None: slice_index = size // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(phantom_3d[slice_index, :, :].T, cmap='gray', origin='lower')
    axes[0].set_title(f'Sagittal (X={slice_index-size//2})')
    axes[1].imshow(phantom_3d[:, slice_index, :].T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal (Y={slice_index-size//2})')
    axes[2].imshow(phantom_3d[:, :, slice_index], cmap='gray', origin='lower')
    axes[2].set_title(f'Axial (Z={slice_index-size//2})')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False) # Non-blocking show
    plt.pause(0.1) # Pause pour permettre l'affichage

def visualize_with_vedo(phantom_3d, mode='isosurface', title="Shepp-Logan 3D - Vedo"):
    try:
        import vedo
    except ImportError:
        print("Vedo n'est pas installé. Impossible de visualiser avec Vedo.")
        return

    size = phantom_3d.shape[0]
    vol = vedo.Volume(phantom_3d, origin=(-0.5,-0.5,-0.5), spacing=(1.0/size, 1.0/size, 1.0/size))
    # Utiliser N=1 pour éviter que les fenêtres successives ne s'écrasent en mode script rapide
    plotter = vedo.Plotter(title=title, axes=1, N=1, sharecam=False) 
    
    if mode == 'isosurface':
        val_min, val_max = phantom_3d.min(), phantom_3d.max()
        iso_values = []
        if val_max > val_min: 
            iso_values.append(val_min + 0.25 * (val_max - val_min)) 
            iso_values.append(val_min + 0.60 * (val_max - val_min)) 
            iso_values.append(val_min + 0.80 * (val_max - val_min)) 
        
        isos = []
        colors = ["lightcoral", "lightskyblue", "lightgreen"] # Couleurs plus distinctes et claires
        opacities = [0.25, 0.4, 0.5] # Opacités ajustées pour mieux voir à travers

        # Ajouter une isosurface pour la coque externe (e1+e2)
        # L'intensité de la coque est e1_intensity + e2_intensity. e2 est négatif.
        # Si e1_intensity ~ 1.0 et e2_intensity ~ -0.7, la coque est à ~0.3
        # On prend une valeur un peu plus basse que la plus faible intensité positive des ellipsoides internes
        coque_val = val_min + 0.15 * (val_max-val_min) if val_max > val_min else 0.1
        if coque_val > val_min and coque_val < (val_min + 0.4 * (val_max-val_min)): # Assurer que c'est une valeur de "coque"
            isos.append(vol.isosurface(value=coque_val).color("silver").opacity(0.15))


        for i, v in enumerate(iso_values):
            if v < val_max and v > val_min + 0.01: # Éviter isosurface à min_val
                 isos.append(vol.isosurface(value=v).color(colors[i % len(colors)]).opacity(opacities[i % len(opacities)]))
        
        if not isos and val_max > val_min: 
            isos.append(vol.isosurface().color("gray").opacity(0.5))
        elif not isos: 
            plotter.add_text("Volume plat ou vide.", pos="bottom-middle")

        plotter.show(isos, "Isosurfaces du fantôme de Shepp-Logan", interactive=True).close()


    elif mode == 'volume':
        min_val, max_val = np.min(phantom_3d), np.max(phantom_3d)
        if min_val == max_val: 
            plotter.add_text("Volume plat.", pos="bottom-middle")
            plotter.show(interactive=True).close()
            return

        alpha_points = [ (min_val + f*(max_val-min_val), op) for f,op in 
                        [(0.0,0.0), (0.1,0.0), (0.25,0.02), (0.4,0.1), (0.6,0.3), (0.8,0.7), (1.0,0.9)] ] # Alpha ajusté
        vol.mode(0).cmap('bone_r').alpha(alpha_points).alpha_unit(1.0) # alpha_unit peut aider
        plotter.add(vol)
        plotter.add_scalar_bar3d(title="Intensité")
        plotter.show("Rendu Volumique du fantôme de Shepp-Logan", interactive=True).close()
    elif mode == 'slices':
        plotter.add_cutter_tool(vol) 
        plotter.show(vol, "Coupes interactives du fantôme de Shepp-Logan", interactive=True).close()
    else: 
        print(f"Mode de visualisation '{mode}' non reconnu.")
        plotter.close()

# --- Exemple d'utilisation pour générer un petit dataset ---
if __name__ == "__main__":
    dataset_output_directory = "shepp_logan_dataset_128_v3" 
    num_phantoms_to_generate = 100 # Encore plus petit pour des tests rapides
    
    generate_dataset(num_phantoms_to_generate, 
                     dataset_output_directory, 
                     phantom_size=128,
                     num_ellipsoids_options=(4,5), 
                     verbose_generation=True,
                     verbose_params=True) 

    print(f"\nVisualisation des fantômes générés (s'ils existent)...")
    phantoms_path = Path(dataset_output_directory) / "phantoms"
    params_path = Path(dataset_output_directory) / "parameters"

    generated_files = sorted(list(phantoms_path.glob("phantom_*.npy")))

    for i, phantom_file in enumerate(generated_files):
        if i >= 3 : break # Visualiser seulement les 3 premiers pour ne pas être submergé
        
        params_file = params_path / f"{phantom_file.stem.replace('phantom', 'params')}.json"

        if phantom_file.exists() and params_file.exists():
            print(f"\n--- Visualisation de {phantom_file.name} ---")
            phantom_data = np.load(phantom_file)
            with open(params_file, 'r') as f:
                params_data = json.load(f)
            
            # print(f"Paramètres ({params_file.name}):")
            # print(json.dumps(params_data, indent=2))

            display_slices_mpl(phantom_data, title=f"{phantom_file.name} (Matplotlib)")
            visualize_with_vedo(phantom_data, mode='isosurface', title=f"{phantom_file.name} (Vedo Isosurface)")
            # visualize_with_vedo(phantom_data, mode='volume', title=f"{phantom_file.name} (Vedo Volume)")
            
        else:
            print(f"Fichier {phantom_file.name} ou {params_file.name} non trouvé.")
    
    if not generated_files:
        print(f"Aucun fantôme n'a été trouvé dans {phantoms_path}.")