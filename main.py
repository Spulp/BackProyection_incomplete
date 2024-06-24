import os
import torch
# Para realizar la retro proyección
from backprojection import features_backprojection, DINOWrapper
from utils import  setup_renderer, sample_view_points, load_mesh, save_features, load_features
# Para la visualización
from sklearn.decomposition import PCA
import polyscope as ps
# para guardar la visualización en html
from pytorch3d.renderer import TexturesVertex
from pytorch3d.vis.plotly_vis import plot_scene

SHAPENET_PATH = "/mnt/e/Shapenet"
SHAPENET_SYNSET_TXT_PATH = os.path.join("utils", "shapenet_synset.txt")
SYNSET_DICT = {}
# abre shapenet_synset.txt, en donde toma cada linea ycrea un diccionario separado por " "
# el primer elemento es el el synset id, el segundo es el nombre humano
with open(SHAPENET_SYNSET_TXT_PATH) as f:
    for line in f:
        line_set = line.strip().split(" ")
        SYNSET_DICT[line_set[1]] = line_set[0]

# crea funcion que abre carpeta segun su nombre
def synset_folder_path(synset_name):
    return os.path.join(SHAPENET_PATH, SYNSET_DICT[synset_name])

def calc_features(name, backprojection=True, visualization=True, quantity=5):
    #########################################
    synset_path = synset_folder_path(name)
    # si no existe la carpeta, retorna un mensaje de error
    if not os.path.exists(synset_path):
        print("No existe la carpeta dada")
        return

    # itera sobre cada carpeta en la carpeta synset_path, y guarda el path en una lista
    synset_files_folder = []
    for folder in os.listdir(synset_path):
        synset_files_folder.append(os.path.join(synset_path, folder))
    print("Cantidad de objetos en la carpeta dada: ", len(synset_files_folder))

    #########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #backprojection = True
    #visualization = True
    #quantity = 5

    #########################################
    if backprojection:
        print("Running Backprojection")
        ############### Parametros
        batch_size = 8

        only_visible = False
        resolution = 224
        render_dist = 1.1
        partitions = 5
        # solo se usan si only_visible = True
        calc_geo_dists = False
        gaussian_sigma=0.001

        views = sample_view_points(render_dist, partitions)
        renderer = setup_renderer(device, res=resolution)
        model = DINOWrapper(device, small=True)
        ############### Retro proyección
        idx = 0
        for obj_file_folder in synset_files_folder:
            if idx == quantity:
                break
            # carga la mesh para obj
            obj_file = os.path.join(obj_file_folder, "models", "model_normalized.obj")
            obj_mesh = load_mesh(obj_file, device)
            # calcula sus caracteristicas 
            features = features_backprojection(renderer=renderer, model=model, mesh=obj_mesh, views=views, batch_size=batch_size,
                                    only_visible=only_visible, render_dist=render_dist, device=device,
                                    calc_geo_dists=calc_geo_dists, gaussian_sigma=gaussian_sigma)
            # guarda las caracteristicas en features
            features_path = os.path.join(obj_file_folder, "features", "features.pt")
            save_features(features_path, features)

            idx += 1

    #########################################
    if visualization:
        print("Running Visualization")
        ############### visualización
        idx = 0
        for obj_file_folder in synset_files_folder:
            if idx == quantity:
                break
            # carga la mesh para obj
            obj_file = os.path.join(obj_file_folder, "models", "model_normalized.obj")
            obj_mesh = load_mesh(obj_file, device)
            # cargar features
            features_path = os.path.join(obj_file_folder, "features", "features.pt")
            features = load_features(features_path)
            # si no encuentra las caracteristicas, imprime un mensaje y continua
            if features is None:
                print("Features not found for", obj_file_folder)
                idx += 1
                continue
            # Perform PCA for visualization
            pca = PCA(n_components=3)
            features_pca = pca.fit_transform(features.cpu().numpy())
            features_pca = (features_pca - features_pca.min(axis=0)) / (features_pca.max(axis=0) - features_pca.min(axis=0))
            # Visualize the mesh with texture using polyscope
            ps.init()
            ps_mesh = ps.register_surface_mesh("airplane", obj_mesh.verts_packed().cpu().numpy(), obj_mesh.faces_packed().cpu().numpy())
            ps_mesh.add_color_quantity("rainbow", features_pca, enabled=True)
            ps.show()

            # la guarda en html
            obj_mesh.textures = TexturesVertex(verts_features=torch.tensor(features_pca, dtype=torch.float32)[None].to(device))
            fig = plot_scene({
                "mesh": {
                    "mesh": obj_mesh,
                }
            })
            fig_html_path = os.path.join(obj_file_folder, "features", "visualization.html")
            fig.write_html(fig_html_path)

            idx += 1

    #########################################
    if backprojection or visualization:
        # imprime las carpetas usadas para verificar
        idx = 0
        for obj_file_folder in synset_files_folder:
            if idx == quantity:
                break
            print(obj_file_folder)
            idx += 1
        print("Done")

# if name main
if __name__ == "__main__":
    calc_features("airplane", backprojection=False, visualization=True, quantity=5)
    calc_features("car", backprojection=False, visualization=True, quantity=2)

