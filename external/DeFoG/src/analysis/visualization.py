import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from datasets.tls_dataset import CellGraph

################################################################################################

from rdkit.Chem.Draw import rdMolDraw2D
import imageio

################################################################################################



class MolecularVisualization:
    def __init__(
        self, 
        remove_h, 
        dataset_infos,
        ###########################################################################
        X_fixed=None, 
        E_fixed=None,
        ###########################################################################
        ):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos
        ###########################################################################
        self.X_fixed = X_fixed
        self.E_fixed = E_fixed  
        ###########################################################################

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

###############################################################################################################
    def visualize(self, path, molecules, num_molecules_to_visualize=1):
        
        nodes_fixed = self.X_fixed
        edges_fixed = self.E_fixed
        RDLogger.DisableLog("rdApp.*")
        
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        nodes_fixed = set(nodes_fixed.squeeze(0).tolist()) if nodes_fixed is not None else set()

        E = self.E_fixed.squeeze(0) if self.E_fixed is not None else None
        edges_fixed = [
            (i, j)
            for i in range(E.shape[0])
            for j in range(i + 1, E.shape[1])
            if E[i, j] > 0
        ] if E is not None else []

        for i in range(min(num_molecules_to_visualize, len(molecules))):
            file_path = os.path.join(path, f"molecule_{i}.png")
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            if mol is None:
                continue

            AllChem.Compute2DCoords(mol)

            # obtain indices of bonds fixed in RDKit
            fixed_bonds = []
            for i,j in edges_fixed:
                b = mol.GetBondBetweenAtoms(i,j)
                if b is not None:
                    fixed_bonds.append(b.GetIdx())

            # green color for fixed atoms and bonds
            highlight_atom_colors = {n: (0.0, 1.0, 0.0) for n in nodes_fixed}
            highlight_bond_colors = {b: (0.0, 1.0, 0.0) for b in fixed_bonds}

            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, 
                mol, 
                highlightAtoms=list(nodes_fixed), 
                highlightBonds=fixed_bonds,
                highlightAtomColors=highlight_atom_colors,
                highlightBondColors=highlight_bond_colors
            )
            drawer.FinishDrawing()
            with open(file_path, "wb") as f:
                f.write(drawer.GetDrawingText())
            print(f"Molecule drawn at {file_path}")
                
    def visualize_chain(self, path, nodes_list, adjacency_matrix, times, trainer=None):
        nodes_fixed = self.X_fixed
        edges_fixed = self.E_fixed

        RDLogger.DisableLog("rdApp.*")
        os.makedirs(path, exist_ok=True)

        nodes_fixed = set(nodes_fixed.squeeze(0).tolist()) if nodes_fixed is not None else set()

        E = self.E_fixed.squeeze(0) if self.E_fixed is not None else None
        edges_fixed = [
            (i, j)
            for i in range(E.shape[0])
            for j in range(i + 1, E.shape[1])
            if E[i, j] > 0
        ] if E is not None else []

        # convert graphs to the rdkit molecules
        mols = [
            self.mol_from_graphs(nodes_list[i], adjacency_matrix[i])
            for i in range(nodes_list.shape[0])
        ]

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            num_atoms = min(len(mol.GetAtoms()), len(coords))  # aseguramos que no se pase del rango
            for j in range(num_atoms):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))
        # draw gif
        save_paths = []
        num_frames = nodes_list.shape[0]

        for frame in range(num_frames):
            mol = mols[frame]

            # obtain indices of bonds fixed in RDKit
            fixed_bonds = []
            for i, j in edges_fixed:
                b = mol.GetBondBetweenAtoms(i, j)
                if b is not None:
                    fixed_bonds.append(b.GetIdx())

            highlight_atom_colors = {n: (0.0, 1.0, 0.0) for n in nodes_fixed}
            highlight_bond_colors = {b: (0.0, 1.0, 0.0) for b in fixed_bonds}

            file_name = os.path.join(path, f"frame_{frame}.png")
            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer,
                mol,
                highlightAtoms=list(nodes_fixed),
                highlightBonds=fixed_bonds,
                highlightAtomColors=highlight_atom_colors,
                highlightBondColors=highlight_bond_colors,
                legend=f"t = {times[frame]:.2f}"
            )
            drawer.FinishDrawing()
            with open(file_name, "wb") as f:
                f.write(drawer.GetDrawingText())

            save_paths.append(file_name)

        # create GIF
        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_name = os.path.basename(os.path.normpath(path))
        gif_path = os.path.join(path, f"{gif_name}.gif")
        print(f"Saving GIF at {gif_path}")
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)

        if wandb.run:
            print(f"Saving {gif_path} to wandb")
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)

        return mols
    
    ###############################################################################################


class NonMolecularVisualization:

    def __init__(
        self, 
        dataset_name, 
        ###########################################################################
        X_fixed=None, 
        E_fixed=None,
        ###########################################################################
        ):
        self.is_tls = "tls" in dataset_name
        ###########################################################################
        self.X_fixed = X_fixed
        self.E_fixed = E_fixed  
        ###########################################################################

    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(
                edge[0], edge[1], color=float(edge_type), weight=3 * edge_type
            )

        return graph

    def visualize_non_molecule(
        self,
        graph,
        pos,
        path,
        iterations=100,
        node_size=100,
        largest_component=False,
        time=None,
    ):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        ##############################################################################################
        fixed_nodes = self.X_fixed.squeeze(0) if self.X_fixed is not None else None  # (N,)
        fixed_edges = self.E_fixed.squeeze(0) if self.E_fixed is not None else None  # (N, N)

        # --- Node colors ---
        node_colors = []
        for n in graph.nodes():
            if fixed_nodes is not None and fixed_nodes[n] != -1:
                node_colors.append((0, 1, 0, 1))  # green
            else:
                val = float(U[n, 1])
                norm_val = (val - vmin) / (vmax - vmin + 1e-8)
                node_colors.append(plt.cm.coolwarm(norm_val))

        # Draw nodes first
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            edgecolors='black'  # node border
        )

        # --- Draw edges separately ---
        # Fixed edges (green)
        if fixed_edges is not None:
            fixed_edge_list = [(u, v) for u, v in graph.edges() if fixed_edges[u, v] != -1]
            nx.draw_networkx_edges(graph, pos, edgelist=fixed_edge_list, edge_color='green', width=2)

        # Remaining edges (grey)
        remaining_edge_list = [
            (u, v) for u, v in graph.edges()
            if fixed_edges is None or fixed_edges[u, v] == -1
        ]
        nx.draw_networkx_edges(graph, pos, edgelist=remaining_edge_list, edge_color='grey', width=1)

        # --- Draw node labels (indices) ---
        labels = {n: str(n) for n in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color='black')

        # --- Add legend ---
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Fixed nodes/edges'),
            Patch(facecolor=plt.cm.coolwarm(0.5), edgecolor='black', label='Free nodes (U[:,1])')
        ]
        plt.legend(handles=legend_elements)
        plt.axis('off')
        plt.show()
        ##############################################################################################
        if time is not None:
            plt.text(
                0.5,
                0.05,  # place below the graph
                f"t = {time:.2f}",
                ha="center",
                va="center",
                transform=plt.gcf().transFigure,
                fontsize=16,
            )

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(
        self, path: str, graphs: list, num_graphs_to_visualize: int, log="graph"
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, "graph_{}.png".format(i))

            if self.is_tls:
                cg = CellGraph.from_dense_graph(graphs[i])
                cg.plot_graph(save_path=file_path, has_legend=True)
            else:
                graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
                self.visualize_non_molecule(graph=graph, pos=None, path=file_path)

            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix, times):

        graphs = []
        for i in range(nodes_list.shape[0]):
            if self.is_tls:
                graphs.append(
                    CellGraph.from_dense_graph((nodes_list[i], adjacency_matrix[i]))
                )
            else:
                graphs.append(self.to_networkx(nodes_list[i], adjacency_matrix[i]))

        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, "fram_{}.png".format(frame))
            if self.is_tls:
                if not graphs[frame].get_pos():  # The last one already has a pos
                    graphs[frame].set_pos(pos=final_pos)
                graphs[frame].plot_graph(
                    save_path=file_name,
                    has_legend=False,
                    verbose=False,
                    time=times[frame],
                )
            else:
                self.visualize_non_molecule(
                    graph=graphs[frame],
                    pos=final_pos,
                    path=file_name,
                    time=times[frame],
                )
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(
            os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
        )
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
        if wandb.run:
            wandb.log(
                {"chain": [wandb.Video(gif_path, caption=gif_path, format="gif")]}
            )
