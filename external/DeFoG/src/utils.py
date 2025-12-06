import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb


#################################################################
import torch.nn.functional as F
import numpy as np
#################################################################


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs("graphs")
        os.makedirs("chains")
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs("graphs/" + args.general.name)
        os.makedirs("chains/" + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = X * norm_values[0] + norm_biases[0]
    E = E * norm_values[1] + norm_biases[1]
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def symmetrize_and_mask_diag(E):
    # symmetrize the edge matrix
    upper_triangular_mask = torch.zeros_like(E)
    indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1)
    if len(E.shape) == 4:
        upper_triangular_mask[:, indices[0], indices[1], :] = 1
    else:
        upper_triangular_mask[:, indices[0], indices[1]] = 1
    E = E * upper_triangular_mask
    E = E + torch.transpose(E, 1, 2)
    # mask the diagonal
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return E


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def to_device(self, device):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device) if self.y is not None else None
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    #######################################################################
    def enforce_hard_constraint(self, fixed_data, node_mask_fixed, edge_mask_fixed):
        """
        Enforces a hard constraint on the current PlaceHolder state (self).
        It overwrites the features in the regions defined by the fixed masks
        with the corresponding clean features from fixed_data.

        """

        x_num_classes = self.X.size(-1)
        e_num_classes = self.E.size(-1)

        fixed_X_onehot = F.one_hot(fixed_data.X, num_classes=x_num_classes).float()
        fixed_E_onehot = F.one_hot(fixed_data.E, num_classes=e_num_classes).float()

        node_mask_expanded = node_mask_fixed.unsqueeze(-1)
        edge_mask_expanded = edge_mask_fixed.unsqueeze(-1)

        self.X = torch.where(node_mask_expanded, fixed_X_onehot, self.X)
        self.E = torch.where(edge_mask_expanded, fixed_E_onehot, self.E)

        return self


    #######################################################################

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def split(self, node_mask):
        """Split a PlaceHolder representing a batch into a list of placeholders representing individual graphs."""
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i], dim=0)
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list


######################################################################

def pad_and_expand(tensor, target_shape, batch_size, value=0):
    """
    Pads the tensor to target_shape and expands it to batch_size.
    tensor: torch.Tensor, shape (N, ...) or (N, N, ...)
    target_shape: tuple of ints, final shape after padding (without batch)
    batch_size: int
    """
    # compute padding for each dim
    padding = []
    for i in reversed(range(len(target_shape))):
        pad = target_shape[i] - tensor.shape[i]
        padding.extend([0, pad])
    tensor = torch.nn.functional.pad(tensor, padding, value=value)

    # expand to batch
    if batch_size > 1:
        tensor = tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor

def load_fixed_subgraph_masks(pt_file, total_nodes):
    """
    Creates masks for fixed nodes and edges based on a .pt file.
    """
    data = torch.load(pt_file)
    X_fixed = data['X']
    E_fixed = data['E']

    num_fixed_nodes = X_fixed.shape[0]

    node_mask_fixed = torch.zeros((total_nodes,), dtype=torch.bool)
    edge_mask_fixed = torch.zeros((total_nodes, total_nodes), dtype=torch.bool)

    node_indices = list(range(num_fixed_nodes))
    node_mask_fixed[node_indices] = True

    for i in range(num_fixed_nodes):
        for j in range(num_fixed_nodes):
            edge_mask_fixed[node_indices[i], node_indices[j]] = True

    return X_fixed, E_fixed, node_mask_fixed, edge_mask_fixed


#######################################################################




def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    if cfg.general.test_only is None:
        name = f"{cfg.general.name}"
    else:
        if cfg.sample.search:
            name = f"{cfg.general.name}_search_{cfg.sample.search}"
        else:
            name = f"{cfg.general.name}_eta{cfg.sample.eta}_{cfg.sample.rdb}_{cfg.sample.time_distortion}"
    kwargs = {
        "name": name,
        "project": f"graph_dfm_{cfg.dataset.name}",
        "config": config_dict,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    config_dict["general"]["local_dir"] = os.getcwd()
    wandb.init(**kwargs)
    wandb.save("*.txt")
