import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
from pathlib import Path
from .run_defog_helper import run_defog_experiments
import pandas as pd

class VUNPredictorMLP:
    def __init__(self, df_existing, hidden_sizes=[32, 32], lr=1e-3, epochs=500, batch_size=32, device=None):
        """
        MLP for predicting VUN given num_steps, distortion, eta, omega.
        
        Args:
            df_existing: DataFrame with existing experiment results to train on
            hidden_sizes: list with sizes of hidden layers
            lr: learning rate
            epochs: number of training epochs
            batch_size: training batch size
            device: torch device to use (cpu or cuda); if None, auto-detect
        """
        self.df = df_existing.copy()
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # One-hot encode de distorsión
        self.enc = OneHotEncoder(sparse_output=False)
        dist_encoded = self.enc.fit_transform(self.df[["distortion"]])

        # Construir X y y
        X = np.column_stack([self.df["num_steps"].values, dist_encoded, self.df[["eta", "omega"]].values])
        y = self.df["vun_mean"].values.reshape(-1, 1)

        # Escalado
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        Xn = self.X_scaler.fit_transform(X)
        yn = self.y_scaler.fit_transform(y)

        self.Xn = torch.tensor(Xn, dtype=torch.float32).to(self.device)
        self.yn = torch.tensor(yn, dtype=torch.float32).to(self.device)

        input_size = Xn.shape[1]
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size

        print(f"Training MLP in {self.device}...")
        self._train()
        print("MLP training completed.")

    def _train(self):
        """ 
        Trains the MLP model.
        """
        dataset = torch.utils.data.TensorDataset(self.Xn, self.yn)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                y_pred = self.model(xb)
                loss = self.criterion(y_pred, yb)
                loss.backward()
                self.optimizer.step()

    def predict_vun(self, num_steps, distortion, eta, omega):
        """
        Predicts VUN for given parameters using the trained MLP.
        
        Args:
            num_steps: number of sampling steps
            distortion: distortion type (string)
            eta: eta parameter
            omega: omega parameter
        Returns:    
            predicted VUN value
        """
        dist_vec = np.zeros(len(self.enc.categories_[0]))
        dist_idx = list(self.enc.categories_[0]).index(distortion)
        dist_vec[dist_idx] = 1.0

        x_raw = np.concatenate([[num_steps], dist_vec, [eta, omega]]).reshape(1, -1)
        x_scaled = self.X_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred_norm = self.model(x_tensor).cpu().numpy()[0, 0]
        y_pred = self.y_scaler.inverse_transform([[y_pred_norm]])[0, 0]
        return y_pred
    
    def optimize_params(self, num_steps):
        """
        Optimizes eta and omega for given num_steps to maximize predicted VUN.
        
        Args:
            num_steps: number of sampling steps
        Returns:
            best configuration dict with keys: num_steps, distortion, eta, omega, pred_score
        """

        eta_bounds = (self.df["eta"].min(), self.df["eta"].max())
        omega_bounds = (self.df["omega"].min(), self.df["omega"].max())
        bounds = [eta_bounds, omega_bounds]

        best_val = -np.inf
        best_cfg = None

        for distortion in self.enc.categories_[0]:
            def obj_fn_global(x):
                eta, omega = x
                return -self.predict_vun(
                    num_steps=num_steps,
                    distortion=distortion,
                    eta=eta,
                    omega=omega
                )

            res = differential_evolution(obj_fn_global, bounds)

            if res.success:
                mu_pred = -res.fun
                if mu_pred > best_val:
                    best_val = mu_pred
                    best_cfg = {
                        "num_steps": num_steps,
                        "distortion": distortion,
                        "eta": res.x[0],
                        "omega": res.x[1],
                        "pred_score": mu_pred
                    }

        return best_cfg
    
def run_dataset_with_model(
    dataset: str,
    outputs_dir: Path,
    defog_src_path: Path,
    num_steps_list,
    checkpoint: str,
    mlp_model=None,
    use_model=True,
):
    """
    Runs Defog experiments on the specified dataset using either model-predicted parameters or default ones.
    
    Args:
        dataset: "planar" or "qm9"
        outputs_dir: Path to the outputs directory
        defog_src_path: Path to the Defog source code
        num_steps_list: list of number of steps to run experiments with
        checkpoint: checkpoint file to use for testing
        mlp_model: VUNPredictorMLP model for parameter prediction (if None, default params are used)
        use_model: whether to use the model for parameter prediction
    """
    outputs_dir.mkdir(exist_ok=True, parents=True)

    experiments = ["qm9_with_h_conditional.ckpt"] if dataset == "qm9" else ["planar.ckpt"]

    for steps in num_steps_list:
        print(f"\nRunning {dataset} experiment with {steps} steps...")

        run_dir = outputs_dir / f"{dataset}_{steps}_steps"

        if mlp_model is not None and use_model:
            if dataset == "planar":
                best_cfg = mlp_model.optimize_params(num_steps=steps)
                eta = best_cfg["eta"]
                omega = best_cfg["omega"]
                distortion = best_cfg["distortion"]
                print(f"Predicted: {best_cfg['pred_score']}")
                print(f"Using model parameters: eta={eta}, omega={omega}, distortion={distortion}")
            else:
                csv_path = defog_src_path.parent / "results" / "tables" / "lookup_table.csv"
                df_lookup = pd.read_csv(csv_path)
                eta = df_lookup.loc[df_lookup["num_steps"] == steps, "eta"].values[0]
                omega = df_lookup.loc[df_lookup["num_steps"] == steps, "omega"].values[0]
                distortion = df_lookup.loc[df_lookup["num_steps"] == steps, "distortion"].values[0]
                print(f"Using lookup table parameters: eta={eta}, omega={omega}, distortion={distortion}")
        else:
            # default parameters
            if dataset == "planar":
                eta = 50.0
                omega = 0.05
                distortion = "polydec"
            else:
                eta = 0.0
                omega = 0.05
                distortion = "polydec"

        overrides = {
            "sample.sample_steps": steps,
            "sample.eta": eta,
            "sample.omega": omega,
            "sample.time_distortion": distortion,
            "general.test_only": checkpoint,
            "hydra.run.dir": str(run_dir.resolve()),
            "+general.num_sample_fold": 5,
            "+general.visualization": False,
        }

        run_defog_experiments(
            experiments=experiments,
            num_steps_list=[steps],
            outputs_dir=outputs_dir,
            defog_src_path=defog_src_path,
            overrides_per_experiment={dataset: overrides},
            name_prefix=f"{dataset}_{steps}",
            verbose=True,
        )