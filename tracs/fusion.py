"""Multi-layer Perceptron for aligning TraCS scores with Backbone predictor, setting a weight for log-linear fusion."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Third Party
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy, spearmanr
from av2.datasets.motion_forecasting.eval import metrics


class FusionNet(nn.Module):

    def __init__(self, input_dim, n_emb=256, threshold=0.5):
        super().__init__()
        self.threshold = 0.5
        self.n_emb = n_emb
        self.n_feat = input_dim - n_emb
        self.emb_proj = nn.Sequential(
            nn.BatchNorm1d(n_emb),
            nn.Linear(n_emb, 32), nn.ReLU(), nn.Dropout(0.5),
        )
        self.net = nn.Sequential(
            nn.BatchNorm1d(self.n_feat + 32),
            nn.Linear(self.n_feat + 32, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),               nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),                 nn.Sigmoid()
        )

    _AGENT_TYPES = ["vehicle", "bus", "cyclist", "motorcyclist", "pedestrian"]

    @staticmethod
    def optimal_weight(log_tracs_scores, log_backbone_scores, predicted_tracks, gt_track, n_grid=100):
        """Vectorized grid search over weights."""

        W = np.linspace(0, 1, n_grid)
        fused_probs = FusionNet.fuse(log_backbone_scores, log_tracs_scores, W)  # (n_grid, K)
        ade_vector = metrics.compute_ade(predicted_tracks, gt_track)            # (K,)

        backbone_probs = np.exp(log_backbone_scores)
        backbone_probs = backbone_probs / backbone_probs.sum()

        scores = (fused_probs * ade_vector[None, :]).sum(axis=-1)  # (n_grid,)
        original_score = (backbone_probs * ade_vector).sum()
        best_weight_idx = np.argmin(scores)
        optimal_score = scores[best_weight_idx]

        return W[best_weight_idx], original_score, optimal_score

    @staticmethod
    def fuse(backbone_log_probs: np.ndarray, tracs_log_probs: np.ndarray, w: np.ndarray, num_outcomes: int = 6) -> np.ndarray:
        # Log-odds against uniform (pointwise mutual information):
        uniform_log_odds = np.log(1/num_outcomes)
        fused_probs = backbone_log_probs + w[..., None] * (tracs_log_probs - uniform_log_odds)

        # For numerical stability
        fused_probs = np.exp(fused_probs - fused_probs.max())

        # Normalize fused probabilities
        return fused_probs / fused_probs.sum(axis=-1, keepdims=True)

    @staticmethod
    def extract_features(predicted_tracks, *, focal_type=None, backbone_scores=None,
                         tracs_scores=None, n_agents=None, past_trajectory=None, embedding=None):
        parts = []

        endpoint_dists = np.linalg.norm(
        predicted_tracks[:, -1, :] - predicted_tracks[:, 0, :], axis=-1
        )  # (6,) predicted displacement per trajectory
        parts.append(np.array([endpoint_dists.mean(), endpoint_dists.std()], dtype=np.float32))  # 2 dims

        # Spearman rank correlation
        rho = spearmanr(backbone_scores, tracs_scores).correlation
        parts.append(np.array([0.0 if np.isnan(rho) else rho], dtype=np.float32))

        # KL divergences in both directions
        b = np.asarray(backbone_scores, dtype=np.float64) + 1e-10
        t = np.asarray(tracs_scores, dtype=np.float64) + 1e-10
        b /= b.sum(); t /= t.sum()
        parts.append(np.array([entropy(b, t), entropy(t, b)], dtype=np.float32))  # 2 dims

        # Relative entropy H(tracs) - H(backbone): negative = TraCS is sharper
        parts.append(np.array([entropy(t, base=2) - entropy(b, base=2)], dtype=np.float32))  # 1 dim

        # Sorted score vectors (in addition to the unsorted ones already appended)
        parts.append(np.sort(np.asarray(backbone_scores, dtype=np.float32)))  # 6 dims
        parts.append(np.sort(np.asarray(tracs_scores, dtype=np.float32)))     # 6 dims

        # Cross-classifier score at the other's top pick
        parts.append(np.asarray(tracs_scores[np.argmax(backbone_scores)], dtype=np.float32)[None])  # 1 dim
        parts.append(np.asarray(backbone_scores[np.argmax(tracs_scores)], dtype=np.float32)[None])  # 1 dim

        b_sorted = np.sort(backbone_scores)[::-1]
        t_sorted = np.sort(tracs_scores)[::-1]
        parts.append(np.array([b_sorted[0] - b_sorted[1],
                                t_sorted[0] - t_sorted[1]], dtype=np.float32))  # 2 dims

        b_ranks = np.argsort(np.argsort(backbone_scores)[::-1])  # rank 0=best
        t_ranks = np.argsort(np.argsort(tracs_scores)[::-1])
        parts.append(np.array([b_ranks[np.argmax(tracs_scores)],   # where backbone ranks TraCS's top
                                t_ranks[np.argmax(backbone_scores)]], dtype=np.float32))  # 2 dims

        # Comment out any line below to disable that feature:
        if focal_type is not None:
            parts.append(
                np.eye(len(FusionNet._AGENT_TYPES), dtype=np.float32)[FusionNet._AGENT_TYPES.index(
                    focal_type) if focal_type in FusionNet._AGENT_TYPES else 0]
            )  # 5 dims
        if backbone_scores is not None:
            scores = np.asarray(backbone_scores, dtype=np.float32)
            parts.append(scores.flatten())  # 6 dims  # Do not sort these!
            parts.append(entropy(scores, base=2)[None])  # 1 dim
        if tracs_scores is not None:
            scores = np.asarray(tracs_scores, dtype=np.float32)
            parts.append(scores.flatten())  # 6 dims  # Do not sort these!
            parts.append(entropy(scores, base=2)[None])  # 1 dim
        if n_agents is not None:
            parts.append(np.array([n_agents], dtype=np.float32))  # 1 dim
        if past_trajectory is not None:  # (n_steps, 2) -> 2*n_steps dims
            p = np.asarray(past_trajectory, dtype=np.float32)
            p = p - p[-1:]  # center at last observed position
            p /= (p.std() + 1e-6)
            parts.append(p.flatten())
        
        # Backbone latent embedding (appended last so hand-crafted feature indices stay stable)
        if embedding is not None:
            parts.append(np.asarray(embedding, dtype=np.float32).flatten())

        return np.concatenate(parts)

    def fit(self, X_train, Y_train, X_val, Y_val, batch_size, n_epochs, min_improvement, patience, device="cuda", verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            self.train()
            for idx in torch.randperm(len(X_train)).split(batch_size):
                optimizer.zero_grad()
                nn.functional.binary_cross_entropy(self(X_train[idx]), Y_train[idx]).backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                train_loss = nn.functional.binary_cross_entropy(self(X_train), Y_train).item()
                val_loss = nn.functional.binary_cross_entropy(self(X_val), Y_val).item()

            scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            improved = val_loss < best_val_loss - min_improvement
            if improved:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or improved:
                lr = optimizer.param_groups[0]["lr"]
                tag = " *" if improved else ""
                if verbose:
                    print(f"Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  "
                        f"lr={lr:.2e}  patience={patience_counter}/{patience}{tag}")

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} — best val={best_val_loss:.4f}")
                break

        self.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if verbose:
            print(f"Restored best model weights (val={best_val_loss:.4f})")

        return np.array(train_losses), np.array(val_losses)

    def forward(self, x):
        feat, emb = x[:, :self.n_feat], x[:, self.n_feat:]
        emb = self.emb_proj(emb)
        return self.net(torch.cat([feat, emb], dim=-1))[..., 0]

    def predict_weight(self, predicted_tracks, *, focal_type=None, backbone_scores=None,
        tracs_scores=None, n_agents=None, past_trajectory=None, embedding=None):
        feats = FusionNet.extract_features(
            predicted_tracks, focal_type=focal_type, backbone_scores=backbone_scores,
            tracs_scores=tracs_scores, n_agents=n_agents, past_trajectory=past_trajectory,
            embedding=embedding,
        )
        x = torch.tensor(feats).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            return self.forward(x).squeeze(0).cpu().numpy()
        
    def predict(self, predicted_tracks, *, focal_type=None, backbone_scores=None,
        tracs_scores=None, n_agents=None, past_trajectory=None, embedding=None):
        w = self.predict_weight(predicted_tracks, focal_type=focal_type, backbone_scores=backbone_scores,
            tracs_scores=tracs_scores, n_agents=n_agents, past_trajectory=past_trajectory,
            embedding=embedding)
        
        if w >= self.threshold:
            return np.asarray(1.0)
        else:
            return np.asarray(0.0)
            