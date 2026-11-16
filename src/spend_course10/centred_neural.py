"""One plain MLP regressing signed date-centred log responses."""
import numpy as np
from spend_baselines.suite_models import Neural


class CentredNeural(Neural):
    def fit_residual(self, x, z, w, validation=None, epochs=None):
        # Reuse the baseline's direct MSE/Adam/early stopping implementation.
        # log1p(expm1(z)) is z, including negative residuals.  There is no
        # purchase classifier or second fitted predictor.
        z = np.asarray(z, dtype=float)
        if not np.isfinite(z).all() or z.min() < -12:
            raise ValueError("Invalid centred log responses")
        converted = None if validation is None else (validation[0], np.expm1(validation[1]))
        return super().fit(x, np.expm1(z), w, validation=converted, epochs=epochs)

    def _predict_tensor(self, x):
        import torch
        self.network.eval()
        with torch.no_grad():
            raw = torch.cat([self.network(chunk)[:, 0] for chunk in x.split(4096)])
        values = raw.cpu().numpy().astype(float)
        # A prediction call contains one whole known forecast population.
        return values - values.mean()
