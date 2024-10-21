from typing import Dict, List, Optional, Tuple, Union
from prdc import compute_prdc
from torch import BoolTensor, FloatTensor
from pytorch_fid.fid_score import calculate_frechet_distance
import numpy as np

Feats = Union[FloatTensor, List[FloatTensor]]
Layout = Tuple[np.ndarray, np.ndarray]
def __to_numpy_array(feats: Feats) -> np.ndarray:
    if isinstance(feats, list):
        # flatten list of batch-processed features
        if isinstance(feats[0], FloatTensor):
            feats = [x.detach().cpu().numpy() for x in feats]
    else:
        feats = feats.detach().cpu().numpy()
    return np.concatenate(feats)


def compute_generative_model_scores(
    feats_real: Feats,
    feats_fake: Feats,
) -> Dict[str, float]:
    """
    Compute precision, recall, density, coverage, and FID.
    """
    feats_real = __to_numpy_array(feats_real)
    feats_fake = __to_numpy_array(feats_fake)

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    results = compute_prdc(
        real_features=feats_real, fake_features=feats_fake, nearest_k=5
    )
    results["fid"] = calculate_frechet_distance(
        mu_real, sigma_real, mu_fake, sigma_fake
    )

    return results