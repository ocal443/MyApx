from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ProjectionResult:
    distances: npt.NDArray[np.floating] | None = None    # (N,)
    projections: npt.NDArray[np.floating] | None = None  # (N, 2)
    params: npt.NDArray[np.floating] | None = None       # (N,)
    indices: npt.NDArray[np.integer] | None = None       # (N,)
