from r3d3.experiment import R3D3ExperimentPlan

from tda.experiments.lid.lid_experiment_plan import (
    experiment_plan as lid_experiment_plan,
)
from tda.experiments.mahalanobis.mahalanobis_experiment_plan import (
    experiment_plan as ma_experiment_plan,
)

experiment_plan = R3D3ExperimentPlan.from_multiple_plans(
    [lid_experiment_plan, ma_experiment_plan]
)
