import numpy as np
from deepsig import aso


def is_a_better_than_b(A, B):
    print(f"Is {A[1]} better than {B[1]}?")
    min_eps = aso(A[0], B[0])
    if min_eps < 0.5:
        print("yes")
    else:
        print("no")
    print(f"min_eps={min_eps}")


baseline = np.array([0.351672133, 0.350331809, 0.35267308, 0.344824149, 0.354718443])

singleMLM_main = np.array([0.338263467, 0.344192233, 0.330652293, 0.336767071, 0.335418264])
multiMLM_main = np.array([0.333798811, 0.336350386, 0.330902119, 0.32777429, 0.341812904])

singleMLM_one_batch = np.array([0.318050717, 0.277818404, 0.318534103, 0.33638409, 0.338945553])
multiMLM_one_batch = np.array([0.219025489, 0.233231048, 0.288300623, 0.229920061, 0.233846487])

is_a_better_than_b((singleMLM_main, "singleMLM_main"), (singleMLM_one_batch, "singleMLM_one_batch"))  # yes
is_a_better_than_b((multiMLM_main, "multiMLM_main"), (multiMLM_one_batch, "multiMLM_one_batch"))  # yes
is_a_better_than_b((multiMLM_main, "singleMLM_main"), (baseline, "baseline"))  # no
is_a_better_than_b((multiMLM_main, "multiMLM_main"), (baseline, "baseline"))  # no
