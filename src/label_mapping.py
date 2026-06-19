import numpy as np

CLASS_NAMES = [
    "Acute Otitis Media",
    "Chronic Otitis Media",
    "Cerumen Impaction",
    "Myringosclerosis",
    "Normal",
]

# Index theo đúng LABEL_MAP trong dataset.py:
# 0: Acute Otitis Media   -> Disease
# 1: Chronic Otitis Media -> Disease
# 2: Cerumen Impaction    -> Normal
# 3: Myringosclerosis     -> Normal
# 4: Normal               -> Normal
FIVE_CLASS_TO_BINARY = {
    0: 1,  # Acute Otitis Media   -> Disease (1)
    1: 1,  # Chronic Otitis Media -> Disease (1)
    2: 0,  # Cerumen Impaction    -> Normal  (0)
    3: 0,  # Myringosclerosis     -> Normal  (0)
    4: 0,  # Normal               -> Normal  (0)
}

BINARY_CLASS_NAME = ["Normal", "Disease"]

def map_5class_to_binary(labels_5class):
    labels_5class = np.asarray(labels_5class)
    binary = np.vectorize(FIVE_CLASS_TO_BINARY.get)(labels_5class)
    return binary

if __name__ == "__main__":
    test_input = np.array([0, 1, 2, 3, 4, 0, 4])
    expected = np.array([1, 1, 0, 0, 0, 1, 0])
    result = map_5class_to_binary(test_input)
    assert np.array_equal(result, expected), f"Mismatch: {result} vs {expected}"
    print("map_5class_to_binary OK:", result)
