import numpy as np
import pandas as pd

def check_answer(out, expected):
    """
    Compare student output with expected output.
    Works for NumPy arrays, scalars, and pandas DataFrames.
    """
    # Case 1: NumPy array
    if isinstance(expected, np.ndarray):
        if np.array_equal(out, expected):
            print("✅ Correct")
        else:
            print("❌ Incorrect")
            print("Your output:\n", out)
            print("Expected:\n", expected)
    
    # Case 2: pandas DataFrame
    elif isinstance(expected, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(out, expected, check_dtype=False)
            print("✅ Correct")
        except AssertionError as e:
            print("❌ Incorrect")
            print("Your output:\n", out)
            print("Expected:\n", expected)
    
    # Case 3: scalar (int, float, etc.)
    else:
        if out == expected:
            print("✅ Correct")
        else:
            print("❌ Incorrect")
            print("Your output:", out)
            print("Expected:", expected)
