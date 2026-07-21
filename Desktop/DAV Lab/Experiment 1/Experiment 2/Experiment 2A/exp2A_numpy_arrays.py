"""
Experiment 2 A) Working with NumPy arrays / NumPy Operations and Array Manipulations

AIM:
To understand and implement various NumPy operations, including array creation, indexing,
slicing, element-wise operations, aggregations, boolean operations, fancy indexing,
reshaping, and structured arrays.
"""
import numpy as np

print("NumPy Version:", np.__version__)

# Creating different types of arrays
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_0d = np.array(42)
arr_ones = np.ones((3, 3))

print("\n1D array:", arr_1d)
print("2D array:\n", arr_2d)
print("0D array:", arr_0d)
print("Ones array:\n", arr_ones)

# Indexing and Slicing
print("\nElement at index 2 in 1D array:", arr_1d[2])
print("Element at row 1, column 2 in 2D array:", arr_2d[1, 2])
print("Slice from 1D array:", arr_1d[1:4])
print("Slice row 1 from 2D array:", arr_2d[1, :])

# Element-wise operations
arr_a = np.array([10, 20, 30])
arr_b = np.array([1, 2, 3])
print("\nAddition:", arr_a + arr_b)
print("Subtraction:", arr_a - arr_b)
print("Multiplication:", arr_a * arr_b)
print("Division:", arr_a / arr_b)
print("Scalar Multiplication:", arr_a * 2)

# Aggregations
print("\nSum:", np.sum(arr_a))
print("Mean:", np.mean(arr_a))
print("Standard Deviation:", np.std(arr_a))

# Element-wise comparison
print("\nElement-wise comparison:", arr_a > arr_b)

# Boolean masking
print("Elements greater than 15:", arr_a[arr_a > 15])

# Fancy Indexing
indices = [0, 2]
print("\nSelected elements:", arr_a[indices])

# Reshape
reshaped_arr = arr_1d.reshape(5, 1)
print("\nReshaped 1D array to 2D:\n", reshaped_arr)

# Structured array
structured_arr = np.array([(25, 90.5), (30, 85.2)], dtype=[('age', 'i4'), ('score', 'f4')])
print("\nStructured array:", structured_arr)

print("\nRESULT: All NumPy operations executed and verified successfully.")
