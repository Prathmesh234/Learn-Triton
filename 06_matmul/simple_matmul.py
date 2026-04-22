# SIMPLE PURE PYTHON MATRIX MULTIPLICATION
# No Triton, no tiling, just the basic algorithm.

def simple_matmul(A, B):
    M = len(A)
    N = len(B[0])
    K = len(A[0])
    C = []
    for row in range(M):
        for col in range(N):
            acc = 0
            for k in range(K):
                product = A[row][k]*B[k][col]
                acc += product
            C.append(acc)
            
        

    
            
    return C

if __name__ == "__main__":
    # Example 3x2 matrix A
    A = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]

    # Example 2x3 matrix B
    B = [
        [7, 8, 9],
        [1, 2, 3]
    ]

    # Result should be 3x3
    result = simple_matmul(A, B)

    print("Matrix A:")
    for row in A: print(row)
    
    print("\nMatrix B:")
    for row in B: print(row)
    
    print("\nResult Matrix C (A @ B):")
    for row in result:
        print(row)
