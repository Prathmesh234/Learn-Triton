def simple_matmul(A, B):
    M = len(A)      # Number of rows in A
    K = len(A[0])   # Number of columns in A (AND rows in B)
    N = len(B[0])   # Number of columns in B

    # Initialize C with zeros
    C = [[0 for _ in range(N)] for _ in range(M)]

    for m in range(M):
        for n in range(N):
            # We are currently focused on C[m][n]
            # To calculate it, we iterate through K
            print(f"Calculating C[{m}][{n}]:")
            
            acc = 0
            for k in range(K):
                # k moves HORIZONTALLY across A
                # k moves VERTICALLY down B
                val_a = A[m][k]
                val_b = B[k][n]
                product = val_a * val_b
                acc += product
                
                print(f"  k={k}: A[{m}][{k}]({val_a}) * B[{k}][{n}]({val_b}) = {product}. Sum so far: {acc}")
            
            C[m][n] = acc
            print(f"  Final C[{m}][{n}] = {acc}\n")
            
    return C

if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    # K is 2 here.
    simple_matmul(A, B)
