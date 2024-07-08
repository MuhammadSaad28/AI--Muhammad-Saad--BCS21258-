# Python program which takes two digits m {row) and n {column) as input
# and generates a two-dimensional array. The element value in the i-th row and j-th
# column of the array should be i*j.
# Note:
# i = 0,1.., m-1
# j = 0,1, n-1.
# Test Data : Rows = 3, Columns = 4
# Expected Result : [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6]]

def two_dimensional_array(rows, columns):
    array = [[0 for col in range(columns)] for row in range(rows)]
    for row in range(rows):
        for col in range(columns):
            array[row][col] = row * col
    return array

def main():
    rows = 3
    columns = 4
    print(two_dimensional_array(rows, columns))
    
if __name__ == "__main__":
    main()
    
# Output: [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6]]