# Python program to get the Fibonacci series between Oto 50. Note : The
# Fibonacci Sequence is the series of numbers :
# 0, 1, 1, 2, 3, 5, 8, 13, 21, ....
# Every next number is found by adding up the two numbers before it.
# Expected Output: 11235813 2134

def fibonacci_series():
    a, b = 0, 1
    while a <= 50:
        a, b = b, a + b
        if(a <= 50):
            print(a, end=" ")
        
def main():
    fibonacci_series()
    
if __name__ == "__main__":
    main()
    
# Output: 1 1 2 3 5 8 13 21 34