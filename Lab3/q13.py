# Python program which accepts a sequence of comma separated 4 digit
# binary numbers as its input and print the numbers that are divisible by 5 in a
# comma separated sequence.
# Sample Data : 0100,0011,1010,1001,1100,1001
# Expected Output: 1010

def print_lines():
    line = str(input("Enter a sequence of comma separated 4 digit binary numbers:"))
    line = line.split(',')
    for i in line:
        if int(i, 2) % 5 == 0:
            print(i, end=',')
    print()
    
def main():
    print_lines()
    
if __name__ == "__main__":
    main()
    
# Output: 1010,