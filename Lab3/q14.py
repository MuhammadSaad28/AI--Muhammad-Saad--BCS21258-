# Python program that accepts a string and calculate the number of digits
# and letters.
# Sample Data: Python 3.2 Expected Output:
# Letters 6
# Digits 2

def print_lines():
    line = str(input("Enter a string:"))
    digits = 0
    letters = 0
    for i in line:
        if i.isdigit():
            digits += 1
        elif i.isalpha():
            letters += 1
    print("Letters", letters)
    print("Digits", digits)
    
def main():
    print_lines()
    
if __name__ == "__main__":
    main()
    
# Output: Enter a string:Python 3.2
# Letters 6
# Digits 2
# Enter a string:hello world
# Letters 10
# Digits 0
