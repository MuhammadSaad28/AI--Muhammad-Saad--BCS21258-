# Python program to count the number of even and odd numbers from a series of numbers.

def count_even_odd_numbers(numbers):
    even_count = 0
    odd_count = 0
    for number in numbers:
        if number % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    return even_count, odd_count

def main():
    numbers = [int(x) for x in input("Enter a series of numbers: ").split()]
    even_count, odd_count = count_even_odd_numbers(numbers)
    print("Number of even numbers: ", even_count)
    print("Number of odd numbers: ", odd_count)
    
if __name__ == "__main__":
    main()
    
# Input: 1 2 3 4 5 6 7 8 9 10
# Output: Number of even numbers:  5
#         Number of odd numbers:  5
    
