# Python program that prints all the numbers from O to 6 except 3 and 6.
# Note : Use 'continue' statement.

def print_numbers():
    for i in range(7):
        if i == 3 or i == 6:
            continue
        print(i)
        
def main():
    print_numbers()
    
if __name__ == "__main__":
    main()
    
# Output: 0
#         1
#         2
#         4
#         5
#         7

