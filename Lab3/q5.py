# Python program that accepts a word from the user and reverse it.

def reverse_string(string):
    return string[::-1]

def main():
    string = input("Enter a word: ")
    print("Reversed word: ", reverse_string(string))
    
if __name__ == "__main__":
    main()

