# Python program that accepts a sequence of lines {blank line to terminate)
# as input and prints the lines as output {all characters in lower case).

def print_lines():
    line = str(input("Enter a sequence of lines:")) 
    print(line.lower())
        
        
def main():
    print_lines()
    
if __name__ == "__main__":
    main()
    
# Output: hello world how are you today