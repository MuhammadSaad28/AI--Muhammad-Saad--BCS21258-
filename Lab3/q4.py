# Python Program to construct the following pattern, using a nested for loop
# *
# **
# ***
# ****
# *****
# ****
# ***
# **
# *

def pattern():
    for i in range(1, 6):
        for j in range(i):
            print("*", end="")
        print()
    for i in range(4, 0, -1):
        for j in range(i):
            print("*", end="")
        print()
        
def main():
    pattern()
    
if __name__ == "__main__":
    main()