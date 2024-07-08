# Python Program to guess a number between 1 to 9. Note:User is prompted to enter a guess. If the user guesses wrong then the prompt appears again until the guess is correct, on successful guess, user will get a "Well guessed!" message, and the program will exit.

import random

def guess_number():
    number = random.randint(1, 9)
    while True:
        guess = int(input("Guess a number between 1 and 9: "))
        if guess == number:
            print("Well guessed!")
            break
        else:
            print("Try again!")
            
def main():
    guess_number()
    
if __name__ == "__main__":
    main()