# Python program to check the validity of password input by users. Validation
# At least 1 letter between [a-z] and 1 letter between [A-Z].
# At least 1 number between [0-9].
# At least 1 character from [$#@].
# Minimum length 6 characters.
# Maximum length 16 characters.

def check_password():
    password = str(input("Enter a password:"))
    if len(password) < 6 or len(password) > 16:
        print("Password must be between 6 and 16 characters.")
        return
    if not any(char.isupper() for char in password):
        print("Password must contain at least one uppercase letter.")
        return
    if not any(char.islower() for char in password):
        print("Password must contain at least one lowercase letter.")
        return
    if not any(char.isdigit() for char in password):
        print("Password must contain at least one digit.")
        return
    if not any(char in "$#@!" for char in password):
        print("Password must contain at least one of the following characters: $, #, @, or !.")
        return
    print("Password is valid.")
    
def main():
    check_password()

if __name__ == "__main__":
    main()
    
# Output: Enter a password:Password1
# Password is valid.