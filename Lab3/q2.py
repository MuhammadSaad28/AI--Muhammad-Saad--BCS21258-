# Python Program to convert temperatures to and from Celsius, Fahrenheit. [Formula: c/5 = f-32/9 [where c=temperature in Celsius and f=temperature in Fahrenheit]] expected output: 60°C is 140 in Fahrenheit. 45°F is 7 in Celsius.

def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5/9
    return celsius

def main():
    celsius = 60
    fahrenheit = celsius_to_fahrenheit(celsius)
    print(f"{celsius}°C is {fahrenheit} in Fahrenheit.")
    fahrenheit = 45
    celsius = fahrenheit_to_celsius(fahrenheit)
    print(f"{fahrenheit}°F is {celsius} in Celsius.")
    
if __name__ == "__main__":
    main()
    
# Output: 60°C is 140.0 in Fahrenheit.    