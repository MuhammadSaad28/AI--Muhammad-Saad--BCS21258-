# While Loop
count = 0
while count < 5:
    print(count, " is less than 5")
    count = count + 1
    
#While Loop in Single Statement
count = 0
while count < 5: print(count, " is less than 5"); count = count + 1

# for in loop
list1 = ['apple', 'banana', 'cherry']
for i in list1:
    print(i)
    
# iterate by index of sequence
for index in range(len(list1)):
    print(list1[index])
    
# Loop Control Statements
# break statement
for i in list1:
    if i == 'banana':
        break
    print(i)
    
# continue statement
for i in list1:
    if i == 'banana':
        continue
    print(i)
    
# Python Functions with and without parameters

# Without Parameters
def greetWithoutParameter():
    print("Hello, Good Morning!")
    
    

# With Parameters
name = "Saad"
def greetWithParameter(name):
    print("Hello, " + name + ". Good Morning!")  
    
# Calling a Function
greetWithoutParameter()
greetWithParameter(name) 

# Passing a List as a Parameter
def greetWithListParameter(names):
    for name in names:
        print("Hello, " + name + ". Good Morning!")
        

names = ["Saad", "Warasat", "Muzammil"] 
greetWithListParameter(names)      


# Return Values
def add(a, b):
    return a + b

sum = add(5, 3)
print("Sum is: ", sum)

# Keyword Arguments
def greet(name, msg):
    print("Hello", name + ', ' + msg)
    
greet(name="Saad", msg="Good Morning!")

# Python Classes and Objects

# Creating a Class
class MyClass:
    x = 5
    
# Creating an Object
p1 = MyClass()
print(p1.x)

# The init() Function
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
p2 = Person("Saad", 20)
print(p2.name)
print(p2.age)

# Object Methods
class Person1:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def myfunc(self):
        print("Hello my name is " + self.name)
        
p3 = Person1("Saad", 20)
p3.myfunc()
 