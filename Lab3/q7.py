# Python program that prints each item and its corresponding type from the following list.

def print_item_type(items):
    for item in items:
        print(item, type(item))
        
def main():
    items = [1, 2.5, "Hello", [1, 2, 3], (1, 2, 3), {"name": "John", "age": 25}]
    print_item_type(items)
    
if __name__ == "__main__":
    main()  
    
# Output: 1 <class 'int'>
#         2.5 <class 'float'>
#         Hello <class 'str'>
#         [1, 2, 3] <class 'list'>
#         (1, 2, 3) <class 'tuple'>
#         {'name': 'John', 'age': 25} <class 'dict'>

