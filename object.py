"""def Name():
    if age >= 18 and age < 65:
        print("You are an adult.")
    elif age >= 13: 
        print("You are a teenager.")
    else:
        print("You are a child.")
age=int(input("Enter your age: "))
Name()"""
# printing today date and time
import datetime
now = datetime.datetime.now()
print("Current date and time: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

