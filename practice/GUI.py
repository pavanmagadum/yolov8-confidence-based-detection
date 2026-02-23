#open window
# import tkinter as tk
# import tkinter
# top = tkinter.Tk() 
# top.title("welcome to tkinter")
# top.geometry("400x300")
# top.mainloop() 

# adding an button
# from tkinter import *
# top = Tk()
# top.geometry("400x300")
# btn1 = Button(top, text="Login")
# btn1.pack(side = LEFT)
# top.mainloop()


#grid method
# from tkinter import *
# parent = Tk()
# parent.geometry("400x300")
# parent.title("Student Registration Form")
# name = Label(parent, text="Name")
# name.grid(row=0, column=0, pady = 10, padx = 5)
# e1 = Entry(parent)
# e1.grid(row=0, column=1 )
# regno = Label(parent, text="Reg No:")
# regno.grid(row=1, column=0, pady = 10, padx = 5,)
# e2 = Entry(parent)
# e2.grid(row=1, column=1)
# btn = Button(parent, text="Submit")
# btn.grid(row=2, column=1)
# parent.mainloop()

# #spalce method
# from tkinter import *
# parent = Tk()
# parent.geometry("400x300")
# parent.title("Student Registration Form")
# name = Label(parent, text="Name:")
# name.place(x=50, y=50)
# e1 = Entry(parent)
# e1.place(x=150, y=50)
# regno = Label(parent, text="Reg No:")
# regno.place(x=50, y=100)
# e2 = Entry(parent)
# e2.place(x=150, y=100)
# btn = Button(parent, text="Submit")
# btn.place(x=150, y=150)
# parent.mainloop()

# function to display the name and reg no
import tkinter as tk
from tkinter import *
def show_name():
    name = entry.get()
    regno = entry1.get()
    result_lable.config(text=f"Hello:\n{name}\nReg No: {regno}")

    # create the main window
root = tk.Tk()
root.title("Student Registration Form")
root.geometry("400x300")
root.configure(bg="lightblue")
 
#title
tk.Label(root, text="Student Registration Form", font=("Arial", 16), bg="lightblue").pack(pady=10)
#entry box
entry = tk.Entry(root, font=("Arial", 12), width=20)
entry.pack(pady=10)
entry1 = tk.Entry(root, font=("Arial", 12), width=20)
entry1.pack(pady=10)

#submit button
tk.Button(root, text="Submit",command = show_name,bg = "blue", fg = "white", 
          font=("Arial", 12)).pack(pady=10)

#result label
result_lable = tk.Label(root, text="", font=("Arial", 12), bg="lightblue")
result_lable.pack(pady=10)
#run the main loop
root.mainloop() 