# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.5.2
# ---

# All the IPython Notebooks in this lecture series are available at https://github.com/rajathkumarmp/Python-Lectures

# # Control Flow Statements

# ## If

# if some_condition:
#     
#     algorithm

x = 12
if x >10:
    print("Hello")

# ## If-else

# if some_condition:
#     
#     algorithm
#     
# else:
#     
#     algorithm

x = 12
if x > 10:
    print("hello")
else:
    print("world")

# ## if-elif

# if some_condition:
#     
#     algorithm
#
# elif some_condition:
#     
#     algorithm
#
# else:
#     
#     algorithm

x = 10
y = 12
if x > y:
    print("x>y")
elif x < y:
    print("x<y")
else:
    print("x=y")

# if statement inside a if statement or if-elif or if-else are called as nested if statements.

x = 10
y = 12
if x > y:
    print("x>y")
elif x < y:
    print("x<y")
    if x==10:
        print("x=10")
    else:
        print("invalid")
else:
    print("x=y")

# ## Loops

# ### For

# for variable in something:
#     
#     algorithm

for i in range(5):
    print(i)

# In the above example, i iterates over the 0,1,2,3,4. Every time it takes each value and executes the algorithm inside the loop. It is also possible to iterate over a nested list illustrated below.

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for list1 in list_of_lists:
    print(list1)

# A use case of a nested for loop in this case would be,

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for list1 in list_of_lists:
    for x in list1:
        print(x)

# ### While

# while some_condition:
#     
#     algorithm

# +
i = 1
while i < 3:
    print(i ** 2)
    i = i+1
print('Bye')

# do-untile
while True:
    #do something
    
    # check 
    if xxxx: break
# -

# ## Break

# As the name says. It is used to break out of a loop when a condition becomes true when executing the loop.

for i in range(100):
    print i
    if i>=7:
        break

# ## Continue

# This continues the rest of the loop. Sometimes when a condition is satisfied there are chances of the loop getting terminated. This can be avoided using continue statement. 

for i in range(10):
    if i>4:
        print("The end.")
        continue
    elif i<7:
        print(i)

# ## List Comprehensions

# Python makes it simple to generate a required list with a single line of code using list comprehensions. For example If i need to generate multiples of say 27 I write the code using for loop as,

res = []
for i in range(1,11):
    x = 27*i
    res.append(x)
print res

# Since you are generating another list altogether and that is what is required, List comprehensions is a more efficient way to solve this problem.

[27*x for x in range(1,11)]

# That's it!. Only remember to enclose it in square brackets

# Understanding the code, The first bit of the code is always the algorithm and then leave a space and then write the necessary loop. But you might be wondering can nested loops be extended to list comprehensions? Yes you can.

[27*x for x in range(1,20) if x<=10]

# Let me add one more loop to make you understand better, 

[27*z for i in range(50) if i==27 for z in range(1,11)]
