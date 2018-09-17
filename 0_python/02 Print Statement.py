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

# # Print Statement

# The **print** statement can be used in the following different ways :
#
#     - print("Hello World")
#     - print("Hello", <Variable Containing the String>)
#     - print("Hello" + <Variable Containing the String>)
#     - print("Hello %s" % <variable containing the string>)

print("Hello World")

# In Python, single, double and triple quotes are used to denote a string.
# Most use single quotes when declaring a single character. 
# Double quotes when declaring a line and triple quotes when declaring a paragraph/multiple lines.

print('Hey')

print("""My name is Rajath Kumar M.P.

I love Python.""")

# Strings can be assigned to variable say _string1_ and _string2_ which can called when using the print statement.

# + {"scrolled": true}
string1 = 'World'
print('Hello', string1)

string2 = '!'
print('Hello', string1, string2)
# -

# String concatenation is the "addition" of two strings. Observe that while concatenating there will be no space between the strings.

print('Hello' + string1 + string2)

# **%s** is used to refer to a variable which contains a string.

print("Hello %s" % string1)

# Similarly, when using other data types
#
#     - %s -> string
#     - %d -> Integer
#     - %f -> Float
#     - %o -> Octal
#     - %x -> Hexadecimal
#     - %e -> exponential
#     
# This can be used for conversions inside the print statement itself.

print("Actual Number = %d" % 18)
print("Float of the number = %f" % 18)
print("Octal equivalent of the number = %o" % 18)
print("Hexadecimal equivalent of the number = %x" % 18)
print("Exponential equivalent of the number = %e" % 18)

# When referring to multiple variables parenthesis is used.

print "Hello %s %s" %(string1,string2)

# ## Other Examples

# The following are other different ways the print statement can be put to use.

print("I want %%d to be printed %s" %'here')

print('_A'*10)

print("Jan\nFeb\nMar\nApr\nMay\nJun\nJul\nAug")

print("\n".join("Jan Feb Mar Apr May Jun Jul Aug".split(" ")))

print("I want \\n to be printed.")

print """
Routine:
\t- Eat
\t- Sleep\n\t- Repeat
"""

# # PrecisionWidth and FieldWidth

# Fieldwidth is the width of the entire number and precision is the width towards the right. One can alter these widths based on the requirements.
#
# The default Precision Width is set to 6.

"%f" % 3.121312312312

# Notice upto 6 decimal points are returned. To specify the number of decimal points, '%(fieldwidth).(precisionwidth)f' is used.

"%.5f" % 3.121312312312

# If the field width is set more than the necessary than the data right aligns itself to adjust to the specified values.

"%9.5f" % 3.121312312312

# Zero padding is done by adding a 0 at the start of fieldwidth.

"%020.5f" % 3.121312312312

# For proper alignment, a space can be left blank in the field width so that when a negative number is used, proper alignment is maintained.

print "% 9f" % 3.121312312312
print "% 9f" % -3.121312312312

# '+' sign can be returned at the beginning of a positive number by adding a + sign at the beginning of the field width.

print "%+9f" % 3.121312312312
print "% 9f" % -3.121312312312

# As mentioned above, the data right aligns itself when the field width mentioned is larger than the actualy field width. But left alignment can be done by specifying a negative symbol in the field width.

"%-9.3f" % 3.121312312312
