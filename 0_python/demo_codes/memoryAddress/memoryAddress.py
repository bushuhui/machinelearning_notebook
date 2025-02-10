#!/usr/bin/python3

a = 4354520
b = 4354520

print("a的地址为:",id(a))
print("b的地址为:",id(b))

c = d = f = 4354521

print("c的地址为:",id(c),"\nd的地址为:",id(d),"\nf的地址为:",id(f))
d = f-1

if (a is b):
    print("\na is b!")
else:
    print("\na is not b!")


if (a is d):
    print("\na is d!")
elif (a == d):
    print("\na is not d, But a==d!")
else:
    print("\na is not d, And a!=d!")



input("\n\n输入任意键退出！")
