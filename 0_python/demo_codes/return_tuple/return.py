#!/usr/bin/python3

def test(*args):
	print("参数为: ",args)
	return args

print(type(test(1,2,3,4)))
