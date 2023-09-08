#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
from student import student

class memory(student):
	def __init__(self,name_,age_,weight_,g):
		student.init(self,name_,age_,weight_,g)
	
	def __add__(self,other)
		return memory
'''

class Vector2d:
	def __init__(self, a_, b_):
		self.a = a_
		self.b = b_
	
	def __str__(self):
		return 'Vector2d(%2.1f,%2.1f)' % (self.a,self.b)
		
	def __add__(self,other):
		return Vector2d(self.a+other.a, self.b+other.b)
		

if __name__ == '__main__':
	a = Vector2d(3.2,4.4)
	b = Vector2d(1.3,-10.2)
	print('{0}+{1}={2}'.format(a,b,a+b))
	input("\n\n 输入任意键退出！")
	
	