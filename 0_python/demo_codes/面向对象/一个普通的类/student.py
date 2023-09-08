#!/usr/bin/python
# -*- coding: UTF-8 -*-

from parents import person
'''
class person:
	name = ''
	age = 0
	weight = 0
	def __init__(self,name_,age_,weight_):
		self.name=name_
		self.age=age_
		self.weight=weight_
	def speek(self):
		print("我叫%s，我今年%d岁了，体重为：%d kg!\n" % (self.name, self.age, self.__weight) )
'''
class student(person):
	grade = ''
	def __init__(self,name_,age_,weight_,g):
		person.__init__(self,name_,age_,weight_)
		self.grade=g
	def speak(self):
		print("我叫%s，我今年%d岁了，体重为：%d kg， 我在读%s年级!\n" % (self.name, self.age, self.weight,self.grade))

if __name__ == '__main__':		
	i = student("张臻炜",27,52,"研究生三")
	i.speak()
	input("\n\n 输入任意键退出！")