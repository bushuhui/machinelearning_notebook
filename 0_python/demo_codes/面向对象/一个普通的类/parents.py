#!/usr/bin/python
# -*- coding: UTF-8 -*-

class person:
	name = ''
	age = 0
	weight = 0
	'''
	__weight 私有属性，外部无法访问
	'''
	def __init__(self,name_,age_,weight_):
		self.name=name_
		self.age=age_
		self.weight=weight_
	def speek(self):
		print("我叫%s，我今年%d岁了，体重为：%d kg!\n" % (self.name, self.age, self.weight) )


	
if __name__ == '__main__':
	i = person("张臻炜",27,52)
	i.speek()
	input("\n\n 输入任意键退出！")
'''	
else:
	input("\n\n import form outside!\n输入任意键退出！")
'''

