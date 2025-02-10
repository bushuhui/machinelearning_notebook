#!/usr/bin/python
# -*- coding: UTF-8 -*-

def travel(listData):
	for i in range(len(listData)):
		print(listData[i])
	return	

data=[
	[1,2,3,4],
	[5,6,7,8],
	[9,10,11,12]
]

travel(data)

transpose=[[row[i] for row in data] for i in range(4)]
print("after transpose:")
travel(transpose)


input("\n\n 输入任意键退出！")