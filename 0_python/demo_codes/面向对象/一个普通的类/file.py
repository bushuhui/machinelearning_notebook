#!/usr/bin/python
# -*- coding: UTF-8 -*-

def fileOpen(str):
	f=open("./overload.py","r")
	for line in f:
		print(line,end='')

	f.close()
	return


if __name__ == '__main__':
	fileOpen("./overload.py")
	input("\n\n 输入任意键退出！")
