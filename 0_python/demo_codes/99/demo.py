#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
for i in range(1, 10):
    if(i>1):
	    print(end='\n') 
    for j in range(1, i+1):
        print ("%d*%d=%d" % (i, j, i*j),end=' ')




input("\n\n输入任意键退出！")