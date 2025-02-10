#!/usr/bin/python3

for i in range(1,5):
	for j in range(1,5):
		for k in range(1,5):
			if (i!=j) and (j!=k) and (k!=i):
				 print (i,j,k)



input("\n\n输入任意键退出！")
