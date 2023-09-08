#!/usr/bin/python3

income=int(input('输入净利润:'))

#rate={1000000:0.01,600000:0.015,400000:0.03,200000:0.05,100000:0.075,0:0.1}

level=[1000000,600000,400000,200000,100000,0]
rate=[0.01,0.015,0.03,0.05,0.075,0.1]

ret = 0
for i in range(0,6):
	if(income>level[i]):
		ret += (income-level[i])*rate[i]
		income=level[i]

print("共发奖金:",ret)
input("\n\n输入任意键退出！")

"""
#!/user/bin/env python
# coding=utf-8

# 计算公司的年度奖金，单位：万元
num = int(raw_input("请输入今年的公司利润："))
obj = {100: 0.01, 60: 0.015, 40: 0.03, 20: 0.05, 10: 0.075, 0: 0.1}
keys = obj.keys()
keys.sort()
keys.reverse()
r = 0
for key in keys:
    if num > key:
        r += (num - key) * obj.get(key)
        num = key
print "今年的奖金为：", r, "万元。"
"""