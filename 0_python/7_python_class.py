# author:gyx
# usage:this file is to learn instance application of class in python

# 类(Class): 用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例
# 对象（Object）：通过类定义的数据结构实例（Instance），对象包括两类成员（类变量和实例变量）和方法。
# 例如我们定义了一个Person类，而具体的人，比如小明，小黄就是Person类的实例
# 属性: 描述该类具有的特征，比如人类具备的属性，身份证，姓名，性别，身高，体重等等都是属性
# 方法: 该类对象的行为，例如这个男孩会打篮球，那个女孩会唱歌等等都是属于方法，常常通过方法改变一些类中的属性值


class student:
    student_info = []

    def __init__(self, name, id, age):
        self.name = name  # self指的是类本身
        self.__id = id  # 将id这一属性私有化
        self.age = age
        self.student_info.append(self.name)
        self.student_info.append(self.__id)
        self.student_info.append(self.age)

    def grade(self):
        return self.__id[0:4]


student1 = student('gyx', '2021300869', 20)  # student类的实例变量
print(student1.student_info)
print(student1.grade())


class Engineering_student(student):
    def speciality(self, speciality):
        self.speciality = speciality
        print(self.name, 'is a', self.speciality, 'student')

    def grade(self):  # 方法重写
        # return self.id# 将id属性私有化后,就已经不能调用id这一属性了
        return self.speciality


student2 = Engineering_student('gyx', '2021300869', 20)
student2.speciality('航空')
print(student2.grade())
