#!/usr/bin/python3


def getfile_fix(filename):
    return filename[filename.rfind('.')+1:]
print(getfile_fix('runoob.txt'))
