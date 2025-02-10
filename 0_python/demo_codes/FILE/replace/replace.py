#!/usr/bin/python3

def file_replace(file_name, rep_word, new_word):

    f_read = open(file_name)

    content = []
    count = 0
    
    for eachline in f_read:
        if rep_word in eachline:
            count = count+eachline.count(rep_word)
            eachline = eachline.replace(rep_word, new_word)

        content.append(eachline)

    decide = input('\n文件 %s 中共有%s个【%s】\n您确定要把所有的【%s】替换为【%s】吗?\n【YES/NO】:' % (file_name, count, rep_word, rep_word, new_word))

    if decide in ['YES', 'Yes', 'yes']:
        f_write = open(file_name, 'w')
        f_write.writelines(content)
        f_write.close()
        f_read.close()

if __name__ == '__main__':
    file_name = input('请输入文件名:')
    rep_word = input('请输入需要替换的单词或字符:')
    new_word = input('请输入新的单词或字符:')
    file_replace(file_name, rep_word, new_word)
