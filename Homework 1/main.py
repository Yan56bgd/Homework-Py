'''
PRACTICE 2. LOOPS, LISTS
'''

import random
import copy
#
# WHILE LOOP, BASE
#

'''
a = 10
while a > 0:
    print("current value is ", a)
    a = int(input())
else:
    print("The input is finished")
'''
#
# FOR LOOP, BASE
#

#for i in range(1,11,1):
#    print(i)

a = []
for i in range(1,10,2):
    a.append(i)

# print(a)
# print(len(a))

# for val in a:
#    print(val)

# for i in range(len(a)):
#    print(i, a[i])

#for i in range(len(a)):
#    a[i] = a[i] ** 2
#print(a)

#for val in a:
#    print(['*']*val)

#for val in a:
#    while val > 0:
#        print(1)
#        val -= 1
#    print("****************************************")

#n = int(input("Input length of list: "))
#a = int(input("Input min elemnt: "))
#b = int(input("Input max elemnt: "))

#l = []
#for i in range(n):
#   l.append(random.randint(a,b))

#l = [random.randint(a,b) for i in range(n)]
#l_0 = [0]*n
#l_0 = [0 for i in range(n)]

#print(l)
#print(l_0)

#for i in range(len(l)):
#    print(l[i] + l[i - 1])
#else:
#    print("FINISHED!")

#copy_l = copy.deepcopy(l)

#a = input().split()
#a = [int(val) for val in a if int(val) >= 2]
#print(a)

#l = [0 for i in range(5)]

#l = [[0 for j in range(5)] for i in range(5)]
#l[2][2] = 1
#for row in l:
#    print(row)

#
# FIND MIN ELEMENT
#

n = 10
l = [random.randint(-10,10) for i in range(n)]

min_el = l[0]
for i in range(1, len(l)):
    if(min_el > l[i]):
        min_el = l[i]

print(l)
print(min(l))
print(min_el)
#print(l)
l.sort()
print(l[0])