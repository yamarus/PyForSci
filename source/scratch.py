import keyword
print(len(keyword.kwlist))
x=1.1+3.2j
print(x.real)

x = [[3, 'строка', 0],
     [1, -13.3+2j, -4],
     [7, 1.347e-7, 2]]
x=0
y=0
z=0

s = {'o', 'l', 4}

s=str(1+4)
print(s)
print(list('text'))
m=0
atom = 'H'
if atom == 'C':     m = 12.
elif atom == 'H':   m = 1.
elif atom == 'O':   m =16.
print(m)
i = 0
while i < 10:
    print('a' + str(i))
    i += 2
range()