import time
import string

x = open("request.txt").readlines()


def resolve(req):
    d = s = o = 0
    for i in range(len(req)):
            if req[i] in string.digits:
                d += 1
            elif req[i] in string.uppercase:
                s += 1
            elif req[i] in string.lowercase:
                s += 1
            else:
                o += 1
    print(d, s, o)
    return float(d), float(s), float(o)


def train(v, r):
    for i in range(len(r)):
        if v[i][3] == 0:
            if r[i] != 0 and i > 0:
                v[i][4] = v[i][5] = r[i - 1] / r[i]
            v[i][0] = v[i][1] = r[i]
            v[i][2] = r[i]
        else:
            if r[i] < v[i][0]:
                v[i][0] = r[i]
            elif r[i] > v[i][1]:
                v[i][1] = r[i]
            if r[i] != 0 and i > 0:
                p = r[i - 1] / r[i]
                if p < v[i][4]:
                    v[i][4] = p
                elif p > v[i][5]:
                    v[i][5] = p
            v[i][2] = (v[i][3] * v[i][2] + r[i]) / (v[i][3] + 1)
        v[i][3] += 1


def check(v, r, k):
    s = []
    o = []
    for i in range(len(r)):
        if abs(r[i] - v[i][2]) / v[i][2] < k:
            s.append(1)
        else:
            s.append(0)
        if r[i] != 0 and i > 0:
            p = r[i - 1] / r[i]
            if p > v[i][4] and p < v[i][5]:
                o.append(1)
            else:
                o.append(0)
    print(s, o)



# [[min_value, max_value, avg, count, min_prop, max_prop]]
vect = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
for i in x:
    train(vect, resolve(i))
print(vect)
for i in x:
    check(vect, resolve(i), 0.3)
