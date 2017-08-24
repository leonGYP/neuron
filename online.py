import string
import datetime

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
    # print(len(req), d, s, o)
    return float(len(req)), float(d), float(s), float(o)


def train(v, r):
    for i in range(len(r)):
        if v[i][0] == 0:
            if r[i] != 0 and i > 0:
                v[i][2] = r[i - 1] / r[i]
            v[i][1] = r[i]
        else:
            v[i][1] = (v[i][1] * v[i][0] + r[i]) / (v[i][0] + 1)
            if r[i] != 0 and i > 0:
                v[i][2] = (v[i][2] * v[i][0] + (r[i - 1] / r[i])) / (v[i][0] + 1)
        v[i][0] += 1


def check(v, r, k, l):
    s = []
    for i in range(len(r)):
        if abs(r[i] - v[i][1]) / v[i][1] < k or abs(r[i] - v[i][1]) <= l:
            s.append(1)
        else:
            s.append(0)
        if r[i] != 0 and i > 0:
            p = r[i - 1] / r[i]
            if abs(p - v[i][2]) / v[i][2] < k:
                s.append(1)
            else:
                s.append(0)
    # print(s)
    # print(sum(s) / float(len(s)))


# [[count, avg, prop]]
vect = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
for i in x:
    train(vect, resolve(i))
# print(vect)
for i in x:
    start = datetime.datetime.now()
    check(vect, resolve(i), 0.1, 2)
    print(datetime.datetime.now() - start)
