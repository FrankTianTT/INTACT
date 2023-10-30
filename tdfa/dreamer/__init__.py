g = [2] * 7 + [4] + [3, 2, 2, 2, 2, 2, 2, 2, 1, 2]
s = [94, 78, 87, 76, 78, 95, 78, 90, 86, 80, 94, 80, 82, 92, 86, 81, 92, 85]


# grade transition:
# 90-100分为4.0分
# 85-89分为3.7分
# 82-84分为3.3分
# 78-81分为3.0分
# 75-77分为2.7分
# 72-74分为2.3分
# 68-70分为2.0分
# 64-67分为1.5分
# 60-63分为1.0分
# 60分以下为0分
def trans(grade):
    if grade >= 90:
        return 4.0
    elif grade >= 85:
        return 3.7
    elif grade >= 82:
        return 3.3
    elif grade >= 78:
        return 3.0
    elif grade >= 75:
        return 2.7
    elif grade >= 72:
        return 2.3
    elif grade >= 68:
        return 2.0
    elif grade >= 64:
        return 1.5
    elif grade >= 60:
        return 1.0
    else:
        return 0.0


def f1(s, g):
    gpa = 0
    for i in range(len(s)):
        gpa += trans(s[i]) * g[i]
    gpa /= sum(g)
    return gpa


def f2(s, g):
    gpa = 0
    for i in range(len(s)):
        gpa += s[i] * g[i]
    gpa /= sum(g)
    return gpa / 100 * 4


def f3(s, g):
    gpa = 0
    for i in range(len(s)):
        gpa += s[i] * g[i]
    gpa /= sum(g)
    return gpa / 20


print(f1(s, g))
print(f2(s, g))
print(f3(s, g))


print(45 / 221)