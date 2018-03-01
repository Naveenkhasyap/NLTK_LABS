
def save_file(student):
    try:
        f = op

def read_file():
    try:
        f = open("students.txt","r")
        for students in read_students(f):
            students.append(student)
        f.close()
    except Exception:
        print("coudnt read file")

def read_students(f):
    for line in f:
        yield line
