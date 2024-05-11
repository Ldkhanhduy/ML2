import numpy as pro
import random

am=abs(int(input("Nhap so hang cho ma tran A:")))
an=abs(int(input("Nhap so cot cho ma tran A:")))
bm=abs(int(input("Nhap so hang cho ma tran B:")))
bn=abs(int(input("Nhap so cot cho ma tran B:")))
while an!=bm:
    print("So cot cua ma tran A phai bang so hang cua \nma tran B moi co the thuc hien nhan hai ma tran duoc")
    an = abs(int(input("Nhap lai so cot cho ma tran A:")))
    bm = abs(int(input("Nhap lai so hang cho ma tran B:")))
else:
    matranketqua = []
    ptumatranketqua = 0
    ptunhantrongptu = 0
    A = random.choices(range(-2, 2), k=am * an)
    B = random.choices(range(-2, 2), k=bm * bn)
    A = pro.reshape(A, (am, an))
    B = pro.reshape(B, (bm, bn))
    print("A=",A)
    print("B=",B)
    for k in range(len(A)):
        matranketqua.insert(k,[])
    for I in range(bn):
        for i in range(len(A)):
            for j in range(len(A[i])):
                ptunhantrongptu=A[i,j]*B[j,I]
                ptumatranketqua+=ptunhantrongptu
            matranketqua[i].insert(I,ptumatranketqua)
            ptumatranketqua=0
    matranketqua = pro.reshape(matranketqua,(am,bn))
    print("A*B=",matranketqua)