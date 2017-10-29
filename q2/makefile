all: main.py
	cython3 --embed -o main.c main.py
	gcc -O3 -I /usr/include/python3.5m/ -o main.o main.c -lpython3.5m -lpthread -lm -lutil -ldl
