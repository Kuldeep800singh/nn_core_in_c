CC = gcc
CFLAGS = -lm

all: login_gate

login_gate: clean
	$(CC) -o build/login_gate main.c $(CFLAGS) 

run:
	./build/login_gate

clean:
	rm -rf build
	mkdir build

