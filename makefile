CC=gcc
BIN=bin
OBJ=obj

all: bin/nn

obj/matrix.o: src/matrix.c src/matrix.h
	[ -d ${OBJ} ] || mkdir -p $(OBJ)
	$(CC) -c -o obj/matrix.o src/matrix.c

obj/neural.o: src/neural.c src/neural.h
	[ -d ${OBJ} ] || mkdir -p $(OBJ)
	$(CC) -c -o obj/neural.o src/neural.c

obj/activation.o: src/activation.c src/activation.h
	[ -d ${OBJ} ] || mkdir -p $(OBJ)
	$(CC) -c -o obj/activation.o src/activation.c

obj/image.o: src/image.c src/image.h
	[ -d ${OBJ} ] || mkdir -p $(OBJ)
	$(CC) -c -o obj/image.o src/image.c

obj/test.o: src/test.c src/test.h
	[ -d ${OBJ} ] || mkdir -p $(OBJ)
	$(CC) -c -o obj/test.o src/test.c

bin/nn: src/main.c obj/matrix.o obj/neural.o obj/activation.o obj/image.o obj/test.o
	[ -d ${BIN} ] || mkdir -p $(BIN)
	$(CC) -o bin/nn src/main.c obj/matrix.o obj/neural.o obj/activation.o obj/image.o obj/test.o

clean:
	rm -f bin/nn
	rm -f obj/*.o
	rm -f -d bin
	rm -f -d obj