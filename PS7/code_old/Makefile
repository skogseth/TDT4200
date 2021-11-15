.PHONY: print


CC  = nvcc
C	= cu
H	= h

MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR := $(shell dirname $(MAKEFILE_DIR))

CFLAGS 	= --compiler-options -Wall -g -I${ROOT_DIR}/includes -I${MAKEFILE_DIR}/includes
LFLAGS  = -g


ifeq ("$(shell uname)", "Darwin")
    LDFLAGS     = -framework Foundation -framework GLUT -framework OpenGL -lm
else
  ifeq ("$(shell uname)", "Linux")
    LDFLAGS     = -L /usr/lib64/ -lglut -lGL -lm -lpthread -lGLU
  endif
endif


PROJECT = morph

SRC_FILES = $(wildcard src/*.${C})
OBJ_FILES = $(patsubst src/%.${C}, build/%.o, ${SRC_FILES})

${PROJECT}: build/${PROJECT}.o $(OBJ_FILES)
		$(CC) -o ${PROJECT} ${CFLAGS} ${LDFLAGS} ${OBJ_FILES}

build/%.o: src/%.${C}
		$(CC) -c $(CFLAGS) -o $@ $<

debug:
	make 'DFLAGS = /usr/lib/debug/malloc.o'

clean:
	rm -f *.o *~ ${PROJECT} core.*
	rm -f *.h.gch

print:
	echo $(ROOT_DIR)
