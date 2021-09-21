.PHONY: clean
.PHONY: print

######################
## GLOBAL VARIABLES ##
######################
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


############################
## PARALLEL MORPH PROGRAM ##
############################
PARALLEL_CC:=mpicc
PARALLEL_FLAGS:=-lm -g

PARALLEL_SRC_FILES:=$(wildcard src/*.c)
PARALLEL_OBJ_FILES:=$(patsubst src/%.c,build/%.o,$(PARALLEL_SRC_FILES))

PARALLEL_INCLUDE_PATHS:=-I$(ROOT_DIR)/inc

build/%.o: src/%.c
	$(PARALLEL_CC) $< $(PARALLEL_FLAGS) $(PARALLEL_INCLUDE_PATHS) -c -o $@

main: $(PARALLEL_OBJ_FILES)
	$(PARALLEL_CC) $^ $(PARALLEL_FLAGS) -o $@

print:
	echo $(ROOT_DIR)
	echo $(SERIAL_OBJ_FILES)
	echo $(PARALLEL_OBJ_FILES)

clean:
	rm -rf build/*.o
