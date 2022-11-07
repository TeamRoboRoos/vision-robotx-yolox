.PHONY: all

CC      = gcc
CFLAGS  = -g -Wall
RM      = rm -f
LIBS 	= -lzmq -ljansson
SOURCE  = utils.h utils.c example_client.c
EXE     = example_client

all: $(EXE)

$(EXE):
	$(CC) $(CFLAGS) $(SOURCE) $(LIBS) -o $(EXE)

run:
	./$(EXE)

clean:
	$(RM) example_client