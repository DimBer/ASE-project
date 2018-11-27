# Makefile

IDIR = include
CC = gcc
CFLAGS = -I$(IDIR) -O0 -lblas -lpthread -lm -Wall  -g

SDIR = src
ODIR = obj

_SOURCES = EMB.c	csr_handling.c	emb_IO.c	emb_mem.c	las2.c svdlib.c	svdutil.c my_svd.c split_gen.c emb_fit.c train.c
SOURCES = $(patsubst %,$(SDIR)/%,$(_SOURCES))

_OBJECTS = $(_SOURCES:.c=.o)
OBJECTS = $(patsubst %,$(ODIR)/%,$(_OBJECTS))

_DEPS = emb_defs.h emb_IO.h emb_mem.h csr_handling.h svdlib.h svdutil.h my_svd.h split_gen.h emb_fit.h train.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

EXE = EMB_min

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXE): $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS)
	rm -f $(ODIR)/*.o
