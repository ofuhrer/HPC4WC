-include ../config.mk


main: main.cpp kernels.hpp parallel.hpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS) -DREALSIZE=$(REALSIZE)

.PHONY: clean
clean:
	$(RM) main in_field.csv out_field.csv plot.png
