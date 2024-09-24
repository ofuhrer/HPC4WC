-include ../config.mk


main: main.cu device.cuh host.cuh kernels.cuh utils.cuh
	$(CUDACXX) -o $@ $< $(CUDAFLAGS) -DREALSIZE=$(REALSIZE)

.PHONY: clean
clean:
	$(RM) main in_field.csv out_field.csv plot.png
