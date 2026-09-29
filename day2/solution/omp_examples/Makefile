all: $(patsubst %.cpp, %.out, $(wildcard *.cpp))

%.out: %.cpp
	g++ -fopenmp $< -o $@ 

clean:
	rm *.out