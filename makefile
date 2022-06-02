a.exe: main.cu makefile
	nvcc main.cu

run: a.exe
	./a.exe