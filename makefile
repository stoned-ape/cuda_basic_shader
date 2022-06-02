a.exe: main.cu makefile
	nvcc main.cu --library cuda  

run: a.exe
	./a.exe