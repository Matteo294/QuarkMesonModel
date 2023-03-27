CXX = g++

CXXFLAGS = -std=c++17 -O3 -I ../../eigen-3.4.0

SOURCES = $(wildcard *.cpp) $(wildcard **/*.cpp)

OBJECTS = $(SOURCES:.cpp=.o)

TARGET = main

$(TARGET) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean :
	rm $(wildcard *.o) $(wildcard **/*.o) data.csv main