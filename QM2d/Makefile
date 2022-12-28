CXX = g++

CXXFLAGS = -std=c++11 -O2

SOURCES = $(wildcard *.cpp) $(wildcard **/*.cpp)

OBJECTS = $(SOURCES:.cpp=.o)

TARGET = main

$(TARGET) : $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean :
	rm $(wildcard *.o) $(wildcard **/*.o) data.csv main