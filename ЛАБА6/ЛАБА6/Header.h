#pragma once

#include <iostream>
#include <iomanip>
#include <string>
using namespace std;

//базовый класс корабль
class ship {
protected:
	string name;
	int id_num;
public:
	void setParameters (string name, int id_num);
	int getValues1() { return id_num; };
	string getValues2() { return name; };										
	virtual void printFunction() = 0;
	virtual void add() = 0;
	virtual void show() = 0;
};
