#pragma once
#include "ship.h"

using namespace std;

//порожденный класс для военного корабля
class military :public ship {
protected:
	string typeOfEquipment;
	int maxNum;
public:
	string getValues5() { return typeOfEquipment; };
	int getValues6() { return maxNum; };
	void printFunction() override;
	void setParam2(string typeOfEquipment, int maxNum);	
	void add() override;
	void show() override;
};