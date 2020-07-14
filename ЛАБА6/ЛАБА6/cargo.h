#pragma once
#include "ship.h"

#include <iostream>
#include <iomanip>
#include <string>
using namespace std;

//порожденный класс для грузового корабля
class cargo :public ship {
protected:
	string typeOfCargo;
	float maxVolume;
public:
	string getValues3() { return typeOfCargo; };
	float getValues4() { return maxVolume; };
	void printFunction() override;
	void setParam1(string typeOfCargo, float maxVolume);
	void add() override;
	void show() override;
};