#include "Header1.h"

void cargo::printFunction()
{
	cout << "Тип корабля - грузовой " << endl;
	cout << "Название корабля: " << name << endl;
	cout << "id номер корабля: " << id_num << endl;
	cout << "Тип груза: " << typeOfCargo << endl;
	cout << "Максимальный объем груза: " << maxVolume << endl;
}

void cargo::setParam1(string typeOfCargo, float maxVolume)
{
	this->typeOfCargo = typeOfCargo;
	this->maxVolume = maxVolume;
}

void cargo::add()
{
	cout << "Название корабля - ";
	cin >> name;
	cout << "id номер корабля - ";
	cin >> id_num;
	cout << "Тип груза - ";
	cin >> typeOfCargo;
	cout << "Максимальный объем груза - ";
	cin >> maxVolume;
}

void cargo::show()
{
	cout << name << " " << id_num << " " << typeOfCargo << " " << maxVolume << endl;
}






