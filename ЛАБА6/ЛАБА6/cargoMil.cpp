#include "cargoMil.h"

void cargoMil::add()
{
	cout << "Название корабля - ";
	cin >> name;
	cout << "id номер корабля - ";
	cin >> id_num;
	cout << "Тип груза - ";
	cin >> typeOfCargo;
	cout << "Максимальный объем груза - ";
	cin >> maxVolume;
	cout << "Тип техники - ";
	cin >> typeOfEquipment;
	cout << "Максимальное количество техники - ";
	cin >> maxNum;
}

void cargoMil::show()
{
	cout << name << " " << id_num << " " << typeOfCargo << " " << maxVolume << " " << typeOfEquipment << " " << maxNum << endl;
}
