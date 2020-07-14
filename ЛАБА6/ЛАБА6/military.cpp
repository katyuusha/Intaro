#include "military.h"

void military::printFunction()
{
		cout << "Тип корабля - военный " << endl;
		cout << "Название корабля: " << name << endl;
		cout << "id номер корабля: " << id_num << endl;
		cout << "Тип техники: " << typeOfEquipment << endl;
		cout << "Максимальное количество техники: " << maxNum << endl;
}

void military::setParam2(string typeOfEquipment, int maxNum)
{
	this->typeOfEquipment = typeOfEquipment;
	this->maxNum = maxNum;
}

void military::add()
{
	cout << "Название корабля - ";
	cin >> name;
	cout << "id номер корабля - ";
	cin >> id_num;
	cout << "Тип техники - ";
	cin >> typeOfEquipment;
	cout << "Максимальное количество техники - ";
	cin >> maxNum;
}

void military::show()
{
	cout << name << " " << id_num << " " << typeOfEquipment << " " << maxNum << endl;
}
