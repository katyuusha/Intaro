#include "Header1.h"

void cargo::printFunction()
{
	cout << "��� ������� - �������� " << endl;
	cout << "�������� �������: " << name << endl;
	cout << "id ����� �������: " << id_num << endl;
	cout << "��� �����: " << typeOfCargo << endl;
	cout << "������������ ����� �����: " << maxVolume << endl;
}

void cargo::setParam1(string typeOfCargo, float maxVolume)
{
	this->typeOfCargo = typeOfCargo;
	this->maxVolume = maxVolume;
}

void cargo::add()
{
	cout << "�������� ������� - ";
	cin >> name;
	cout << "id ����� ������� - ";
	cin >> id_num;
	cout << "��� ����� - ";
	cin >> typeOfCargo;
	cout << "������������ ����� ����� - ";
	cin >> maxVolume;
}

void cargo::show()
{
	cout << name << " " << id_num << " " << typeOfCargo << " " << maxVolume << endl;
}






